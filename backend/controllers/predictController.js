const path = require('path');
const fs = require('fs');
const Prediction = require('../models/Prediction');
const aiService = require('../services/aiService');

// Validated crop species supported by the research project's Vision Transformer domain heads
const ALLOWED_CROPS = ['Tomato', 'Potato', 'Rice', 'Wheat', 'Maize'];

/**
 * Handle POST /api/predict
 * Validates upload & crop, invokes AI adapter, stores record in MongoDB, and returns prediction.
 */
async function predictCropLeaf(req, res, next) {
  try {
    // 1. Validate file upload presence
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No leaf image uploaded. Please upload a valid JPG, JPEG, or PNG image.',
      });
    }

    // 2. Validate crop species against domain whitelist
    const crop = req.body.crop ? req.body.crop.trim() : 'Tomato';
    if (!ALLOWED_CROPS.includes(crop)) {
      return res.status(400).json({
        success: false,
        error: `Invalid crop species "${crop}". Allowed options: ${ALLOWED_CROPS.join(', ')}.`,
      });
    }

    const imagePathOnDisk = req.file.path;
    const publicImageUrl = `/uploads/${path.basename(req.file.filename)}`;

    // 3. Call modular AI inference service (routes to real endpoint or mock development mode)
    const aiResult = await aiService.predictDisease(imagePathOnDisk, crop);

    // 4. Save prediction record to MongoDB
    let savedRecord = null;
    try {
      savedRecord = await Prediction.create({
        crop: aiResult.crop,
        disease: aiResult.disease,
        severity: aiResult.severity,
        confidence: aiResult.confidence,
        severityLevel: aiResult.severityLevel,
        imagePath: publicImageUrl,
        originalFilename: req.file.originalname,
        segmentationMask: aiResult.segmentationMask || null,
        maskType: aiResult.maskType || null,
        isMock: Boolean(aiResult.isMock),
      });
    } catch (dbError) {
      console.warn('Warning: Could not persist prediction to MongoDB:', dbError.message);
      // Non-blocking: prediction output is still returned even if DB has a momentary glitch
    }

    // 5. Send standardized response matching Member 2/3 API contract
    return res.status(200).json({
      success: true,
      data: {
        id: savedRecord ? savedRecord._id : null,
        crop: aiResult.crop,
        disease: aiResult.disease,
        severity: aiResult.severity,
        confidence: aiResult.confidence,
        severityLevel: aiResult.severityLevel,
        segmentationMask: aiResult.segmentationMask || null,
        maskType: aiResult.maskType || null,
        isMock: Boolean(aiResult.isMock),
        imagePath: publicImageUrl,
        timestamp: savedRecord ? savedRecord.timestamp : new Date(),
      },
    });
  } catch (error) {
    next(error);
  }
}

/**
 * Handle GET /api/predictions
 * Retrieves previous prediction history, sorted with newest first.
 */
async function getPredictions(req, res, next) {
  try {
    const predictions = await Prediction.find().sort({ timestamp: -1 }).limit(50);

    return res.status(200).json({
      success: true,
      count: predictions.length,
      data: predictions,
    });
  } catch (error) {
    console.warn('MongoDB query warning in getPredictions:', error.message);
    return res.status(200).json({
      success: false,
      count: 0,
      data: [],
      message: 'Database currently offline or unreachable. History cannot be retrieved.',
    });
  }
}

/**
 * Handle DELETE /api/predictions/:id
 * Removes a specific prediction record from MongoDB.
 */
async function deletePrediction(req, res, next) {
  try {
    const { id } = req.params;
    const record = await Prediction.findByIdAndDelete(id);

    if (!record) {
      return res.status(404).json({
        success: false,
        error: 'Prediction record not found.',
      });
    }

    return res.status(200).json({
      success: true,
      message: 'Prediction record deleted successfully.',
    });
  } catch (error) {
    next(error);
  }
}

module.exports = {
  predictCropLeaf,
  getPredictions,
  deletePrediction,
};
