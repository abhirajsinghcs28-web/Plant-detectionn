const path = require('path');
const fs = require('fs');
const Prediction = require('../models/Prediction');
const aiService = require('../services/aiService');

// Validated crop species supported by the research project's Vision Transformer domain heads
const ALLOWED_CROPS = ['Tomato', 'Potato', 'Rice', 'Wheat', 'Maize', 'Crop Leaf', 'General'];

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

    // 2. Validate crop species against domain whitelist if provided
    let crop = 'Crop Leaf';
    if (req.body.crop && req.body.crop.trim() !== '') {
      const trimmedCrop = req.body.crop.trim();
      const matchedCrop = ALLOWED_CROPS.find(c => c.toLowerCase() === trimmedCrop.toLowerCase());
      if (!matchedCrop) {
        return res.status(400).json({
          success: false,
          error: `Invalid crop species "${trimmedCrop}". Allowed options: ${ALLOWED_CROPS.filter(c => c !== 'Crop Leaf' && c !== 'General').join(', ')}, or Crop Leaf.`,
        });
      }
      crop = matchedCrop;
    }

    const imagePathOnDisk = req.file.path;
    const publicImageUrl = `/uploads/${path.basename(req.file.filename)}`;

    // 3. Call modular AI inference service (routes to real endpoint or mock development mode)
    const aiResult = await aiService.predictDisease(imagePathOnDisk, crop);

    // Compute numerical severity score (for DB schema backwards compatibility)
    const numericSeverity = typeof aiResult.severityScore === 'number' 
      ? aiResult.severityScore 
      : (typeof aiResult.severity === 'number' 
          ? aiResult.severity 
          : (aiResult.severityLevel?.toLowerCase() === 'healthy' ? 0 
             : aiResult.severityLevel?.toLowerCase() === 'mild' ? 15 
             : aiResult.severityLevel?.toLowerCase() === 'moderate' ? 35 : 75));

    // Normalize confidence percentage (e.g. 97.2)
    const confidenceVal = typeof aiResult.confidence === 'number'
      ? (aiResult.confidence <= 1.0 ? Number((aiResult.confidence * 100).toFixed(1)) : Number(aiResult.confidence.toFixed(1)))
      : 95.0;

    // 4. Save prediction record to MongoDB
    let savedRecord = null;
    try {
      savedRecord = await Prediction.create({
        crop: aiResult.crop || crop,
        disease: aiResult.disease || 'Severity Analysis Only',
        severity: numericSeverity,
        confidence: confidenceVal > 1 ? confidenceVal / 100 : confidenceVal,
        severityLevel: aiResult.severityLevel || (typeof aiResult.severity === 'string' ? aiResult.severity : 'Moderate'),
        probabilities: aiResult.probabilities || null,
        imagePath: publicImageUrl,
        originalFilename: req.file.originalname,
        segmentationMask: aiResult.segmentationMask || null,
        maskType: aiResult.maskType || null,
        isMock: Boolean(aiResult.isMock),
      });
    } catch (dbError) {
      console.warn('Warning: Could not persist prediction to MongoDB:', dbError.message);
    }

    // 5. Send standardized response matching Member 2/3 and user requirements
    return res.status(200).json({
      success: true,
      data: {
        id: savedRecord ? savedRecord._id : null,
        severity: aiResult.severity || aiResult.severityLevel,
        severityLevel: aiResult.severityLevel || (typeof aiResult.severity === 'string' ? aiResult.severity : 'Moderate'),
        severityScore: numericSeverity,
        confidence: confidenceVal,
        probabilities: aiResult.probabilities || {
          healthy: aiResult.severityLevel === 'Healthy' ? confidenceVal : 1.0,
          mild: aiResult.severityLevel === 'Mild' ? confidenceVal : 2.0,
          moderate: aiResult.severityLevel === 'Moderate' ? confidenceVal : 5.0,
          severe: aiResult.severityLevel === 'Severe' ? confidenceVal : 92.0,
        },
        crop: aiResult.crop || crop,
        disease: aiResult.disease || 'Severity Analysis Only',
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
