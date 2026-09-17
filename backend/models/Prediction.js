const mongoose = require('mongoose');

/**
 * Prediction Schema
 * Stores each leaf disease analysis request, prediction outputs, and file references.
 * Contains isMock flag to distinguish development mock data from real Vision Transformer inference.
 */
const predictionSchema = new mongoose.Schema({
  crop: {
    type: String,
    required: [true, 'Crop name is required'],
    trim: true,
  },
  disease: {
    type: String,
    required: [true, 'Disease name is required'],
    trim: true,
  },
  severity: {
    type: Number,
    required: [true, 'Severity percentage is required'],
    min: [0, 'Severity cannot be negative'],
    max: [100, 'Severity cannot exceed 100%'],
  },
  confidence: {
    type: Number,
    required: [true, 'Confidence score is required'],
    min: 0,
    max: 1,
  },
  severityLevel: {
    type: String,
    enum: ['Healthy', 'Mild', 'Moderate', 'Severe'],
    default: 'Moderate',
  },
  imagePath: {
    type: String,
    required: [true, 'Image path is required'],
  },
  originalFilename: {
    type: String,
  },
  segmentationMask: {
    type: String,
    default: null, // Data URI or path to mask image; null if not provided by model
  },
  maskType: {
    type: String,
    enum: ['segmentation_mask', 'attention_heatmap', null],
    default: null,
  },
  isMock: {
    type: Boolean,
    default: false, // True if produced under AI_MODE=mock
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('Prediction', predictionSchema);
