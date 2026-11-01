const mongoose = require('mongoose');

/**
 * Prediction Schema
 * Stores each leaf disease analysis request, prediction outputs, and file references.
 * Contains isMock flag to distinguish development mock data from real Vision Transformer inference.
 */
const predictionSchema = new mongoose.Schema({
  crop: {
    type: String,
    default: 'Crop Leaf',
    trim: true,
  },
  disease: {
    type: String,
    default: 'Severity Analysis Only',
    trim: true,
  },
  severity: {
    type: Number,
    default: 0,
    min: [0, 'Severity cannot be negative'],
    max: [100, 'Severity cannot exceed 100%'],
  },
  confidence: {
    type: Number,
    required: [true, 'Confidence score is required'],
  },
  severityLevel: {
    type: String,
    enum: ['Healthy', 'Mild', 'Moderate', 'Severe', 'healthy', 'mild', 'moderate', 'severe'],
    default: 'Moderate',
  },
  probabilities: {
    type: Object,
    default: null,
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
    default: null,
  },
  maskType: {
    type: String,
    enum: ['segmentation_mask', 'attention_heatmap', null],
    default: null,
  },
  isMock: {
    type: Boolean,
    default: false,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('Prediction', predictionSchema);
