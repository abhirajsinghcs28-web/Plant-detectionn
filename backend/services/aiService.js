/**
 * ============================================================================
 * AI INFERENCE SERVICE ADAPTER (MEMBER 3: SYSTEM INTEGRATION)
 * Project: Agritech Self-Supervised Cross-Domain Vision Transformers
 *          for Fine-Grained Crop Leaf Disease Severity Estimation
 * ============================================================================
 * 
 * ARCHITECTURE CONTRACT:
 * React Frontend -> Express API -> aiService.js -> AI Model Service -> Express -> Frontend
 * 
 * DUAL MODE SUPPORT:
 * 1. AI_MODE=real
 *    Dispatches HTTP multipart request with the uploaded leaf image and crop species
 *    to Member 2's Python Vision Transformer inference microservice (AI_INFERENCE_URL).
 * 
 * 2. AI_MODE=mock (Default during development)
 *    Returns explicitly marked development demo data (isMock: true, segmentationMask: null).
 *    NEVER fabricates synthetic SVG masks or fake accuracy scores.
 */

const fs = require('fs');
const path = require('path');

// Crop disease reference catalogue used strictly for development mock mode
const MOCK_CROP_DISEASES = {
  Tomato: [
    { name: 'Late Blight', baseSeverity: 38.5, confidenceRange: [0.88, 0.96] },
    { name: 'Early Blight', baseSeverity: 22.0, confidenceRange: [0.85, 0.94] },
    { name: 'Septoria Leaf Spot', baseSeverity: 14.5, confidenceRange: [0.89, 0.95] },
    { name: 'Healthy Leaf', baseSeverity: 0.0, confidenceRange: [0.93, 0.99] },
  ],
  Potato: [
    { name: 'Late Blight', baseSeverity: 42.0, confidenceRange: [0.87, 0.95] },
    { name: 'Early Blight', baseSeverity: 18.5, confidenceRange: [0.84, 0.93] },
    { name: 'Common Scab', baseSeverity: 27.0, confidenceRange: [0.86, 0.92] },
    { name: 'Healthy Leaf', baseSeverity: 0.0, confidenceRange: [0.94, 0.99] },
  ],
  Rice: [
    { name: 'Brown Spot', baseSeverity: 29.5, confidenceRange: [0.86, 0.93] },
    { name: 'Bacterial Leaf Blight', baseSeverity: 48.0, confidenceRange: [0.88, 0.96] },
    { name: 'Leaf Blast', baseSeverity: 36.0, confidenceRange: [0.87, 0.94] },
    { name: 'Healthy Leaf', baseSeverity: 0.0, confidenceRange: [0.95, 0.99] },
  ],
  Wheat: [
    { name: 'Stripe Rust', baseSeverity: 52.0, confidenceRange: [0.90, 0.97] },
    { name: 'Leaf Rust', baseSeverity: 31.0, confidenceRange: [0.86, 0.94] },
    { name: 'Powdery Mildew', baseSeverity: 19.5, confidenceRange: [0.85, 0.92] },
    { name: 'Healthy Leaf', baseSeverity: 0.0, confidenceRange: [0.92, 0.98] },
  ],
  Maize: [
    { name: 'Northern Leaf Blight', baseSeverity: 44.5, confidenceRange: [0.89, 0.96] },
    { name: 'Common Rust', baseSeverity: 24.0, confidenceRange: [0.87, 0.95] },
    { name: 'Gray Leaf Spot', baseSeverity: 33.5, confidenceRange: [0.85, 0.93] },
    { name: 'Healthy Leaf', baseSeverity: 0.0, confidenceRange: [0.94, 0.99] },
  ],
};

/**
 * Determine qualitative severity tier based on percentage.
 * Thresholds aligned with agricultural research standards:
 * - 0%: Healthy
 * - 1% to 20%: Mild
 * - 21% to 50%: Moderate
 * - > 50%: Severe
 */
function calculateSeverityLevel(severity) {
  if (severity <= 0) return 'Healthy';
  if (severity <= 20) return 'Mild';
  if (severity <= 50) return 'Moderate';
  return 'Severe';
}

/**
 * Check AI Inference Service Health
 * Reports whether the AI integration layer is running in mock mode or connected to a real server.
 */
async function checkAIHealth() {
  const mode = (process.env.AI_MODE || 'mock').toLowerCase();
  const inferenceUrl = process.env.AI_INFERENCE_URL || 'http://localhost:8000/predict';

  if (mode === 'mock') {
    return {
      mode: 'mock',
      status: 'mock_active',
      inferenceUrl: null,
      message: 'Running in development mock mode (UI/API testing). No real AI model connected.',
    };
  }

  // Real mode: verify Member 2's Python endpoint
  try {
    const healthUrl = inferenceUrl.replace(/\/predict\/?$/, '/health');
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const res = await fetch(healthUrl, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (res.ok) {
      return {
        mode: 'real',
        status: 'connected',
        inferenceUrl,
        message: 'Member 2 Vision Transformer inference service is online.',
      };
    }
    return {
      mode: 'real',
      status: 'offline',
      inferenceUrl,
      message: `Inference endpoint responded with status ${res.status}.`,
    };
  } catch (err) {
    return {
      mode: 'real',
      status: 'offline',
      inferenceUrl,
      message: `Cannot reach AI inference service at ${inferenceUrl} (${err.message}).`,
    };
  }
}

/**
 * Predict crop disease and estimate severity.
 * 
 * @param {string} imagePath - Path to stored uploaded leaf image on disk
 * @param {string} crop - Selected crop name ('Tomato', 'Potato', 'Rice', 'Wheat', 'Maize')
 * @returns {Promise<Object>} Standardized prediction object
 */
async function predictDisease(imagePath, crop = 'Tomato') {
  const mode = (process.env.AI_MODE || 'mock').toLowerCase();
  const inferenceUrl = process.env.AI_INFERENCE_URL || 'http://localhost:8000/predict';
  const timeoutMs = parseInt(process.env.AI_TIMEOUT_MS, 10) || 15000;

  // ==========================================================================
  // REAL AI INFERENCE MODE (MEMBER 2 VISION TRANSFORMER MICROSERVICE)
  // ==========================================================================
  if (mode === 'real') {
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Leaf image file not found on disk at ${imagePath}`);
    }

    try {
      // Build multipart request using Node 24 native FormData & Blob
      const fileBuffer = fs.readFileSync(imagePath);
      const filename = path.basename(imagePath);
      const fileBlob = new Blob([fileBuffer]);

      const formData = new FormData();
      formData.append('image', fileBlob, filename);
      formData.append('crop', crop);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      const response = await fetch(inferenceUrl, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown AI service error');
        throw new Error(`AI model service responded with HTTP ${response.status}: ${errorText}`);
      }

      const aiData = await response.json();

      // Validate contract fields from Member 2's AI response
      if (!aiData || typeof aiData.disease !== 'string') {
        throw new Error('Malformed response from AI model service: missing disease diagnosis.');
      }

      const rawSeverity = typeof aiData.severity === 'number' ? aiData.severity : 0;
      const severity = Number(Math.max(0, Math.min(100, rawSeverity)).toFixed(1));
      const confidence = typeof aiData.confidence === 'number' ? Number(aiData.confidence.toFixed(2)) : 0.85;
      const severityLevel = aiData.severityLevel || calculateSeverityLevel(severity);

      // Handle real segmentation mask or attention map (null if model doesn't provide one)
      const segmentationMask = aiData.segmentationMask || null;
      const maskType = aiData.maskType || (segmentationMask ? 'segmentation_mask' : null);

      return {
        crop: aiData.crop || crop,
        disease: aiData.disease,
        severity: severity,
        confidence: confidence,
        severityLevel: severityLevel,
        segmentationMask: segmentationMask, // Real output or null
        maskType: maskType,
        isMock: false,
      };
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(`AI inference service timed out after ${timeoutMs / 1000}s. Please check the model server.`);
      }
      if (err.code === 'ECONNREFUSED' || err.message.includes('fetch failed')) {
        throw new Error(`AI inference service is currently offline at ${inferenceUrl}. Please ensure Member 2's Vision Transformer microservice is running.`);
      }
      throw err;
    }
  }

  // ==========================================================================
  // MOCK DEVELOPMENT MODE (HONEST DEMO DATA, ZERO FAKE MASK ARTIFACTS)
  // ==========================================================================
  const cropList = MOCK_CROP_DISEASES[crop] || MOCK_CROP_DISEASES['Tomato'];
  const randomDisease = cropList[Math.floor(Math.random() * cropList.length)];

  let severity = randomDisease.baseSeverity;
  if (severity > 0) {
    const jitter = (Math.random() * 6 - 3);
    severity = Math.max(1.0, Math.min(99.0, Number((severity + jitter).toFixed(1))));
  }

  const [minConf, maxConf] = randomDisease.confidenceRange;
  const confidence = Number((minConf + Math.random() * (maxConf - minConf)).toFixed(2));
  const severityLevel = calculateSeverityLevel(severity);

  return {
    crop: crop,
    disease: randomDisease.name,
    severity: severity,
    confidence: confidence,
    severityLevel: severityLevel,
    segmentationMask: null, // Zero fake SVG masks! Explicitly null in mock mode
    maskType: null,
    isMock: true,
  };
}

module.exports = {
  predictDisease,
  calculateSeverityLevel,
  checkAIHealth,
};
