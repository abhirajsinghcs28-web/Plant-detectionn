import React from 'react';
import SeverityDisplay from './SeverityDisplay';

/**
 * Result Card Component
 * Summarizes the diagnosis, severity metrics, and honest inference provenance.
 */
export default function ResultCard({ result }) {
  if (!result) return null;

  const { crop, disease, severity, confidence, severityLevel, timestamp, isMock } = result;

  return (
    <div className="card result-card">
      <div className="card-title">
        <span>Inference & Severity Assessment</span>
        <span className="timestamp-text">
          {timestamp ? new Date(timestamp).toLocaleString() : new Date().toLocaleString()}
        </span>
      </div>

      {/* Honest Provenance Banner */}
      {isMock ? (
        <div className="alert alert-warning" role="alert">
          <strong>⚠️ DEMO / MOCK INFERENCE:</strong> This assessment was generated using local
          development simulation data (<code>AI_MODE=mock</code>). The actual Vision Transformer
          model from Member 2 has not yet been connected to this backend.
        </div>
      ) : (
        <div className="alert alert-success" role="alert">
          <strong>🟢 REAL AI INFERENCE:</strong> Assessment produced by the connected Vision
          Transformer model inference pipeline.
        </div>
      )}

      <div className="result-summary-grid">
        <div className="summary-item">
          <span className="summary-label">Target Crop Specimen</span>
          <span className="summary-value crop-highlight">{crop}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Diagnosed Condition</span>
          <span className="summary-value disease-highlight">{disease}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Severity Classification</span>
          <span className="summary-value">{severityLevel}</span>
        </div>
      </div>

      {/* Visual Severity and Confidence Breakdown */}
      <SeverityDisplay
        severity={severity}
        severityLevel={severityLevel}
        confidence={confidence}
      />

      {/* Scientific Diagnostics Assessment */}
      <div className="research-notes-box">
        <h4>Agronomic Assessment:</h4>
        <p>
          {disease === 'Healthy Leaf'
            ? `The ${crop} specimen displays uniform foliar coloration with no detectable necrotic lesions or chlorosis. Standard canopy monitoring advised.`
            : `Specimen displays symptoms characteristic of ${crop} ${disease}. The estimated severity index is ${typeof severity === 'number' ? severity.toFixed(1) : 0}% (${severityLevel} tier). Locally validated fungicide/bactericide protocols should be cross-referenced.`}
        </p>
      </div>
    </div>
  );
}
