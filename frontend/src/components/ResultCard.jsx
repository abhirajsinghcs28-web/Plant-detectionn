import React from 'react';
import SeverityDisplay from './SeverityDisplay';

/**
 * Result Card Component
 * Formatted specifically for Vision Transformer 4-tier severity estimation.
 * Displays:
 * 1. Estimated Leaf Severity (Healthy, Mild, Moderate, Severe)
 * 2. Model Confidence Score (%)
 * 3. Probability distribution across all four classes
 * 4. Contextual explanation
 * 5. Agricultural disclaimer
 */
export default function ResultCard({ result }) {
  if (!result) return null;

  // Extract fields with case-insensitive normalization
  const rawSeverity = result.severityLevel || result.severity || 'Moderate';
  const severityLevel =
    typeof rawSeverity === 'string'
      ? rawSeverity.charAt(0).toUpperCase() + rawSeverity.slice(1).toLowerCase()
      : 'Moderate';

  const confidence = result.confidence;
  const probabilities = result.probabilities || null;
  const timestamp = result.timestamp;
  const severityScore = typeof result.severityScore === 'number' ? result.severityScore : null;
  const necrosisPct = typeof result.necrosisPct === 'number' ? result.necrosisPct : null;
  const chlorosisPct = typeof result.chlorosisPct === 'number' ? result.chlorosisPct : null;
  const modelUsed = result.modelUsed || 'PROPOSED_vit_ce_reg (Vision Transformer v2 - 94.1% Accuracy)';

  // Color mapping
  const getBadgeStyle = (level) => {
    switch (level.toLowerCase()) {
      case 'healthy':
        return { color: '#16a34a', bg: '#dcfce7', border: '#86efac' };
      case 'mild':
        return { color: '#2563eb', bg: '#dbeafe', border: '#93c5fd' };
      case 'moderate':
        return { color: '#d97706', bg: '#fef3c7', border: '#fcd34d' };
      case 'severe':
        return { color: '#dc2626', bg: '#fee2e2', border: '#fca5a5' };
      default:
        return { color: '#d97706', bg: '#fef3c7', border: '#fcd34d' };
    }
  };

  const badgeStyle = getBadgeStyle(severityLevel);

  // Results explanation based on model prediction
  const getSeverityExplanation = (level) => {
    switch (level.toLowerCase()) {
      case 'healthy':
        return 'The model detected little to no visible symptoms in the uploaded leaf image.';
      case 'mild':
        return 'The model detected early or limited visible symptoms.';
      case 'moderate':
        return 'The model detected noticeable symptoms that may require further attention.';
      case 'severe':
        return 'The model detected extensive visible symptoms.';
      default:
        return 'The model evaluated the visible foliar symptoms of the uploaded leaf.';
    }
  };

  const formattedConfidence =
    typeof confidence === 'number'
      ? `${confidence <= 1.0 ? (confidence * 100).toFixed(1) : confidence.toFixed(1)}%`
      : 'N/A';

  return (
    <div className="card result-card">
      <div className="result-card-header">
        <div className="header-meta">
          <span className="analysis-tag">VISION TRANSFORMER INFERENCE</span>
          <span className="timestamp-text">
            {timestamp ? new Date(timestamp).toLocaleString() : new Date().toLocaleString()}
          </span>
        </div>
      </div>

      {/* Primary Severity Banner */}
      <div
        className="estimated-severity-banner"
        style={{
          backgroundColor: badgeStyle.bg,
          borderColor: badgeStyle.border,
        }}
      >
        <div className="banner-left">
          <span className="banner-subtitle">ESTIMATED LEAF SEVERITY</span>
          <h2 className="banner-title" style={{ color: badgeStyle.color }}>
            {severityLevel}
          </h2>
        </div>
        <div className="banner-right">
          <span className="confidence-pill-label">Model Confidence</span>
          <span className="confidence-pill-val" style={{ color: badgeStyle.color }}>
            {formattedConfidence}
          </span>
        </div>
      </div>

      {/* Quantitative Continuous Symptom Metrics (From PROPOSED_vit_ce_reg v2) */}
      {severityScore !== null && (
        <div className="symptom-metrics-banner">
          <div className="symptom-metric-chip">
            <span className="metric-chip-label">Symptom Coverage:</span>
            <span className="metric-chip-value"><strong>{severityScore}%</strong></span>
          </div>
          {necrosisPct !== null && (
            <div className="symptom-metric-chip">
              <span className="metric-chip-label">Necrosis (Dark Lesion):</span>
              <span className="metric-chip-value">{necrosisPct}%</span>
            </div>
          )}
          {chlorosisPct !== null && (
            <div className="symptom-metric-chip">
              <span className="metric-chip-label">Chlorosis (Yellowing):</span>
              <span className="metric-chip-value">{chlorosisPct}%</span>
            </div>
          )}
          <div className="symptom-metric-chip model-chip">
            <span className="metric-chip-label">Model:</span>
            <span className="metric-chip-value">PROPOSED ViT (v2 - 94.1% Acc)</span>
          </div>
        </div>
      )}

      {/* Visual Severity Indicator & 4-Class Breakdown */}
      <SeverityDisplay
        severityLevel={severityLevel}
        confidence={confidence}
        probabilities={probabilities}
      />

      {/* Results Explanation Section */}
      <div className="results-explanation-card">
        <h4 className="explanation-title">Analysis Assessment</h4>
        <p className="explanation-text">{getSeverityExplanation(severityLevel)}</p>
      </div>

      {/* Agricultural Disclaimer */}
      <div className="agricultural-disclaimer-box">
        <strong>Notice & Disclaimer:</strong> This is an AI-based severity estimate using a Vision Transformer model and should not be treated as a professional agricultural diagnosis. Field conditions, lighting variations, and cultivar differences can impact visual assessment.
      </div>
    </div>
  );
}
