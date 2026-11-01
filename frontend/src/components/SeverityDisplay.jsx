import React from 'react';

/**
 * Visual Severity Indicator & 4-Class Probability Distribution
 * Implements:
 * 1. 4-level linear severity indicator: Healthy -> Mild -> Moderate -> Severe
 * 2. Highlight on predicted severity tier
 * 3. Probability distribution progress bars for all 4 classes
 */
export default function SeverityDisplay({ severityLevel, confidence, probabilities }) {
  const tiers = [
    { key: 'healthy', label: 'Healthy', color: '#16a34a', bgLight: '#dcfce7', borderColor: '#86efac' },
    { key: 'mild', label: 'Mild', color: '#2563eb', bgLight: '#dbeafe', borderColor: '#93c5fd' },
    { key: 'moderate', label: 'Moderate', color: '#d97706', bgLight: '#fef3c7', borderColor: '#fcd34d' },
    { key: 'severe', label: 'Severe', color: '#dc2626', bgLight: '#fee2e2', borderColor: '#fca5a5' },
  ];

  const currentLevelNorm = (severityLevel || 'Moderate').toLowerCase();

  // Normalize probabilities into { healthy: float, mild: float, moderate: float, severe: float }
  const getProbabilityValue = (key) => {
    if (!probabilities) {
      return key === currentLevelNorm ? (confidence || 95.0) : 0.0;
    }
    // Check lowercase, uppercase, and capitalized keys
    const val =
      probabilities[key] ??
      probabilities[key.toUpperCase()] ??
      probabilities[key.charAt(0).toUpperCase() + key.slice(1)] ??
      0;

    // Normalize: if in [0, 1] range, convert to percentage
    return typeof val === 'number' && val <= 1.0 ? Number((val * 100).toFixed(1)) : Number(val.toFixed(1));
  };

  const formattedConfidence =
    typeof confidence === 'number'
      ? `${confidence <= 1.0 ? (confidence * 100).toFixed(1) : confidence.toFixed(1)}%`
      : 'N/A';

  return (
    <div className="severity-display-section">
      {/* 4-Level Visual Indicator (Healthy -> Mild -> Moderate -> Severe) */}
      <div className="severity-stepper-container">
        <div className="stepper-label-heading">Severity Progression Spectrum</div>
        <div className="severity-stepper">
          {tiers.map((tier, index) => {
            const isSelected = tier.key === currentLevelNorm;
            return (
              <React.Fragment key={tier.key}>
                <div
                  className={`severity-step-node ${isSelected ? 'active-step' : ''}`}
                  style={{
                    borderColor: isSelected ? tier.color : '#cbd5e1',
                    backgroundColor: isSelected ? tier.bgLight : '#ffffff',
                    color: isSelected ? tier.color : '#64748b',
                  }}
                >
                  <span className="step-number">{index + 1}</span>
                  <span className="step-name">{tier.label}</span>
                  {isSelected && <span className="step-badge-indicator">PREDICTED</span>}
                </div>
                {index < tiers.length - 1 && (
                  <div className="severity-step-arrow" aria-hidden="true">
                    →
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Model Confidence & Probability Distribution Breakdown */}
      <div className="probability-breakdown-card">
        <div className="prob-header-row">
          <span className="prob-title">Four-Class Probability Distribution</span>
          <span className="prob-confidence-badge">
            Model Confidence: <strong>{formattedConfidence}</strong>
          </span>
        </div>

        <div className="probability-bars-list">
          {tiers.map((tier) => {
            const isPredicted = tier.key === currentLevelNorm;
            const prob = getProbabilityValue(tier.key);

            return (
              <div
                key={tier.key}
                className={`probability-bar-row ${isPredicted ? 'row-highlighted' : ''}`}
              >
                <div className="prob-bar-label">
                  <span
                    className="tier-color-dot"
                    style={{ backgroundColor: tier.color }}
                  ></span>
                  <span className="tier-text">{tier.label}</span>
                  {isPredicted && <span className="pred-tag">Target</span>}
                </div>

                <div className="prob-progress-track">
                  <div
                    className="prob-progress-fill"
                    style={{
                      width: `${Math.min(Math.max(prob, 0), 100)}%`,
                      backgroundColor: tier.color,
                    }}
                  ></div>
                </div>

                <div className="prob-value-text" style={{ color: isPredicted ? tier.color : '#334155' }}>
                  {prob.toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
