import React from 'react';

/**
 * Visual Severity Gauge and Metric Display
 */
export default function SeverityDisplay({ severity, severityLevel, confidence }) {
  // Determine color class based on severity level
  const getBadgeClass = (level) => {
    switch (level?.toLowerCase()) {
      case 'healthy':
        return 'badge-healthy';
      case 'mild':
        return 'badge-mild';
      case 'moderate':
        return 'badge-moderate';
      case 'severe':
        return 'badge-severe';
      default:
        return 'badge-moderate';
    }
  };

  const getBarColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'healthy':
        return '#16a34a';
      case 'mild':
        return '#2563eb';
      case 'moderate':
        return '#d97706';
      case 'severe':
        return '#dc2626';
      default:
        return '#d97706';
    }
  };

  const formattedSeverity = typeof severity === 'number' ? severity.toFixed(1) : '0.0';
  const formattedConfidence =
    typeof confidence === 'number' ? `${(confidence * 100).toFixed(0)}%` : 'N/A';

  return (
    <div className="severity-display-box">
      <div className="severity-header-row">
        <div>
          <div className="metric-label">ESTIMATED SEVERITY</div>
          <div className="severity-large-value">{formattedSeverity}%</div>
        </div>
        <div className="severity-meta-right">
          <div>
            <span className="metric-label">CLASSIFICATION: </span>
            <span className={`badge ${getBadgeClass(severityLevel)}`}>
              {severityLevel || 'Moderate'}
            </span>
          </div>
          <div style={{ marginTop: '6px' }}>
            <span className="metric-label">MODEL CONFIDENCE: </span>
            <strong className="confidence-value">{formattedConfidence}</strong>
          </div>
        </div>
      </div>

      {/* Severity Progress Bar */}
      <div className="severity-bar-container">
        <div
          className="severity-bar-fill"
          style={{
            width: `${Math.min(Math.max(severity, 0), 100)}%`,
            backgroundColor: getBarColor(severityLevel),
          }}
        ></div>
      </div>

      {/* Scientific scale guide */}
      <div className="severity-scale-ticks">
        <span className="tick-label">0% (Healthy)</span>
        <span className="tick-label">20% (Mild)</span>
        <span className="tick-label">50% (Moderate)</span>
        <span className="tick-label">100% (Severe)</span>
      </div>
    </div>
  );
}
