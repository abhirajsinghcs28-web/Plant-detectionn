import React from 'react';
import ResultCard from '../components/ResultCard';

/**
 * Dedicated Results Inspection Page
 * Focused entirely on Vision Transformer Severity Estimation.
 */
export default function Results({ result, onBackToAnalyze }) {
  if (!result) {
    return (
      <div className="card">
        <div className="empty-state">
          <p>No active severity estimation result to display.</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onBackToAnalyze}
            style={{ marginTop: '12px' }}
          >
            Go to Severity Analysis
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="results-toolbar">
        <button type="button" className="btn btn-secondary" onClick={onBackToAnalyze}>
          ← Back to Severity Analysis
        </button>
        <div className="specimen-id-tag">
          Specimen Record: <code>{result.id || result._id || 'LIVE_ANALYSIS'}</code>
        </div>
      </div>

      <ResultCard result={result} />
    </div>
  );
}
