import React from 'react';
import ResultCard from '../components/ResultCard';
import SegmentationViewer from '../components/SegmentationViewer';

/**
 * Dedicated Results Inspection Page
 */
export default function Results({ result, onBackToAnalyze }) {
  if (!result) {
    return (
      <div className="card">
        <div className="empty-state">
          <p>No active prediction result to display.</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onBackToAnalyze}
            style={{ marginTop: '12px' }}
          >
            Go to Leaf Analysis
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="results-toolbar">
        <button type="button" className="btn btn-secondary" onClick={onBackToAnalyze}>
          ← Back to Analysis
        </button>
        <div className="specimen-id-tag">
          Specimen ID: <code>{result.id || result._id || 'LIVE_EVAL'}</code>
        </div>
      </div>

      <ResultCard result={result} />

      <SegmentationViewer
        originalImageUrl={result.imagePath}
        segmentationMaskUrl={result.segmentationMask}
        maskType={result.maskType}
      />
    </div>
  );
}
