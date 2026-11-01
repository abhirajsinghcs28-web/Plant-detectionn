import React from 'react';

/**
 * Clean History List Component
 * Renders previous predictions stored in MongoDB with actions to view or delete.
 * Transparently tags mock development entries vs real model inference.
 */
export default function HistoryList({
  predictions,
  isLoading,
  onSelectPrediction,
  onDeletePrediction,
  onRefresh,
}) {
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

  return (
    <div className="card history-card">
      <div className="card-title">
        <span>Stored Prediction History (MongoDB)</span>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onRefresh}
          disabled={isLoading}
        >
          {isLoading ? 'Refreshing...' : 'Refresh Records'}
        </button>
      </div>

      {isLoading ? (
        <div className="empty-state">Loading prediction history from database...</div>
      ) : !predictions || predictions.length === 0 ? (
        <div className="empty-state">
          <p>No prediction records found in MongoDB.</p>
          <p className="empty-hint">
            Analyze a leaf specimen from the <strong>Analyze Severity</strong> tab to log your first
            result.
          </p>
        </div>
      ) : (
        <div className="table-responsive">
          <table className="research-table">
            <thead>
              <tr>
                <th>Thumbnail</th>
                <th>Estimated Severity</th>
                <th>Model Confidence</th>
                <th>Probability Breakdown</th>
                <th>Inference Mode</th>
                <th>Date & Time</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((item) => {
                const level = item.severityLevel || (typeof item.severity === 'string' ? item.severity : 'Moderate');
                const conf = item.confidence
                  ? `${item.confidence <= 1.0 ? (item.confidence * 100).toFixed(1) : item.confidence.toFixed(1)}%`
                  : 'N/A';

                return (
                  <tr key={item._id}>
                    <td className="thumbnail-cell">
                      <img
                        src={item.imagePath}
                        alt="Leaf Specimen"
                        className="history-thumb"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    </td>
                    <td>
                      <span className={`badge ${getBadgeClass(level)}`}>
                        {level}
                      </span>
                    </td>
                    <td>
                      <strong>{conf}</strong>
                    </td>
                    <td>
                      {item.probabilities ? (
                        <div className="prob-micro-pills">
                          <span title="Healthy">H: {item.probabilities.healthy ?? 0}%</span>
                          <span title="Mild">M: {item.probabilities.mild ?? 0}%</span>
                          <span title="Moderate">Mod: {item.probabilities.moderate ?? 0}%</span>
                          <span title="Severe">Sev: {item.probabilities.severe ?? 0}%</span>
                        </div>
                      ) : (
                        <span className="text-muted">Direct Tier Assessment</span>
                      )}
                    </td>
                    <td>
                      {item.isMock ? (
                        <span className="badge badge-mock" title="Generated in development mock mode">Mock</span>
                      ) : (
                        <span className="badge badge-real" title="Produced by connected Vision Transformer">ViT Real</span>
                      )}
                    </td>
                    <td className="date-cell">
                      {new Date(item.timestamp).toLocaleDateString()}{' '}
                      <span className="time-sub">
                        {new Date(item.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </span>
                    </td>
                    <td className="actions-cell">
                      <button
                        type="button"
                        className="btn btn-sm btn-secondary"
                        onClick={() => onSelectPrediction(item)}
                        title="Load this result into inspection view"
                      >
                        Inspect
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-danger"
                        onClick={() => onDeletePrediction(item._id)}
                        title="Delete record from MongoDB"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
