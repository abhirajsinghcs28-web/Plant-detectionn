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
            Analyze a leaf specimen from the <strong>Analyze Leaf</strong> tab to log your first
            result.
          </p>
        </div>
      ) : (
        <div className="table-responsive">
          <table className="research-table">
            <thead>
              <tr>
                <th>Thumbnail</th>
                <th>Specimen</th>
                <th>Diagnosis & Provenance</th>
                <th>Severity (%)</th>
                <th>Tier</th>
                <th>Confidence</th>
                <th>Date & Time</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((item) => (
                <tr key={item._id}>
                  <td className="thumbnail-cell">
                    <img
                      src={item.imagePath}
                      alt={item.crop}
                      className="history-thumb"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </td>
                  <td>
                    <strong>{item.crop}</strong>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span>{item.disease}</span>
                      {item.isMock ? (
                        <span className="badge badge-mock" title="Generated in development mock mode">Mock</span>
                      ) : (
                        <span className="badge badge-real" title="Produced by connected Vision Transformer">AI Real</span>
                      )}
                    </div>
                  </td>
                  <td>
                    <strong>{item.severity ? item.severity.toFixed(1) : 0}%</strong>
                  </td>
                  <td>
                    <span className={`badge ${getBadgeClass(item.severityLevel)}`}>
                      {item.severityLevel || 'Moderate'}
                    </span>
                  </td>
                  <td>{item.confidence ? `${(item.confidence * 100).toFixed(0)}%` : 'N/A'}</td>
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
                      title="Load this result into viewer"
                    >
                      View
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
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
