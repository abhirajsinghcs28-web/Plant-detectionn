import React, { useState, useEffect } from 'react';
import HistoryList from '../components/HistoryList';
import { fetchPredictions, deletePrediction } from '../services/api';

/**
 * Prediction History Page
 * Connects to MongoDB to view, manage, and inspect historical assessments.
 */
export default function History({ onSelectPrediction }) {
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  const loadHistory = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetchPredictions();
      if (res && res.success) {
        setPredictions(res.data || []);
      } else {
        setPredictions([]);
        if (res.message) {
          setErrorMessage(res.message);
        }
      }
    } catch (err) {
      setErrorMessage(err.message || 'Failed to connect to MongoDB prediction store.');
      setPredictions([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this prediction record from MongoDB?')) {
      return;
    }

    try {
      await deletePrediction(id);
      setSuccessMessage('Record removed successfully.');
      setTimeout(() => setSuccessMessage(null), 3500);
      // Update local state immediately
      setPredictions((prev) => prev.filter((p) => p._id !== id));
    } catch (err) {
      setErrorMessage(err.message || 'Could not delete record.');
    }
  };

  return (
    <div className="history-page">
      <div className="research-intro-banner">
        <h2>Prediction Log & Database Records</h2>
        <p>
          Historical repository of analyzed crop leaf specimens, severity percentages, and
          segmentation masks stored in MongoDB.
        </p>
      </div>

      {errorMessage && (
        <div className="alert alert-error">
          <strong>Database Notice:</strong> {errorMessage}
        </div>
      )}

      {successMessage && (
        <div className="alert alert-success">
          {successMessage}
        </div>
      )}

      <HistoryList
        predictions={predictions}
        isLoading={isLoading}
        onSelectPrediction={onSelectPrediction}
        onDeletePrediction={handleDelete}
        onRefresh={loadHistory}
      />
    </div>
  );
}
