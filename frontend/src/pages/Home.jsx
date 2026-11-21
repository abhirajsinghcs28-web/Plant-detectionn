import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import ResultCard from '../components/ResultCard';
import { predictLeaf } from '../services/api';

/**
 * Main Analysis Page (Home)
 * Focused on AI-Powered Crop Leaf Severity Analysis using Vision Transformers.
 */
export default function Home({ onPredictionCompleted, currentResult, setCurrentResult }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedCrop, setSelectedCrop] = useState('Tomato');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // File selection handler
  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setErrorMessage(null);
  };

  // Clear image handler
  const handleClear = () => {
    setFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setErrorMessage(null);
  };

  // Analyze leaf severity submission
  const handleAnalyze = async (e) => {
    e.preventDefault();

    if (!file) {
      setErrorMessage('Please upload a crop leaf image before starting analysis.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('crop', selectedCrop || 'Tomato');

      const response = await predictLeaf(formData);

      if (response && response.success) {
        setCurrentResult(response.data);
        if (onPredictionCompleted) {
          onPredictionCompleted(response.data);
        }
      } else {
        throw new Error(response.error || 'Severity estimation returned an unsuccessful status.');
      }
    } catch (err) {
      setErrorMessage(err.message || 'Analysis failed. Please check backend server and try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="home-page">
      {/* Intro Section with Clear Messaging */}
      <div className="research-intro-banner">
        <h2>AI-Powered Crop Leaf Severity Analysis</h2>
        <p>
          Upload a crop leaf image to estimate the severity of visible symptoms using our Vision Transformer-based AI model.
        </p>
      </div>

      {/* Error alert */}
      {errorMessage && (
        <div className="alert alert-error" role="alert">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}

      {/* Main Analysis Form */}
      <div className="card upload-workflow-card">
        <div className="card-title">Specimen Ingestion & Severity Analysis</div>

        <div className="model-spec-panel">
          <div className="spec-badge">PROPOSED ViT (v2 FULL BENCHMARK - 94.1% ACCURACY)</div>
          <div className="spec-grid">
            <div className="spec-item">
              <span className="spec-label">Model Architecture</span>
              <span className="spec-val">PROPOSED_vit_ce_reg (ViT-B/16 Dual Head)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Validation Benchmark</span>
              <span className="spec-val">94.1% Acc • 1.60% MAE • 0.973 QWK</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Severity Tiers (Grade Edges)</span>
              <span className="spec-val">Healthy (&lt;1%) • Mild (1-5%) • Moderate (5-20%) • Severe (≥20%)</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Interpretability</span>
              <span className="spec-val">Multi-layer Attention Rollout (1.12 Lesion Ratio)</span>
            </div>
          </div>
        </div>

        <div className="crop-selection-section">
          <div className="crop-select-header">
            <label className="crop-select-label">Crop Specimen (Optional):</label>
            <span className="crop-select-hint">Calibrated on PlantVillage v2 (Tomato, Corn, Grape, Apple)</span>
          </div>
          <div className="crop-pills-container">
            {[
              { id: 'Tomato', label: 'Tomato', icon: '🍅' },
              { id: 'Corn_(maize)', label: 'Corn (Maize)', icon: '🌽' },
              { id: 'Grape', label: 'Grape', icon: '🍇' },
              { id: 'Apple', label: 'Apple', icon: '🍎' },
              { id: 'Crop Leaf', label: 'General Leaf', icon: '🍃' },
              { id: 'Potato', label: 'Potato', icon: '🥔' }
            ].map((c) => (
              <button
                key={c.id}
                type="button"
                className={`crop-pill-btn ${selectedCrop === c.id ? 'active' : ''}`}
                onClick={() => setSelectedCrop(c.id)}
              >
                {c.icon} {c.label}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleAnalyze}>
          <ImageUpload
            selectedFile={file}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onClear={handleClear}
            onError={setErrorMessage}
            disabled={isAnalyzing}
          />

          <div className="form-submit-row">
            <button
              type="submit"
              className="btn btn-primary btn-large"
              disabled={!file || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <span className="spinner"></span> Analyzing Severity...
                </>
              ) : (
                'Analyze Severity'
              )}
            </button>

            {currentResult && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  handleClear();
                  setCurrentResult(null);
                }}
              >
                Reset Specimen
              </button>
            )}
          </div>
        </form>
      </div>

      {/* Results Section rendered immediately upon inference completion */}
      {currentResult && (
        <div className="results-container" id="prediction-results">
          <ResultCard result={currentResult} />
        </div>
      )}
    </div>
  );
}
