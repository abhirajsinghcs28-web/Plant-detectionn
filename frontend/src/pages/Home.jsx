import React, { useState } from 'react';
import CropSelector from '../components/CropSelector';
import ImageUpload from '../components/ImageUpload';
import ResultCard from '../components/ResultCard';
import SegmentationViewer from '../components/SegmentationViewer';
import { predictLeaf } from '../services/api';

/**
 * Main Analysis Page (Home)
 * Orchestrates crop selection, leaf upload, AI inference dispatch, and results visualization.
 */
export default function Home({ onPredictionCompleted, currentResult, setCurrentResult }) {
  const [crop, setCrop] = useState('Tomato');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
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

  // Analyze leaf submission
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
      formData.append('crop', crop);

      const response = await predictLeaf(formData);

      if (response && response.success) {
        setCurrentResult(response.data);
        if (onPredictionCompleted) {
          onPredictionCompleted(response.data);
        }
      } else {
        throw new Error(response.error || 'Inference returned unsuccessful status.');
      }
    } catch (err) {
      setErrorMessage(err.message || 'Prediction failed. Please check backend server and try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="home-page">
      {/* Intro section */}
      <div className="research-intro-banner">
        <h2>Fine-Grained Crop Leaf Disease Severity Estimation</h2>
        <p>
          Upload a clear photograph of a crop leaf specimen to quantify disease presence,
          categorize infection severity, and generate localized attention segmentation masks using
          cross-domain Vision Transformer representations.
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
        <div className="card-title">Specimen Input & Analysis Setup</div>

        <form onSubmit={handleAnalyze}>
          <div className="grid-2-col">
            <CropSelector
              selectedCrop={crop}
              onCropChange={setCrop}
              disabled={isAnalyzing}
            />

            <div className="form-info-panel">
              <label>Domain Head Configuration:</label>
              <div className="info-box">
                <div>Selected Model Target: <strong>{crop} Domain ViT</strong></div>
                <div>Model Resolution: <strong>224 × 224 RGB</strong></div>
                <div>Task: <strong>Fine-Grained Severity Estimation</strong></div>
              </div>
            </div>
          </div>

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
                  <span className="spinner"></span> Processing Inference...
                </>
              ) : (
                'Analyze Leaf'
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

          <SegmentationViewer
            originalImageUrl={currentResult.imagePath || previewUrl}
            segmentationMaskUrl={currentResult.segmentationMask}
            maskType={currentResult.maskType}
          />
        </div>
      )}
    </div>
  );
}
