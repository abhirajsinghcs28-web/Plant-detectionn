import React, { useRef, useState } from 'react';

/**
 * Image Upload Component
 * Supports drag-and-drop, standard file selection, image preview, and sample testing.
 */
export default function ImageUpload({
  selectedFile,
  previewUrl,
  onFileSelect,
  onClear,
  onError,
  disabled,
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // Validate and handle file
  const processFile = (file) => {
    if (!file) return;

    // Validate mime type and extension
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const hasValidExt = /\.(jpe?g|png)$/i.test(file.name);

    if (!validTypes.includes(file.type) && !hasValidExt) {
      onError('Please upload a valid JPG, JPEG, or PNG image.');
      return;
    }

    // Validate size (10 MB max)
    if (file.size > 10 * 1024 * 1024) {
      onError('Uploaded image is too large. Maximum allowed size is 10MB.');
      return;
    }

    // Clear previous error and invoke parent handler
    onError(null);
    onFileSelect(file);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Helper to load sample test leaves
  const loadSampleLeaf = async (samplePath, sampleName) => {
    try {
      onError(null);
      const res = await fetch(samplePath);
      const blob = await res.blob();
      const file = new File([blob], sampleName, { type: 'image/jpeg' });
      onFileSelect(file);
    } catch (err) {
      onError('Could not load sample leaf image: ' + err.message);
    }
  };

  return (
    <div className="image-upload-wrapper">
      <div className="upload-header-text">
        <label>Crop Leaf Image Upload</label>
        <p className="upload-guide-hint">Upload a clear image of a single crop leaf for the best analysis.</p>
      </div>

      {!previewUrl ? (
        <div
          className={`upload-dropzone ${isDragging ? 'dragging' : ''} ${
            disabled ? 'disabled' : ''
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            style={{ display: 'none' }}
            disabled={disabled}
          />
          <div className="dropzone-content">
            <div className="upload-icon-box">🍃</div>
            <p className="upload-main-text">
              <strong>Drag and drop image here</strong> or <span className="browse-link">browse</span> from your computer
            </p>
            <p className="upload-sub-text">Supported formats: JPG, JPEG, PNG (Max size: 10MB)</p>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-image-wrapper">
            <img src={previewUrl} alt="Leaf Preview" className="preview-image" />
          </div>
          <div className="preview-details">
            <div className="preview-meta">
              <span className="file-name">{selectedFile?.name || 'Uploaded Leaf Specimen'}</span>
              <span className="file-size">
                {selectedFile?.size
                  ? `${(selectedFile.size / 1024).toFixed(1)} KB`
                  : 'Ready for Severity Analysis'}
              </span>
            </div>
            <div className="preview-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
              >
                Change Image
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={onClear}
                disabled={disabled}
              >
                Remove
              </button>
            </div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              style={{ display: 'none' }}
              disabled={disabled}
            />
          </div>
        </div>
      )}

      {/* Quick Test Samples */}
      <div className="sample-test-bar">
        <span className="sample-label">Try sample test leaf:</span>
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          onClick={() =>
            loadSampleLeaf('/sample-leaves/tomato_late_blight.jpg', 'specimen_with_symptoms.jpg')
          }
          disabled={disabled}
        >
          Sample Leaf with Symptoms
        </button>
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          onClick={() =>
            loadSampleLeaf('/sample-leaves/healthy_leaf.jpg', 'healthy_leaf_specimen.jpg')
          }
          disabled={disabled}
        >
          Sample Healthy Leaf
        </button>
      </div>
    </div>
  );
}
