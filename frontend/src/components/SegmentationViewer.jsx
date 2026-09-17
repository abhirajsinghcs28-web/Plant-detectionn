import React, { useState, useEffect } from 'react';

/**
 * Segmentation & Attention Map Viewer Component
 * Displays the original leaf specimen alongside or superimposed with the segmentation mask.
 * Strictly adheres to scientific honesty: never displays synthetic/fake masks.
 */
export default function SegmentationViewer({ originalImageUrl, segmentationMaskUrl, maskType }) {
  const hasMask = Boolean(segmentationMaskUrl);
  const [viewMode, setViewMode] = useState(hasMask ? 'sideBySide' : 'original');
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);

  // Synchronize viewMode if segmentationMaskUrl changes
  useEffect(() => {
    if (!hasMask) {
      setViewMode('original');
    } else if (viewMode === 'original') {
      setViewMode('sideBySide');
    }
  }, [hasMask]);

  const maskLabel =
    maskType === 'attention_heatmap'
      ? 'Vision Transformer Attention Heatmap'
      : 'Disease Lesion Segmentation Mask';

  return (
    <div className="card segmentation-viewer-card">
      <div className="card-title">
        <span>Disease Lesion Segmentation & Attention Map Viewer</span>
        <div className="view-mode-tabs">
          <button
            type="button"
            className={`tab-btn ${viewMode === 'original' ? 'active' : ''}`}
            onClick={() => setViewMode('original')}
          >
            Original Specimen
          </button>
          <button
            type="button"
            className={`tab-btn ${viewMode === 'sideBySide' ? 'active' : ''}`}
            onClick={() => setViewMode('sideBySide')}
            disabled={!hasMask}
            title={!hasMask ? 'Unavailable: No mask returned by AI model' : 'Side-by-side comparison'}
          >
            Side-by-Side
          </button>
          <button
            type="button"
            className={`tab-btn ${viewMode === 'overlay' ? 'active' : ''}`}
            onClick={() => setViewMode('overlay')}
            disabled={!hasMask}
            title={!hasMask ? 'Unavailable: No mask returned by AI model' : 'Overlay mask on specimen'}
          >
            Overlay
          </button>
          <button
            type="button"
            className={`tab-btn ${viewMode === 'mask' ? 'active' : ''}`}
            onClick={() => setViewMode('mask')}
            disabled={!hasMask}
            title={!hasMask ? 'Unavailable: No mask returned by AI model' : 'Mask only view'}
          >
            Mask Only
          </button>
        </div>
      </div>

      {!hasMask && (
        <div className="alert alert-info">
          <strong>Segmentation / Attention Heatmap Unavailable:</strong> The inference service
          did not provide a spatial lesion mask or attention map for this specimen.
          <br />
          <span style={{ fontSize: '0.85rem', color: '#475569' }}>
            Note for Member 2 integration: Return <code>segmentationMask</code> (as a data URI or image URL)
            and optional <code>maskType</code> (<code>'attention_heatmap'</code> or <code>'segmentation_mask'</code>)
            in the prediction response to activate overlay visualization.
          </span>
        </div>
      )}

      {/* Original Only View (Always available) */}
      {viewMode === 'original' && (
        <div className="single-view-container">
          <div className="image-frame large-frame">
            <img src={originalImageUrl} alt="Original Specimen" className="specimen-img" />
          </div>
          <div className="panel-caption">Original RGB Leaf Specimen</div>
        </div>
      )}

      {/* Side-by-Side View (Only when mask is present) */}
      {viewMode === 'sideBySide' && hasMask && (
        <div className="viewer-grid-dual">
          <div className="viewer-panel">
            <div className="panel-header">Specimen Image</div>
            <div className="image-frame">
              <img src={originalImageUrl} alt="Original Specimen" className="specimen-img" />
            </div>
            <div className="panel-caption">Original RGB Leaf Input</div>
          </div>

          <div className="viewer-panel">
            <div className="panel-header">{maskLabel}</div>
            <div className="image-frame mask-frame">
              <img src={segmentationMaskUrl} alt={maskLabel} className="specimen-img" />
            </div>
            <div className="panel-caption">{maskLabel}</div>
          </div>
        </div>
      )}

      {/* Overlay View (Only when mask is present) */}
      {viewMode === 'overlay' && hasMask && (
        <div className="overlay-container">
          <div className="overlay-controls">
            <label htmlFor="opacity-range">
              Mask Transparency: {Math.round(overlayOpacity * 100)}%
            </label>
            <input
              id="opacity-range"
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
            />
          </div>

          <div className="overlay-stage">
            <img src={originalImageUrl} alt="Original specimen" className="overlay-base" />
            <img
              src={segmentationMaskUrl}
              alt="Segmentation mask overlay"
              className="overlay-top"
              style={{ opacity: overlayOpacity }}
            />
          </div>
          <div className="panel-caption">
            Superimposed {maskLabel}
          </div>
        </div>
      )}

      {/* Mask Only View (Only when mask is present) */}
      {viewMode === 'mask' && hasMask && (
        <div className="single-view-container">
          <div className="image-frame large-frame mask-frame">
            <img src={segmentationMaskUrl} alt={maskLabel} className="specimen-img" />
          </div>
          <div className="panel-caption">{maskLabel}</div>
        </div>
      )}
    </div>
  );
}
