import React from 'react';

// List of supported crops - easy to modify or expand
export const SUPPORTED_CROPS = [
  { id: 'Tomato', label: 'Tomato (Solanum lycopersicum)' },
  { id: 'Potato', label: 'Potato (Solanum tuberosum)' },
  { id: 'Rice', label: 'Rice (Oryza sativa)' },
  { id: 'Wheat', label: 'Wheat (Triticum aestivum)' },
  { id: 'Maize', label: 'Maize / Corn (Zea mays)' },
];

/**
 * Clean Crop Selector Dropdown
 */
export default function CropSelector({ selectedCrop, onCropChange, disabled }) {
  return (
    <div className="form-group crop-selector-group">
      <label htmlFor="crop-select">Select Crop Specimen:</label>
      <select
        id="crop-select"
        value={selectedCrop}
        onChange={(e) => onCropChange(e.target.value)}
        disabled={disabled}
        className="form-control"
      >
        {SUPPORTED_CROPS.map((crop) => (
          <option key={crop.id} value={crop.id}>
            {crop.label}
          </option>
        ))}
      </select>
      <p className="field-hint">
        Select the agricultural crop to target the corresponding Vision Transformer domain head.
      </p>
    </div>
  );
}
