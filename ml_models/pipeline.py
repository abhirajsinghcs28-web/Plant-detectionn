"""
=============================================================================
UNIFIED CROP DISEASE INFERENCE PIPELINE (MEMBER 2 CORE ENGINE)
Project: Agritech Self-Supervised Cross-Domain Vision Transformers
         for Fine-Grained Crop Leaf Disease Severity Estimation
=============================================================================
Unifies the Pretrained Vision Transformer (ViT-B/16) and the CNN Baseline
into a high-performance inference engine.
Conforms strictly to the Member 3 Express backend API contract.
=============================================================================
"""

import time
import io
import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple

from .cnn_baseline import NumpyCNNBaseline
from .vision_transformer import NumpyVisionTransformer
from .dataset_and_transforms import preprocess_image, render_scientific_heatmap
from .taxonomy import CROPS, calculate_severity_level, get_disease_info

class CropDiseaseInferencePipeline:
    """
    Unified Inference Engine coordinating CNN Baseline and Pretrained ViT models.
    """
    def __init__(self):
        # Initialize both model engines
        self.vit_model = NumpyVisionTransformer()
        self.cnn_model = NumpyCNNBaseline()

    def predict(
        self,
        image_input,
        crop: str = "Tomato",
        model_name: str = "vit",
        colormap: str = "turbo"
    ) -> Dict[str, Any]:
        """
        Run inference using either the Vision Transformer or CNN Baseline.
        Conforms strictly to Member 3 API contract:
        {
          "crop": "Tomato",
          "disease": "Late Blight",
          "severity": 36.8,
          "confidence": 0.94,
          "severityLevel": "Moderate",
          "segmentationMask": "data:image/png;base64,...",
          "maskType": "attention_heatmap"
        }
        """
        start_time = time.time()
        # Crop verification
        if crop not in CROPS:
            crop = "Tomato"

        # Preprocess input image to (224, 224, 3) in [0, 1]
        rgb_img = preprocess_image(image_input, target_size=(224, 224))

        model_name_lower = (model_name or "vit").lower()
        if "cnn" in model_name_lower:
            # Execute CNN Baseline
            raw_result = self.cnn_model.predict(rgb_img, crop=crop)
            spatial_map = raw_result["activation_map"]
            model_display = self.cnn_model.model_name
            params = self.cnn_model.total_parameters
            gflops = self.cnn_model.gflops
        else:
            # Execute Vision Transformer (Default)
            raw_result = self.vit_model.predict(rgb_img, crop=crop)
            spatial_map = raw_result["attention_map"]
            model_display = self.vit_model.model_name
            params = self.vit_model.total_parameters
            gflops = self.vit_model.gflops

        # Render continuous floating-point attention or Grad-CAM map into Base64 PNG
        heatmap_uri = render_scientific_heatmap(spatial_map, target_size=(256, 256), colormap=colormap)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "crop": raw_result["crop"],
            "disease": raw_result["disease"],
            "pathogen": raw_result.get("pathogen", ""),
            "severity": float(raw_result["severity"]),
            "confidence": float(raw_result["confidence"]),
            "severityLevel": raw_result["severityLevel"],
            "segmentationMask": heatmap_uri,
            "maskType": "attention_heatmap",
            "modelUsed": model_display,
            "latencyMs": latency_ms,
            "parameters": params,
            "gflops": gflops,
            "management": raw_result.get("management", "")
        }

    def compare(self, image_input, crop: str = "Tomato") -> Dict[str, Any]:
        """
        Run both models on the same leaf specimen and return side-by-side comparison.
        """
        vit_res = self.predict(image_input, crop=crop, model_name="vit")
        cnn_res = self.predict(image_input, crop=crop, model_name="cnn")

        severity_diff = round(abs(vit_res["severity"] - cnn_res["severity"]), 2)
        confidence_diff = round(abs(vit_res["confidence"] - cnn_res["confidence"]), 2)
        class_match = (vit_res["disease"] == cnn_res["disease"])

        return {
            "crop": crop,
            "agreement": class_match,
            "severity_delta": severity_diff,
            "confidence_delta": confidence_diff,
            "vit": vit_res,
            "cnn": cnn_res
        }

# Global singleton instance
pipeline = CropDiseaseInferencePipeline()

