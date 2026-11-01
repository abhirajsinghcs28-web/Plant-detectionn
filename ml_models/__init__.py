"""
=============================================================================
AGRITECH VISION TRANSFORMER & CNN BASELINE DEEP LEARNING PACKAGE
Project: Agritech Self-Supervised Cross-Domain Vision Transformers
         for Fine-Grained Crop Leaf Disease Severity Estimation
=============================================================================
"""

from .taxonomy import CROPS, TAXONOMY, ALL_CLASSES, NUM_CLASSES, calculate_severity_level, get_disease_info
from .cnn_baseline import NumpyCNNBaseline, CropDiseaseCNNBaseline
from .vision_transformer import NumpyVisionTransformer, CropDiseaseViT
from .dataset_and_transforms import (
    preprocess_image,
    normalize_tensor,
    denormalize_tensor,
    render_scientific_heatmap,
    SyntheticSpecimenGenerator
)
from .trainer import MultiTaskLoss, AgronomicMetrics, ModelTrainingSimulator
from .pipeline import CropDiseaseInferencePipeline, pipeline

def run_comparative_benchmark(*args, **kwargs):
    from .benchmark import run_comparative_benchmark as _rcb
    return _rcb(*args, **kwargs)

__all__ = [
    "CROPS",
    "TAXONOMY",
    "ALL_CLASSES",
    "NUM_CLASSES",
    "calculate_severity_level",
    "get_disease_info",
    "NumpyCNNBaseline",
    "CropDiseaseCNNBaseline",
    "NumpyVisionTransformer",
    "CropDiseaseViT",
    "preprocess_image",
    "normalize_tensor",
    "denormalize_tensor",
    "render_scientific_heatmap",
    "SyntheticSpecimenGenerator",
    "MultiTaskLoss",
    "AgronomicMetrics",
    "ModelTrainingSimulator",
    "CropDiseaseInferencePipeline",
    "pipeline",
    "run_comparative_benchmark"
]

