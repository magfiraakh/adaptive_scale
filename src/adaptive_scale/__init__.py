"""
Adaptive Scale — Estimasi luas fisik pothole dari citra UAV.

Package ini mengintegrasikan:
- YOLOv11-Seg untuk deteksi dan segmentasi pothole
- Regression Scale Head (RSH) untuk prediksi GSD per frame
- Pipeline estimasi luas fisik dalam meter persegi

Konvensi resolusi:
- GSD dihitung dari resolusi asli gambar, bukan resolusi input YOLO
- Mask piksel untuk area fisik harus mengacu ke resolusi asli

Formula inti:
    GSD_gt (m/px) = (altitude_m * sensor_width_mm)
                    / (focal_length_mm * resolution_px_asli)
                    / 1000

    A_fisik (m²)  = N_piksel_mask_asli * GSD²
"""

__version__ = "1.0.0"
__author__ = "Tesis S2 Teknik Informatika"

from .models import (
    YOLOv11Scale,
    RegressionScaleHead,
    MetadataEncoder,
    ModelOutput,
    compute_gsd_gt_batch,
)

from .training import (
    MultiTaskLoss,
    Trainer,
)

from .datasets import (
    PotholeDataset,
    collate_fn,
)

from .utils import (
    compute_physical_area,
    upscale_mask_to_original,
    pixels_to_area_m2,
)

from .evaluation import (
    regression_metrics,
    area_metrics,
    consistency_metrics,
    Evaluator,
)

__all__ = [
    # models
    "YOLOv11Scale",
    "RegressionScaleHead",
    "MetadataEncoder",
    "ModelOutput",
    "compute_gsd_gt_batch",
    # training
    "MultiTaskLoss",
    "Trainer",
    # datasets
    "PotholeDataset",
    "collate_fn",
    # utils
    "compute_physical_area",
    "upscale_mask_to_original",
    "pixels_to_area_m2",
    # evaluation
    "regression_metrics",
    "area_metrics",
    "consistency_metrics",
    "Evaluator",
]