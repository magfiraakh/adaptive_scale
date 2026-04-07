"""
Models package — Arsitektur YOLOv11Scale dengan Regression Scale Head.

Ekspor utama:
- YOLOv11Scale
- RegressionScaleHead
- MetadataEncoder
- ModelOutput
- compute_gsd_gt_batch
"""

from .yolo_scale import (
    YOLOv11Scale,
    RegressionScaleHead,
    MetadataEncoder,
    ConvBNAct,
    TinyBackbone,
    TinyNeck,
    DetectHead,
    SegHead,
    ModelOutput,
    compute_gsd_gt_batch,
    META_IDX_ALTITUDE,
    META_IDX_FOCAL_LEN,
    META_IDX_SENSOR_W,
    META_IDX_RESOLUTION,
)

__all__ = [
    "YOLOv11Scale",
    "RegressionScaleHead",
    "MetadataEncoder",
    "ConvBNAct",
    "TinyBackbone",
    "TinyNeck",
    "DetectHead",
    "SegHead",
    "ModelOutput",
    "compute_gsd_gt_batch",
    "META_IDX_ALTITUDE",
    "META_IDX_FOCAL_LEN",
    "META_IDX_SENSOR_W",
    "META_IDX_RESOLUTION",
]