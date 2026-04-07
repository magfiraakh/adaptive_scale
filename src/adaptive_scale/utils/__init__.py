"""
Utils package — Postprocessing dan konversi area.

Ekspor utama:
- compute_physical_area
- upscale_mask_to_original
- pixels_to_area_m2
- write_area_csv
- bbox_to_mask_xyxy

Konvensi resolusi:
- GSD dalam m/px dihitung dari resolusi asli
- Mask untuk perhitungan area fisik harus berada pada resolusi asli
- Jika mask berasal dari output YOLO, upscale dulu ke resolusi asli
"""

from .postprocess import (
    compute_physical_area,
    upscale_mask_to_original,
    pixels_to_area_m2,
    write_area_csv,
    bbox_to_mask_xyxy,
)

__all__ = [
    "compute_physical_area",
    "upscale_mask_to_original",
    "pixels_to_area_m2",
    "write_area_csv",
    "bbox_to_mask_xyxy",
]