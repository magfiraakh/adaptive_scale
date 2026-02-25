from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch


def bbox_to_mask_xyxy(box: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x1, y1, x2, y2 = box.int().tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    m = torch.zeros((h, w), dtype=torch.float32)
    m[y1:y2, x1:x2] = 1.0
    return m


def pixel_area_from_mask(mask: torch.Tensor) -> float:
    return float((mask > 0.5).sum().item())


def compute_areas_for_image(
    image_id: str,
    dets: List[Dict],
    mpp_pred: float,
    image_hw: tuple[int, int],
) -> List[Dict]:
    rows = []
    h, w = image_hw
    for i, det in enumerate(dets):
        conf = float(det.get("conf", 0.0))
        mask = det.get("mask")
        if mask is None:
            mask = bbox_to_mask_xyxy(det["bbox_xyxy"], h=h, w=w)
        area_px = pixel_area_from_mask(mask)
        area_m2 = area_px * (mpp_pred**2)
        rows.append(
            {
                "image_id": image_id,
                "pothole_id": i,
                "conf": conf,
                "area_px": area_px,
                "mpp_pred": mpp_pred,
                "area_m2_pred": area_m2,
            }
        )
    return rows


def write_area_csv(rows: List[Dict], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "pothole_id", "conf", "area_px", "mpp_pred", "area_m2_pred"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
