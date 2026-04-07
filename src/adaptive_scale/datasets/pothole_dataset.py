"""Flexible dataset loader for UAV pothole detection/segmentation + mpp regression.

Expected annotation format (JSON per image), example keys:
{
  "image_id": "frame_0001",
  "image_path": "images/frame_0001.jpg",
  "mpp": 0.0123,
  "objects": [
    {"bbox": [cx, cy, w, h], "class_id": 0, "mask_path": "masks/frame_0001_obj0.png", "area_m2": 1.2}
  ]
}

TODO: adapt parse_annotation to your exact schema if needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PotholeDataset(Dataset):
    def __init__(
        self,
        root: str,
        annotation_file: str,
        image_size: int = 640,
        use_segmentation: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.use_segmentation = use_segmentation
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.root / rel_path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_mask(self, rel_path: str) -> torch.Tensor:
        mask = Image.open(self.root / rel_path).convert("L").resize((self.image_size, self.image_size))
        arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(arr)

    def parse_annotation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = self._load_image(sample["image_path"])
        objects = sample.get("objects", [])

        bboxes = []
        classes = []
        masks = []
        gt_area_m2 = []
        for obj in objects:
            bboxes.append(obj["bbox"])  # normalized cx, cy, w, h assumed
            classes.append(obj.get("class_id", 0))
            gt_area_m2.append(obj.get("area_m2", -1.0))
            if self.use_segmentation and obj.get("mask_path"):
                masks.append(self._load_mask(obj["mask_path"]))

        item = {
            "image_id": sample.get("image_id", Path(sample["image_path"]).stem),
            "image": image,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long),
            "masks": torch.stack(masks) if masks else None,
            "mpp": torch.tensor(sample["mpp"], dtype=torch.float32),
            "gt_area_m2": torch.tensor(gt_area_m2, dtype=torch.float32) if gt_area_m2 else torch.zeros((0,), dtype=torch.float32),
        }
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.parse_annotation(self.samples[idx])


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    return {
        "image_ids": [b["image_id"] for b in batch],
        "images": images,
        "bboxes": [b["bboxes"] for b in batch],
        "classes": [b["classes"] for b in batch],
        "masks": [b["masks"] for b in batch],
        "mpp": torch.stack([b["mpp"] for b in batch]),
        "gt_area_m2": [b["gt_area_m2"] for b in batch],
    }
