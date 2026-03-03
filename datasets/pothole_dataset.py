"""
Flexible dataset loader for UAV pothole detection/segmentation + mpp regression.

Supports TWO annotation schemas:
1) COCO-style dict:
   {
     "images": [{"id":1,"file_name":"xxx.jpg","width":640,"height":640,"mpp":0.0073}, ...],
     "annotations": [
        {"image_id":1, "category_id":0, "bbox_xyxy":[x1,y1,x2,y2], "polygon_xy":[...], "area":1234, ...},
        ...
     ],
     "categories": [...]
   }

2) Legacy list-per-image:
   [
     {"image_id":"frame_0001","image_path":"images/frame_0001.jpg","mpp":0.0123,
      "objects":[{"bbox":[cx,cy,w,h],"class_id":0,"mask_path":"...","area_m2":1.2}, ...]
     },
     ...
   ]

Output item keys (used by train/eval/infer):
- image_id: str|int
- image: FloatTensor [3,H,W] in [0,1]
- bboxes: FloatTensor [N,4] normalized cx,cy,w,h
- classes: LongTensor [N]
- masks: FloatTensor [N,H,W] or None
- mpp: FloatTensor []
- gt_area_m2: FloatTensor [N] (if available, else -1)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def _xyxy_to_cxcywh_norm(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> List[float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    # normalize
    return [cx / w, cy / h, bw / w, bh / h]


def _coco_bbox_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    # COCO bbox is [x, y, width, height]
    x, y, bw, bh = bbox
    return x, y, x + bw, y + bh


def _rasterize_polygon_xy(polygon_xy: List[float], width: int, height: int) -> np.ndarray:
    """
    polygon_xy: [x1,y1,x2,y2,...] in image pixel coords.
    Returns mask (H,W) with {0,1}.
    """
    if polygon_xy is None or len(polygon_xy) < 6:
        return np.zeros((height, width), dtype=np.float32)

    pts = [(float(polygon_xy[i]), float(polygon_xy[i + 1])) for i in range(0, len(polygon_xy), 2)]
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, outline=1, fill=1)
    return np.asarray(img, dtype=np.float32)


class PotholeDataset(Dataset):
    def __init__(
        self,
        root: str,
        annotation_file: str,
        image_size: int = 640,
        use_segmentation: bool = True,
        require_mpp: bool = True,
        default_mpp: Optional[float] = None,
    ):
        """
        root:
          Folder base tempat image berada.
          Untuk COCO, file_name akan di-join ke root (root/file_name).
        require_mpp:
          Jika True dan mpp tidak ditemukan, akan error (disarankan untuk training scale head).
        default_mpp:
          Dipakai jika require_mpp=False dan mpp tidak ada di annotation.
        """
        self.root = Path(root)
        self.image_size = image_size
        self.use_segmentation = use_segmentation
        self.require_mpp = require_mpp
        self.default_mpp = default_mpp

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Detect schema
        self.is_coco = isinstance(data, dict) and "images" in data and "annotations" in data

        if self.is_coco:
            self.coco = data
            self._build_coco_index()
        else:
            # legacy list
            if not isinstance(data, list):
                raise ValueError(
                    "Annotation format not recognized. Expected COCO dict with keys "
                    "'images'/'annotations' OR a legacy list-per-image."
                )
            self.samples = data

    def _build_coco_index(self) -> None:
        images = self.coco.get("images", [])
        anns = self.coco.get("annotations", [])

        self.images_by_id: Dict[Union[int, str], Dict[str, Any]] = {}
        self.image_ids: List[Union[int, str]] = []

        for im in images:
            im_id = im.get("id")
            self.images_by_id[im_id] = im
            self.image_ids.append(im_id)

        self.anns_by_image_id: Dict[Union[int, str], List[Dict[str, Any]]] = {}
        for a in anns:
            iid = a.get("image_id")
            self.anns_by_image_id.setdefault(iid, []).append(a)

    def __len__(self) -> int:
        return len(self.image_ids) if self.is_coco else len(self.samples)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.root / rel_path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_mask_file(self, rel_path: str) -> torch.Tensor:
        mask = Image.open(self.root / rel_path).convert("L").resize((self.image_size, self.image_size))
        arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(arr)

    def _resize_mask_np(self, mask: np.ndarray, orig_w: int, orig_h: int) -> torch.Tensor:
        # mask (H,W) float32 -> resize to (image_size,image_size)
        img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((self.image_size, self.image_size))
        arr = (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)
        return torch.from_numpy(arr)

    def _get_mpp(self, img_info: Dict[str, Any], legacy_sample: Optional[Dict[str, Any]] = None) -> float:
        if self.is_coco:
            mpp = img_info.get("mpp", None)
        else:
            mpp = None if legacy_sample is None else legacy_sample.get("mpp", None)

        if mpp is None:
            if self.require_mpp:
                raise ValueError(
                    "mpp tidak ditemukan di annotation. "
                    "Untuk COCO, tambahkan field images[i]['mpp']. "
                    "Atau set require_mpp=False dan isi default_mpp."
                )
            if self.default_mpp is None:
                raise ValueError("require_mpp=False tapi default_mpp belum diisi.")
            return float(self.default_mpp)

        return float(mpp)

    def _parse_legacy(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = self._load_image(sample["image_path"])
        objects = sample.get("objects", [])

        bboxes, classes, masks, gt_area_m2 = [], [], [], []
        for obj in objects:
            bboxes.append(obj["bbox"])  # normalized cx,cy,w,h assumed
            classes.append(obj.get("class_id", 0))
            gt_area_m2.append(obj.get("area_m2", -1.0))
            if self.use_segmentation and obj.get("mask_path"):
                masks.append(self._load_mask_file(obj["mask_path"]))

        mpp = self._get_mpp(img_info={}, legacy_sample=sample)
        item = {
            "image_id": sample.get("image_id", Path(sample["image_path"]).stem),
            "image": image,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long),
            "masks": torch.stack(masks) if masks else None,
            "mpp": torch.tensor(mpp, dtype=torch.float32),
            "gt_area_m2": torch.tensor(gt_area_m2, dtype=torch.float32) if gt_area_m2 else torch.zeros((0,), dtype=torch.float32),
        }
        return item

    def _parse_coco(self, img_id: Union[int, str]) -> Dict[str, Any]:
        img_info = self.images_by_id[img_id]
        rel_path = img_info["file_name"]
        orig_w = int(img_info.get("width", self.image_size))
        orig_h = int(img_info.get("height", self.image_size))

        image = self._load_image(rel_path)
        anns = self.anns_by_image_id.get(img_id, [])

        bboxes, classes, masks, gt_area_m2 = [], [], [], []

        mpp = self._get_mpp(img_info=img_info)

        for a in anns:
            # class
            cls = int(a.get("category_id", a.get("class_id", 0)))
            classes.append(cls)

            # bbox: prefer bbox_xyxy if available
            if "bbox_xyxy" in a and a["bbox_xyxy"] is not None:
                x1, y1, x2, y2 = map(float, a["bbox_xyxy"])
            elif "bbox" in a and a["bbox"] is not None:
                x1, y1, x2, y2 = _coco_bbox_to_xyxy(list(map(float, a["bbox"])))
            else:
                # fallback: if polygon exists, compute bbox from polygon points
                poly = a.get("polygon_xy", None)
                if poly and len(poly) >= 6:
                    xs = poly[0::2]
                    ys = poly[1::2]
                    x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
                else:
                    x1 = y1 = x2 = y2 = 0.0

            bboxes.append(_xyxy_to_cxcywh_norm(x1, y1, x2, y2, orig_w, orig_h))

            # mask from polygon if requested
            if self.use_segmentation:
                poly = a.get("polygon_xy", None)
                if poly and len(poly) >= 6:
                    mask_np = _rasterize_polygon_xy(poly, orig_w, orig_h)
                    masks.append(self._resize_mask_np(mask_np, orig_w, orig_h))
                else:
                    masks.append(torch.zeros((self.image_size, self.image_size), dtype=torch.float32))

            # area_m2 if available, else compute from COCO pixel area if present
            if "area_m2" in a and a["area_m2"] is not None:
                gt_area_m2.append(float(a["area_m2"]))
            elif "area" in a and a["area"] is not None:
                # COCO 'area' typically in pixel^2 (orig scale)
                area_px = float(a["area"])
                gt_area_m2.append(area_px * (mpp ** 2))
            else:
                gt_area_m2.append(-1.0)

        item = {
            "image_id": img_info.get("id", rel_path),
            "image": image,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long),
            "masks": torch.stack(masks) if (self.use_segmentation and len(masks) > 0) else None,
            "mpp": torch.tensor(mpp, dtype=torch.float32),
            "gt_area_m2": torch.tensor(gt_area_m2, dtype=torch.float32) if gt_area_m2 else torch.zeros((0,), dtype=torch.float32),
        }
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_coco:
            img_id = self.image_ids[idx]
            return self._parse_coco(img_id)
        return self._parse_legacy(self.samples[idx])


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