"""YOLOv11-style multi-task skeleton with a global MPP regression head.

This module keeps the architecture intentionally lightweight and extensible.
Replace placeholder blocks with your preferred YOLOv11 backbone/neck/head details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        p = k // 2 if p is None else p
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyBackbone(nn.Module):
    """Placeholder backbone that outputs P3/P4/P5-like features."""

    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        self.stem = ConvBNAct(in_channels, width, 3, 2)
        self.stage2 = nn.Sequential(ConvBNAct(width, width * 2, 3, 2), ConvBNAct(width * 2, width * 2))
        self.stage3 = nn.Sequential(ConvBNAct(width * 2, width * 4, 3, 2), ConvBNAct(width * 4, width * 4))
        self.stage4 = nn.Sequential(ConvBNAct(width * 4, width * 8, 3, 2), ConvBNAct(width * 8, width * 8))
        self.stage5 = nn.Sequential(ConvBNAct(width * 8, width * 16, 3, 2), ConvBNAct(width * 16, width * 16))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return {"p3": p3, "p4": p4, "p5": p5}


class TinyNeck(nn.Module):
    """Simple FPN-like neck with shared multi-scale features."""

    def __init__(self, width: int = 32):
        super().__init__()
        c3, c4, c5 = width * 4, width * 8, width * 16
        self.reduce_p5 = ConvBNAct(c5, c4, 1, 1, 0)
        self.out_p4 = ConvBNAct(c4 + c4, c4)
        self.reduce_p4 = ConvBNAct(c4, c3, 1, 1, 0)
        self.out_p3 = ConvBNAct(c3 + c3, c3)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3, p4, p5 = feats["p3"], feats["p4"], feats["p5"]
        p5_up = F.interpolate(self.reduce_p5(p5), size=p4.shape[-2:], mode="nearest")
        n4 = self.out_p4(torch.cat([p4, p5_up], dim=1))
        n4_up = F.interpolate(self.reduce_p4(n4), size=p3.shape[-2:], mode="nearest")
        n3 = self.out_p3(torch.cat([p3, n4_up], dim=1))
        return {"p3": n3, "p4": n4, "p5": p5}


class DetectHead(nn.Module):
    """Detection head skeleton: outputs dense predictions for each scale.

    Output shape per level: [B, A*(5+num_classes), H, W].
    """

    def __init__(self, in_channels: List[int], num_classes: int, anchors: int = 3):
        super().__init__()
        out_c = anchors * (5 + num_classes)
        self.heads = nn.ModuleList([nn.Conv2d(c, out_c, 1) for c in in_channels])

    def forward(self, neck_feats: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        feats = [neck_feats["p3"], neck_feats["p4"], neck_feats["p5"]]
        return [head(f) for head, f in zip(self.heads, feats)]


class SegHead(nn.Module):
    """Optional semantic/instance mask head (prototype-style placeholder)."""

    def __init__(self, in_channels: int, num_masks: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, in_channels),
            nn.Conv2d(in_channels, num_masks, kernel_size=1),
        )

    def forward(self, p3: torch.Tensor, image_hw: torch.Size) -> torch.Tensor:
        logits = self.net(p3)
        return F.interpolate(logits, size=image_hw, mode="bilinear", align_corners=False)


class GlobalScaleHead(nn.Module):
    """Predict global meters-per-pixel (mpp) from fused P3/P4/P5 features.

    We choose global mpp because UAV altitude/camera intrinsics are often shared
    across the full frame, giving a stable scalar that regularizes area estimates.
    """

    def __init__(self, in_channels: List[int], hidden: int = 128, uncertainty: bool = True):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(c, hidden, k=1, s=1, p=0) for c in in_channels])
        out_dim = 2 if uncertainty else 1
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
        self.uncertainty = uncertainty

    def forward(self, neck_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pooled = []
        for proj, k in zip(self.proj, ["p3", "p4", "p5"]):
            x = proj(neck_feats[k])
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            pooled.append(x)
        fused = torch.cat(pooled, dim=1)
        raw = self.regressor(fused)
        mpp = F.softplus(raw[:, :1]) + 1e-6
        out = {"mpp": mpp.squeeze(1)}
        if self.uncertainty:
            out["log_var"] = raw[:, 1].clamp(-10, 10)
        return out


@dataclass
class ModelOutput:
    det_preds: List[torch.Tensor]
    seg_logits: Optional[torch.Tensor]
    mpp: torch.Tensor
    mpp_log_var: Optional[torch.Tensor]


class YOLOv11Scale(nn.Module):
    def __init__(self, num_classes: int, with_seg: bool = True, width: int = 32, uncertainty: bool = True):
        super().__init__()
        self.backbone = TinyBackbone(width=width)
        self.neck = TinyNeck(width=width)
        self.detect = DetectHead([width * 4, width * 8, width * 16], num_classes=num_classes)
        self.seg_head = SegHead(width * 4, num_masks=1) if with_seg else None
        self.scale_head = GlobalScaleHead([width * 4, width * 8, width * 16], uncertainty=uncertainty)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        image_hw = x.shape[-2:]
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        det_preds = self.detect(neck_feats)
        seg_logits = self.seg_head(neck_feats["p3"], image_hw) if self.seg_head is not None else None
        scale_out = self.scale_head(neck_feats)
        return ModelOutput(
            det_preds=det_preds,
            seg_logits=seg_logits,
            mpp=scale_out["mpp"],
            mpp_log_var=scale_out.get("log_var"),
        )
