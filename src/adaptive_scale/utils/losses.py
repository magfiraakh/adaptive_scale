from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Combined loss: detection + segmentation + mpp regression.

    Detection loss here is a placeholder; replace `detection_loss` with your
    YOLOv11 target assignment + box/objectness/class losses.
    """

    def __init__(self, seg_weight: float = 1.0, scale_weight: float = 1.0, use_uncertainty: bool = True):
        super().__init__()
        self.seg_weight = seg_weight
        self.scale_weight = scale_weight
        self.use_uncertainty = use_uncertainty
        self.huber = nn.SmoothL1Loss(beta=0.05)

    def detection_loss(self, det_preds: List[torch.Tensor], targets: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        # TODO: replace this with actual YOLO loss components.
        return torch.stack([p.abs().mean() for p in det_preds]).mean() * 0.0

    def segmentation_loss(self, seg_logits: Optional[torch.Tensor], gt_masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        if seg_logits is None:
            return torch.tensor(0.0, device=gt_masks[0].device if gt_masks and gt_masks[0] is not None else "cpu")

        losses = []
        for i, masks in enumerate(gt_masks):
            if masks is None or masks.numel() == 0:
                continue
            target = masks.max(dim=0, keepdim=True).values.to(seg_logits.device)  # union mask per image
            pred = seg_logits[i : i + 1]
            bce = F.binary_cross_entropy_with_logits(pred, target)
            probs = torch.sigmoid(pred)
            inter = (probs * target).sum()
            dice = 1 - (2 * inter + 1) / (probs.sum() + target.sum() + 1)
            losses.append(bce + dice)
        if not losses:
            return torch.tensor(0.0, device=seg_logits.device)
        return torch.stack(losses).mean()

    def scale_loss(self, pred_mpp: torch.Tensor, gt_mpp: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_uncertainty and log_var is not None:
            base = F.smooth_l1_loss(pred_mpp, gt_mpp, reduction="none", beta=0.05)
            return (torch.exp(-log_var) * base + log_var).mean()
        return self.huber(pred_mpp, gt_mpp)

    def forward(self, outputs, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l_det = self.detection_loss(outputs.det_preds, batch)
        l_seg = self.segmentation_loss(outputs.seg_logits, batch["masks"])
        l_scale = self.scale_loss(outputs.mpp, batch["mpp"].to(outputs.mpp.device), outputs.mpp_log_var)
        total = l_det + self.seg_weight * l_seg + self.scale_weight * l_scale
        return {
            "loss_total": total,
            "loss_det": l_det.detach(),
            "loss_seg": l_seg.detach(),
            "loss_scale": l_scale.detach(),
        }
