# src/adaptive_scale/models/yolo_scale.py
"""
YOLOv11-style multi-task model dengan Regression Scale Head (RSH).

ARSITEKTUR KESELURUHAN:
    Input: image [B, 3, H, W] + metadata [B, 4]
        │
        ├── TinyBackbone ──────────────────────────────────────────────
        │   stem → stage2 → P3 → P4 → P5
        │   Channel (width=32): P3=128, P4=256, P5=512
        │
        ├── TinyNeck (FPN) ─────────────────────────────────────────────
        │   P5 → upsample → fuse P4 → N4
        │   N4 → upsample → fuse P3 → N3
        │   Output: N3=128ch, N4=256ch, N5=512ch
        │
        ├── DetectHead(N3, N4, N5) ─────────────────────────────────────
        │   Output: List[Tensor] — [B, A*(5+C), H, W] per skala
        │
        ├── SegHead(N3) ────────────────────────────────────────────────
        │   Output: [B, num_masks, H_out, W_out]
        │
        └── RegressionScaleHead(N3, N4, N5, metadata) ─────────────────
            Visual branch : GAP setiap skala → concat → [B, hidden*3]
            Meta branch   : MetadataEncoder  → [B, meta_dim]
            Fusion        : concat → FC → GSD [B]
            Output        : mpp [B], log_var [B] (opsional)

PERUBAHAN DARI VERSI EXISTING (GlobalScaleHead):
    1. GlobalScaleHead → RegressionScaleHead
       - Tambah MetadataEncoder branch
       - Fusion visual + metadata sebelum regressor
       - forward() menerima (neck_feats, metadata)
    2. YOLOv11Scale.forward() menerima metadata [B, 4] sebagai input
    3. compute_gsd_gt_batch() tersedia sebagai fungsi standalone
    4. ModelOutput tambah field gsd_gt untuk traceability

KONVENSI RESOLUSI (KRITIS):
    metadata[:, 3] = resolution HARUS resolusi ASLI (mis. 3840px untuk DJI Mavic 3)
    BUKAN resolusi YOLO (mis. 640px).
    GSD yang diprediksi berlaku untuk resolusi asli.
    Mask piksel juga idealnya dihitung pada resolusi asli sebelum hitung luas.

FORMULA GSD:
    GSD (m/px) = (altitude_m × sensor_width_mm)
                 / (focal_length_mm × resolution_px_asli)

Karena sensor_width_mm dan focal_length_mm sama-sama dalam mm,
maka satuannya saling cancel, hasil akhir langsung m/px.

FORMULA LUAS FISIK:
    A (m²) = N_piksel_mask_resolusi_asli × GSD²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# KONSTANTA INDEKS METADATA
# ═══════════════════════════════════════════════════════════════════════

META_IDX_ALTITUDE   = 0   # altitude_m       (meter)
META_IDX_FOCAL_LEN  = 1   # focal_length_mm  (mm)
META_IDX_SENSOR_W   = 2   # sensor_width_mm  (mm)
META_IDX_RESOLUTION = 3   # resolution_px    (piksel, resolusi ASLI)

# Default DJI Mavic 3
DJI_MAVIC3_FOCAL_MM   = 12.29
DJI_MAVIC3_SENSOR_MM  = 17.3
DJI_MAVIC3_RES_PX     = 3840
YOLO_INPUT_RES_PX     = 640
RESOLUTION_SCALE      = DJI_MAVIC3_RES_PX / YOLO_INPUT_RES_PX  # = 6.0


# ═══════════════════════════════════════════════════════════════════════
# GSD GROUND TRUTH CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

def compute_gsd_gt_batch(metadata: torch.Tensor) -> torch.Tensor:
    """
    Hitung GSD ground truth dari metadata penerbangan.

    FORMULA:
        GSD (m/px) = (altitude_m × sensor_width_mm)
                     / (focal_length_mm × resolution_px)

    Catatan penting:
        Tidak perlu /1000.
        Alasannya, sensor_width_mm dan focal_length_mm sama-sama dalam mm,
        sehingga rasio (sensor_width_mm / focal_length_mm) menjadi tak berdimensi.
        Maka:
            altitude_m × (sensor_width_mm / focal_length_mm) / resolution_px
        langsung menghasilkan meter per piksel.

    Args:
        metadata: Tensor [B, 4]
            [:, 0] altitude_m       — ketinggian terbang (meter)
            [:, 1] focal_length_mm  — panjang fokus lensa (mm)
            [:, 2] sensor_width_mm  — lebar sensor fisik (mm)
            [:, 3] resolution_px    — resolusi lebar gambar ASLI (piksel)

    Returns:
        gsd_gt: Tensor [B] dalam satuan m/px

    Contoh:
        >>> meta = torch.tensor([
        ...     [ 5.0, 12.29, 17.3, 3840.0],
        ...     [10.0, 12.29, 17.3, 3840.0],
        ...     [15.0, 12.29, 17.3, 3840.0],
        ...     [20.0, 12.29, 17.3, 3840.0],
        ... ], dtype=torch.float32)
        >>> gsd = compute_gsd_gt_batch(meta)
        >>> for h, g in zip([5, 10, 15, 20], gsd.tolist()):
        ...     print(f"H={h:2d}m → GSD={g:.6f} m/px")
        H= 5m → GSD=0.001832 m/px
        H=10m → GSD=0.003665 m/px
        H=15m → GSD=0.005497 m/px
        H=20m → GSD=0.007330 m/px
    """
    altitude_m      = metadata[:, META_IDX_ALTITUDE]    # [B]
    focal_length_mm = metadata[:, META_IDX_FOCAL_LEN]   # [B]
    sensor_width_mm = metadata[:, META_IDX_SENSOR_W]    # [B]
    resolution_px   = metadata[:, META_IDX_RESOLUTION]  # [B]

    gsd_gt = (altitude_m * sensor_width_mm) / (focal_length_mm * resolution_px)
    return gsd_gt  # [B]


def compute_gsd_gt_scalar(
    altitude_m: float,
    focal_length_mm: float = DJI_MAVIC3_FOCAL_MM,
    sensor_width_mm: float = DJI_MAVIC3_SENSOR_MM,
    resolution_px: int = DJI_MAVIC3_RES_PX,
) -> float:
    """
    Versi scalar dari compute_gsd_gt_batch untuk debugging dan validasi.

    Contoh:
        >>> gsd = compute_gsd_gt_scalar(altitude_m=10.0)
        >>> print(f"{gsd:.6f} m/px")
        0.003665 m/px
    """
    return (altitude_m * sensor_width_mm) / (focal_length_mm * resolution_px)


def compute_physical_area_from_mask(
    mask_logits: torch.Tensor,
    gsd: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    """
    Hitung luas fisik (m²) dari mask dan GSD.

    Args:
        mask_logits: Tensor [B, 1, H, W] atau [B, H, W]
        gsd: Tensor [B] dalam m/px
        threshold: ambang binarisasi mask
        from_logits: jika True, gunakan sigmoid dulu

    Returns:
        area_m2: Tensor [B]
    """
    if mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits[:, 0]  # [B, H, W]
    elif mask_logits.ndim != 3:
        raise ValueError("mask_logits harus berbentuk [B, 1, H, W] atau [B, H, W]")

    if gsd.ndim != 1:
        raise ValueError("gsd harus berbentuk [B]")

    probs = torch.sigmoid(mask_logits) if from_logits else mask_logits
    mask_bin = (probs > threshold).float()
    pixel_count = mask_bin.flatten(1).sum(dim=1)  # [B]
    area_m2 = pixel_count * (gsd ** 2)
    return area_m2


# ═══════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class ConvBNAct(nn.Module):
    """
    Conv2d + BatchNorm2d + SiLU.
    Building block dasar.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
    ) -> None:
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
    """
    Placeholder backbone — output P3/P4/P5.

    Channel output dengan width=32:
        P3: 128ch  (stride 8  dari input)
        P4: 256ch  (stride 16 dari input)
        P5: 512ch  (stride 32 dari input)

    CATATAN UNTUK PRODUKSI:
        Ganti dengan backbone YOLOv11-Seg asli dari Ultralytics.
        Sesuaikan in_channels di RegressionScaleHead dan DetectHead
        dengan channel output backbone yang digunakan.

        YOLOv11n-seg: P3=128,  P4=256,  P5=512
        YOLOv11s-seg: P3=256,  P4=512,  P5=512
        YOLOv11m-seg: P3=384,  P4=768,  P5=768
    """

    def __init__(self, in_channels: int = 3, width: int = 32) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, width, 3, 2)
        self.stage2 = nn.Sequential(
            ConvBNAct(width, width * 2, 3, 2),
            ConvBNAct(width * 2, width * 2),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(width * 2, width * 4, 3, 2),
            ConvBNAct(width * 4, width * 4),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(width * 4, width * 8, 3, 2),
            ConvBNAct(width * 8, width * 8),
        )
        self.stage5 = nn.Sequential(
            ConvBNAct(width * 8, width * 16, 3, 2),
            ConvBNAct(width * 16, width * 16),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return {"p3": p3, "p4": p4, "p5": p5}


class TinyNeck(nn.Module):
    """
    Simple FPN-like neck.

    Top-down pathway:
        P5 → reduce → upsample → cat(P4) → N4
        N4 → reduce → upsample → cat(P3) → N3

    Output channel sama dengan input P3/P4, sedangkan P5 dipertahankan.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        c3, c4, c5 = width * 4, width * 8, width * 16

        self.reduce_p5 = ConvBNAct(c5, c4, 1, 1, 0)
        self.out_p4    = ConvBNAct(c4 + c4, c4)
        self.reduce_p4 = ConvBNAct(c4, c3, 1, 1, 0)
        self.out_p3    = ConvBNAct(c3 + c3, c3)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3, p4, p5 = feats["p3"], feats["p4"], feats["p5"]

        p5_up = F.interpolate(
            self.reduce_p5(p5),
            size=p4.shape[-2:],
            mode="nearest",
        )
        n4 = self.out_p4(torch.cat([p4, p5_up], dim=1))

        n4_up = F.interpolate(
            self.reduce_p4(n4),
            size=p3.shape[-2:],
            mode="nearest",
        )
        n3 = self.out_p3(torch.cat([p3, n4_up], dim=1))

        return {"p3": n3, "p4": n4, "p5": p5}


class DetectHead(nn.Module):
    """
    Detection head — output dense predictions per skala.

    Output shape per level: [B, anchors*(5+num_classes), H, W]
        5 = [x, y, w, h, objectness]

    Catatan:
        Detection loss/NMS belum diimplementasikan penuh di file ini.
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        anchors: int = 3,
    ) -> None:
        super().__init__()
        out_c = anchors * (5 + num_classes)
        self.heads = nn.ModuleList([
            nn.Conv2d(c, out_c, 1) for c in in_channels
        ])

    def forward(self, neck_feats: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        feats = [neck_feats["p3"], neck_feats["p4"], neck_feats["p5"]]
        return [head(f) for head, f in zip(self.heads, feats)]


class SegHead(nn.Module):
    """
    Segmentation head — output mask logits.

    Input:
        neck_feats["p3"] dengan stride 8 terhadap input model.

    Output:
        mask_logits [B, num_masks, H_out, W_out]

    Catatan:
        - Secara default, head akan meng-upsample ke resolusi input model.
        - Untuk kalkulasi luas fisik yang akurat, gunakan `seg_output_size`
          sesuai resolusi asli gambar pada saat forward().
    """

    def __init__(
        self,
        in_channels: int,
        num_masks: int = 1,
        mid_channels: int = 64,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, 3, 1),
            ConvBNAct(mid_channels, mid_channels, 3, 1),
            nn.Conv2d(mid_channels, num_masks, 1),
        )

    def forward(
        self,
        neck_feats: Dict[str, torch.Tensor],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = neck_feats["p3"]
        logits = self.block(x)

        if output_size is None:
            output_size = (x.shape[-2] * 8, x.shape[-1] * 8)

        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class MetadataEncoder(nn.Module):
    """
    Encode metadata numerik [altitude, focal_len, sensor_width, resolution]
    menjadi embedding padat.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 32,
        out_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


class RegressionScaleHead(nn.Module):
    """
    Regression Scale Head (RSH).

    Alur:
        1. Global average pooling pada N3/N4/N5
        2. Tiap skala diproyeksikan ke dimensi hidden yang sama
        3. Metadata diencode terpisah
        4. Fusi visual + metadata
        5. Regress GSD positif (m/px)
        6. Opsional: prediksi log-variance untuk heteroscedastic uncertainty
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 128,
        meta_dim: int = 32,
        predict_log_var: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("in_channels harus berisi 3 elemen untuk P3/P4/P5")

        self.visual_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, hidden_dim),
                nn.SiLU(inplace=True),
            )
            for c in in_channels
        ])

        self.meta_encoder = MetadataEncoder(
            in_dim=4,
            hidden_dim=max(meta_dim, 16),
            out_dim=meta_dim,
        )

        fusion_in = hidden_dim * 3 + meta_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )

        self.gsd_regressor = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1) if predict_log_var else None

    def _pool_and_project(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)  # [B, C]
        return proj(pooled)  # [B, hidden]

    def forward(
        self,
        neck_feats: Dict[str, torch.Tensor],
        metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feats = [neck_feats["p3"], neck_feats["p4"], neck_feats["p5"]]

        visual_vecs = [
            self._pool_and_project(f, proj)
            for f, proj in zip(feats, self.visual_proj)
        ]
        visual_feat = torch.cat(visual_vecs, dim=1)
        meta_feat = self.meta_encoder(metadata)

        fused = self.fusion(torch.cat([visual_feat, meta_feat], dim=1))

        # Softplus memastikan output GSD selalu positif
        gsd_pred = F.softplus(self.gsd_regressor(fused)).squeeze(1) + 1e-8
        log_var = self.log_var_head(fused).squeeze(1) if self.log_var_head is not None else None

        return gsd_pred, log_var


@dataclass
class ModelOutput:
    """
    Container output utama model multi-task.
    """
    det_preds: List[torch.Tensor]
    mask_logits: torch.Tensor
    gsd_pred: torch.Tensor
    log_var: Optional[torch.Tensor] = None
    gsd_gt: Optional[torch.Tensor] = None


class YOLOv11Scale(nn.Module):
    """
    YOLOv11-style multi-task model dengan Regression Scale Head.

    Args:
        num_classes: jumlah kelas untuk deteksi
        num_masks: jumlah channel mask output
        in_channels: channel input citra
        width: base width backbone/neck
        anchors: jumlah anchor per skala
        rsh_hidden_dim: hidden dim untuk visual branch RSH
        meta_dim: dimensi embedding metadata
        predict_log_var: apakah memprediksi uncertainty log-variance
    """

    def __init__(
        self,
        num_classes: int,
        num_masks: int = 1,
        in_channels: int = 3,
        width: int = 32,
        anchors: int = 3,
        rsh_hidden_dim: int = 128,
        meta_dim: int = 32,
        predict_log_var: bool = True,
    ) -> None:
        super().__init__()
        c3, c4, c5 = width * 4, width * 8, width * 16

        self.backbone = TinyBackbone(in_channels=in_channels, width=width)
        self.neck = TinyNeck(width=width)
        self.detect_head = DetectHead(
            in_channels=[c3, c4, c5],
            num_classes=num_classes,
            anchors=anchors,
        )
        self.seg_head = SegHead(
            in_channels=c3,
            num_masks=num_masks,
            mid_channels=max(c3 // 2, 32),
        )
        self.scale_head = RegressionScaleHead(
            in_channels=[c3, c4, c5],
            hidden_dim=rsh_hidden_dim,
            meta_dim=meta_dim,
            predict_log_var=predict_log_var,
        )

    @staticmethod
    def _validate_metadata(metadata: torch.Tensor) -> None:
        if metadata.ndim != 2 or metadata.shape[1] != 4:
            raise ValueError(
                "metadata harus berbentuk [B, 4] = "
                "[altitude_m, focal_length_mm, sensor_width_mm, resolution_px_asli]"
            )
        if torch.any(metadata[:, META_IDX_RESOLUTION] <= 0):
            raise ValueError("metadata[:, 3] / resolution_px harus > 0")
        if torch.any(metadata[:, META_IDX_FOCAL_LEN] <= 0):
            raise ValueError("metadata[:, 1] / focal_length_mm harus > 0")
        if torch.any(metadata[:, META_IDX_SENSOR_W] <= 0):
            raise ValueError("metadata[:, 2] / sensor_width_mm harus > 0")

    @staticmethod
    def infer_original_size(
        images: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        Infer ukuran output segmentasi dari resolusi metadata.

        Asumsi:
            - metadata[:, 3] menyimpan lebar gambar asli
            - rasio aspek dipertahankan saat resize ke input model
            - seluruh batch punya resolusi asli yang sama

        Jika batch memiliki resolusi berbeda, fallback ke resolusi input model.
        """
        in_h, in_w = images.shape[-2:]
        res_w = metadata[:, META_IDX_RESOLUTION]

        if not torch.allclose(res_w, res_w[0].expand_as(res_w)):
            return in_h, in_w

        scale = float(res_w[0].item()) / float(in_w)
        out_h = int(round(in_h * scale))
        out_w = int(round(in_w * scale))
        return out_h, out_w

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor,
        seg_output_size: Optional[Tuple[int, int]] = None,
        return_gsd_gt: bool = True,
    ) -> ModelOutput:
        """
        Args:
            images: Tensor [B, 3, H, W]
            metadata: Tensor [B, 4]
            seg_output_size: ukuran target mask (H_orig, W_orig)
            return_gsd_gt: hitung GSD GT dari metadata untuk traceability
        """
        if images.ndim != 4:
            raise ValueError("images harus berbentuk [B, C, H, W]")
        if images.shape[0] != metadata.shape[0]:
            raise ValueError("Batch size images dan metadata harus sama")

        if not metadata.is_floating_point():
            metadata = metadata.float()

        metadata = metadata.to(images.device)
        self._validate_metadata(metadata)

        feats = self.backbone(images)
        neck_feats = self.neck(feats)

        det_preds = self.detect_head(neck_feats)

        if seg_output_size is None:
            seg_output_size = self.infer_original_size(images, metadata)

        mask_logits = self.seg_head(
            neck_feats,
            output_size=seg_output_size,
        )

        gsd_pred, log_var = self.scale_head(neck_feats, metadata)
        gsd_gt = compute_gsd_gt_batch(metadata) if return_gsd_gt else None

        return ModelOutput(
            det_preds=det_preds,
            mask_logits=mask_logits,
            gsd_pred=gsd_pred,
            log_var=log_var,
            gsd_gt=gsd_gt,
        )


def build_yolo_scale_tiny(
    num_classes: int,
    num_masks: int = 1,
    predict_log_var: bool = True,
) -> YOLOv11Scale:
    """
    Factory helper untuk eksperimen cepat.
    """
    return YOLOv11Scale(
        num_classes=num_classes,
        num_masks=num_masks,
        in_channels=3,
        width=32,
        anchors=3,
        rsh_hidden_dim=128,
        meta_dim=32,
        predict_log_var=predict_log_var,
    )


if __name__ == "__main__":
    # Smoke test sederhana
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_yolo_scale_tiny(
        num_classes=1,
        num_masks=1,
        predict_log_var=True,
    ).to(device)

    images = torch.randn(2, 3, 640, 640, device=device)
    metadata = torch.tensor([
        [10.0, 12.29, 17.3, 3840.0],
        [15.0, 12.29, 17.3, 3840.0],
    ], dtype=torch.float32, device=device)

    out = model(images, metadata)

    print("=== Smoke Test ===")
    for i, p in enumerate(out.det_preds, start=3):
        print(f"det P{i}:      {tuple(p.shape)}")
    print(f"mask_logits:   {tuple(out.mask_logits.shape)}")
    print(f"gsd_pred:      {tuple(out.gsd_pred.shape)} | {out.gsd_pred.detach().cpu().tolist()}")
    if out.log_var is not None:
        print(f"log_var:       {tuple(out.log_var.shape)}")
    if out.gsd_gt is not None:
        print(f"gsd_gt:        {tuple(out.gsd_gt.shape)} | {out.gsd_gt.detach().cpu().tolist()}")

    area_m2 = compute_physical_area_from_mask(out.mask_logits, out.gsd_gt)
    print(f"area_m2:       {tuple(area_m2.shape)} | {area_m2.detach().cpu().tolist()}")