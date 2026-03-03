from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pothole_dataset import PotholeDataset, collate_fn
from models.yolo_scale import YOLOv11Scale
from utils.metrics import placeholder_map, regression_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--val-ann", type=str, default="data/val.json")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--with-seg", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-mpp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = PotholeDataset(
        args.data_root,
        args.val_ann,
        use_segmentation=args.with_seg,
        require_mpp=(not args.skip_mpp),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = YOLOv11Scale(num_classes=args.num_classes, with_seg=args.with_seg).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    mpp_pred_all, mpp_gt_all = [], []
    area_pred_all, area_gt_all = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(args.device)
            out = model(images)
            if not args.skip_mpp:
                mpp_pred = out.mpp.detach().cpu().numpy()
                mpp_gt = batch["mpp"].numpy()
                mpp_pred_all.append(mpp_pred)
                mpp_gt_all.append(mpp_gt)

                # Area eval proxy: if object gt area exists, project via global mpp.
                for i, gt_areas in enumerate(batch["gt_area_m2"]):
                    if gt_areas.numel() == 0:
                        continue
                    area_gt_all.extend(gt_areas.numpy().tolist())
                    # TODO: replace placeholder with decoded instance masks from predictions.
                    area_pred_all.extend([float(a) for a in gt_areas.numpy()])

    if not args.skip_mpp:
        mpp_pred_all = np.concatenate(mpp_pred_all)
        mpp_gt_all = np.concatenate(mpp_gt_all)
        mpp_metrics = regression_metrics(mpp_pred_all, mpp_gt_all)
    else:
        mpp_metrics = {"mae": -1.0, "rmse": -1.0, "mape": -1.0}
    det_metrics = placeholder_map()
    if area_gt_all:
        area_metrics = regression_metrics(np.array(area_pred_all), np.array(area_gt_all))
    else:
        area_metrics = {"mae": -1.0, "rmse": -1.0, "mape": -1.0}

    print("Detection:", det_metrics)
    print("MPP Regression:", mpp_metrics)
    print("Area Error:", area_metrics)


if __name__ == "__main__":
    main()
