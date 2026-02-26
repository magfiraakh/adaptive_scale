# apeakek
from __future__ import annotations

import argparse
import typing

import torch
from torch.utils.data import DataLoader

from datasets.pothole_dataset import PotholeDataset, collate_fn
from models.yolo_scale import YOLOv11Scale
from utils.postprocess import compute_areas_for_image, write_area_csv


def decode_detections_placeholder(det_preds: typing.List[torch.Tensor], image_hw: tuple[int, int]) -> typing.List[typing.List[typing.Dict]]:
    """Decode YOLO outputs into per-image detections.

    TODO: implement proper anchor/grid decoding + NMS.
    Current placeholder returns an empty list per image.
    """
    b = det_preds[0].shape[0]
    return [[] for _ in range(b)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--ann", type=str, default="data/test.json")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--with-seg", action="store_true")
    p.add_argument("--out-csv", type=str, default="runs/infer/pothole_areas.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = PotholeDataset(args.data_root, args.ann, use_segmentation=args.with_seg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = YOLOv11Scale(num_classes=args.num_classes, with_seg=args.with_seg).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(args.device)
            out = model(images)
            decoded = decode_detections_placeholder(out.det_preds, image_hw=images.shape[-2:])
            for i, image_id in enumerate(batch["image_ids"]):
                rows.extend(
                    compute_areas_for_image(
                        image_id=image_id,
                        dets=decoded[i],
                        mpp_pred=float(out.mpp[i].item()),
                        image_hw=tuple(images.shape[-2:]),
                    )
                )

    write_area_csv(rows, args.out_csv)
    print(f"wrote {len(rows)} pothole rows to {args.out_csv}")


if __name__ == "__main__":
    main()
