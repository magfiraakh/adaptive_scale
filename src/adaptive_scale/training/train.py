from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.adaptive_scale.datasets.pothole_dataset import PotholeDataset, collate_fn
from models.yolo_scale import YOLOv11Scale
from src.adaptive_scale.utils.losses import MultiTaskLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--train-ann", type=str, default="data/train.json")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--with-seg", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="runs/train")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = PotholeDataset(args.data_root, args.train_ann, use_segmentation=args.with_seg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = YOLOv11Scale(num_classes=args.num_classes, with_seg=args.with_seg).to(args.device)
    criterion = MultiTaskLoss(seg_weight=1.0, scale_weight=1.0, use_uncertainty=True)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch["images"] = batch["images"].to(args.device)
            batch["mpp"] = batch["mpp"].to(args.device)
            outputs = model(batch["images"])
            losses = criterion(outputs, batch)

            optim.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            optim.step()

        print(
            f"epoch={epoch+1}/{args.epochs} total={losses['loss_total'].item():.4f} "
            f"det={losses['loss_det'].item():.4f} seg={losses['loss_seg'].item():.4f} "
            f"scale={losses['loss_scale'].item():.4f}"
        )

    ckpt = save_dir / "last.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"saved checkpoint to: {ckpt}")


if __name__ == "__main__":
    main()
