import argparse
import torch
from torch.utils.data import DataLoader

from datasets.pothole_dataset import PotholeDataset, collate_fn
from models.yolo_scale import YOLOv11Scale
from utils.losses import MultiTaskLoss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--ann", required=True)
    p.add_argument("--init-ckpt", required=True)
    p.add_argument("--out-ckpt", default="/kaggle/working/SetAseg_SetBscale.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ds = PotholeDataset(
        args.data_root,
        args.ann,
        use_segmentation=False,
        require_mpp=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = YOLOv11Scale(num_classes=args.num_classes, with_seg=True).to(device)
    ckpt = torch.load(args.init_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze scale head (and logvar head if exists)
    for p in model.scale_head.parameters():
        p.requires_grad = True
    if getattr(model, "mpp_logvar_head", None) is not None:
        for p in model.mpp_logvar_head.parameters():
            p.requires_grad = True

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = MultiTaskLoss(seg_weight=0.0, scale_weight=1.0, use_uncertainty=True)

    model.train()
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        n = 0
        for batch in dl:
            batch["images"] = batch["images"].to(device)
            batch["mpp"] = batch["mpp"].to(device)

            out = model(batch["images"])
            losses = criterion(out, batch)
            loss = losses["loss_total"]

            optim.zero_grad()
            loss.backward()
            optim.step()

            tot += float(loss.item())
            n += 1
        print(f"epoch {ep}/{args.epochs} scale_loss={tot/max(n,1):.6f}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_ckpt)
    print("Saved:", args.out_ckpt)

if __name__ == "__main__":
    main()