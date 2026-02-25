# UAV Pothole Area Measurement (YOLOv11 + Adaptive Scale Head)

End-to-end PyTorch scaffold for joint pothole detection, optional segmentation, and meters-per-pixel (mpp) regression.

## Repo structure

```
repo/
  models/
  datasets/
  train.py
  eval.py
  infer.py
  utils/
  configs/
```

## Why global mpp?

This implementation predicts a **global per-image mpp** from shared P3/P4/P5 neck features.
For UAV imagery, camera pose/intrinsics are usually coherent across a frame, so global mpp is stable and easier to supervise.

## Training

```bash
python train.py --data-root data --train-ann data/train.json --with-seg
```

## Evaluation

```bash
python eval.py --data-root data --val-ann data/val.json --ckpt runs/train/last.pt --with-seg
```

## Inference

```bash
python infer.py --data-root data --ann data/test.json --ckpt runs/train/last.pt --out-csv runs/infer/pothole_areas.csv
```

CSV columns:
- image_id
- pothole_id
- conf
- area_px
- mpp_pred
- area_m2_pred

## TODOs

- Replace placeholder detection decode/loss with full YOLOv11 assigner + NMS.
- Align segmentation with instance masks for per-object area.
- Plug in true mAP evaluation backend.
- Adapt dataset parser to your annotation schema if needed.
