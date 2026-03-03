import json
import math
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw


def rasterize_polygon(polygon_xy, width, height):
    """
    polygon_xy format dataset kamu: [[x,y],[x,y],...]
    fallback: [x1,y1,x2,y2,...]
    return mask (H,W) {0,1}
    """
    if polygon_xy is None or len(polygon_xy) < 3:
        return np.zeros((height, width), dtype=np.uint8)

    # list-of-pairs
    if isinstance(polygon_xy[0], (list, tuple)) and len(polygon_xy[0]) == 2:
        pts = [(float(x), float(y)) for x, y in polygon_xy]
    else:
        # flat list
        if len(polygon_xy) < 6:
            return np.zeros((height, width), dtype=np.uint8)
        pts = [(float(polygon_xy[i]), float(polygon_xy[i + 1])) for i in range(0, len(polygon_xy), 2)]

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def regression_metrics(y_pred, y_true, eps=1e-9):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = math.sqrt(np.mean((y_pred - y_true) ** 2))
    mape = np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path COCO _annotations.json (SetB) yang sudah punya images[].mpp")
    ap.add_argument("--pred", required=True, help="Path areas_setB.csv (hasil infer)")
    ap.add_argument("--out", default="eval_area_setB.csv", help="Output CSV evaluasi")
    args = ap.parse_args()

    # --- Load COCO ---
    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # map image_id -> info (w,h,mpp_gt)
    img_info = {}
    for im in images:
        iid = im["id"]
        w = int(im.get("width", 640))
        h = int(im.get("height", 640))
        mpp = im.get("mpp", None)
        if mpp is None:
            raise ValueError(f"Image id {iid} tidak punya field 'mpp'. Pastikan COCO SetB sudah digabung mpp.")
        img_info[iid] = {"width": w, "height": h, "mpp_gt": float(mpp)}

    # group annotations per image_id
    anns_by_iid = {}
    for a in anns:
        iid = a["image_id"]
        anns_by_iid.setdefault(iid, []).append(a)

    # --- Compute GT areas per image (sum semua polygon di image tsb) ---
    gt_rows = []
    for iid, info in img_info.items():
        w, h = info["width"], info["height"]
        mpp_gt = info["mpp_gt"]

        # sum mask area (pixel) untuk semua object di image (SetB biasanya 1 object)
        total_area_px = 0.0
        for a in anns_by_iid.get(iid, []):
            poly = a.get("polygon_xy", None)
            if poly is None:
                continue
            mask = rasterize_polygon(poly, w, h)  # (H,W)
            total_area_px += float(mask.sum())

        gt_area_m2 = total_area_px * (mpp_gt ** 2)

        gt_rows.append({
            "image_id": iid,
            "gt_area_px": total_area_px,
            "mpp_gt": mpp_gt,
            "gt_area_m2": gt_area_m2,
        })

    gt_df = pd.DataFrame(gt_rows)

    # --- Load predictions ---
    pred_df = pd.read_csv(args.pred)

    # pastikan tipe image_id match (di pred bisa string/angka)
    # coba konversi ke int kalau bisa
    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return x

    pred_df["image_id"] = pred_df["image_id"].apply(to_int_safe)
    gt_df["image_id"] = gt_df["image_id"].apply(to_int_safe)

    # --- Merge ---
    df = pred_df.merge(gt_df, on="image_id", how="inner")

    if len(df) == 0:
        raise ValueError("Hasil merge kosong. Cek apakah image_id di CSV pred cocok dengan COCO images[id].")

    # --- Errors ---
    df["abs_err_area_px"] = (df["area_px"] - df["gt_area_px"]).abs()
    df["abs_err_area_m2"] = (df["area_m2_pred"] - df["gt_area_m2"]).abs()
    df["pct_err_area_m2"] = df["abs_err_area_m2"] / (df["gt_area_m2"].abs() + 1e-9) * 100.0

    # --- Metrics summary (area m2) ---
    metrics = regression_metrics(df["area_m2_pred"].values, df["gt_area_m2"].values)

    # save
    df.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print("N images evaluated:", len(df))
    print("Area(m2) metrics:", metrics)
    print("Mean abs err (m2):", float(df["abs_err_area_m2"].mean()))
    print("Mean pct err (m2):", float(df["pct_err_area_m2"].mean()))


if __name__ == "__main__":
    main()