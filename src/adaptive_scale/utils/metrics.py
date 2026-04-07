from __future__ import annotations

from typing import Dict

import numpy as np


def regression_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(gt), 1e-8, None)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def placeholder_map() -> Dict[str, float]:
    """TODO: wire pycocotools/torchmetrics mAP with decoded predictions."""
    return {"mAP50": -1.0, "mAP50_95": -1.0}
