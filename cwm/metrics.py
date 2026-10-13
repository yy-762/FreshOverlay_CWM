from __future__ import annotations

import numpy as np

TARGET_NAMES = ["sinr_dB", "host_rate_Mbps", "passive_rate_kbps", "aoi_ms", "cbr", "contender_load"]


def regression_metrics(pred: np.ndarray, true: np.ndarray, train_std: np.ndarray):
    err = pred - true
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    train_nrmse = rmse / np.maximum(np.asarray(train_std), 1e-8)
    eval_std = np.std(true, axis=0)
    eval_nrmse = rmse / np.maximum(eval_std, 1e-8)
    r2 = 1.0 - np.sum(err ** 2, axis=0) / np.maximum(
        np.sum((true - np.mean(true, axis=0)) ** 2, axis=0), 1e-8
    )
    return {"rmse": rmse, "mae": mae, "nrmse_std": train_nrmse,
            "nrmse_eval_std": eval_nrmse, "r2": r2}
