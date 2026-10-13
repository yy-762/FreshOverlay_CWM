from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cwm.checkpoint import load_checkpoint
from cwm.config import resolve_data_paths, resolve_output_dir
from cwm.data import RolloutWindowDataset, load_stage2_mat, sort_by_trajectory
from cwm.dynamics import compose_next_x_normalized, project_y_raw
from cwm.metrics import TARGET_NAMES, regression_metrics
from cwm.utils import ensure_dir, load_yaml, resolve_device


def rollout(model, x0, aseq, normalizer):
    ys = []
    x = x0
    for h in range(aseq.shape[1]):
        y = model(x, aseq[:, h])
        ys.append(y)
        x = compose_next_x_normalized(y, x, normalizer, project_physical=True)
    return torch.stack(ys, 1)


def evaluate_split(split_name, mat_path, model, normalizer, device, horizon, batch_size, max_windows, out_dir):
    arr = load_stage2_mat(mat_path, with_aux=False)
    ds = RolloutWindowDataset(arr, horizon, normalizer, max_windows=max_windows, seed=99)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    pred_all = [[] for _ in range(horizon)]
    true_all = [[] for _ in range(horizon)]
    model.eval()
    with torch.inference_mode():
        for x0, aseq, yseq in tqdm(loader, desc=f"Evaluate {split_name}"):
            x0 = x0.to(device); aseq = aseq.to(device)
            pred_norm = rollout(model, x0, aseq, normalizer)
            pred = project_y_raw(normalizer.denormalize_y_t(pred_norm)).cpu().numpy()
            true = normalizer.denormalize_y_np(yseq.numpy())
            for h in range(horizon):
                pred_all[h].append(pred[:, h])
                true_all[h].append(true[:, h])

    rows = []
    train_std = normalizer.y_std
    for h in range(horizon):
        p = np.concatenate(pred_all[h], 0)
        t = np.concatenate(true_all[h], 0)
        m = regression_metrics(p, t, train_std)
        for j, name in enumerate(TARGET_NAMES):
            rows.append({
                "split": split_name,
                "horizon": h + 1,
                "target": name,
                "RMSE": float(m["rmse"][j]),
                "MAE": float(m["mae"][j]),
                "NRMSE_std": float(m["nrmse_std"][j]),
                "NRMSE_eval_std": float(m["nrmse_eval_std"][j]),
                "R2": float(m["r2"][j]),
                "windows": len(ds),
            })
        if h == 0:
            # Save representative scatter data for paper-quality plotting later.
            n = min(10000, len(p))
            idx = np.linspace(0, len(p)-1, n, dtype=int)
            pd.DataFrame({
                "true_host_rate_Mbps": t[idx,1], "pred_host_rate_Mbps": p[idx,1],
                "true_passive_rate_kbps": t[idx,2], "pred_passive_rate_kbps": p[idx,2],
                "true_aoi_ms": t[idx,3], "pred_aoi_ms": p[idx,3],
            }).to_csv(out_dir / f"{split_name}_one_step_scatter.csv", index=False)
    if split_name == "test":
        plot_test_trace(arr, model, normalizer, device, out_dir)
    return rows


def plot_nrmse(df, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for target in ["host_rate_Mbps", "passive_rate_kbps", "aoi_ms", "cbr"]:
        q = df[df.target == target]
        ax.plot(q.horizon, q.NRMSE_eval_std, marker="o", linewidth=2, label=target)
    ax.set_xlabel("Prediction horizon H")
    ax.set_ylabel("NRMSE (normalized by held-out std.)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_generalization(df, out_path):
    targets = ["host_rate_Mbps", "passive_rate_kbps", "aoi_ms", "cbr", "contender_load"]
    labels = ["Host rate", "Passive rate", "AoI", "CBR", "Contender load"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, target, label in zip(axes.flat, targets, labels):
        for split in ["test", "unseen"]:
            q = df[(df.split == split) & (df.target == target)]
            ax.plot(q.horizon, q.NRMSE_eval_std, "o-", label=split)
        ax.set(xlabel="Prediction step", ylabel="NRMSE", title=label)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    axes.flat[-1].axis("off")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_test_trace(arr, model, normalizer, device, out_dir, length=100):
    arr = sort_by_trajectory(arr)
    key = arr.Meta[:, [0, 3, 4]]
    breaks = np.r_[0, np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1, len(key)]
    groups = [(int(s), int(e)) for s, e in zip(breaks[:-1], breaks[1:]) if e - s >= length]
    if not groups:
        raise ValueError("No held-out trajectory is long enough for a trace plot")
    # Pick a trajectory near the median number of observed AoI resets, so the
    # illustration is neither a cherry-picked best case nor an all-stale case.
    reset_counts = np.array([
        np.count_nonzero(np.diff(arr.Y[s:s + length, 3]) < -10.0)
        for s, _ in groups
    ])
    median_resets = np.median(reset_counts)
    start, _ = groups[int(np.argmin(np.abs(reset_counts - median_resets)))]
    sl = slice(start, start + length)
    with torch.inference_mode():
        x = torch.as_tensor(normalizer.normalize_x_np(arr.X[sl]), device=device)
        a = torch.as_tensor(normalizer.normalize_a_np(arr.A[sl]), device=device)
        pred = project_y_raw(normalizer.denormalize_y_t(model(x, a))).cpu().numpy()
    observed = arr.Y[sl]
    time_index = arr.Meta[sl, 5].astype(int)
    trace = pd.DataFrame({
        "time_index": time_index,
        "observed_throughput_mbps": observed[:, 1] + observed[:, 2] / 1000,
        "predicted_throughput_mbps": pred[:, 1] + pred[:, 2] / 1000,
        "observed_aoi_ms": observed[:, 3],
        "predicted_aoi_ms": pred[:, 3],
    })
    trace.to_csv(out_dir / "test_one_step_trace.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, stem, label in [(axes[0], "throughput_mbps", "Throughput (Mbps)"),
                            (axes[1], "aoi_ms", "AoI (ms)")]:
        ax.plot(time_index, trace[f"observed_{stem}"], label="Observed", lw=1.5)
        ax.plot(time_index, trace[f"predicted_{stem}"], label="One-step predicted", lw=1.5)
        ax.set(xlabel="Decision epoch", ylabel=label)
        ax.legend(frameon=False)
    fig.savefig(out_dir / "fig_test_one_step_trace.png", dpi=220)
    plt.close(fig)


def scatter_plot(csv_path, out_path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(df.true_aoi_ms, df.pred_aoi_ms, s=7, alpha=0.25)
    lo = float(min(df.true_aoi_ms.min(), df.pred_aoi_ms.min()))
    hi = float(max(df.true_aoi_ms.max(), df.pred_aoi_ms.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed AoI (ms)")
    ax.set_ylabel("Predicted AoI (ms)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    _, paths = resolve_data_paths(cfg, args.config)
    out = ensure_dir(Path(args.output_dir) if args.output_dir else resolve_output_dir(cfg, args.config) / "evaluation")
    device = resolve_device(cfg["hardware"]["device"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_output_dir(cfg, args.config) / "checkpoints" / "best.pt"
    _, model, normalizer = load_checkpoint(ckpt_path, device)
    model.eval()

    H = int(cfg["evaluation"]["max_horizon"])
    all_rows = []
    for name in ["test", "unseen"]:
        all_rows += evaluate_split(
            name, paths[name], model, normalizer, device, H,
            int(cfg["evaluation"]["batch_size"]),
            cfg["evaluation"].get("max_windows", None), out,
        )
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "prediction_metrics.csv", index=False)
    plot_nrmse(df[df.split == "test"], out / "fig_prediction_nrmse_test.png")
    plot_generalization(df, out / "fig_prediction_generalization.png")
    scatter_plot(out / "test_one_step_scatter.csv", out / "fig_pred_vs_observed_aoi.png")
    print(df.to_string(index=False))
    print(f"Saved evaluation to {out}")


if __name__ == "__main__":
    main()
