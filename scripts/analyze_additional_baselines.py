from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/baselines/formal"
KEY = ["density", "resource", "seed"]
METRICS = ["aggregate_throughput_mbps", "avg_aoi_ms", "p95_aoi_ms",
           "freshness_probability", "violation_rate", "idle_rate"]
ORDER = ["fixed", "mcs_only", "masked_dqn", "cwm_h1", "cwm_h3", "exact_h1"]
LABELS = {"fixed": "Fixed", "mcs_only": "MCS-only", "masked_dqn": "Masked DQN",
          "cwm_h1": "CWM-H1", "cwm_h3": "CWM-H3", "exact_h1": "Exact-H1"}


def validate(df: pd.DataFrame, method: str) -> pd.DataFrame:
    q = df[df.method == method].copy().sort_values(KEY).reset_index(drop=True)
    if len(q) != 75 or q.duplicated(KEY).any() or q[KEY + METRICS].isna().any().any():
        raise ValueError(f"Expected 75 complete unique episodes for {method}, got {len(q)}")
    if not np.allclose(q.violation_rate, 0):
        raise ValueError(f"Non-zero feasibility violations for {method}")
    return q


def ci95(values: pd.Series):
    x = values.to_numpy(float); mean = float(x.mean())
    if np.allclose(x, x[0]):
        return mean, mean
    return t.interval(0.95, len(x) - 1, loc=mean, scale=x.std(ddof=1) / np.sqrt(len(x)))


def main() -> None:
    original = pd.read_csv(ROOT / "outputs/closed_loop/safe_idle_fast/episode_metrics.csv")
    original = original[original.experiment == "comparison"]
    beam4 = pd.read_csv(ROOT / "outputs/closed_loop/beam4_formal/episode_metrics.csv")
    dqn = pd.read_csv(ROOT / "outputs/baselines/formal_masked_dqn/episode_metrics.csv")
    exact = pd.read_csv(ROOT / "outputs/baselines/formal_exact_h1/episode_metrics.csv")
    frames = {
        "fixed": validate(original, "fixed"), "mcs_only": validate(original, "mcs_only"),
        "cwm_h1": validate(original, "cwm_h1"), "cwm_h3": validate(beam4, "cwm_h3"),
        "masked_dqn": validate(dqn, "masked_dqn"), "exact_h1": validate(exact, "exact_h1"),
    }
    combined = pd.concat([frames[name] for name in ORDER], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "episode_metrics_all_controllers.csv", index=False)
    summary = combined.groupby("method")[METRICS].mean().loc[ORDER].reset_index()
    summary.to_csv(OUT / "controller_summary.csv", index=False)
    baseline = frames["cwm_h3"]
    comparisons = []
    for name in ORDER:
        if name == "cwm_h3":
            continue
        paired = baseline[KEY + METRICS].merge(frames[name][KEY + METRICS], on=KEY,
                                               suffixes=("_cwm", "_baseline"), validate="one_to_one")
        for metric in METRICS:
            difference = paired[f"{metric}_baseline"] - paired[f"{metric}_cwm"]
            seed_difference = difference.groupby(paired.seed).mean()
            lo, hi = ci95(seed_difference)
            comparisons.append({"baseline": name, "metric": metric,
                                "baseline_minus_cwm_h3": float(seed_difference.mean()),
                                "ci95_low": float(lo), "ci95_high": float(hi),
                                "episodes": len(paired), "seed_clusters": len(seed_difference)})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_vs_cwm_h3.csv", index=False)
    density = combined.groupby(["method", "density"])[["aggregate_throughput_mbps", "avg_aoi_ms"]].mean().reset_index()
    colors = dict(zip(ORDER, ["#999999", "#CC79A7", "#E69F00", "#56B4E9", "#0072B2", "#009E73"]))
    markers = dict(zip(ORDER, ["X", "s", "D", "^", "o", "P"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for name in ORDER:
        q = density[density.method == name]
        axes[0].plot(q.density, q.aggregate_throughput_mbps, marker=markers[name], color=colors[name],
                     linewidth=2, label=LABELS[name])
        if name != "fixed":
            axes[1].plot(q.density, q.avg_aoi_ms, marker=markers[name], color=colors[name], linewidth=2)
    axes[0].set(xlabel="Robots per resource", ylabel="Aggregate throughput (Mbps)", title="Throughput scaling")
    axes[1].set(xlabel="Robots per resource", ylabel="Average AoI (ms)", title="Freshness scaling")
    fixed_aoi = float(frames["fixed"].avg_aoi_ms.mean())
    axes[1].text(0.02, 0.96, f"Fixed: {fixed_aoi:.0f} ms (off-scale)", transform=axes[1].transAxes,
                 va="top", color=colors["fixed"], fontsize=9)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(OUT / "fig_all_baselines_density_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(summary.round(4).to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
