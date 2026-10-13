from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


METRICS = [
    "aggregate_throughput_mbps",
    "avg_aoi_ms",
    "p95_aoi_ms",
    "freshness_probability",
    "violation_rate",
    "idle_rate",
    "decision_ms_mean",
]


def analyze(input_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    comp = df.loc[df.experiment == "comparison"].copy()
    expected = {"fixed", "mcs_only", "cwm_h1", "cwm_h3"}
    if set(comp.method) != expected or comp.duplicated(["method", "seed", "density", "resource"]).any():
        raise ValueError("Comparison methods or paired episode keys are incomplete")
    if comp.groupby("method").size().nunique() != 1:
        raise ValueError("Comparison methods have different episode counts")

    rows = []
    for baseline in ["cwm_h1", "mcs_only", "fixed"]:
        for metric in METRICS:
            paired = comp.pivot(index=["seed", "density", "resource"], columns="method", values=metric)
            if paired[["cwm_h3", baseline]].isna().any().any():
                raise ValueError(f"Missing paired setting for {baseline}: {metric}")
            # Density and resource are repeated conditions.  Aggregate their
            # paired differences within seed before estimating uncertainty.
            per_seed = (paired["cwm_h3"] - paired[baseline]).groupby(level="seed").mean()
            n = len(per_seed)
            mean = float(per_seed.mean())
            sem = float(per_seed.std(ddof=1) / np.sqrt(n))
            half_width = float(t.ppf(0.975, n - 1) * sem)
            rows.append({
                "comparison": f"cwm_h3_minus_{baseline}",
                "metric": metric,
                "seeds": n,
                "mean_paired_difference": mean,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            })
    paired_summary = pd.DataFrame(rows)
    density_summary = (comp.groupby(["method", "density"])[METRICS]
                       .agg(["mean", "std", "count"]))
    density_summary.columns = [f"{metric}_{stat}" for metric, stat in density_summary.columns]
    density_summary = density_summary.reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_summary.to_csv(out_dir / "paired_comparisons.csv", index=False)
    density_summary.to_csv(out_dir / "density_summary.csv", index=False)
    return paired_summary, density_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/closed_loop/safe_idle_fast/episode_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/closed_loop/safe_idle_fast"))
    args = parser.parse_args()
    paired, density = analyze(args.input, args.output_dir)
    print(paired.to_string(index=False))
    print(f"Density rows: {len(density)}")


if __name__ == "__main__":
    main()
