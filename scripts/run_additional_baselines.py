from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from cwm.checkpoint import load_checkpoint
from cwm.dqn import load_dqn
from evaluate_action_fidelity import exact_one_step_utility
from run_closed_loop_evaluation import (
    ContinuousPhyLUT, Controllers, IDLE_ACTION, SimConfig, init_env, observe, step,
)


class AdditionalControllers:
    def __init__(self, base, lut, dqn_checkpoint, device):
        self.base = base
        self.lut = lut
        self.device = torch.device(device)
        checkpoint, self.dqn = load_dqn(dqn_checkpoint, device)
        if not np.array_equal(checkpoint["actions"], lut.actions):
            raise ValueError("DQN action order differs from the evaluation PHY LUT")
        self.mean = np.asarray(checkpoint["state_mean"], np.float32)
        self.std = np.asarray(checkpoint["state_std"], np.float32)

    def exact_h1(self, env, x, cfg, lam=0.5):
        background = self.base.mcs_only(x)
        selected = background.copy()
        for robot, state in enumerate(x):
            candidates = self.lut.feasible(float(state[0]))
            if not len(candidates):
                selected[robot] = IDLE_ACTION
                continue
            scores = exact_one_step_utility(env, robot, candidates, background, self.lut, cfg, lam)
            selected[robot] = candidates[int(np.argmax(scores))]
        return selected

    @torch.inference_mode()
    def masked_dqn(self, x):
        state = ((x - self.mean) / self.std).astype(np.float32)
        q = self.dqn(torch.as_tensor(state, device=self.device)).cpu().numpy()
        selected = np.repeat(IDLE_ACTION[None, :], len(x), axis=0)
        for i, sinr in enumerate(x[:, 0]):
            *_, mask = self.lut.query(np.repeat(float(sinr), len(self.lut.actions)), self.lut.actions)
            valid = np.flatnonzero(mask)
            if len(valid):
                selected[i] = self.lut.actions[valid[np.argmax(q[i, valid])]]
        return selected


def run_episode(method, density, resource, seed, mode, ctrl, lut, cfg, keep_aoi=False):
    env = init_env(density, resource, seed, mode)
    throughput = []
    aois = []
    violations = idle_count = avoidable_idle = 0
    decisions = []
    for _ in range(cfg.steps):
        x = observe(env)
        start = time.perf_counter()
        actions = ctrl.exact_h1(env, x, cfg) if method == "exact_h1" else ctrl.masked_dqn(x)
        decisions.append((time.perf_counter() - start) * 1000)
        result = step(env, actions, lut, cfg)
        throughput.append(result["host_rate"].sum() + result["passive_rate"].sum() / 1000)
        aois.append(1000 * result["aoi"])
        idle_count += int(result["idle"].sum())
        for i in np.flatnonzero(result["idle"]):
            *_, any_valid = lut.query(np.repeat(float(x[i, 0]), len(lut.actions)), lut.actions)
            avoidable_idle += int(any_valid.any())
        violations += int((~result["feasible"]).sum())
    aoi = np.concatenate(aois)
    row = {
        "method": method, "lambda_a": 0.5, "horizon": 1,
        "density": density, "resource": resource, "seed": seed,
        "aggregate_throughput_mbps": float(np.mean(throughput)),
        "avg_aoi_ms": float(aoi.mean()), "p95_aoi_ms": float(np.percentile(aoi, 95)),
        "freshness_probability": float(np.mean(aoi <= 100)),
        "violation_rate": float(violations / aoi.size),
        "idle_rate": float(idle_count / aoi.size),
        "avoidable_idle_rate": float(avoidable_idle / aoi.size),
        "decision_ms_mean": float(np.mean(decisions)),
        "decision_ms_p95": float(np.percentile(decisions, 95)),
        "experiment": "comparison",
    }
    return row, aoi.astype(np.float32) if keep_aoi else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-checkpoint", type=Path,
                        default=Path("outputs/baselines/masked_dqn/masked_dqn.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines/formal"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--method", choices=["exact_h1", "masked_dqn", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dqn_path = args.dqn_checkpoint if args.dqn_checkpoint.is_absolute() else root / args.dqn_checkpoint
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lut = ContinuousPhyLUT(root / "data/stage1_phy_lut/phy_lut_4d.csv")
    _, model, normalizer = load_checkpoint(root / "outputs/default_run/checkpoints/best.pt", device)
    base = Controllers(model, normalizer, lut, device, beam=4)
    ctrl = AdditionalControllers(base, lut, dqn_path, device)
    if args.quick:
        cfg = SimConfig(steps=20); densities = [5]; seeds = [101]; resources = [1]
    elif args.calibrate:
        cfg = SimConfig(steps=100); densities = [5, 20, 40]; seeds = [101, 102]; resources = [1, 2, 3]
    else:
        cfg = SimConfig(steps=300); densities = [5, 10, 20, 30, 40]
        seeds = [101, 102, 103, 104, 105]; resources = [1, 2, 3]
    methods = ["exact_h1", "masked_dqn"] if args.method == "all" else [args.method]
    rows = []
    for method in methods:
        for density in densities:
            for seed in seeds:
                for resource in resources:
                    mode = "bursty" if 1 + (density + seed + resource) % 2 == 2 else "periodic"
                    row, _ = run_episode(method, density, resource, seed, mode, ctrl, lut, cfg)
                    rows.append(row)
                    print(f"{method}: density={density} seed={seed} resource={resource}", flush=True)
    name = "quick_episode_metrics.csv" if args.quick else ("calibration_episode_metrics.csv" if args.calibrate else "episode_metrics.csv")
    pd.DataFrame(rows).to_csv(output / name, index=False)
    print(output / name)


if __name__ == "__main__":
    main()
