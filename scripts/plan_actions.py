from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cwm.checkpoint import load_checkpoint
from cwm.config import resolve_data_paths, resolve_output_dir
from cwm.data import load_stage2_mat
from cwm.phy_lut import Stage1PhyLUT
from cwm.planner import CWMPlanner, PlannerConfig
from cwm.utils import load_yaml, resolve_device


def main():
    ap = argparse.ArgumentParser(description="Run CWM counterfactual planning on Stage-2 states.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=["val", "test", "unseen"], default="test")
    ap.add_argument("--num-states", type=int, default=200)
    ap.add_argument("--lambda-a", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    _, paths = resolve_data_paths(cfg, args.config)
    device = resolve_device(cfg["hardware"]["device"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_output_dir(cfg, args.config) / "checkpoints" / "best.pt"
    ckpt, model, normalizer = load_checkpoint(ckpt_path, device)
    action_library = np.asarray(ckpt["action_library"], dtype=np.float32)

    planner_keys = {k: cfg["planner"][k] for k in ["horizon","beam_size","beta","lambda_a","R0_mbps","A0_ms","source_count"]}
    pcfg = PlannerConfig(**planner_keys)
    if args.lambda_a is not None: pcfg.lambda_a = args.lambda_a
    if args.horizon is not None: pcfg.horizon = args.horizon

    lut = None
    lut_path = Path(cfg["planner"].get("stage1_lut_csv", ""))
    if lut_path and not lut_path.is_absolute():
        lut_path = Path(args.config).resolve().parent.parent / lut_path
    if str(lut_path) and lut_path.exists():
        lut = Stage1PhyLUT(lut_path, eps_host=cfg["planner"]["eps_host"], eps_passive=cfg["planner"]["eps_passive"])
        action_library = lut.action_library()
        print(f"Feasibility mask: {lut_path}; candidate actions from LUT: {len(action_library)}")
    else:
        print("WARNING: Stage-1 LUT not found; planner will evaluate all actions without feasibility masking.")

    planner = CWMPlanner(model, normalizer, action_library, device, phy_lut=lut, cfg=pcfg)
    arr = load_stage2_mat(paths[args.split], with_aux=False)
    n = min(args.num_states, len(arr.X))
    idx = np.linspace(0, len(arr.X)-1, n, dtype=int)
    rows = []
    for count, i in enumerate(idx, 1):
        action, score = planner.select_action(arr.X[i])
        rows.append({
            "row_index": int(i), "sinr_dB": float(arr.X[i,0]), "aoi_ms": float(arr.X[i,3]),
            "cbr": float(arr.X[i,7]), "load": float(arr.X[i,8]),
            "delta_dB": float(action[0]), "rb_kbps": float(action[1]), "mcs": int(round(action[2])),
            "predicted_plan_score": float(score), "lambda_a": pcfg.lambda_a, "horizon": pcfg.horizon,
        })
        if count % 20 == 0:
            print(f"planned {count}/{n}")
    out = resolve_output_dir(cfg, args.config) / "planning"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}_planned_actions_H{pcfg.horizon}_lambda{pcfg.lambda_a}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
