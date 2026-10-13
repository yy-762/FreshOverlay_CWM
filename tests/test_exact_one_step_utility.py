from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_action_fidelity import exact_one_step_utility  # noqa: E402
from run_closed_loop_evaluation import (  # noqa: E402
    ContinuousPhyLUT, IDLE_ACTION, SimConfig, init_env, step,
)


@pytest.mark.parametrize("traffic_mode", ["periodic", "bursty"])
def test_exact_one_step_expected_utility_matches_monte_carlo(traffic_mode):
    lut_path = ROOT / "data/stage1_phy_lut/phy_lut_4d.csv"
    if not lut_path.is_file():
        pytest.skip("requires the external Stage 1 PHY LUT")
    lut = ContinuousPhyLUT(lut_path)
    env = init_env(10, 3, 201, traffic_mode)
    cfg = SimConfig()
    background = np.asarray([
        acts[0] if len(acts := lut.feasible(s)) else IDLE_ACTION
        for s in env["sinr"]
    ], dtype=np.float32)
    for _ in range(5):
        step(env, background, lut, cfg)
    candidates = [(i, lut.feasible(s)) for i, s in enumerate(env["sinr"])]
    robot, actions = next((i, acts) for i, acts in candidates if len(acts))
    candidate = actions[0]
    joint = background.copy()
    joint[robot] = candidate
    expected = exact_one_step_utility(env, robot, actions[:1], background, lut, cfg, 0.5)[0]
    _, pp, _, _, _ = lut.query([env["sinr"][robot]], [candidate])
    quality = float(candidate[0] > 0) * (1 - pp[0])
    realized = []
    for seed in range(400):
        trial = copy.deepcopy(env)
        trial["rng"] = np.random.default_rng(10_000 + seed)
        out = step(trial, joint, lut, cfg)
        i = robot
        realized.append(
            np.log1p(out["host_rate"][i] / 1.0)
            + 0.25 * np.log1p(out["passive_rate"][i] / 0.25)
            - 0.5 * (out["aoi"][i] * 1000 / 500.0 - 0.75 * quality)
        )
    assert abs(float(np.mean(realized)) - expected) < 0.02
