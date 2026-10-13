from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from cwm.checkpoint import load_checkpoint
from cwm.dynamics import project_y_raw
from run_closed_loop_evaluation import (
    ContinuousPhyLUT, Controllers, SimConfig, dcf, init_env, observe, step,
)


def exact_one_step_utility(env, robot, actions, background_actions, lut, cfg, lam):
    """Expected local utility using the explicit PHY/MAC simulator.

    Other robots keep a fixed MCS-only action at this decision epoch. Periodic
    status generation is deterministic; packet delivery is integrated out.
    """
    n = env["n"]
    next_epoch = env["t"] + 1
    now = env["t"] * cfg.dt
    if env["mode"] == "periodic":
        generated = ((next_epoch - env["phase"]) % env["period"]) == 0
        p_generated = float(generated[robot])
    else:
        p_burst_next = 0.80 if env["burst"][robot] else 0.08
        p_generated = 0.75 * p_burst_next
    stale_aoi = min(float(env["aoi"][robot]) + cfg.dt, cfg.max_aoi)
    pending_without_generation = bool(env["pending"][robot])
    old_latest = float(env["latest"][robot])
    no_generation_fresh_aoi = (
        min(max(now + cfg.dt - old_latest, cfg.dt), cfg.max_aoi)
        if pending_without_generation else stale_aoi
    )
    generated_fresh_aoi = cfg.dt
    scores = np.empty(len(actions), dtype=float)
    for j, candidate in enumerate(actions):
        joint = background_actions.copy()
        joint[robot] = candidate
        active = joint[:, 0] >= 0
        hp, pp, duration, _, _ = lut.query(env["sinr"][active], joint[active])
        mac = dcf(hp, duration, env["activity"])
        active_index = np.flatnonzero(active)
        local_index = int(np.flatnonzero(active_index == robot)[0])
        p_host = float(mac["p_host"][local_index])
        p_passive = 1 - (1 - p_host * (1 - pp[local_index])) ** max(cfg.dt / mac["t_slot"], 1)
        host_rate = p_host * (8 * lut.host_payload_bytes) / mac["t_slot"] / 1e6
        pending_probability = p_generated + (1 - p_generated) * pending_without_generation
        passive_rate = p_passive * pending_probability * lut.passive_payload_bits / cfg.dt / 1e3
        expected_aoi_no_generation = (
            (1 - p_passive) * stale_aoi + p_passive * no_generation_fresh_aoi
            if pending_without_generation else stale_aoi
        )
        expected_aoi_generated = (1 - p_passive) * stale_aoi + p_passive * generated_fresh_aoi
        expected_aoi = ((1 - p_generated) * expected_aoi_no_generation
                        + p_generated * expected_aoi_generated)
        quality = float(candidate[0] > 0) * (1 - pp[local_index])
        scores[j] = (np.log1p(host_rate / 1.0)
                     + 0.25 * np.log1p(passive_rate / 0.25)
                     - lam * (expected_aoi * 1000 / 500.0 - 0.75 * quality))
    return scores


@torch.inference_mode()
def predicted_one_step_utility(model, normalizer, device, state, actions, lut, lam):
    x = torch.as_tensor(normalizer.normalize_x_np(np.repeat(state[None, :], len(actions), axis=0)), device=device)
    a = torch.as_tensor(normalizer.normalize_a_np(actions), device=device)
    y = project_y_raw(normalizer.denormalize_y_t(model(x, a))).cpu().numpy()
    _, pp, _, _, _ = lut.query(np.repeat(float(state[0]), len(actions)), actions)
    quality = (actions[:, 0] > 0) * (1 - pp)
    return (np.log1p(y[:, 1] / 1.0)
            + 0.25 * np.log1p(y[:, 2] / 0.25)
            - lam * (y[:, 3] / 500.0 - 0.75 * quality))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/action_fidelity"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/default_run/checkpoints/best.pt"))
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--sample-interval", type=int, default=30)
    parser.add_argument("--samples-per-setting", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lut = ContinuousPhyLUT(root / "data/stage1_phy_lut/phy_lut_4d.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint, model, normalizer = load_checkpoint(checkpoint_path, device)
    trained_actions = np.asarray(checkpoint["action_library"])
    controller = Controllers(model, normalizer, lut, device, beam=4)
    cfg = SimConfig()
    rows = []
    for density in (5, 10, 20, 30, 40):
        for resource in (1, 2, 3):
            for seed in (201, 202, 203, 204, 205):
                env = init_env(density, resource, seed, "periodic")
                sampled = 0
                horizon = args.warmup_steps + args.sample_interval * args.samples_per_setting
                for epoch in range(horizon):
                    x = observe(env)
                    background = controller.mcs_only(x)
                    if epoch >= args.warmup_steps and (epoch - args.warmup_steps) % args.sample_interval == 0:
                        eligible = []
                        for i in range(len(x)):
                            _, _, _, _, valid = lut.query(np.repeat(float(x[i, 0]), len(lut.actions)), lut.actions)
                            if valid.any():
                                eligible.append((i, lut.actions[valid]))
                        if eligible:
                            i, candidates = eligible[seed % len(eligible)]
                            exact = exact_one_step_utility(env, i, candidates, background, lut, cfg, 0.5)
                            predicted = predicted_one_step_utility(model, normalizer, device, x[i], candidates, lut, 0.5)
                            best_exact = int(np.argmax(exact))
                            best_predicted = int(np.argmax(predicted))
                            trained = np.any(np.all(candidates[:, None, :] == trained_actions[None, :, :], axis=2), axis=1)
                            mcs_action = background[i]
                            fixed_action = controller.fixed(x[i:i + 1])[0]
                            mcs_index = np.flatnonzero(np.all(candidates == mcs_action, axis=1))
                            fixed_index = np.flatnonzero(np.all(candidates == fixed_action, axis=1))
                            if len(mcs_index) != 1 or len(fixed_index) != 1:
                                raise AssertionError("Reference controller selected an action outside the feasible set")
                            regret = max(0.0, float(exact[best_exact] - exact[best_predicted]))
                            utility_range = float(exact.max() - exact.min())
                            rows.append({
                                "density": density, "resource": resource, "seed": seed,
                                "epoch": epoch, "robot": i, "candidates": len(candidates),
                                "sinr_db": float(x[i, 0]),
                                "embedding_candidates": int(np.sum(candidates[:, 0] > 0)),
                                "trained_candidates": int(np.sum(trained)),
                                "cwm_action_in_training": int(trained[best_predicted]),
                                "exact_action_in_training": int(trained[best_exact]),
                                "exact_delta_db": float(candidates[best_exact, 0]),
                                "cwm_delta_db": float(candidates[best_predicted, 0]),
                                "mcs_delta_db": float(mcs_action[0]),
                                "action_agreement": int(best_predicted == best_exact),
                                "mcs_exact_action_agreement": int(mcs_index[0] == best_exact),
                                "cwm_mcs_action_agreement": int(best_predicted == mcs_index[0]),
                                "exact_utility_regret": regret,
                                "mcs_only_regret": max(0.0, float(exact[best_exact] - exact[mcs_index[0]])),
                                "fixed_regret": max(0.0, float(exact[best_exact] - exact[fixed_index[0]])),
                                "random_action_expected_regret": float(exact.max() - exact.mean()),
                                "candidate_utility_range": utility_range,
                                "within_1pct_of_range": int(regret <= 0.01 * utility_range + 1e-9),
                                "exact_best_utility": float(exact[best_exact]),
                                "selected_exact_utility": float(exact[best_predicted]),
                            })
                            sampled += 1
                    step(env, background, lut, cfg)
                print(f"density={density} resource={resource} seed={seed} samples={sampled}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "action_fidelity_episodes.csv", index=False)
    summary = df.groupby("density").agg(
        states=("action_agreement", "size"),
        action_agreement=("action_agreement", "mean"),
        mcs_exact_action_agreement=("mcs_exact_action_agreement", "mean"),
        cwm_mcs_action_agreement=("cwm_mcs_action_agreement", "mean"),
        mean_regret=("exact_utility_regret", "mean"),
        p95_regret=("exact_utility_regret", lambda x: np.percentile(x, 95)),
        mcs_only_mean_regret=("mcs_only_regret", "mean"),
        fixed_mean_regret=("fixed_regret", "mean"),
        random_action_expected_regret=("random_action_expected_regret", "mean"),
        within_1pct_of_range=("within_1pct_of_range", "mean"),
        mean_candidates=("candidates", "mean"),
        embedding_available=("embedding_candidates", lambda x: np.mean(x > 0)),
        cwm_embeds=("cwm_delta_db", lambda x: np.mean(x > 0)),
        mcs_embeds=("mcs_delta_db", lambda x: np.mean(x > 0)),
        mean_trained_candidates=("trained_candidates", "mean"),
        cwm_action_in_training=("cwm_action_in_training", "mean"),
        exact_action_in_training=("exact_action_in_training", "mean"),
    )
    summary.loc["all"] = [len(df), df.action_agreement.mean(),
                          df.mcs_exact_action_agreement.mean(), df.cwm_mcs_action_agreement.mean(),
                          df.exact_utility_regret.mean(),
                          np.percentile(df.exact_utility_regret, 95),
                          df.mcs_only_regret.mean(), df.fixed_regret.mean(),
                          df.random_action_expected_regret.mean(), df.within_1pct_of_range.mean(),
                          df.candidates.mean(), np.mean(df.embedding_candidates > 0),
                          np.mean(df.cwm_delta_db > 0), np.mean(df.mcs_delta_db > 0),
                          df.trained_candidates.mean(), df.cwm_action_in_training.mean(),
                          df.exact_action_in_training.mean()]
    summary.to_csv(args.output_dir / "action_fidelity_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
