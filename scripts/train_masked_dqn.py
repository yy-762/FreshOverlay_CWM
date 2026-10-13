from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import _bootstrap  # noqa: F401
from cwm.checkpoint import load_checkpoint
from cwm.dqn import MaskedQNetwork, ReplayBuffer, save_dqn
from run_closed_loop_evaluation import (
    ContinuousPhyLUT, IDLE_ACTION, SimConfig, init_env, observe, step,
)


def feasible_mask(states: np.ndarray, lut: ContinuousPhyLUT) -> np.ndarray:
    masks = np.zeros((len(states), len(lut.actions)), bool)
    for i, sinr in enumerate(states[:, 0]):
        *_, masks[i] = lut.query(np.repeat(float(sinr), len(lut.actions)), lut.actions)
    return masks


def normalized(states, mean, std):
    return ((states - mean) / std).astype(np.float32)


def local_reward(out, actions, lut, states, lam=0.5):
    quality = np.zeros(len(actions), np.float32)
    active = actions[:, 0] >= 0
    if active.any():
        _, passive_per, _, _, _ = lut.query(states[active, 0], actions[active])
        quality[active] = (actions[active, 0] > 0) * (1 - passive_per)
    return (np.log1p(out["host_rate"] / 1.0)
            + 0.25 * np.log1p(out["passive_rate"] / 0.25)
            - lam * (out["aoi"] * 1000 / 500.0 - 0.75 * quality)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines/masked_dqn"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut = ContinuousPhyLUT(root / "data/stage1_phy_lut/phy_lut_4d.csv")
    _, _, normalizer = load_checkpoint(root / "outputs/default_run/checkpoints/best.pt", "cpu")
    state_mean = normalizer.x_mean.astype(np.float32)
    state_std = normalizer.x_std.astype(np.float32)
    online = MaskedQNetwork(9, len(lut.actions), 128).to(device)
    target = MaskedQNetwork(9, len(lut.actions), 128).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=3e-4, weight_decay=1e-5)
    replay = ReplayBuffer(200_000, 9, len(lut.actions), args.seed)
    cfg = SimConfig(steps=args.steps)
    gamma = 0.97
    batch_size = 512
    warmup = 5_000
    update_count = 0
    history_path = args.output_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["episode", "density", "resource", "seed", "epsilon",
                                    "mean_reward", "active_transitions", "replay_size", "mean_loss"])
    settings = [(d, r, s) for s in range(1, 81) for d in (5, 10, 20, 30) for r in (1, 2, 3)]
    rng.shuffle(settings)
    for episode in range(args.episodes):
        density, resource, seed = settings[episode % len(settings)]
        mode = "bursty" if 1 + (density + seed + resource) % 2 == 2 else "periodic"
        env = init_env(density, resource, seed, mode)
        epsilon = max(0.05, 1.0 - 0.95 * episode / max(args.episodes * 0.75, 1))
        rewards = []
        losses = []
        transitions = 0
        for epoch in range(args.steps):
            state = observe(env)
            mask = feasible_mask(state, lut)
            norm_state = normalized(state, state_mean, state_std)
            with torch.inference_mode():
                q = online(torch.as_tensor(norm_state, device=device)).cpu().numpy()
            action_index = np.full(len(state), -1, np.int64)
            actions = np.repeat(IDLE_ACTION[None, :], len(state), axis=0)
            for i in range(len(state)):
                valid = np.flatnonzero(mask[i])
                if not len(valid):
                    continue
                action_index[i] = int(rng.choice(valid) if rng.random() < epsilon else valid[np.argmax(q[i, valid])])
                actions[i] = lut.actions[action_index[i]]
            out = step(env, actions, lut, cfg)
            next_state = observe(env)
            next_mask = feasible_mask(next_state, lut)
            reward = local_reward(out, actions, lut, state)
            active = action_index >= 0
            if active.any():
                replay.add_batch(norm_state[active], action_index[active], reward[active],
                                 normalized(next_state[active], state_mean, state_std),
                                 np.full(active.sum(), epoch == args.steps - 1, np.float32),
                                 next_mask[active])
                rewards.extend(reward[active].tolist())
                transitions += int(active.sum())
            if replay.size >= warmup:
                batch = replay.sample(batch_size, device)
                chosen = online(batch.state).gather(1, batch.action[:, None]).squeeze(1)
                with torch.no_grad():
                    next_online = online(batch.next_state).masked_fill(~batch.next_mask, -torch.inf)
                    has_next = batch.next_mask.any(dim=1)
                    next_action = next_online.argmax(dim=1)
                    next_value = target(batch.next_state).gather(1, next_action[:, None]).squeeze(1)
                    next_value = torch.where(has_next, next_value, torch.zeros_like(next_value))
                    td_target = batch.reward + gamma * (1 - batch.done) * next_value
                loss = F.smooth_l1_loss(chosen, td_target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                update_count += 1
                if update_count % 500 == 0:
                    target.load_state_dict(online.state_dict())
        row = [episode + 1, density, resource, seed, epsilon,
               float(np.mean(rewards)) if rewards else np.nan, transitions, replay.size,
               float(np.mean(losses)) if losses else np.nan]
        with open(history_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        if (episode + 1) % 10 == 0:
            print(f"episode={episode+1}/{args.episodes} reward={row[5]:.4f} "
                  f"epsilon={epsilon:.3f} replay={replay.size} loss={row[-1]:.5f}", flush=True)
    target.load_state_dict(online.state_dict())
    save_dqn(args.output_dir / "masked_dqn.pt", online, state_mean, state_std, lut.actions, {
        "episodes": args.episodes, "steps": args.steps, "seed": args.seed,
        "gamma": gamma, "reward_lambda": 0.5, "training_densities": [5, 10, 20, 30],
        "training_seeds": [1, 80], "feasibility_mask": True,
    })
    print(args.output_dir / "masked_dqn.pt")


if __name__ == "__main__":
    main()
