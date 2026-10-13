from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class MaskedQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int = 128):
        super().__init__()
        self.config = {"state_dim": state_dim, "action_count": action_count, "hidden_dim": hidden_dim}
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class ReplayBatch:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    next_mask: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_count: int, seed: int):
        self.capacity = capacity
        self.state = np.empty((capacity, state_dim), np.float32)
        self.action = np.empty(capacity, np.int64)
        self.reward = np.empty(capacity, np.float32)
        self.next_state = np.empty((capacity, state_dim), np.float32)
        self.done = np.empty(capacity, np.float32)
        self.next_mask = np.empty((capacity, action_count), bool)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(self, state, action, reward, next_state, done, next_mask) -> None:
        count = len(state)
        indices = (np.arange(count) + self.position) % self.capacity
        self.state[indices] = state
        self.action[indices] = action
        self.reward[indices] = reward
        self.next_state[indices] = next_state
        self.done[indices] = done
        self.next_mask[indices] = next_mask
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        index = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            state=torch.as_tensor(self.state[index], device=device),
            action=torch.as_tensor(self.action[index], device=device),
            reward=torch.as_tensor(self.reward[index], device=device),
            next_state=torch.as_tensor(self.next_state[index], device=device),
            done=torch.as_tensor(self.done[index], device=device),
            next_mask=torch.as_tensor(self.next_mask[index], device=device),
        )


def save_dqn(path, model, state_mean, state_std, actions, metadata) -> None:
    torch.save({
        "model_state": model.state_dict(), "model_config": model.config,
        "state_mean": np.asarray(state_mean, np.float32),
        "state_std": np.asarray(state_std, np.float32),
        "actions": np.asarray(actions, np.float32), "metadata": metadata,
    }, path)


def load_dqn(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MaskedQNetwork(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return checkpoint, model
