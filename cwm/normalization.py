from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class NormalizerState:
    x_mean: list
    x_std: list
    a_mean: list
    a_std: list
    y_mean: list
    y_std: list


class Normalizer:
    """Training-only normalizer.

    Continuous X features are standardized; resource one-hot dimensions 4:7
    remain exactly 0/1, matching the paper's statement that continuous features
    are normalized while the resource identity is fixed context.
    """

    X_ONEHOT = (4, 5, 6)

    def __init__(self, x_mean, x_std, a_mean, a_std, y_mean, y_std):
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)
        self.a_mean = np.asarray(a_mean, dtype=np.float32)
        self.a_std = np.asarray(a_std, dtype=np.float32)
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)

    @classmethod
    def fit(cls, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> "Normalizer":
        xm = X.mean(0).astype(np.float32)
        xs = X.std(0).astype(np.float32)
        am = A.mean(0).astype(np.float32)
        ast = A.std(0).astype(np.float32)
        ym = Y.mean(0).astype(np.float32)
        ys = Y.std(0).astype(np.float32)
        xs[xs < 1e-8] = 1.0
        ast[ast < 1e-8] = 1.0
        ys[ys < 1e-8] = 1.0
        # Preserve resource one-hot exactly.
        for j in cls.X_ONEHOT:
            xm[j] = 0.0
            xs[j] = 1.0
        return cls(xm, xs, am, ast, ym, ys)

    def normalize_x_np(self, x):
        return (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_std

    def normalize_a_np(self, a):
        return (np.asarray(a, dtype=np.float32) - self.a_mean) / self.a_std

    def normalize_y_np(self, y):
        return (np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std

    def denormalize_x_np(self, x):
        return np.asarray(x, dtype=np.float32) * self.x_std + self.x_mean

    def denormalize_a_np(self, a):
        return np.asarray(a, dtype=np.float32) * self.a_std + self.a_mean

    def denormalize_y_np(self, y):
        return np.asarray(y, dtype=np.float32) * self.y_std + self.y_mean

    def _tensor_stats(self, kind: str, ref: torch.Tensor):
        mean = torch.as_tensor(getattr(self, f"{kind}_mean"), device=ref.device, dtype=ref.dtype)
        std = torch.as_tensor(getattr(self, f"{kind}_std"), device=ref.device, dtype=ref.dtype)
        return mean, std

    def normalize_x_t(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._tensor_stats("x", x)
        return (x - m) / s

    def normalize_a_t(self, a: torch.Tensor) -> torch.Tensor:
        m, s = self._tensor_stats("a", a)
        return (a - m) / s

    def normalize_y_t(self, y: torch.Tensor) -> torch.Tensor:
        m, s = self._tensor_stats("y", y)
        return (y - m) / s

    def denormalize_x_t(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._tensor_stats("x", x)
        return x * s + m

    def denormalize_a_t(self, a: torch.Tensor) -> torch.Tensor:
        m, s = self._tensor_stats("a", a)
        return a * s + m

    def denormalize_y_t(self, y: torch.Tensor) -> torch.Tensor:
        m, s = self._tensor_stats("y", y)
        return y * s + m

    def state_dict(self) -> Dict:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "a_mean": self.a_mean.tolist(),
            "a_std": self.a_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, d: Dict) -> "Normalizer":
        return cls(d["x_mean"], d["x_std"], d["a_mean"], d["a_std"], d["y_mean"], d["y_std"])
