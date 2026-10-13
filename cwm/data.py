from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TransitionArrays:
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    Meta: np.ndarray
    Aux: Optional[np.ndarray] = None


def _read_numeric_v73(h5: h5py.File, key: str) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Missing dataset '{key}' in MATLAB v7.3 file")
    arr = np.asarray(h5[key])
    # MATLAB writes matrices in column-major layout; h5py exposes dimensions reversed.
    if arr.ndim == 2:
        arr = arr.T
    elif arr.ndim > 2:
        arr = np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))
    return np.asarray(arr)


def load_stage2_mat(path: str | Path, with_aux: bool = False) -> TransitionArrays:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        with h5py.File(path, "r") as h5:
            X = _read_numeric_v73(h5, "X").astype(np.float32, copy=False)
            A = _read_numeric_v73(h5, "A").astype(np.float32, copy=False)
            Y = _read_numeric_v73(h5, "Y").astype(np.float32, copy=False)
            Meta = _read_numeric_v73(h5, "Meta").astype(np.float64, copy=False)
            Aux = None
            if with_aux and "Aux" in h5:
                Aux = _read_numeric_v73(h5, "Aux").astype(np.float32, copy=False)
    except OSError:
        # Backward-compatible fallback for MATLAB v5/v7 files (the formal Stage-2
        # generator uses -v7.3, but older bootstrap datasets may use the classic MAT format).
        from scipy.io import loadmat
        m = loadmat(path, variable_names=["X", "A", "Y", "Meta", "Aux"] if with_aux else ["X", "A", "Y", "Meta"])
        X = np.asarray(m["X"], dtype=np.float32)
        A = np.asarray(m["A"], dtype=np.float32)
        Y = np.asarray(m["Y"], dtype=np.float32)
        Meta = np.asarray(m["Meta"], dtype=np.float64)
        Aux = np.asarray(m["Aux"], dtype=np.float32) if with_aux and "Aux" in m else None
    if X.shape[1] != 9 or A.shape[1] != 3 or Y.shape[1] != 6:
        raise ValueError(
            f"Unexpected Stage-2 shapes: X={X.shape}, A={A.shape}, Y={Y.shape}. "
            "Expected X[:,9], A[:,3], Y[:,6]."
        )
    if not (len(X) == len(A) == len(Y) == len(Meta)):
        raise ValueError("Stage-2 arrays have inconsistent row counts")
    return TransitionArrays(X=X, A=A, Y=Y, Meta=Meta, Aux=Aux)


def _is_sorted_by_trajectory(meta: np.ndarray) -> bool:
    if len(meta) < 2:
        return True
    key = meta[:, [0, 3, 4, 5]]  # episode, resource, robot, time
    prev = key[:-1]
    nxt = key[1:]
    # Lexicographic non-decreasing check.
    for c in range(key.shape[1]):
        lt = nxt[:, c] < prev[:, c]
        eq_prefix = np.ones(len(prev), dtype=bool)
        if c > 0:
            eq_prefix = np.all(nxt[:, :c] == prev[:, :c], axis=1)
        if np.any(lt & eq_prefix):
            return False
    return True


def sort_by_trajectory(arrs: TransitionArrays) -> TransitionArrays:
    meta = arrs.Meta
    order = np.lexsort((meta[:, 5], meta[:, 4], meta[:, 3], meta[:, 0]))
    aux = arrs.Aux[order] if arrs.Aux is not None else None
    return TransitionArrays(
        X=arrs.X[order], A=arrs.A[order], Y=arrs.Y[order], Meta=meta[order], Aux=aux
    )


def build_window_starts(meta: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = len(meta)
    if n < horizon:
        return np.empty(0, dtype=np.int64)
    # Dataset is expected to be sorted by trajectory/time. A valid start has the
    # same (episode,resource,robot) for H rows and consecutive time indices.
    valid = np.ones(n - horizon + 1, dtype=bool)
    base = meta[: n - horizon + 1]
    key_cols = [0, 3, 4]
    t_col = 5
    for h in range(1, horizon):
        cur = meta[h : h + n - horizon + 1]
        valid &= np.all(cur[:, key_cols] == base[:, key_cols], axis=1)
        valid &= cur[:, t_col] == base[:, t_col] + h
    return np.flatnonzero(valid).astype(np.int64)


class RolloutWindowDataset(Dataset):
    def __init__(
        self,
        arrays: TransitionArrays,
        horizon: int,
        normalizer=None,
        max_windows: Optional[int] = None,
        seed: int = 2026,
    ):
        if not _is_sorted_by_trajectory(arrays.Meta):
            arrays = sort_by_trajectory(arrays)
        self.arrays = arrays
        self.horizon = int(horizon)
        starts = build_window_starts(arrays.Meta, self.horizon)
        if max_windows is not None and len(starts) > max_windows:
            rng = np.random.default_rng(seed)
            starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
        self.starts = starts
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        H = self.horizon
        x0 = self.arrays.X[s]
        aseq = self.arrays.A[s : s + H]
        yseq = self.arrays.Y[s : s + H]
        if self.normalizer is not None:
            x0 = self.normalizer.normalize_x_np(x0)
            aseq = self.normalizer.normalize_a_np(aseq)
            yseq = self.normalizer.normalize_y_np(yseq)
        return (
            torch.from_numpy(np.asarray(x0, dtype=np.float32)),
            torch.from_numpy(np.asarray(aseq, dtype=np.float32)),
            torch.from_numpy(np.asarray(yseq, dtype=np.float32)),
        )


class OneStepDataset(Dataset):
    def __init__(self, arrays: TransitionArrays, normalizer=None):
        self.arrays = arrays
        self.normalizer = normalizer

    def __len__(self):
        return len(self.arrays.X)

    def __getitem__(self, idx):
        x = self.arrays.X[idx]
        a = self.arrays.A[idx]
        y = self.arrays.Y[idx]
        if self.normalizer is not None:
            x = self.normalizer.normalize_x_np(x)
            a = self.normalizer.normalize_a_np(a)
            y = self.normalizer.normalize_y_np(y)
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(a, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )


def unique_action_library(A: np.ndarray) -> np.ndarray:
    # Actions are discrete but stored as float; round to avoid float noise.
    key = np.round(A.astype(np.float64), decimals=6)
    return np.unique(key, axis=0).astype(np.float32)
