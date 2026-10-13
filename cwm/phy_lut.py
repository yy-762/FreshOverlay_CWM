from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class LUTQueryResult:
    host_per: np.ndarray
    passive_per: np.ndarray
    feasible: np.ndarray


class Stage1PhyLUT:
    """Discrete Stage-1 LUT used only for the feasibility mask in predictive control."""

    def __init__(self, csv_path, eps_host=0.1, eps_passive=0.1):
        self.path = Path(csv_path)
        df = pd.read_csv(self.path)
        aliases = {
            "SINR_dB": ["SINR_dB", "sinr_dB"],
            "MCS": ["MCS", "mcs"],
            "delta_dB": ["delta_dB", "Delta_dB"],
            "Rb_kbps": ["Rb_kbps", "rb_kbps"],
            "host": ["wifiGlaze_PER", "host_PER", "hostPER"],
            "passive": ["glaze_PER", "passive_PER", "passivePER"],
        }
        def pick(name):
            for c in aliases[name]:
                if c in df.columns:
                    return c
            raise KeyError(f"Cannot find LUT column for {name}; columns={list(df.columns)}")
        self.c_sinr = pick("SINR_dB")
        self.c_mcs = pick("MCS")
        self.c_delta = pick("delta_dB")
        self.c_rb = pick("Rb_kbps")
        self.c_host = pick("host")
        self.c_passive = pick("passive")
        self.eps_host = eps_host
        self.eps_passive = eps_passive
        self.df = df.copy()
        self.sinr_grid = np.sort(df[self.c_sinr].unique().astype(float))
        self._map = {}
        for _, r in df.iterrows():
            key = (float(r[self.c_sinr]), float(r[self.c_delta]), float(r[self.c_rb]), int(r[self.c_mcs]))
            if "embeddingFeasible" in df.columns:
                ev = r["embeddingFeasible"]
                if isinstance(ev, str):
                    emb = ev.strip().lower() in {"1", "true", "yes", "y"}
                else:
                    emb = bool(ev) if not pd.isna(ev) else False
            else:
                emb = True
            self._map[key] = (float(r[self.c_host]), float(r[self.c_passive]), emb)

    def nearest_sinr(self, x):
        x = np.asarray(x, dtype=float)
        idx = np.abs(self.sinr_grid[:, None] - x.reshape(1, -1)).argmin(axis=0)
        return self.sinr_grid[idx]

    def query(self, sinr_dB, actions):
        actions = np.asarray(actions, dtype=float)
        sinr = np.broadcast_to(np.asarray(sinr_dB, dtype=float).reshape(-1), (len(actions),)) if np.asarray(sinr_dB).size == 1 else np.asarray(sinr_dB, dtype=float).reshape(-1)
        if len(sinr) != len(actions):
            raise ValueError("sinr_dB must be scalar or same length as actions")
        qsinr = self.nearest_sinr(sinr)
        h = np.ones(len(actions), dtype=np.float32)
        p = np.ones(len(actions), dtype=np.float32)
        f = np.zeros(len(actions), dtype=bool)
        for i, (s, a) in enumerate(zip(qsinr, actions)):
            d, rb, m = float(a[0]), float(a[1]), int(round(a[2]))
            v = self._map.get((float(s), d, rb, m), None)
            if v is None:
                continue
            hp, pp, emb = v
            h[i] = hp; p[i] = pp
            if d == 0:
                f[i] = hp <= self.eps_host
            else:
                f[i] = emb and hp <= self.eps_host and pp <= self.eps_passive
        return LUTQueryResult(h, p, f)

    def action_library(self):
        cols = self.df[[self.c_delta, self.c_rb, self.c_mcs]].to_numpy(dtype=np.float32)
        return np.unique(np.round(cols, 6), axis=0).astype(np.float32)
