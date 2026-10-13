from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .dynamics import compose_next_x_normalized, project_y_raw


@dataclass
class PlannerConfig:
    horizon: int = 3
    beam_size: int = 32
    beta: float = 0.95
    lambda_a: float = 0.5
    R0_mbps: float = 100.0
    A0_ms: float = 100.0
    source_count: float = 1.0


class CWMPlanner:
    def __init__(self, model, normalizer, action_library, device, phy_lut=None, cfg: Optional[PlannerConfig] = None):
        self.model = model.eval()
        self.normalizer = normalizer
        self.actions_raw = np.asarray(action_library, dtype=np.float32)
        self.device = torch.device(device)
        self.phy_lut = phy_lut
        self.cfg = cfg or PlannerConfig()

    def _feasible_actions(self, x_raw):
        if self.phy_lut is None:
            return self.actions_raw
        q = self.phy_lut.query(float(x_raw[0]), self.actions_raw)
        acts = self.actions_raw[q.feasible]
        if len(acts) == 0:
            # Robust fallback: choose all delta=0 host-only representations; if none, all actions.
            acts = self.actions_raw[self.actions_raw[:, 0] == 0]
            if len(acts) == 0:
                acts = self.actions_raw
        return acts

    def _utility_from_y_raw(self, y_raw):
        r_loc = y_raw[..., 1] + y_raw[..., 2] / 1000.0
        aoi = y_raw[..., 3]
        return r_loc / self.cfg.R0_mbps - self.cfg.lambda_a * aoi / (self.cfg.source_count * self.cfg.A0_ms)

    @torch.inference_mode()
    def select_action(self, x_raw: np.ndarray):
        x_raw = np.asarray(x_raw, dtype=np.float32).reshape(-1)
        if x_raw.shape[0] != 9:
            raise ValueError("x_raw must have 9 Stage-2 state/context features")

        # Beam items are (score, x_raw, first_action)
        beams = [(0.0, x_raw.copy(), None)]
        for h in range(self.cfg.horizon):
            expansions = []
            for score, state_raw, first_action in beams:
                acts = self._feasible_actions(state_raw)
                x_batch_raw = np.repeat(state_raw[None, :], len(acts), axis=0)
                x = torch.as_tensor(self.normalizer.normalize_x_np(x_batch_raw), device=self.device)
                a = torch.as_tensor(self.normalizer.normalize_a_np(acts), device=self.device)
                y_norm = self.model(x, a)
                y_raw_t = project_y_raw(self.normalizer.denormalize_y_t(y_norm))
                util = self._utility_from_y_raw(y_raw_t).cpu().numpy()
                x_next_norm = compose_next_x_normalized(y_norm, x, self.normalizer, project_physical=True)
                x_next_raw = self.normalizer.denormalize_x_t(x_next_norm).cpu().numpy()
                for j in range(len(acts)):
                    fa = acts[j].copy() if first_action is None else first_action
                    expansions.append((score + (self.cfg.beta ** h) * float(util[j]), x_next_raw[j], fa))
            expansions.sort(key=lambda z: z[0], reverse=True)
            beams = expansions[: self.cfg.beam_size]
        best = beams[0]
        return best[2], best[0]
