from __future__ import annotations

import torch


# X = [sinr, host_rate, passive_rate, aoi, resource_1,resource_2,resource_3,cbr,load]
# Y = [sinr_next,host_rate_next,passive_rate_next,aoi_next,cbr_next,load_next]


def compose_next_x_normalized(y_norm, x_prev_norm, normalizer, project_physical: bool = True):
    """Convert predicted normalized Y to next normalized X.

    Resource identity is copied from the current X, as it is fixed over the
    rollout. Physical projection is applied in raw units to prevent autoregressive
    numerical drift from creating impossible negative rates/AoI or invalid CBR.
    """
    y_raw = normalizer.denormalize_y_t(y_norm)
    x_prev_raw = normalizer.denormalize_x_t(x_prev_norm)
    if project_physical:
        y_raw = project_y_raw(y_raw)
    x_next_raw = torch.cat([y_raw[..., 0:4], x_prev_raw[..., 4:7], y_raw[..., 4:6]], dim=-1)
    return normalizer.normalize_x_t(x_next_raw)


def project_y_raw(y_raw: torch.Tensor) -> torch.Tensor:
    # Functional (non-inplace) projection keeps autograd valid during rollout training.
    return torch.stack([
        y_raw[..., 0],
        torch.clamp(y_raw[..., 1], min=0.0),
        torch.clamp(y_raw[..., 2], min=0.0),
        torch.clamp(y_raw[..., 3], min=0.0, max=2000.0),
        torch.clamp(y_raw[..., 4], min=0.0, max=1.0),
        torch.clamp(y_raw[..., 5], min=0.0, max=1.0),
    ], dim=-1)
