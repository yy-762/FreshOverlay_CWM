from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_rollout_loss(
    preds,
    targets,
    horizon_decay: float = 0.85,
    lambda_context: float = 1.0,
):
    """Eq.-32 style multi-step loss on normalized targets.

    preds/targets: [B,H,6]. First four targets are local service state; the last
    two are dynamic resource context. All continuous quantities have already been
    standardized, so unit weights are meaningful across outputs.
    """
    H = preds.shape[1]
    total = preds.new_tensor(0.0)
    wsum = 0.0
    per_h = []
    for h in range(H):
        w = horizon_decay ** h
        state = F.mse_loss(preds[:, h, :4], targets[:, h, :4])
        context = F.mse_loss(preds[:, h, 4:], targets[:, h, 4:])
        loss = state + lambda_context * context
        total = total + w * loss
        wsum += w
        per_h.append(loss.detach())
    return total / wsum, torch.stack(per_h)
