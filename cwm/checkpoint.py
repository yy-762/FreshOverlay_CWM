from __future__ import annotations

from pathlib import Path

import torch

from .model import ConditionalWorldModel
from .normalization import Normalizer


def save_checkpoint(path, model, normalizer, action_library, config, epoch, metrics, optimizer=None, scheduler=None):
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "normalizer": normalizer.state_dict(),
        "action_library": action_library,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConditionalWorldModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
    return ckpt, model, normalizer
