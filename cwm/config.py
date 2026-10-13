from __future__ import annotations

from pathlib import Path


def resolve_data_paths(cfg, config_path):
    base = Path(config_path).resolve().parent.parent
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = (base / root).resolve()
    out = {}
    for k in ["train", "val", "test", "unseen"]:
        p = Path(cfg["data"][k])
        out[k] = p if p.is_absolute() else root / p
    return root, out


def resolve_output_dir(cfg, config_path):
    base = Path(config_path).resolve().parent.parent
    p = Path(cfg["output_dir"])
    return p if p.is_absolute() else (base / p).resolve()
