from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from cwm.config import resolve_data_paths
from cwm.data import load_stage2_mat
from cwm.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Python:", sys.version.replace("\n", " "))
    print("OS:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA build:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
        print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    _, paths = resolve_data_paths(cfg, args.config)
    for k, p in paths.items():
        print(f"{k:>7} dataset: {p} ->", "OK" if p.exists() else "MISSING")
    if paths["train"].exists():
        arr = load_stage2_mat(paths["train"], with_aux=False)
        print("Train shapes:", arr.X.shape, arr.A.shape, arr.Y.shape, arr.Meta.shape)
        print("Finite X/A/Y:", np.isfinite(arr.X).all(), np.isfinite(arr.A).all(), np.isfinite(arr.Y).all())


if __name__ == "__main__":
    main()
