from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cwm.checkpoint import save_checkpoint
from cwm.config import resolve_data_paths, resolve_output_dir
from cwm.data import RolloutWindowDataset, OneStepDataset, load_stage2_mat, unique_action_library
from cwm.dynamics import compose_next_x_normalized
from cwm.losses import weighted_rollout_loss
from cwm.model import ConditionalWorldModel
from cwm.normalization import Normalizer
from cwm.utils import ensure_dir, load_yaml, resolve_device, save_json, seed_everything


def make_loader(ds, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def autocast_context(device, enabled, dtype_name):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if dtype_name.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def rollout_forward(model, x0, aseq, normalizer, project_physical=True):
    preds = []
    x = x0
    H = aseq.shape[1]
    for h in range(H):
        y = model(x, aseq[:, h])
        preds.append(y)
        x = compose_next_x_normalized(y, x, normalizer, project_physical=project_physical)
    return torch.stack(preds, dim=1)


@torch.inference_mode()
def validate(model, loader, normalizer, device, cfg):
    model.eval()
    total = 0.0
    n = 0
    per_h_acc = None
    for x0, aseq, yseq in loader:
        x0 = x0.to(device, non_blocking=True)
        aseq = aseq.to(device, non_blocking=True)
        yseq = yseq.to(device, non_blocking=True)
        preds = rollout_forward(model, x0, aseq, normalizer, project_physical=cfg["training"]["physical_projection"])
        loss, per_h = weighted_rollout_loss(
            preds, yseq,
            horizon_decay=cfg["training"]["horizon_decay"],
            lambda_context=cfg["training"]["lambda_context"],
        )
        bs = x0.shape[0]
        total += float(loss) * bs
        n += bs
        ph = per_h.cpu().numpy() * bs
        per_h_acc = ph if per_h_acc is None else per_h_acc + ph
    return total / max(n, 1), per_h_acc / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ablation", choices=["no_action", "no_context", "no_aoi"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.ablation:
        cfg["model"][f"ablate_{args.ablation.removeprefix('no_')}"] = True
        cfg["output_dir"] = f"outputs/ablations/{args.ablation}"
    seed_everything(int(cfg["seed"]))

    root, paths = resolve_data_paths(cfg, args.config)
    out_dir = ensure_dir(resolve_output_dir(cfg, args.config))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    print("Loading Stage-2 datasets...")
    train_arr = load_stage2_mat(paths["train"], with_aux=False)
    val_arr = load_stage2_mat(paths["val"], with_aux=False)
    normalizer = Normalizer.fit(train_arr.X, train_arr.A, train_arr.Y)
    action_library = unique_action_library(train_arr.A)
    save_json(normalizer.state_dict(), out_dir / "normalizer.json")
    np.save(out_dir / "action_library.npy", action_library)

    H = int(cfg["training"]["rollout_horizon"])
    train_roll = RolloutWindowDataset(
        train_arr, H, normalizer,
        max_windows=cfg["data"].get("max_train_windows", None),
        seed=int(cfg["seed"]),
    )
    val_roll = RolloutWindowDataset(
        val_arr, H, normalizer,
        max_windows=cfg["data"].get("max_val_windows", None),
        seed=int(cfg["seed"]) + 1,
    )
    train_one = OneStepDataset(train_arr, normalizer)

    device = resolve_device(cfg["hardware"]["device"])
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")
        if cfg["hardware"].get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    model = ConditionalWorldModel(**cfg["model"]).to(device)
    if cfg["hardware"].get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = CosineAnnealingLR(opt, T_max=int(cfg["training"]["epochs"]), eta_min=float(cfg["training"]["min_lr"]))
    use_amp = bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"
    amp_dtype = cfg["hardware"].get("amp_dtype", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype.lower() == "fp16"))

    num_workers = int(cfg["data"].get("num_workers", 0))
    pin = device.type == "cuda"
    roll_loader = make_loader(train_roll, int(cfg["training"]["batch_size"]), True, num_workers, pin)
    val_loader = make_loader(val_roll, int(cfg["training"]["val_batch_size"]), False, num_workers, pin)
    one_loader = make_loader(train_one, int(cfg["training"]["batch_size"]), True, num_workers, pin)

    print(f"Train transitions: {len(train_arr.X):,}")
    print(f"Train H={H} windows: {len(train_roll):,}")
    print(f"Val H={H} windows: {len(val_roll):,}")
    print(f"Action library: {len(action_library)} unique actions")

    best_val = math.inf
    patience = 0
    history_path = out_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "phase", "train_loss", "val_loss", "lr", "seconds"] + [f"val_h{h+1}" for h in range(H)])

    epochs = int(cfg["training"]["epochs"])
    warmup = int(cfg["training"]["one_step_warmup_epochs"])
    grad_clip = float(cfg["training"]["grad_clip"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_total = 0.0
        seen = 0
        phase = "one_step_warmup" if epoch <= warmup else "autoregressive_rollout"
        loader = one_loader if epoch <= warmup else roll_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} {phase}")
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            if epoch <= warmup:
                x, a, y = [z.to(device, non_blocking=True) for z in batch]
                with autocast_context(device, use_amp, amp_dtype):
                    pred = model(x, a)
                    # Same state/context decomposition as rollout objective.
                    state = torch.mean((pred[:, :4] - y[:, :4]) ** 2)
                    context = torch.mean((pred[:, 4:] - y[:, 4:]) ** 2)
                    loss = state + float(cfg["training"]["lambda_context"]) * context
                bs = x.shape[0]
            else:
                x0, aseq, yseq = [z.to(device, non_blocking=True) for z in batch]
                with autocast_context(device, use_amp, amp_dtype):
                    preds = rollout_forward(
                        model, x0, aseq, normalizer,
                        project_physical=cfg["training"]["physical_projection"],
                    )
                    loss, _ = weighted_rollout_loss(
                        preds, yseq,
                        horizon_decay=float(cfg["training"]["horizon_decay"]),
                        lambda_context=float(cfg["training"]["lambda_context"]),
                    )
                bs = x0.shape[0]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            train_total += float(loss.detach()) * bs
            seen += bs
            pbar.set_postfix(loss=f"{train_total/max(seen,1):.4f}")

        train_loss = train_total / max(seen, 1)
        val_loss, val_per_h = validate(model, val_loader, normalizer, device, cfg)
        scheduler.step()
        elapsed = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(f"epoch={epoch} train={train_loss:.5f} val={val_loss:.5f} lr={lr:.3e} time={elapsed:.1f}s")

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, phase, train_loss, val_loss, lr, elapsed] + val_per_h.tolist())

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_per_h": val_per_h.tolist()}
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        save_checkpoint(ckpt_dir / "last.pt", raw_model, normalizer, action_library, cfg, epoch, metrics, opt, scheduler)
        if val_loss < best_val - float(cfg["training"]["min_delta"]):
            best_val = val_loss
            patience = 0
            save_checkpoint(ckpt_dir / "best.pt", raw_model, normalizer, action_library, cfg, epoch, metrics)
            print("  -> new best checkpoint")
        else:
            patience += 1
            if patience >= int(cfg["training"]["early_stop_patience"]):
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    print(f"Done. Best validation loss: {best_val:.6f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
