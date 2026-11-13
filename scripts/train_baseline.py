#!/usr/bin/env python3
"""GeoResQ Milestone 3 — Train & Evaluate Official-Compatible Baseline UNet.

Features:
- Auto-discovers BRIGHT v0.1 subset in /kaggle/input or /kaggle/working
- Deterministic seeding, FP16 AMP on CUDA (with FP32 loss stability), train-only class weighting
- Periodic & resumable checkpoints (`last_checkpoint.pth` + `best_model.pth`)
- Selects `best_model.pth` strictly by validation score (zero test leakage)
- Evaluates `best_model.pth` once on val & unseen test (`noto-earthquake`) at completion
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_subset_dir(explicit_dir: Path | None) -> Path:
    if explicit_dir is not None and (explicit_dir / "pre-event").exists():
        return explicit_dir

    candidates = [
        Path("/kaggle/working/bright_subset_v0_1"),
        Path("/kaggle/input/georesq-bright-subset/bright_subset_v0_1"),
        Path("/kaggle/input/georesq-bright-subset"),
    ]
    for c in candidates:
        if (c / "pre-event").exists() and (c / "target").exists():
            return c

    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            for mf in root.rglob("manifest_v0_1.csv"):
                if (mf.parent / "pre-event").exists():
                    return mf.parent

    raise FileNotFoundError(
        "Could not find bright_subset_v0_1. Attach your Kaggle subset (+ Add Input) or pass --data-dir."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GeoResQ Baseline UNet.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/training.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/runs/baseline_unet_v0_1"))
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for number of epochs.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config & model instantiation without data.")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from georesq.data.dataset import (
        BrightDamageDataset,
        compute_class_weights_from_manifest,
        verify_zero_event_leakage,
    )
    from georesq.models.unet import BaselineUNet
    from georesq.training.loop import (
        count_model_parameters,
        evaluate_model,
        load_checkpoint,
        load_yaml_config,
        save_checkpoint,
        save_training_curves,
        set_global_seed,
        write_history_csv,
    )
    from georesq.training.losses import BrightComboLoss

    cfg = load_yaml_config(args.config)
    exp_cfg = cfg["experiment"]  # type: ignore[index]
    m_cfg = cfg["model"]  # type: ignore[index]
    o_cfg = cfg["optimization"]  # type: ignore[index]

    seed = int(exp_cfg["seed"])  # type: ignore[index]
    set_global_seed(seed)
    epochs = int(args.epochs) if args.epochs is not None else int(o_cfg["epochs"])  # type: ignore[index]

    model = BaselineUNet(
        in_channels=int(m_cfg["in_channels"]),  # type: ignore[index]
        num_classes=int(m_cfg["num_classes"]),  # type: ignore[index]
        base_channels=int(m_cfg["base_channels"]),  # type: ignore[index]
    )
    param_info = count_model_parameters(model)

    if args.dry_run:
        mf_check = REPO_ROOT / str(exp_cfg["manifest_relpath"])  # type: ignore[index]
        leak_info = verify_zero_event_leakage(mf_check)
        weights = compute_class_weights_from_manifest(
            mf_check, power=float(o_cfg["class_weight_power"])  # type: ignore[index]
        )
        print(
            json.dumps(
                {
                    "dry_run": "PASSED",
                    "architecture": m_cfg["architecture"],  # type: ignore[index]
                    "param_info": param_info,
                    "class_weights": weights,
                    "split_events": leak_info,
                },
                indent=2,
            )
        )
        return 0

    data_dir = resolve_subset_dir(args.data_dir)
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else (
            (data_dir / "manifest_v0_1.csv")
            if (data_dir / "manifest_v0_1.csv").exists()
            else (REPO_ROOT / str(exp_cfg["manifest_relpath"]))  # type: ignore[index]
        )
    )
    verify_zero_event_leakage(manifest_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    use_amp = bool(o_cfg["use_amp"]) and (device.type == "cuda")  # type: ignore[index]

    class_weights = (
        compute_class_weights_from_manifest(
            manifest_path, power=float(o_cfg["class_weight_power"])  # type: ignore[index]
        )
        if o_cfg.get("use_train_class_weights", True)  # type: ignore[attr-defined]
        else None
    )

    train_ds = BrightDamageDataset(
        data_root=data_dir,
        manifest_path=manifest_path,
        split="train",
        crop_size=int(o_cfg["crop_size"]),  # type: ignore[index]
        norm_mode=str(m_cfg["norm_mode"]),  # type: ignore[index]
        stack_sar_3ch=bool(m_cfg["stack_sar_3ch"]),  # type: ignore[index]
        input_modality=str(m_cfg["input_modality"]),  # type: ignore[index]
        seed=seed,
    )
    val_ds = BrightDamageDataset(
        data_root=data_dir,
        manifest_path=manifest_path,
        split="val",
        augment=False,
        norm_mode=str(m_cfg["norm_mode"]),  # type: ignore[index]
        stack_sar_3ch=bool(m_cfg["stack_sar_3ch"]),  # type: ignore[index]
        input_modality=str(m_cfg["input_modality"]),  # type: ignore[index]
        seed=seed,
    )
    test_ds = BrightDamageDataset(
        data_root=data_dir,
        manifest_path=manifest_path,
        split="test",
        augment=False,
        norm_mode=str(m_cfg["norm_mode"]),  # type: ignore[index]
        stack_sar_3ch=bool(m_cfg["stack_sar_3ch"]),  # type: ignore[index]
        input_modality=str(m_cfg["input_modality"]),  # type: ignore[index]
        seed=seed,
    )

    num_workers = int(o_cfg.get("num_workers", 2)) if device.type == "cuda" else 0  # type: ignore[attr-defined]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(o_cfg["train_batch_size"]),  # type: ignore[index]
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(o_cfg["eval_batch_size"]),  # type: ignore[index]
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(o_cfg["eval_batch_size"]),  # type: ignore[index]
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(device)
    criterion = BrightComboLoss(
        class_weights=class_weights,
        lovasz_weight=float(o_cfg["lovasz_weight"]),  # type: ignore[index]
        ignore_index=255,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(o_cfg["learning_rate"]),  # type: ignore[index]
        weight_decay=float(o_cfg["weight_decay"]),  # type: ignore[index]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(o_cfg["min_lr"])  # type: ignore[index]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config_snapshot.yaml")

    start_epoch = 1
    best_score = -1.0
    prev_train_sec = 0.0
    history: list[dict[str, object]] = []
    sel_key = str(o_cfg.get("selection_metric", "mIoU_foreground3"))  # type: ignore[attr-defined]

    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = int(ckpt["epoch"]) + 1  # type: ignore[arg-type]
        best_score = float(ckpt["best_score"])  # type: ignore[arg-type]
        prev_train_sec = float(ckpt.get("meta", {}).get("cumulative_train_sec", 0.0))  # type: ignore[union-attr]
        history = list(ckpt.get("history", []))  # type: ignore[arg-type]
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best {sel_key}={best_score:.4f})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"Starting Baseline UNet Training on {gpu_name} | epochs={epochs} | "
        f"train={len(train_ds)} tiles | val={len(val_ds)} tiles | AMP={use_amp}"
    )
    t_train_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.perf_counter()
        train_ds.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            pre = batch["pre_opt"].to(device, non_blocking=True)
            post = batch["post_sar"].to(device, non_blocking=True)
            targets = batch["clf_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pre, post)
            loss = criterion(logits.float(), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(1, steps)
        val_metrics = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp)
        score = float(val_metrics[sel_key])  # type: ignore[arg-type]
        ep_sec = time.perf_counter() - ep_t0

        row_log: dict[str, object] = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics["loss"],
            "val_mIoU_all4": val_metrics["mIoU_all4"],
            "val_mIoU_foreground3": val_metrics["mIoU_foreground3"],
            "val_macro_f1_foreground3": val_metrics["macro_f1_foreground3"],
            "val_bright_loc_f1": val_metrics["bright_loc_f1"],
            "epoch_seconds": round(ep_sec, 2),
        }
        history.append(row_log)

        cum_sec = round(prev_train_sec + (time.perf_counter() - t_train_start), 2)
        is_best = score > best_score
        if is_best:
            best_score = score
            save_checkpoint(
                out_dir / "best_model.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_score,
                history,
                extra_meta={
                    "selected_by": sel_key,
                    "val_metrics": val_metrics,
                    "cumulative_train_sec": cum_sec,
                },
            )

        if epoch % int(o_cfg.get("save_every_epochs", 5)) == 0 or epoch == epochs:  # type: ignore[attr-defined]
            save_checkpoint(
                out_dir / "last_checkpoint.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_score,
                history,
                extra_meta={"cumulative_train_sec": cum_sec},
            )

        tag = " *BEST*" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_mIoU_all4={100*float(val_metrics['mIoU_all4']):.2f}% | "
            f"val_mIoU_fg3={100*score:.2f}% | {ep_sec:.1f}s{tag}"
        )

    total_train_sec = round(prev_train_sec + (time.perf_counter() - t_train_start), 2)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        if device.type == "cuda"
        else 0.0
    )

    best_ckpt = load_checkpoint(out_dir / "best_model.pth", model, device=device)
    best_epoch = int(best_ckpt["epoch"])  # type: ignore[arg-type]
    final_val = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp)
    final_test = evaluate_model(model, test_loader, criterion, device, use_amp=use_amp)

    write_history_csv(history, out_dir / "baseline_unet_v0_1_history.csv")
    save_training_curves(
        history,
        out_dir / "baseline_unet_v0_1_curves.png",
        title="GeoResQ Stage B — Baseline UNet (v0.1 Subset)",
    )

    summary_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": exp_cfg["name"],  # type: ignore[index]
        "dataset_version": "10.5281/zenodo.20072020",
        "subset_version": exp_cfg["subset_version"],  # type: ignore[index]
        "seed": seed,
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "amp_enabled": use_amp,
            "peak_vram_mb": peak_vram_mb,
            "training_duration_seconds": total_train_sec,
        },
        "model_info": param_info,
        "hyperparameters": o_cfg,
        "train_class_weights": class_weights,
        "documented_deviations": cfg["documented_deviations"],
        "loss_progression": {
            "first_epoch_train_loss": history[0]["train_loss"] if history else None,
            "final_epoch_train_loss": history[-1]["train_loss"] if history else None,
            "loss_decreased": (
                float(history[-1]["train_loss"]) < float(history[0]["train_loss"])  # type: ignore[arg-type]
                if len(history) >= 2
                else None
            ),
        },
        "best_checkpoint": {
            "epoch": best_epoch,
            "selection_metric": sel_key,
            "best_val_score": round(best_score, 6),
        },
        "final_validation_metrics": final_val,
        "unseen_event_test_metrics_noto_earthquake": final_test,
    }

    summary_path = out_dir / "baseline_unet_v0_1_summary.json"
    summary_path.write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")
    print("\n=== FINAL BASELINE SUMMARY ===")
    print(json.dumps(summary_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
