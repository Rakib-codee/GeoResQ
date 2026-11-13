#!/usr/bin/env python3
"""GeoResQ Milestone 4 — Controlled Multimodal Optical-SAR Ablation Runner.

Trains & evaluates all 4 controlled variants under identical seed, splits, loss,
and hyperparameters, then aggregates them alongside BaselineUNet into:
  - ablation_v0_1_summary.json
  - ablation_v0_1_table.csv
  - ablation_v0_1_comparison.png
  - ablation_v0_1_predictions.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DAMAGE_CMAP = np.array(
    [
        [20, 24, 35],    # 0: Background (dark slate)
        [46, 204, 113],  # 1: Intact (emerald green)
        [243, 156, 18],  # 2: Damaged (amber orange)
        [231, 76, 60],   # 3: Destroyed (crimson red)
    ],
    dtype=np.uint8,
)


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
        "Could not find bright_subset_v0_1. Attach your Kaggle subset or pass --data-dir."
    )


def colorize_mask(mask_2d: np.ndarray) -> np.ndarray:
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(4):
        rgb[mask_2d == cls_id] = DAMAGE_CMAP[cls_id]
    rgb[mask_2d == 255] = np.array([140, 140, 140], dtype=np.uint8)
    return rgb


def save_ablation_bar_chart(rows: list[dict[str, object]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_labels = [str(r["short_name"]) for r in rows]
    x = np.arange(len(short_labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    for ax, prefix, panel_title in [
        (axes[0], "val", "Validation Events (Beirut + Libya)"),
        (axes[1], "test", "Unseen Test Event (Noto Earthquake — GSI & Umbra)"),
    ]:
        miou_fg = [100.0 * float(r[f"{prefix}_mIoU_fg3"]) for r in rows]
        f1_fg = [100.0 * float(r[f"{prefix}_macro_f1_fg3"]) for r in rows]
        loc_f1 = [100.0 * float(r[f"{prefix}_loc_f1"]) for r in rows]

        b1 = ax.bar(x - width, miou_fg, width, label="mIoU (3 Damage Classes)%")
        b2 = ax.bar(x, f1_fg, width, label="Macro-F1 (3 Damage Classes)%")
        b3 = ax.bar(x + width, loc_f1, width, label="Building Localization F1%")

        for bars in (b1, b2, b3):
            for bar in bars:
                h_val = bar.get_height()
                ax.annotate(
                    f"{h_val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h_val),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=16, ha="right", fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_title(panel_title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8.5)
        ymax = max(max(miou_fg + f1_fg + loc_f1, default=10.0) * 1.18, 15.0)
        ax.set_ylim(0, min(105.0, ymax))

    fig.suptitle(
        "GeoResQ Milestone 4 — Controlled Optical-SAR Modality & Fusion Ablation (BRIGHT v0.1)",
        fontsize=12.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_predictions(
    data_dir: Path,
    val_ds,
    test_ds,
    variant_checkpoints: dict[str, tuple[dict[str, object], Path]],
    device,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    from georesq.models import build_model
    from georesq.training.loop import load_checkpoint

    models: dict[str, object] = {}
    for v_id, (m_cfg, ckpt_path) in variant_checkpoints.items():
        cfg_no_dl = dict(m_cfg)
        cfg_no_dl["pretrained"] = False
        m = build_model(cfg_no_dl).to(device)
        load_checkpoint(ckpt_path, m, device=device)
        m.eval()
        models[v_id] = m

    def pick_rich_indices(ds, per_event: int = 1, max_total: int = 2) -> list[int]:
        scored = []
        for idx, r in enumerate(ds.rows):
            lbl = json.loads(str(r["label_information"]))
            fg = (
                int(lbl.get("intact", 0))
                + int(lbl.get("damaged", 0))
                + int(lbl.get("destroyed", 0))
            )
            scored.append((fg, str(r["disaster_event"]), idx))
        scored.sort(reverse=True)
        chosen: list[int] = []
        seen_events: dict[str, int] = {}
        for _, ev, idx in scored:
            if seen_events.get(ev, 0) < per_event:
                chosen.append(idx)
                seen_events[ev] = seen_events.get(ev, 0) + 1
            if len(chosen) >= max_total:
                break
        return chosen

    val_idxs = pick_rich_indices(val_ds, per_event=1, max_total=2)
    test_idxs = pick_rich_indices(test_ds, per_event=2, max_total=2)
    samples = [("VAL", val_ds, i) for i in val_idxs] + [
        ("TEST", test_ds, i) for i in test_idxs
    ]

    ordered_v = [
        ("optical_only", "Optical-Only"),
        ("sar_only", "SAR-Only"),
        ("dual_stream_concat", "Naive Concat"),
        ("dual_stream_gated_fusion", "Prior-Gated Fusion"),
    ]
    present_v = [(vid, vlbl) for vid, vlbl in ordered_v if vid in models]
    n_cols = 3 + len(present_v)
    n_rows = len(samples)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.75 * n_rows), constrained_layout=True
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Pre-Event Optical", "Post-Event SAR", "Ground Truth"] + [
        vlbl for _, vlbl in present_v
    ]

    for row_i, (split_tag, ds, idx) in enumerate(samples):
        row_meta = ds.rows[idx]
        tile_id = str(row_meta["sample_id"])
        raw_opt = np.asarray(Image.open(data_dir / row_meta["pre_event_path"]))[:, :, :3]
        raw_sar = np.asarray(Image.open(data_dir / row_meta["post_event_path"]))
        if raw_sar.ndim == 3:
            raw_sar = raw_sar[:, :, 0]

        item = ds[idx]
        gt_mask = item["clf_mask"].numpy()
        pre_t = item["pre_opt"].unsqueeze(0).to(device)
        post_t = item["post_sar"].unsqueeze(0).to(device)

        axes[row_i, 0].imshow(raw_opt)
        axes[row_i, 0].set_ylabel(
            f"[{split_tag}]\n{tile_id}", fontsize=8.5, fontweight="bold"
        )
        axes[row_i, 1].imshow(raw_sar, cmap="gray")
        axes[row_i, 2].imshow(colorize_mask(gt_mask))

        with torch.no_grad():
            for col_j, (vid, _) in enumerate(present_v, start=3):
                logits = models[vid](pre_t, post_t)  # type: ignore[operator]
                pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                axes[row_i, col_j].imshow(colorize_mask(pred_mask))

        for col_k in range(n_cols):
            axes[row_i, col_k].set_xticks([])
            axes[row_i, col_k].set_yticks([])
            if row_i == 0:
                axes[row_i, col_k].set_title(
                    col_titles[col_k], fontsize=9.5, fontweight="bold"
                )

    patches = [
        mpatches.Patch(color=DAMAGE_CMAP[0] / 255.0, label="0: Background"),
        mpatches.Patch(color=DAMAGE_CMAP[1] / 255.0, label="1: Intact"),
        mpatches.Patch(color=DAMAGE_CMAP[2] / 255.0, label="2: Damaged"),
        mpatches.Patch(color=DAMAGE_CMAP[3] / 255.0, label="3: Destroyed"),
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "GeoResQ Qualitative Ablation — Research Decision-Support Prototype (Requires Human Review)",
        fontsize=11.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summarize_row(
    variant_id: str, short_name: str, summary: dict[str, object]
) -> dict[str, object]:
    m_info = summary["model_info"]  # type: ignore[index]
    hw_info = summary["hardware"]  # type: ignore[index]
    val_m = summary["final_validation_metrics"]  # type: ignore[index]
    test_m = summary["unseen_event_test_metrics_noto_earthquake"]  # type: ignore[index]
    best_c = summary["best_checkpoint"]  # type: ignore[index]
    return {
        "variant_id": variant_id,
        "short_name": short_name,
        "params_M": round(int(m_info["total_params"]) / 1e6, 2),  # type: ignore[index]
        "peak_vram_mb": hw_info["peak_vram_mb"],  # type: ignore[index]
        "train_sec": hw_info["training_duration_seconds"],  # type: ignore[index]
        "ms_per_tile": val_m["ms_per_tile"],  # type: ignore[index]
        "best_epoch": best_c["epoch"],  # type: ignore[index]
        "val_mIoU_all4": val_m["mIoU_all4"],  # type: ignore[index]
        "val_mIoU_fg3": val_m["mIoU_foreground3"],  # type: ignore[index]
        "val_macro_f1_fg3": val_m["macro_f1_foreground3"],  # type: ignore[index]
        "val_loc_f1": val_m["bright_loc_f1"],  # type: ignore[index]
        "val_intact_f1": val_m["per_class"]["intact"]["f1"],  # type: ignore[index]
        "val_damaged_f1": val_m["per_class"]["damaged"]["f1"],  # type: ignore[index]
        "val_destroyed_f1": val_m["per_class"]["destroyed"]["f1"],  # type: ignore[index]
        "test_mIoU_all4": test_m["mIoU_all4"],  # type: ignore[index]
        "test_mIoU_fg3": test_m["mIoU_foreground3"],  # type: ignore[index]
        "test_macro_f1_fg3": test_m["macro_f1_foreground3"],  # type: ignore[index]
        "test_loc_f1": test_m["bright_loc_f1"],  # type: ignore[index]
        "test_intact_f1": test_m["per_class"]["intact"]["f1"],  # type: ignore[index]
        "test_damaged_f1": test_m["per_class"]["damaged"]["f1"],  # type: ignore[index]
        "test_destroyed_f1": test_m["per_class"]["destroyed"]["f1"],  # type: ignore[index]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GeoResQ Milestone 4 Ablation Suite.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/ablation_v0_1.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/runs/ablation_v0_1"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variant IDs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from georesq.data.dataset import (
        BrightDamageDataset,
        compute_class_weights_from_manifest,
        verify_zero_event_leakage,
    )
    from georesq.models import build_model, get_parameter_groups
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
    s_cfg = cfg["shared_data"]  # type: ignore[index]
    o_cfg = cfg["optimization"]  # type: ignore[index]
    all_variants: dict[str, dict[str, object]] = cfg["variants"]  # type: ignore[assignment,index]

    selected_ids = args.variants if args.variants else list(all_variants.keys())
    seed = int(exp_cfg["seed"])  # type: ignore[index]
    epochs = int(args.epochs) if args.epochs is not None else int(o_cfg["epochs"])  # type: ignore[index]

    if args.dry_run:
        mf_check = REPO_ROOT / str(exp_cfg["manifest_relpath"])  # type: ignore[index]
        leak_info = verify_zero_event_leakage(mf_check)
        dry_summary = {}
        for vid in selected_ids:
            v_cfg = dict(all_variants[vid])
            v_cfg["pretrained"] = False
            v_cfg["stack_sar_3ch"] = s_cfg["stack_sar_3ch"]
            m = build_model(v_cfg)
            dry_summary[vid] = count_model_parameters(m)
        print(
            json.dumps(
                {
                    "dry_run": "PASSED",
                    "split_events": leak_info,
                    "variants": dry_summary,
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
    class_weights = compute_class_weights_from_manifest(
        manifest_path, power=float(o_cfg["class_weight_power"])  # type: ignore[index]
    )

    val_ds = BrightDamageDataset(
        data_root=data_dir,
        manifest_path=manifest_path,
        split="val",
        augment=False,
        norm_mode=str(s_cfg["norm_mode"]),  # type: ignore[index]
        stack_sar_3ch=bool(s_cfg["stack_sar_3ch"]),  # type: ignore[index]
        input_modality=str(s_cfg["input_modality"]),  # type: ignore[index]
        seed=seed,
    )
    test_ds = BrightDamageDataset(
        data_root=data_dir,
        manifest_path=manifest_path,
        split="test",
        augment=False,
        norm_mode=str(s_cfg["norm_mode"]),  # type: ignore[index]
        stack_sar_3ch=bool(s_cfg["stack_sar_3ch"]),  # type: ignore[index]
        input_modality=str(s_cfg["input_modality"]),  # type: ignore[index]
        seed=seed,
    )

    num_workers = int(o_cfg.get("num_workers", 2)) if device.type == "cuda" else 0  # type: ignore[attr-defined]
    val_loader = DataLoader(
        val_ds, batch_size=int(o_cfg["eval_batch_size"]), shuffle=False, num_workers=num_workers  # type: ignore[index]
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(o_cfg["eval_batch_size"]), shuffle=False, num_workers=num_workers  # type: ignore[index]
    )

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    variant_summaries: dict[str, dict[str, object]] = {}
    variant_ckpts: dict[str, tuple[dict[str, object], Path]] = {}

    short_map = {
        "optical_only": "Optical-Only (R18)",
        "sar_only": "SAR-Only (R18)",
        "dual_stream_concat": "Dual Concat (R18)",
        "dual_stream_gated_fusion": "Prior-Gated Fusion (Ours)",
    }

    for vid in selected_ids:
        v_cfg = dict(all_variants[vid])
        v_cfg["stack_sar_3ch"] = s_cfg["stack_sar_3ch"]
        v_dir = out_root / vid
        v_dir.mkdir(parents=True, exist_ok=True)
        v_sum_path = v_dir / "summary.json"
        v_best_path = v_dir / "best_model.pth"
        v_last_path = v_dir / "last_checkpoint.pth"

        if v_sum_path.exists() and v_best_path.exists():
            print(f"\n[SKIP] {vid} already completed — loading {v_sum_path}")
            variant_summaries[vid] = json.loads(v_sum_path.read_text(encoding="utf-8"))
            variant_ckpts[vid] = (v_cfg, v_best_path)
            continue

        set_global_seed(seed)
        train_ds = BrightDamageDataset(
            data_root=data_dir,
            manifest_path=manifest_path,
            split="train",
            crop_size=int(o_cfg["crop_size"]),  # type: ignore[index]
            norm_mode=str(s_cfg["norm_mode"]),  # type: ignore[index]
            stack_sar_3ch=bool(s_cfg["stack_sar_3ch"]),  # type: ignore[index]
            input_modality=str(s_cfg["input_modality"]),  # type: ignore[index]
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(o_cfg["train_batch_size"]),  # type: ignore[index]
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(v_cfg).to(device)
        param_info = count_model_parameters(model)
        criterion = BrightComboLoss(
            class_weights=class_weights,
            lovasz_weight=float(o_cfg["lovasz_weight"]),  # type: ignore[index]
            ignore_index=255,
        ).to(device)

        param_groups = get_parameter_groups(
            model,
            base_lr=float(o_cfg["learning_rate"]),  # type: ignore[index]
            backbone_lr_scale=float(o_cfg.get("backbone_lr_scale", 0.33)),  # type: ignore[attr-defined]
            weight_decay=float(o_cfg["weight_decay"]),  # type: ignore[index]
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=float(o_cfg["min_lr"])  # type: ignore[index]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch = 1
        best_score = -1.0
        prev_sec = 0.0
        history: list[dict[str, object]] = []
        sel_key = str(o_cfg.get("selection_metric", "mIoU_foreground3"))  # type: ignore[attr-defined]

        if v_last_path.exists():
            ckpt = load_checkpoint(v_last_path, model, optimizer, scheduler, scaler, device)
            start_epoch = int(ckpt["epoch"]) + 1  # type: ignore[arg-type]
            best_score = float(ckpt["best_score"])  # type: ignore[arg-type]
            prev_sec = float(ckpt.get("meta", {}).get("cumulative_train_sec", 0.0))  # type: ignore[union-attr]
            history = list(ckpt.get("history", []))  # type: ignore[arg-type]
            print(f"[RESUME] {vid} from epoch {start_epoch}")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        print(
            f"\n=== Training Variant: {vid} ({v_cfg['label']}) | "
            f"params={param_info['total_params']/1e6:.2f}M | epochs={epochs} ==="
        )
        t0 = time.perf_counter()

        for epoch in range(start_epoch, epochs + 1):
            ep_t0 = time.perf_counter()
            train_ds.set_epoch(epoch)
            model.train()
            run_loss = 0.0
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

                run_loss += float(loss.item())
                steps += 1

            scheduler.step()
            train_loss = run_loss / max(1, steps)
            val_metrics = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp)
            score = float(val_metrics[sel_key])  # type: ignore[arg-type]
            ep_sec = time.perf_counter() - ep_t0

            history.append(
                {
                    "epoch": epoch,
                    "lr": round(float(optimizer.param_groups[-1]["lr"]), 7),
                    "train_loss": round(train_loss, 6),
                    "val_loss": val_metrics["loss"],
                    "val_mIoU_all4": val_metrics["mIoU_all4"],
                    "val_mIoU_foreground3": val_metrics["mIoU_foreground3"],
                    "val_macro_f1_foreground3": val_metrics["macro_f1_foreground3"],
                    "val_bright_loc_f1": val_metrics["bright_loc_f1"],
                    "epoch_seconds": round(ep_sec, 2),
                }
            )

            cum_sec = round(prev_sec + (time.perf_counter() - t0), 2)
            is_best = score > best_score
            if is_best:
                best_score = score
                save_checkpoint(
                    v_best_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_score,
                    history,
                    extra_meta={"cumulative_train_sec": cum_sec},
                )

            if epoch % int(o_cfg.get("save_every_epochs", 5)) == 0 or epoch == epochs:  # type: ignore[attr-defined]
                save_checkpoint(
                    v_last_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_score,
                    history,
                    extra_meta={"cumulative_train_sec": cum_sec},
                )

            if epoch in (1, 5, 10, 15, 20, epochs) or is_best:
                tag = " *BEST*" if is_best else ""
                print(
                    f"  [{vid}] Ep {epoch:02d}/{epochs:02d} | "
                    f"tr_loss={train_loss:.4f} | va_mIoU_fg3={100*score:.2f}% | "
                    f"va_loc_f1={100*float(val_metrics['bright_loc_f1']):.2f}% | {ep_sec:.1f}s{tag}"
                )

        total_sec = round(prev_sec + (time.perf_counter() - t0), 2)
        peak_vram = (
            round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if device.type == "cuda"
            else 0.0
        )

        best_ckpt = load_checkpoint(v_best_path, model, device=device)
        best_ep = int(best_ckpt["epoch"])  # type: ignore[arg-type]
        final_val = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp)
        final_test = evaluate_model(model, test_loader, criterion, device, use_amp=use_amp)

        write_history_csv(history, v_dir / "history.csv")
        save_training_curves(
            history, v_dir / "curves.png", title=f"GeoResQ Ablation — {v_cfg['label']}"
        )

        v_summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "variant_id": vid,
            "label": v_cfg["label"],
            "seed": seed,
            "hardware": {
                "device": str(device),
                "gpu_name": gpu_name,
                "amp_enabled": use_amp,
                "peak_vram_mb": peak_vram,
                "training_duration_seconds": total_sec,
            },
            "model_info": param_info,
            "best_checkpoint": {
                "epoch": best_ep,
                "selection_metric": sel_key,
                "best_val_score": round(best_score, 6),
            },
            "loss_progression": {
                "first_epoch_train_loss": history[0]["train_loss"],
                "final_epoch_train_loss": history[-1]["train_loss"],
                "loss_decreased": float(history[-1]["train_loss"]) < float(history[0]["train_loss"]),  # type: ignore[arg-type]
            },
            "final_validation_metrics": final_val,
            "unseen_event_test_metrics_noto_earthquake": final_test,
        }
        v_sum_path.write_text(json.dumps(v_summary, indent=2) + "\n", encoding="utf-8")
        variant_summaries[vid] = v_summary
        variant_ckpts[vid] = (v_cfg, v_best_path)

        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_rows: list[dict[str, object]] = []
    base_rel = str(exp_cfg.get("baseline_summary_relpath", ""))  # type: ignore[attr-defined]
    base_path = REPO_ROOT / base_rel
    if base_rel and base_path.exists():
        base_sum = json.loads(base_path.read_text(encoding="utf-8"))
        comparison_rows.append(
            summarize_row("baseline_unet", "Scratch UNet (6ch)", base_sum)
        )

    for vid in selected_ids:
        comparison_rows.append(
            summarize_row(vid, short_map.get(vid, vid), variant_summaries[vid])
        )

    csv_path = out_root / "ablation_v0_1_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    save_ablation_bar_chart(comparison_rows, out_root / "ablation_v0_1_comparison.png")
    save_qualitative_predictions(
        data_dir,
        val_ds,
        test_ds,
        variant_ckpts,
        device,
        out_root / "ablation_v0_1_predictions.png",
    )

    suite_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": exp_cfg["suite_name"],  # type: ignore[index]
        "subset_version": exp_cfg["subset_version"],  # type: ignore[index]
        "seed": seed,
        "comparison_table": comparison_rows,
        "variant_details": variant_summaries,
    }
    suite_json = out_root / "ablation_v0_1_summary.json"
    suite_json.write_text(json.dumps(suite_report, indent=2) + "\n", encoding="utf-8")

    print("\n=== MILESTONE 4 CONTROLLED ABLATION TABLE ===")
    print(json.dumps(comparison_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
