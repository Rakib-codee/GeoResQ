#!/usr/bin/env python3
"""GeoResQ Milestone 5 — Medium-Scale Multi-Hazard Training & Evaluation (BRIGHT v0.2).

Trains & evaluates the 4 controlled v0.2 models on 294 triplets across all 11
open-license BRIGHT disaster events, reporting overall and per-event unseen
generalization (full 79-tile Noto Earthquake + 25-tile Marshall Wildfire).
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

from georesq.evaluation.metrics import DamageEvaluator

DAMAGE_CMAP = np.array(
    [
        [20, 24, 35],    # 0: Background
        [46, 204, 113],  # 1: Intact
        [243, 156, 18],  # 2: Damaged
        [231, 76, 60],   # 3: Destroyed
    ],
    dtype=np.uint8,
)


def resolve_subset_v0_2(explicit_dir: Path | None) -> Path:
    if explicit_dir is not None and (explicit_dir / "pre-event").exists():
        return explicit_dir

    candidates = [
        Path("/kaggle/working/bright_subset_v0_2"),
        Path("/kaggle/input/georesq-bright-v0-2/bright_subset_v0_2"),
        Path("/kaggle/input/georesq-bright-v0-2"),
    ]
    for c in candidates:
        if (c / "pre-event").exists() and (c / "target").exists():
            return c

    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            for mf in root.rglob("manifest_v0_2.csv"):
                if (mf.parent / "pre-event").exists():
                    return mf.parent

    raise FileNotFoundError(
        "Could not find bright_subset_v0_2. Attach your v0.2 Kaggle output (+ Add Input) or pass --data-dir."
    )


def colorize_mask(mask_2d: np.ndarray) -> np.ndarray:
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(4):
        rgb[mask_2d == cls_id] = DAMAGE_CMAP[cls_id]
    rgb[mask_2d == 255] = np.array([140, 140, 140], dtype=np.uint8)
    return rgb


def evaluate_with_per_event(
    model, loader, criterion, device, use_amp: bool = False
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    import torch

    model.eval()
    global_eval = DamageEvaluator(num_classes=4, ignore_index=255)
    event_evals: dict[str, DamageEvaluator] = {}
    total_loss = 0.0
    n_batches = 0
    n_imgs = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre_opt"].to(device, non_blocking=True)
            post = batch["post_sar"].to(device, non_blocking=True)
            targets = batch["clf_mask"].to(device, non_blocking=True)
            events = batch["disaster_event"]

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(pre, post)
            loss = criterion(logits.float(), targets)
            preds = torch.argmax(logits, dim=1)

            total_loss += float(loss.item())
            n_batches += 1
            n_imgs += int(pre.size(0))

            tgt_np = targets.cpu().numpy()
            prd_np = preds.cpu().numpy()
            global_eval.update(tgt_np, prd_np)

            for b_i, ev_name in enumerate(events):
                ev_str = str(ev_name)
                if ev_str not in event_evals:
                    event_evals[ev_str] = DamageEvaluator(num_classes=4, ignore_index=255)
                event_evals[ev_str].update(tgt_np[b_i : b_i + 1], prd_np[b_i : b_i + 1])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(1e-6, time.perf_counter() - t0)

    summary = global_eval.compute_summary()
    summary["loss"] = round(total_loss / max(1, n_batches), 6)
    summary["inference_seconds"] = round(elapsed, 4)
    summary["ms_per_tile"] = round(1000.0 * elapsed / max(1, n_imgs), 2)
    summary["tiles_per_sec"] = round(n_imgs / elapsed, 2)

    per_event_summary = {
        ev: event_evals[ev].compute_summary() for ev in sorted(event_evals.keys())
    }
    return summary, per_event_summary


def save_v02_bar_chart(rows: list[dict[str, object]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_labels = [str(r["short_name"]) for r in rows]
    x = np.arange(len(short_labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    panels = [
        (axes[0], "val", "Validation (40 Tiles: Beirut, Libya, Congo)"),
        (axes[1], "noto", "Unseen Noto Earthquake (All 79 Tiles: GSI & Umbra)"),
        (axes[2], "marshall", "Unseen Marshall Wildfire (25 Tiles: NAIP & Capella)"),
    ]
    for ax, prefix, title in panels:
        miou_fg = [100.0 * float(r[f"{prefix}_mIoU_fg3"]) for r in rows]
        f1_fg = [100.0 * float(r[f"{prefix}_macro_f1_fg3"]) for r in rows]
        loc_f1 = [100.0 * float(r[f"{prefix}_loc_f1"]) for r in rows]

        b1 = ax.bar(x - width, miou_fg, width, label="mIoU (3 Damage)%")
        b2 = ax.bar(x, f1_fg, width, label="Macro-F1 (3 Damage)%")
        b3 = ax.bar(x + width, loc_f1, width, label="Building Loc F1%")

        for bars in (b1, b2, b3):
            for bar in bars:
                hv = bar.get_height()
                ax.annotate(
                    f"{hv:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, hv),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ymax = max(max(miou_fg + f1_fg + loc_f1, default=15.0) * 1.18, 20.0)
        ax.set_ylim(0, min(105.0, ymax))

    fig.suptitle(
        "GeoResQ Milestone 5 — Multi-Hazard Medium-Scale Benchmark (BRIGHT v0.2: 294 Triplets, 11 Disasters)",
        fontsize=12.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_v02_predictions(
    data_dir: Path,
    val_ds,
    test_ds,
    variant_ckpts: dict[str, tuple[dict[str, object], Path]],
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
    for vid, (m_cfg, ckpt_path) in variant_ckpts.items():
        cfg_local = dict(m_cfg)
        cfg_local["pretrained"] = False
        m = build_model(cfg_local).to(device)
        load_checkpoint(ckpt_path, m, device=device)
        m.eval()
        models[vid] = m

    def pick_one_per_event(ds, target_events: list[str]) -> list[int]:
        chosen: list[int] = []
        for tev in target_events:
            cands = []
            for idx, r in enumerate(ds.rows):
                if str(r["disaster_event"]) == tev:
                    lbl = json.loads(str(r["label_information"]))
                    dmg = int(lbl.get("damaged", 0)) + int(lbl.get("destroyed", 0))
                    fg = dmg + int(lbl.get("intact", 0))
                    cands.append((dmg > 0, fg, idx))
            if cands:
                cands.sort(reverse=True)
                chosen.append(cands[0][2])
        return chosen

    val_idxs = pick_one_per_event(val_ds, ["beirut-explosion", "libya-flood"])
    test_idxs = pick_one_per_event(test_ds, ["noto-earthquake", "marshall-wildfire"])
    samples = [("VAL", val_ds, i) for i in val_idxs] + [
        ("UNSEEN TEST", test_ds, i) for i in test_idxs
    ]

    ordered = [
        ("optical_only_v0_2", "Optical-Only"),
        ("sar_only_v0_2", "SAR-Only"),
        ("dual_concat_v0_2", "Dual Concat"),
        ("lite_gated_fusion_v0_2", "Lite Gated Fusion (Ours)"),
    ]
    present = [(vid, lbl) for vid, lbl in ordered if vid in models]
    n_cols = 3 + len(present)
    n_rows = len(samples)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.75 * n_rows), constrained_layout=True
    )
    col_titles = ["Pre-Event Optical", "Post-Event SAR", "Ground Truth"] + [
        lbl for _, lbl in present
    ]

    for r_i, (sp_tag, ds, idx) in enumerate(samples):
        meta = ds.rows[idx]
        tid = str(meta["sample_id"])
        raw_opt = np.asarray(Image.open(data_dir / meta["pre_event_path"]))[:, :, :3]
        raw_sar = np.asarray(Image.open(data_dir / meta["post_event_path"]))
        if raw_sar.ndim == 3:
            raw_sar = raw_sar[:, :, 0]

        item = ds[idx]
        gt = item["clf_mask"].numpy()
        pre_t = item["pre_opt"].unsqueeze(0).to(device)
        post_t = item["post_sar"].unsqueeze(0).to(device)

        axes[r_i, 0].imshow(raw_opt)
        axes[r_i, 0].set_ylabel(f"[{sp_tag}]\n{tid}", fontsize=8, fontweight="bold")
        axes[r_i, 1].imshow(raw_sar, cmap="gray")
        axes[r_i, 2].imshow(colorize_mask(gt))

        with torch.no_grad():
            for c_j, (vid, _) in enumerate(present, start=3):
                logits = models[vid](pre_t, post_t)  # type: ignore[operator]
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                axes[r_i, c_j].imshow(colorize_mask(pred))

        for c_k in range(n_cols):
            axes[r_i, c_k].set_xticks([])
            axes[r_i, c_k].set_yticks([])
            if r_i == 0:
                axes[r_i, c_k].set_title(col_titles[c_k], fontsize=9.5, fontweight="bold")

    patches = [
        mpatches.Patch(color=DAMAGE_CMAP[0] / 255.0, label="0: Background"),
        mpatches.Patch(color=DAMAGE_CMAP[1] / 255.0, label="1: Intact"),
        mpatches.Patch(color=DAMAGE_CMAP[2] / 255.0, label="2: Damaged"),
        mpatches.Patch(color=DAMAGE_CMAP[3] / 255.0, label="3: Destroyed"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "GeoResQ Milestone 5 Qualitative Predictions (v0.2) — Research Decision-Support Prototype (Human Review Required)",
        fontsize=11,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_table_row(vid: str, short_name: str, v_sum: dict[str, object]) -> dict[str, object]:
    m_info = v_sum["model_info"]  # type: ignore[index]
    hw = v_sum["hardware"]  # type: ignore[index]
    val_m = v_sum["final_validation_metrics"]  # type: ignore[index]
    test_m = v_sum["unseen_test_combined_metrics"]  # type: ignore[index]
    ev_test = v_sum["unseen_test_per_event"]  # type: ignore[index]
    noto = ev_test["noto-earthquake"]  # type: ignore[index]
    marsh = ev_test["marshall-wildfire"]  # type: ignore[index]

    return {
        "variant_id": vid,
        "short_name": short_name,
        "params_M": round(int(m_info["total_params"]) / 1e6, 2),  # type: ignore[index]
        "peak_vram_mb": hw["peak_vram_mb"],  # type: ignore[index]
        "train_sec": hw["training_duration_seconds"],  # type: ignore[index]
        "ms_per_tile": val_m["ms_per_tile"],  # type: ignore[index]
        "best_epoch": v_sum["best_checkpoint"]["epoch"],  # type: ignore[index]
        "val_bg_recall": val_m["per_class"]["background"]["recall"],  # type: ignore[index]
        "val_mIoU_all4": val_m["mIoU_all4"],  # type: ignore[index]
        "val_mIoU_fg3": val_m["mIoU_foreground3"],  # type: ignore[index]
        "val_macro_f1_fg3": val_m["macro_f1_foreground3"],  # type: ignore[index]
        "val_loc_f1": val_m["bright_loc_f1"],  # type: ignore[index]
        "val_intact_f1": val_m["per_class"]["intact"]["f1"],  # type: ignore[index]
        "val_damaged_f1": val_m["per_class"]["damaged"]["f1"],  # type: ignore[index]
        "val_destroyed_f1": val_m["per_class"]["destroyed"]["f1"],  # type: ignore[index]
        "test104_mIoU_fg3": test_m["mIoU_foreground3"],  # type: ignore[index]
        "test104_macro_f1_fg3": test_m["macro_f1_foreground3"],  # type: ignore[index]
        "test104_loc_f1": test_m["bright_loc_f1"],  # type: ignore[index]
        "noto_mIoU_fg3": noto["mIoU_foreground3"],
        "noto_macro_f1_fg3": noto["macro_f1_foreground3"],
        "noto_loc_f1": noto["bright_loc_f1"],
        "noto_damaged_f1": noto["per_class"]["damaged"]["f1"],
        "noto_destroyed_f1": noto["per_class"]["destroyed"]["f1"],
        "marshall_mIoU_fg3": marsh["mIoU_foreground3"],
        "marshall_macro_f1_fg3": marsh["macro_f1_foreground3"],
        "marshall_loc_f1": marsh["bright_loc_f1"],
        "marshall_destroyed_f1": marsh["per_class"]["destroyed"]["f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & evaluate GeoResQ v0.2 benchmark.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/benchmark_v0_2.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/runs/benchmark_v0_2"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None)
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
    from georesq.training.losses import BrightComboLoss, BuildingLocBCEDiceLoss

    cfg = load_yaml_config(args.config)
    exp_cfg = cfg["experiment"]  # type: ignore[index]
    s_cfg = cfg["shared_data"]  # type: ignore[index]
    o_cfg = cfg["optimization"]  # type: ignore[index]
    all_variants: dict[str, dict[str, object]] = cfg["variants"]  # type: ignore[assignment,index]

    selected_ids = args.variants if args.variants else list(all_variants.keys())
    seed = int(exp_cfg["seed"])  # type: ignore[index]
    epochs = int(args.epochs) if args.epochs is not None else int(o_cfg["epochs"])  # type: ignore[index]
    val_every = int(o_cfg.get("val_every_epochs", 2))  # type: ignore[attr-defined]
    aux_loc_w = float(o_cfg.get("aux_loc_weight", 0.5))  # type: ignore[attr-defined]

    if args.dry_run:
        mf_cand = REPO_ROOT / str(exp_cfg["manifest_relpath"])  # type: ignore[index]
        if not mf_cand.exists():
            mf_cand = REPO_ROOT / "results/metrics/manifest_v0_1.csv"
        leak_info = verify_zero_event_leakage(mf_cand)
        weights = compute_class_weights_from_manifest(
            mf_cand, power=float(o_cfg["class_weight_power"])  # type: ignore[index]
        )
        dry_models = {}
        for vid in selected_ids:
            vc = dict(all_variants[vid])
            vc["pretrained"] = False
            vc["stack_sar_3ch"] = s_cfg["stack_sar_3ch"]
            m = build_model(vc)
            dry_models[vid] = count_model_parameters(m)
        print(
            json.dumps(
                {
                    "dry_run": "PASSED",
                    "manifest_checked": str(mf_cand.name),
                    "class_weights_p015": weights,
                    "split_events": leak_info,
                    "variants": dry_models,
                },
                indent=2,
            )
        )
        return 0

    data_dir = resolve_subset_v0_2(args.data_dir)
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else (
            (data_dir / "manifest_v0_2.csv")
            if (data_dir / "manifest_v0_2.csv").exists()
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
        loc_criterion = BuildingLocBCEDiceLoss(ignore_index=255).to(device)

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
            f"\n=== Training {vid} ({v_cfg['label']}) | "
            f"params={param_info['total_params']/1e6:.2f}M | train={len(train_ds)} | epochs={epochs} ==="
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
                loc_targets = batch["loc_mask"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if hasattr(model, "forward_with_aux"):
                        clf_logits, loc_logits = model.forward_with_aux(pre, post)
                    else:
                        clf_logits, loc_logits = model(pre, post), None

                loss = criterion(clf_logits.float(), targets)
                if loc_logits is not None and aux_loc_w > 0.0:
                    loss = loss + aux_loc_w * loc_criterion(loc_logits.float(), loc_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                run_loss += float(loss.item())
                steps += 1

            scheduler.step()
            train_loss = run_loss / max(1, steps)

            should_eval = (
                epoch == 1
                or (epoch % val_every == 0)
                or (epoch >= epochs - 2)
                or (epoch == epochs)
            )
            if should_eval:
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
                tag = " *BEST*" if is_best else ""
                print(
                    f"  [{vid}] Ep {epoch:02d}/{epochs:02d} | tr_loss={train_loss:.4f} | "
                    f"va_mIoU_fg3={100*score:.2f}% | va_loc_f1={100*float(val_metrics['bright_loc_f1']):.2f}% | "
                    f"va_bg_rec={100*float(val_metrics['per_class']['background']['recall']):.1f}% | {ep_sec:.1f}s{tag}"  # type: ignore[index]
                )

            if epoch % int(o_cfg.get("save_every_epochs", 5)) == 0 or epoch == epochs:  # type: ignore[attr-defined]
                cum_sec = round(prev_sec + (time.perf_counter() - t0), 2)
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

        total_sec = round(prev_sec + (time.perf_counter() - t0), 2)
        peak_vram = (
            round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if device.type == "cuda"
            else 0.0
        )

        best_ckpt = load_checkpoint(v_best_path, model, device=device)
        best_ep = int(best_ckpt["epoch"])  # type: ignore[arg-type]

        final_val, val_per_event = evaluate_with_per_event(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        final_test, test_per_event = evaluate_with_per_event(
            model, test_loader, criterion, device, use_amp=use_amp
        )

        write_history_csv(history, v_dir / "history.csv")
        save_training_curves(history, v_dir / "curves.png", title=f"GeoResQ v0.2 — {v_cfg['label']}")

        v_summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "variant_id": vid,
            "label": v_cfg["label"],
            "short_name": v_cfg["short_name"],
            "subset_version": "v0.2",
            "seed": seed,
            "train_class_weights": class_weights,
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
            "validation_per_event": val_per_event,
            "unseen_test_combined_metrics": final_test,
            "unseen_test_per_event": test_per_event,
        }
        v_sum_path.write_text(json.dumps(v_summary, indent=2) + "\n", encoding="utf-8")
        variant_summaries[vid] = v_summary
        variant_ckpts[vid] = (v_cfg, v_best_path)

        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table_rows = [
        build_table_row(vid, str(all_variants[vid]["short_name"]), variant_summaries[vid])
        for vid in selected_ids
    ]

    csv_path = out_root / "benchmark_v0_2_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    save_v02_bar_chart(table_rows, out_root / "benchmark_v0_2_comparison.png")
    save_v02_predictions(
        data_dir,
        val_ds,
        test_ds,
        variant_ckpts,
        device,
        out_root / "benchmark_v0_2_predictions.png",
    )

    suite_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": exp_cfg["suite_name"],  # type: ignore[index]
        "subset_version": "v0.2",
        "seed": seed,
        "train_class_weights": class_weights,
        "comparison_table": table_rows,
        "variant_details": variant_summaries,
    }
    (out_root / "benchmark_v0_2_summary.json").write_text(
        json.dumps(suite_payload, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== MILESTONE 5 BENCHMARK v0.2 TABLE ===")
    print(json.dumps(table_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
