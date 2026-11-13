#!/usr/bin/env python3
"""Milestone 7 — Cross-Event Generalization, Confusion Matrices & Failure Mode Analysis.

Evaluates the 4 frozen Milestone 5 v0.2 checkpoints across all 5 held-out disasters
(144 tiles: 40 Val [`beirut-explosion`, `congo-volcano`, `libya-flood`] + 104 Unseen Test
[`marshall-wildfire`, `noto-earthquake`]), producing:
  1. `results/metrics/failure_analysis_v0_2_summary.json` (per-hazard generalization +
     4x4 confusion matrices + decoupled localization vs. severity error taxonomy)
  2. `results/metrics/per_tile_audit_v0_2.csv` (144-tile exact audit & failure tags)
  3. `results/figures/cross_event_and_confusion_v0_2.png`
  4. `results/figures/failure_modes_gallery_v0_2.png`
  5. Compact fp16 deployment weights in `/kaggle/working/exports/` for Milestone 8 CLI.
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

from run_stress_test import (  # noqa: E402
    DAMAGE_COLORS,
    mask_to_rgb,
    resolve_runs_dir,
    resolve_subset_v0_2,
    sync_m5_artifacts_if_present,
)

HAZARD_LABELS = {
    "beirut-explosion": "Beirut Blast\n(Val, n=15)",
    "congo-volcano": "Congo Volcano\n(Val, n=10)",
    "libya-flood": "Libya Flood\n(Val, n=15)",
    "marshall-wildfire": "Marshall Wildfire\n(Test, n=25)",
    "noto-earthquake": "Noto Earthquake\n(Test, n=79)",
}


def compute_single_tile_stats(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = (gt != 255) & (gt >= 0) & (gt < 4)
    g = gt[valid]
    p = pred[valid]
    eps = 1e-7

    gt_b = g > 0
    pr_b = p > 0
    loc_tp = float(np.logical_and(gt_b, pr_b).sum())
    loc_fp = float(np.logical_and(~gt_b, pr_b).sum())
    loc_fn = float(np.logical_and(gt_b, ~pr_b).sum())
    loc_f1 = (2.0 * loc_tp) / (2.0 * loc_tp + loc_fp + loc_fn + eps)

    # Active foreground IoU across classes present in either GT or prediction
    ious = []
    for c in (1, 2, 3):
        tp = float(np.logical_and(g == c, p == c).sum())
        union = float(np.logical_or(g == c, p == c).sum())
        if union > 0:
            ious.append(tp / (union + eps))
    active_fg_iou = float(np.mean(ious)) if ious else 1.0

    d_tp = float(np.logical_and(g == 3, p == 3).sum())
    d_fp = float(np.logical_and(g != 3, p == 3).sum())
    d_fn = float(np.logical_and(g == 3, p != 3).sum())
    dest_f1 = (2.0 * d_tp) / (2.0 * d_tp + d_fp + d_fn + eps) if (d_tp + d_fn) > 0 else 0.0

    dest_to_intact = float(np.logical_and(g == 3, p == 1).sum())
    dmg_to_intact = float(np.logical_and(g == 2, p == 1).sum())

    return {
        "loc_f1": round(loc_f1, 5),
        "active_fg_iou": round(active_fg_iou, 5),
        "destroyed_f1": round(dest_f1, 5),
        "loc_fp_px": loc_fp,
        "loc_fn_px": loc_fn,
        "dest_to_intact_px": dest_to_intact,
        "dmg_to_intact_px": dmg_to_intact,
    }


def classify_tile_mode(
    gt: np.ndarray,
    gated_stats: dict[str, float],
) -> str:
    tot = max(1, gt.size)
    bldg_px = int((gt > 0).sum())
    dmg_px = int((gt == 2).sum())
    dest_px = int((gt == 3).sum())

    if bldg_px < int(0.003 * tot):
        return "sparse_or_clean_background"
    if gated_stats["active_fg_iou"] >= 0.36 and gated_stats["loc_f1"] >= 0.72:
        return "cross_event_success"
    if dmg_px > 8000 and gated_stats["dmg_to_intact_px"] > 0.35 * dmg_px:
        return "sub_roof_damaged_as_intact"
    if dest_px > 5000 and gated_stats["dest_to_intact_px"] > 0.35 * dest_px:
        return "rubble_destroyed_as_intact"
    if gated_stats["loc_fp_px"] > 0.60 * bldg_px:
        return "sar_layover_or_clutter_fp"
    if gated_stats["loc_fn_px"] > 0.45 * bldg_px:
        return "missed_small_or_sparse_buildings"
    return "moderate_mixed_boundary_error"


def save_cross_event_and_confusion_figure(
    event_order: list[str],
    per_event_results: dict[str, dict[str, dict[str, object]]],
    test_summaries: dict[str, dict[str, object]],
    variant_names: dict[str, str],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18.0, 10.5), constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig)

    ax_miou = fig.add_subplot(gs[0, :3])
    ax_loc = fig.add_subplot(gs[0, 3:])

    vids = list(variant_names.keys())
    palette = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
    x = np.arange(len(event_order))
    width = 0.19

    for i, vid in enumerate(vids):
        offset = (i - 1.5) * width
        m_vals = [float(per_event_results[ev][vid]["mIoU_foreground3"]) * 100.0 for ev in event_order]
        l_vals = [float(per_event_results[ev][vid]["bright_loc_f1"]) * 100.0 for ev in event_order]
        ax_miou.bar(x + offset, m_vals, width, label=variant_names[vid], color=palette[i])
        ax_loc.bar(x + offset, l_vals, width, label=variant_names[vid], color=palette[i])

    for ax, ylabel, title in [
        (ax_miou, "Foreground Damage mIoU (3-cls %)", "A. Cross-Hazard Damage mIoU Across 5 Held-Out Disasters"),
        (ax_loc, "Building Localization F1 (%)", "B. Cross-Hazard Building Localization F1 Across 5 Held-Out Disasters"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([HAZARD_LABELS.get(ev, ev) for ev in event_order], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8.3, ncol=2)

    # Bottom row: 3 row-normalized 4x4 confusion matrices on Unseen Test-104
    cm_vids = ["optical_only_v0_2", "dual_concat_v0_2", "lite_gated_fusion_v0_2"]
    cls_short = ["Bg (0)", "Intact (1)", "Damaged (2)", "Destroyed (3)"]
    for idx, vid in enumerate(cm_vids):
        ax_cm = fig.add_subplot(gs[1, idx * 2 : (idx + 1) * 2])
        cm = np.asarray(test_summaries[vid]["confusion_matrix"], dtype=np.float64)
        row_pct = 100.0 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax_cm.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=100.0)
        ax_cm.set_xticks(range(4))
        ax_cm.set_yticks(range(4))
        ax_cm.set_xticklabels(cls_short, rotation=20, ha="right", fontsize=8.5)
        ax_cm.set_yticklabels(cls_short, fontsize=8.5)
        ax_cm.set_xlabel("Predicted Class", fontsize=9)
        if idx == 0:
            ax_cm.set_ylabel("True Class (Row-Normalized %)", fontsize=9)
        ax_cm.set_title(f"Unseen Test-104 Confusion — {variant_names[vid]}", fontsize=10.5, fontweight="bold")
        for r in range(4):
            for c in range(4):
                val = row_pct[r, c]
                txt_col = "white" if val > 52.0 else "#111827"
                ax_cm.text(c, r, f"{val:.1f}%", ha="center", va="center", color=txt_col, fontsize=8.5, fontweight="bold")
        if idx == 2:
            fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row %")

    fig.suptitle(
        "GeoResQ Milestone 7 — Cross-Hazard Generalization & Unseen Test-104 Error Routing",
        fontsize=13.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_failure_gallery(
    data_dir: Path,
    selected_cases: list[tuple[str, dict[str, object], dict[str, str]]],
    models: dict[str, object],
    norm_mode: str,
    stack_sar: bool,
    device,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from georesq.data.dataset import _read_rgb_tif, _read_single_band_tif
    from georesq.preprocessing.transforms import normalize_multimodal

    fig, axes = plt.subplots(len(selected_cases), 6, figsize=(18.5, 3.05 * len(selected_cases)), constrained_layout=True)
    if len(selected_cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Pre-Event Optical",
        "Post-Event SAR",
        "Ground Truth",
        "Optical-Only (v0.2)",
        "Dual Concat (v0.2)",
        "Lite Gated Fusion (Ours)",
    ]

    for r_i, (case_title, audit_row, mf_row) in enumerate(selected_cases):
        pre_u8 = _read_rgb_tif(data_dir / mf_row["pre_event_path"])
        post_u8 = _read_single_band_tif(data_dir / mf_row["post_event_path"])
        gt_u8 = _read_single_band_tif(data_dir / mf_row["target_path"])

        pre_chw, post_chw = normalize_multimodal(pre_u8, post_u8, norm_mode=norm_mode, stack_sar_3ch=stack_sar)
        pre_t = torch.from_numpy(pre_chw).unsqueeze(0).to(device)
        post_t = torch.from_numpy(post_chw).unsqueeze(0).to(device)

        with torch.no_grad():
            p_opt = torch.argmax(models["optical_only_v0_2"](pre_t, post_t), dim=1).squeeze(0).cpu().numpy()  # type: ignore[operator]
            p_cat = torch.argmax(models["dual_concat_v0_2"](pre_t, post_t), dim=1).squeeze(0).cpu().numpy()  # type: ignore[operator]
            p_gat = torch.argmax(models["lite_gated_fusion_v0_2"](pre_t, post_t), dim=1).squeeze(0).cpu().numpy()  # type: ignore[operator]

        axes[r_i, 0].imshow(pre_u8)
        axes[r_i, 0].set_ylabel(
            f"{case_title}\n{mf_row['sample_id']}\nLocF1={audit_row['gated_loc_f1']:.2f} | DestF1={audit_row['gated_destroyed_f1']:.2f}",
            fontsize=8.2,
            fontweight="bold",
        )
        axes[r_i, 1].imshow(post_u8, cmap="gray", vmin=0, vmax=255)
        axes[r_i, 2].imshow(mask_to_rgb(gt_u8))
        axes[r_i, 3].imshow(mask_to_rgb(p_opt))
        axes[r_i, 4].imshow(mask_to_rgb(p_cat))
        axes[r_i, 5].imshow(mask_to_rgb(p_gat))

        for c_i in range(6):
            if r_i == 0:
                axes[r_i, c_i].set_title(col_titles[c_i], fontsize=10, fontweight="bold")
            axes[r_i, c_i].set_xticks([])
            axes[r_i, c_i].set_yticks([])

    fig.suptitle(
        "GeoResQ Milestone 7 — Automated Cross-Hazard Success & Physical Failure Mode Gallery\n"
        "[Dark=Background | Green=Intact | Orange=Damaged | Red=Destroyed]",
        fontsize=12.8,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Milestone 7 Cross-Event & Failure Analysis.")
    parser.add_argument("--benchmark-config", type=Path, default=REPO_ROOT / "configs/benchmark_v0_2.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--exports-dir", type=Path, default=Path("/kaggle/working/exports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from georesq.evaluation.metrics import DamageEvaluator, decompose_confusion_errors
    from georesq.training.loop import load_yaml_config, set_global_seed

    bench_cfg = load_yaml_config(args.benchmark_config)
    exp_cfg = bench_cfg["experiment"]  # type: ignore[index]
    shared_cfg = bench_cfg["shared_data"]  # type: ignore[index]
    variants_cfg: dict[str, dict[str, object]] = bench_cfg["variants"]  # type: ignore[assignment,index]
    seed = int(exp_cfg["seed"])  # type: ignore[index]

    if args.dry_run:
        dummy_cm = [
            [90000, 1200, 100, 200],
            [1500, 7500, 100, 300],
            [400, 1200, 50, 250],
            [500, 900, 50, 1100],
        ]
        decomp = decompose_confusion_errors(dummy_cm)
        print(
            json.dumps(
                {
                    "dry_run": "PASSED",
                    "variants": list(variants_cfg.keys()),
                    "sample_error_decomposition": decomp,
                },
                indent=2,
            )
        )
        return 0

    import torch

    from georesq.data.dataset import _read_rgb_tif, _read_single_band_tif, verify_zero_event_leakage
    from georesq.models import build_model
    from georesq.preprocessing.transforms import normalize_multimodal
    from georesq.training.loop import export_fp16_inference_checkpoint, load_checkpoint

    variant_ids = list(variants_cfg.keys())
    data_dir = resolve_subset_v0_2(args.data_dir)
    runs_dir = resolve_runs_dir(args.runs_dir, variant_ids)
    sync_m5_artifacts_if_present(runs_dir, args.results_dir)

    manifest_path = (
        (data_dir / "manifest_v0_2.csv")
        if (data_dir / "manifest_v0_2.csv").exists()
        else (REPO_ROOT / str(exp_cfg["manifest_relpath"]))  # type: ignore[index]
    )
    verify_zero_event_leakage(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        heldout_rows = [r for r in csv.DictReader(f) if r["split"] in ("val", "test")]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    norm_mode = str(shared_cfg["norm_mode"])  # type: ignore[index]
    stack_sar = bool(shared_cfg["stack_sar_3ch"])  # type: ignore[index]
    set_global_seed(seed)

    models: dict[str, object] = {}
    variant_names: dict[str, str] = {}
    exported_fp16: dict[str, dict[str, object]] = {}

    for vid in variant_ids:
        v_cfg = dict(variants_cfg[vid])
        v_cfg["pretrained"] = False
        v_cfg["stack_sar_3ch"] = stack_sar
        m = build_model(v_cfg).to(device)
        ckpt_path = runs_dir / vid / "best_model.pth"
        load_checkpoint(ckpt_path, m, device=device)
        m.eval()
        models[vid] = m
        variant_names[vid] = str(v_cfg["short_name"])

        if vid in ("lite_gated_fusion_v0_2", "dual_concat_v0_2"):
            dst_fp16 = args.exports_dir / f"{vid}_fp16.pth"
            mb = export_fp16_inference_checkpoint(ckpt_path, dst_fp16, v_cfg, norm_mode=norm_mode)
            exported_fp16[vid] = {"path": str(dst_fp16), "size_mb": mb}

    event_order = [
        "beirut-explosion",
        "congo-volcano",
        "libya-flood",
        "marshall-wildfire",
        "noto-earthquake",
    ]
    split_evals = {
        sp: {vid: DamageEvaluator(4, 255) for vid in variant_ids} for sp in ("val", "test")
    }
    event_evals = {
        ev: {vid: DamageEvaluator(4, 255) for vid in variant_ids} for ev in event_order
    }

    per_tile_rows: list[dict[str, object]] = []
    row_lookup: dict[str, dict[str, str]] = {}

    t0 = time.perf_counter()
    print(f"=== Running Milestone 7 Per-Tile & Cross-Hazard Audit ({len(heldout_rows)} held-out tiles) ===")

    with torch.no_grad():
        for r in heldout_rows:
            sid = r["sample_id"]
            sp = r["split"]
            ev = r["disaster_event"]
            row_lookup[sid] = r

            pre_u8 = _read_rgb_tif(data_dir / r["pre_event_path"])
            post_u8 = _read_single_band_tif(data_dir / r["post_event_path"])
            gt = _read_single_band_tif(data_dir / r["target_path"]).astype(np.int64)

            pre_chw, post_chw = normalize_multimodal(pre_u8, post_u8, norm_mode=norm_mode, stack_sar_3ch=stack_sar)
            pre_t = torch.from_numpy(pre_chw).unsqueeze(0).to(device)
            post_t = torch.from_numpy(post_chw).unsqueeze(0).to(device)

            tile_stats: dict[str, dict[str, float]] = {}
            for vid in variant_ids:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = models[vid](pre_t, post_t)  # type: ignore[operator]
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                split_evals[sp][vid].update(gt, pred)
                if ev in event_evals:
                    event_evals[ev][vid].update(gt, pred)
                tile_stats[vid] = compute_single_tile_stats(gt, pred)

            tot_px = max(1, gt.size)
            bldg_pct = round(100.0 * float((gt > 0).sum()) / tot_px, 3)
            dmg_pct = round(100.0 * float((gt == 2).sum()) / tot_px, 3)
            dest_pct = round(100.0 * float((gt == 3).sum()) / tot_px, 3)

            g_st = tile_stats["lite_gated_fusion_v0_2"]
            o_st = tile_stats["optical_only_v0_2"]
            c_st = tile_stats["dual_concat_v0_2"]
            s_st = tile_stats["sar_only_v0_2"]

            mode_tag = classify_tile_mode(gt, g_st)
            per_tile_rows.append(
                {
                    "sample_id": sid,
                    "split": sp,
                    "disaster_event": ev,
                    "gt_bldg_pct": bldg_pct,
                    "gt_damaged_pct": dmg_pct,
                    "gt_destroyed_pct": dest_pct,
                    "opt_fg_iou": o_st["active_fg_iou"],
                    "sar_fg_iou": s_st["active_fg_iou"],
                    "concat_fg_iou": c_st["active_fg_iou"],
                    "gated_fg_iou": g_st["active_fg_iou"],
                    "gated_loc_f1": g_st["loc_f1"],
                    "gated_destroyed_f1": g_st["destroyed_f1"],
                    "multimodal_gain_iou": round(g_st["active_fg_iou"] - o_st["active_fg_iou"], 5),
                    "dest_to_intact_px": int(g_st["dest_to_intact_px"]),
                    "dmg_to_intact_px": int(g_st["dmg_to_intact_px"]),
                    "loc_fp_px": int(g_st["loc_fp_px"]),
                    "loc_fn_px": int(g_st["loc_fn_px"]),
                    "mode_tag": mode_tag,
                }
            )

    val_summaries: dict[str, dict[str, object]] = {}
    test_summaries: dict[str, dict[str, object]] = {}
    for vid in variant_ids:
        vs = split_evals["val"][vid].compute_summary()
        vs["error_decomposition"] = decompose_confusion_errors(vs["confusion_matrix"])  # type: ignore[arg-type]
        val_summaries[vid] = vs

        ts = split_evals["test"][vid].compute_summary()
        ts["error_decomposition"] = decompose_confusion_errors(ts["confusion_matrix"])  # type: ignore[arg-type]
        test_summaries[vid] = ts

    per_event_results: dict[str, dict[str, dict[str, object]]] = {}
    for ev in event_order:
        per_event_results[ev] = {}
        for vid in variant_ids:
            es = event_evals[ev][vid].compute_summary()
            es["error_decomposition"] = decompose_confusion_errors(es["confusion_matrix"])  # type: ignore[arg-type]
            per_event_results[ev][vid] = es

    # Deterministically select 5 representative tiles for the visual failure taxonomy gallery
    def pick_best(predicate, key_fn):
        cands = [r for r in per_tile_rows if predicate(r)]
        if not cands:
            cands = per_tile_rows
        return max(cands, key=key_fn)

    used_ids: set[str] = set()
    gallery_specs = [
        (
            "1. Cross-Event Success\n(Wildfire Ash vs Roof)",
            lambda r: r["disaster_event"] == "marshall-wildfire" and r["gt_destroyed_pct"] > 1.0,
            lambda r: float(r["gated_destroyed_f1"]) + float(r["gated_loc_f1"]),
        ),
        (
            "2. Seismic Rubble\n(Destroyed -> Intact)",
            lambda r: r["disaster_event"] == "noto-earthquake" and r["sample_id"] not in used_ids,
            lambda r: int(r["dest_to_intact_px"]),
        ),
        (
            "3. Sub-Roof Damage\n(Damaged -> Intact)",
            lambda r: r["split"] in ("val", "test") and r["sample_id"] not in used_ids,
            lambda r: int(r["dmg_to_intact_px"]),
        ),
        (
            "4. Dense Urban / SAR\nLayover False Alarms",
            lambda r: r["gt_bldg_pct"] > 5.0 and r["sample_id"] not in used_ids,
            lambda r: int(r["loc_fp_px"]),
        ),
        (
            "5. Sparse / Volcanic\nMissed Footprints",
            lambda r: r["disaster_event"] in ("congo-volcano", "noto-earthquake") and r["gt_bldg_pct"] > 1.5 and r["sample_id"] not in used_ids,
            lambda r: int(r["loc_fn_px"]),
        ),
    ]

    selected_cases: list[tuple[str, dict[str, object], dict[str, str]]] = []
    for title, pred_fn, score_fn in gallery_specs:
        picked = pick_best(pred_fn, score_fn)
        sid = str(picked["sample_id"])
        used_ids.add(sid)
        selected_cases.append((title, picked, row_lookup[sid]))

    # Mode distribution counts across 144 held-out tiles
    mode_counts: dict[str, int] = {}
    for r in per_tile_rows:
        tag = str(r["mode_tag"])
        mode_counts[tag] = mode_counts.get(tag, 0) + 1

    elapsed = round(time.perf_counter() - t0, 2)
    metrics_dir = args.results_dir / "metrics"
    figures_dir = args.results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "per_tile_audit_v0_2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_tile_rows)

    save_cross_event_and_confusion_figure(
        event_order,
        per_event_results,
        test_summaries,
        variant_names,
        figures_dir / "cross_event_and_confusion_v0_2.png",
    )
    save_failure_gallery(
        data_dir,
        selected_cases,
        models,
        norm_mode,
        stack_sar,
        device,
        figures_dir / "failure_modes_gallery_v0_2.png",
    )

    summary_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": "failure_analysis_v0_2",
        "heldout_tiles_evaluated": len(heldout_rows),
        "duration_seconds": elapsed,
        "exported_fp16_checkpoints": exported_fp16,
        "tile_mode_distribution_lite_gated": mode_counts,
        "selected_gallery_tiles": [
            {"case": t.replace("\n", " "), "sample_id": str(ar["sample_id"]), "event": str(ar["disaster_event"])}
            for t, ar, _ in selected_cases
        ],
        "validation_40_summary": val_summaries,
        "unseen_test_104_summary": test_summaries,
        "per_event_heldout_summary": per_event_results,
    }
    (metrics_dir / "failure_analysis_v0_2_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== MILESTONE 7 ERROR DECOMPOSITION (UNSEEN TEST-104) ===")
    print(
        json.dumps(
            {
                "duration_sec": elapsed,
                "exported_fp16_checkpoints": exported_fp16,
                "tile_mode_counts": mode_counts,
                "lite_gated_test104_error_decomposition": test_summaries["lite_gated_fusion_v0_2"]["error_decomposition"],
                "dual_concat_test104_error_decomposition": test_summaries["dual_concat_v0_2"]["error_decomposition"],
                "optical_only_test104_error_decomposition": test_summaries["optical_only_v0_2"]["error_decomposition"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
