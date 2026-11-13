#!/usr/bin/env python3
"""Milestone 6 — All-Weather & Modality Degradation Stress-Test Runner (BRIGHT v0.2).

Evaluates the 4 frozen Milestone 5 checkpoints (`optical_only_v0_2`, `sar_only_v0_2`,
`dual_concat_v0_2`, `lite_gated_fusion_v0_2`) WITHOUT retraining across 5 realistic
operational sensor degradations on the 104-tile unseen test split (`noto` + `marshall`):
  1. Pre-Optical Cloud Occlusion (`s = 0.0 -> 1.0`)
  2. Wildfire Smoke & Atmospheric Haze (`s = 0.0 -> 1.0`)
  3. Archive Optical Swath / Tile Dropout (`s = 0.0 -> 1.0`)
  4. Post-SAR Multiplicative Nakagami Speckle (`ENL = inf, 8, 4, 2, 1`)
  5. Compound Storm (`Cloud + Smoke + SAR Speckle`)

Also inspects `LitePriorGatedFusionBlock` learned LayerScale `|gamma|` magnitudes and
spatial `opt_prior_gate` shifts under corruption, saving all JSON/CSV/PNG artifacts to
`results/metrics/` and `results/figures/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DAMAGE_COLORS = np.array(
    [
        [24, 28, 36],     # 0: Background
        [46, 204, 113],   # 1: Intact
        [243, 156, 18],   # 2: Damaged
        [231, 76, 60],    # 3: Destroyed
    ],
    dtype=np.uint8,
)

MODEL_STYLE = {
    "optical_only_v0_2": {"color": "#3b82f6", "marker": "o", "ls": "--"},
    "sar_only_v0_2": {"color": "#f59e0b", "marker": "s", "ls": ":"},
    "dual_concat_v0_2": {"color": "#10b981", "marker": "^", "ls": "-."},
    "lite_gated_fusion_v0_2": {"color": "#ef4444", "marker": "D", "ls": "-"},
}


def resolve_subset_v0_2(explicit_dir: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(explicit_dir)
    candidates.extend(
        [
            Path("/kaggle/working/data/bright_subset_v0_2"),
            Path("/kaggle/input/bright-subset-v0-2"),
            Path("/kaggle/input/bright-v0-2/bright_subset_v0_2"),
            Path("/content/data/bright_subset_v0_2"),
            REPO_ROOT / "data/bright_subset_v0_2",
        ]
    )
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for p in sorted(kaggle_in.rglob("manifest_v0_2.csv")):
            if (p.parent / "pre-event").exists():
                candidates.append(p.parent)
    for c in candidates:
        if (c / "pre-event").exists() and (c / "post-event").exists() and (c / "target").exists():
            return c
    raise FileNotFoundError(
        "Could not locate `bright_subset_v0_2`. Pass `--data-dir /path/to/bright_subset_v0_2`."
    )


def resolve_runs_dir(explicit_dir: Path | None, required_variants: list[str]) -> Path:
    candidates: list[Path] = []
    if explicit_dir is not None:
        candidates.append(explicit_dir)
    candidates.extend(
        [
            Path("/kaggle/working/runs/benchmark_v0_2"),
            Path("/content/runs/benchmark_v0_2"),
            REPO_ROOT / "runs/benchmark_v0_2",
        ]
    )
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for p in sorted(kaggle_in.rglob("best_model.pth")):
            if p.parent.name in required_variants:
                candidates.append(p.parent.parent)
    for c in candidates:
        if all((c / vid / "best_model.pth").exists() for vid in required_variants):
            return c
    raise FileNotFoundError(
        "Could not locate trained v0.2 checkpoints (`best_model.pth` for all 4 variants). "
        "Pass `--runs-dir /kaggle/working/runs/benchmark_v0_2`."
    )


def sync_m5_artifacts_if_present(runs_dir: Path, results_dir: Path) -> list[str]:
    """Copy Milestone 5 summary/table/figures from runs_dir into results/ if present."""
    copied: list[str] = []
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "benchmark_v0_2_summary.json": metrics_dir / "benchmark_v0_2_summary.json",
        "benchmark_v0_2_table.csv": metrics_dir / "benchmark_v0_2_table.csv",
        "benchmark_v0_2_comparison.png": figures_dir / "benchmark_v0_2_comparison.png",
        "benchmark_v0_2_predictions.png": figures_dir / "benchmark_v0_2_predictions.png",
    }
    for fname, dst in mapping.items():
        src = runs_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(results_dir)))
    return copied


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(4):
        out[mask == cls_id] = DAMAGE_COLORS[cls_id]
    out[mask == 255] = np.array([120, 120, 120], dtype=np.uint8)
    return out


def save_stress_curves_figure(
    corruptions_cfg: dict[str, dict[str, object]],
    severities: list[float],
    results_grid: dict[str, dict[str, list[dict[str, object]]]],
    audc_summary: dict[str, dict[str, float]],
    variant_names: dict[str, str],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), constrained_layout=True)
    flat_axes = axes.ravel()

    corr_keys = list(corruptions_cfg.keys())
    for idx, c_key in enumerate(corr_keys[:5]):
        ax = flat_axes[idx]
        c_meta = corruptions_cfg[c_key]
        for vid, short_name in variant_names.items():
            st = MODEL_STYLE.get(vid, {"color": "#333333", "marker": "o", "ls": "-"})
            pts = results_grid[c_key][vid]
            ys = [float(p["mIoU_foreground3"]) * 100.0 for p in pts]
            ax.plot(
                severities,
                ys,
                label=short_name,
                color=st["color"],
                marker=st["marker"],
                linestyle=st["ls"],
                linewidth=2.2,
                markersize=6,
            )
        ax.set_title(str(c_meta["title"]), fontsize=11, fontweight="bold")
        ax.set_xlabel(str(c_meta["x_label"]), fontsize=9.5)
        ax.set_ylabel("Unseen Test-104 Damage mIoU (3-cls %)", fontsize=9.5)
        ax.set_xticks(severities)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8.5, loc="best")

    # Panel 6: Mean AUDC across all 5 corruption families (Damage mIoU % & Building Loc-F1 %)
    ax6 = flat_axes[5]
    vids = list(variant_names.keys())
    x = np.arange(len(vids))
    width = 0.36
    mean_miou_audc = [audc_summary[v]["mean_audc_mIoU_fg3_pct"] for v in vids]
    mean_loc_audc = [audc_summary[v]["mean_audc_loc_f1_pct"] for v in vids]

    b1 = ax6.bar(x - width / 2, mean_miou_audc, width, label="Mean AUDC mIoU_fg3 (%)", color="#2563eb")
    b2 = ax6.bar(x + width / 2, mean_loc_audc, width, label="Mean AUDC Loc-F1 (%)", color="#dc2626")
    ax6.set_xticks(x)
    ax6.set_xticklabels([variant_names[v].replace(" (v0.2)", "") for v in vids], rotation=14, ha="right", fontsize=8.8)
    ax6.set_ylabel("Area Under Degradation Curve (%)")
    ax6.set_title("Overall All-Weather Robustness (5-Corruption Mean AUDC)", fontsize=11, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)
    ax6.legend(fontsize=8.5, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax6.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    fig.suptitle(
        "GeoResQ Milestone 6 — All-Weather & Modality Degradation Stress Test (104 Unseen Test Tiles: Noto + Marshall)",
        fontsize=13.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_gate_panel(
    data_dir: Path,
    test_rows: list[dict[str, str]],
    models: dict[str, object],
    norm_mode: str,
    seed: int,
    device,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    from georesq.data.dataset import _read_rgb_tif, _read_single_band_tif
    from georesq.preprocessing.corruptions import corrupt_triplet
    from georesq.preprocessing.transforms import normalize_multimodal

    # Pick a deterministic building-and-damage-rich unseen test tile
    best_idx = 0
    best_score = -1
    for i, r in enumerate(test_rows):
        info = json.loads(r["label_information"])
        sc = int(info.get("destroyed", 0)) * 3 + int(info.get("intact", 0))
        if r["disaster_event"] == "marshall-wildfire" and sc > best_score:
            best_score = sc
            best_idx = i

    row = test_rows[best_idx]
    pre_u8 = _read_rgb_tif(data_dir / row["pre_event_path"])
    post_u8 = _read_single_band_tif(data_dir / row["post_event_path"])
    gt_u8 = _read_single_band_tif(data_dir / row["target_path"])

    scenarios = [
        ("Clean Archive + Clean SAR", "clean", 0.0),
        ("Pre-Optical Cloud (s=0.75)", "cloud_occlusion", 0.75),
        ("Smoke & Haze (s=0.75)", "smoke_haze", 0.75),
        ("Single-Look SAR Speckle (L=1)", "sar_speckle", 1.0),
    ]

    fig, axes = plt.subplots(len(scenarios), 6, figsize=(18.5, 12.2), constrained_layout=True)
    col_titles = [
        "Pre-Event Optical",
        "Post-Event SAR",
        "Ground Truth",
        "Optical-Only (v0.2)",
        "Dual Concat (v0.2)",
        "Lite Gated + Prior Gate",
    ]

    for r_i, (sc_label, c_name, sev) in enumerate(scenarios):
        pre_c, post_c, _ = corrupt_triplet(pre_u8, post_u8, c_name, sev, seed=seed, sample_idx=best_idx)
        pre_chw, post_chw = normalize_multimodal(pre_c, post_c, norm_mode=norm_mode, stack_sar_3ch=True)
        pre_t = torch.from_numpy(pre_chw).unsqueeze(0).to(device)
        post_t = torch.from_numpy(post_chw).unsqueeze(0).to(device)

        with torch.no_grad():
            p_opt = int(0)  # placeholder
            pred_opt = torch.argmax(models["optical_only_v0_2"](pre_t, post_t), dim=1).squeeze(0).cpu().numpy()  # type: ignore[operator]
            pred_cat = torch.argmax(models["dual_concat_v0_2"](pre_t, post_t), dim=1).squeeze(0).cpu().numpy()  # type: ignore[operator]
            gated_logits, gate_maps, _ = models["lite_gated_fusion_v0_2"].extract_fusion_diagnostics(pre_t, post_t)  # type: ignore[attr-defined]
            pred_gat = torch.argmax(gated_logits, dim=1).squeeze(0).cpu().numpy()
            # Upsample stage-1 (1/4) spatial prior gate to full resolution for visualization
            g_map = F.interpolate(
                gate_maps[1], size=gt_u8.shape, mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            _ = p_opt

        axes[r_i, 0].imshow(pre_c)
        axes[r_i, 0].set_ylabel(sc_label, fontsize=9.5, fontweight="bold")
        axes[r_i, 1].imshow(post_c, cmap="gray", vmin=0, vmax=255)
        axes[r_i, 2].imshow(mask_to_rgb(gt_u8))
        axes[r_i, 3].imshow(mask_to_rgb(pred_opt))
        axes[r_i, 4].imshow(mask_to_rgb(pred_cat))

        # Overlay Lite Gated prediction with subtle inset / alpha blend of prior gate contours
        axes[r_i, 5].imshow(mask_to_rgb(pred_gat))
        axes[r_i, 5].contour(g_map, levels=[0.45, 0.60], colors=["#38bdf8", "#fde047"], linewidths=0.7, alpha=0.8)
        axes[r_i, 5].set_xlabel(f"mean gate={float(g_map.mean()):.3f}", fontsize=8)

        for c_i in range(6):
            if r_i == 0:
                axes[r_i, c_i].set_title(col_titles[c_i], fontsize=10, fontweight="bold")
            axes[r_i, c_i].set_xticks([])
            axes[r_i, c_i].set_yticks([])

    fig.suptitle(
        f"Qualitative Degradation & Spatial Prior-Gate Response — {row['sample_id']} ({row['disaster_event']})\n"
        "[Green=Intact | Orange=Damaged | Red=Destroyed | Cyan/Yellow Contours = LitePriorGate Activation]",
        fontsize=12.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Milestone 6 All-Weather Stress-Test Suite.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stress_test_v0_2.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from georesq.evaluation.metrics import DamageEvaluator
    from georesq.preprocessing.corruptions import corrupt_triplet, severity_to_sar_enl
    from georesq.training.loop import load_yaml_config, set_global_seed

    cfg = load_yaml_config(args.config)
    exp_cfg = cfg["experiment"]  # type: ignore[index]
    bench_cfg = load_yaml_config(REPO_ROOT / str(exp_cfg["benchmark_config"]))  # type: ignore[index]
    severities = [float(x) for x in cfg["severities"]]  # type: ignore[index]
    corruptions_cfg: dict[str, dict[str, object]] = cfg["corruptions"]  # type: ignore[assignment,index]
    variants_cfg: dict[str, dict[str, object]] = bench_cfg["variants"]  # type: ignore[assignment,index]
    shared_cfg: dict[str, object] = bench_cfg["shared_data"]  # type: ignore[assignment,index]
    seed = int(exp_cfg["seed"])  # type: ignore[index]

    if args.dry_run:
        dummy_opt = np.full((128, 128, 3), 110, dtype=np.uint8)
        dummy_sar = np.full((128, 128), 90, dtype=np.uint8)
        checks = {}
        for c_key in corruptions_cfg:
            _, _, diag = corrupt_triplet(dummy_opt, dummy_sar, c_key, 0.75, seed=seed, sample_idx=0)
            checks[c_key] = diag
        print(
            json.dumps(
                {
                    "dry_run": "PASSED",
                    "severities": severities,
                    "variants": list(variants_cfg.keys()),
                    "sample_corruption_diagnostics_s075": checks,
                },
                indent=2,
            )
        )
        return 0

    import torch
    from torch.utils.data import DataLoader, Dataset

    from georesq.data.dataset import _read_rgb_tif, _read_single_band_tif, verify_zero_event_leakage
    from georesq.models import build_model
    from georesq.preprocessing.transforms import normalize_multimodal
    from georesq.training.loop import load_checkpoint

    variant_ids = list(variants_cfg.keys())
    data_dir = resolve_subset_v0_2(args.data_dir)
    runs_dir = resolve_runs_dir(args.runs_dir, variant_ids)
    copied_m5 = sync_m5_artifacts_if_present(runs_dir, args.results_dir)
    if copied_m5:
        print(f"[SYNC] Copied Milestone 5 artifacts to results/: {copied_m5}")

    manifest_path = (
        (data_dir / "manifest_v0_2.csv")
        if (data_dir / "manifest_v0_2.csv").exists()
        else (REPO_ROOT / str(exp_cfg["manifest_relpath"]))  # type: ignore[index]
    )
    verify_zero_event_leakage(manifest_path)

    split_name = str(exp_cfg.get("split", "test"))  # type: ignore[attr-defined]
    with manifest_path.open("r", encoding="utf-8") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == split_name]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = bool(exp_cfg.get("use_amp", True)) and (device.type == "cuda")  # type: ignore[attr-defined]
    norm_mode = str(shared_cfg["norm_mode"])
    stack_sar = bool(shared_cfg["stack_sar_3ch"])

    set_global_seed(seed)

    # Load all 4 frozen Milestone 5 models onto GPU simultaneously (~420 MB total VRAM)
    models: dict[str, object] = {}
    variant_names: dict[str, str] = {}
    for vid in variant_ids:
        v_cfg = dict(variants_cfg[vid])
        v_cfg["pretrained"] = False
        v_cfg["stack_sar_3ch"] = stack_sar
        m = build_model(v_cfg).to(device)
        load_checkpoint(runs_dir / vid / "best_model.pth", m, device=device)
        m.eval()
        models[vid] = m
        variant_names[vid] = str(v_cfg["short_name"])

    # Extract learned |gamma| LayerScale profile from Lite Gated Fusion
    learned_gammas: dict[str, float] = {}
    gated_model = models.get("lite_gated_fusion_v0_2")
    if gated_model is not None and getattr(gated_model, "fusion_blocks", None) is not None:
        stage_labels = ["stage0_1_2", "stage1_1_4", "stage2_1_8", "stage3_1_16", "stage4_1_32"]
        for s_lbl, blk in zip(stage_labels, gated_model.fusion_blocks):  # type: ignore[attr-defined]
            if hasattr(blk, "gamma"):
                learned_gammas[s_lbl] = round(float(blk.gamma.detach().abs().mean().item()), 5)
                learned_gammas[f"{s_lbl}_signed_mean"] = round(float(blk.gamma.detach().mean().item()), 5)

    class CorruptedEvalDataset(Dataset):
        def __init__(self, rows: list[dict[str, str]], corruption: str, severity: float) -> None:
            self.rows = rows
            self.corruption = corruption
            self.severity = severity

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, object]:
            r = self.rows[idx]
            pre_u8 = _read_rgb_tif(data_dir / r["pre_event_path"])
            post_u8 = _read_single_band_tif(data_dir / r["post_event_path"])
            mask = _read_single_band_tif(data_dir / r["target_path"]).astype(np.int64)
            pre_c, post_c, diag = corrupt_triplet(
                pre_u8, post_u8, self.corruption, self.severity, seed=seed, sample_idx=idx
            )
            pre_chw, post_chw = normalize_multimodal(
                pre_c, post_c, norm_mode=norm_mode, stack_sar_3ch=stack_sar
            )
            return {
                "pre_opt": torch.from_numpy(pre_chw),
                "post_sar": torch.from_numpy(post_chw),
                "clf_mask": torch.from_numpy(mask),
                "opt_mae_u8": float(diag["opt_mae_u8"]),
                "sar_mae_u8": float(diag["sar_mae_u8"]),
            }

    batch_size = int(exp_cfg.get("eval_batch_size", 4))  # type: ignore[attr-defined]
    num_workers = int(exp_cfg.get("num_workers", 2)) if device.type == "cuda" else 0  # type: ignore[attr-defined]

    def run_pass(
        corruption: str,
        severity: float,
        active_vids: list[str],
    ) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
        ds = CorruptedEvalDataset(test_rows, corruption, severity)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        evaluators = {vid: DamageEvaluator(num_classes=4, ignore_index=255) for vid in active_vids}
        opt_maes: list[float] = []
        sar_maes: list[float] = []
        gate_bldg_vals: list[float] = []
        gate_bg_vals: list[float] = []

        with torch.no_grad():
            for batch in loader:
                pre = batch["pre_opt"].to(device, non_blocking=True)
                post = batch["post_sar"].to(device, non_blocking=True)
                gt_np = batch["clf_mask"].numpy()
                opt_maes.extend([float(x) for x in batch["opt_mae_u8"]])
                sar_maes.extend([float(x) for x in batch["sar_mae_u8"]])

                for vid in active_vids:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        if vid == "lite_gated_fusion_v0_2":
                            logits, gates, _ = models[vid].extract_fusion_diagnostics(pre, post)  # type: ignore[attr-defined]
                            # Inspect stage-1 (1/4 resolution) spatial prior gate on building vs background
                            g1 = torch.nn.functional.interpolate(
                                gates[1].float(), size=gt_np.shape[-2:], mode="bilinear", align_corners=False
                            ).squeeze(1).cpu().numpy()
                            b_mask = (gt_np > 0) & (gt_np != 255)
                            bg_mask = gt_np == 0
                            if b_mask.any():
                                gate_bldg_vals.append(float(g1[b_mask].mean()))
                            if bg_mask.any():
                                gate_bg_vals.append(float(g1[bg_mask].mean()))
                        else:
                            logits = models[vid](pre, post)  # type: ignore[operator]
                    preds_np = torch.argmax(logits, dim=1).cpu().numpy()
                    evaluators[vid].update(gt_np, preds_np)

        summaries = {vid: evaluators[vid].compute_summary() for vid in active_vids}
        pass_diag = {
            "mean_opt_mae_u8": round(float(np.mean(opt_maes)), 3),
            "mean_sar_mae_u8": round(float(np.mean(sar_maes)), 3),
            "sar_enl": (
                round(severity_to_sar_enl(severity), 3)
                if corruption == "sar_speckle" and severity > 0
                else -1.0
            ),
            "lite_gate_stage1_building_mean": round(float(np.mean(gate_bldg_vals)), 5) if gate_bldg_vals else 0.0,
            "lite_gate_stage1_bg_mean": round(float(np.mean(gate_bg_vals)), 5) if gate_bg_vals else 0.0,
        }
        return summaries, pass_diag

    t0 = time.perf_counter()
    print(f"=== Running Milestone 6 Clean Baseline Pass ({len(test_rows)} unseen test tiles) ===")
    clean_summaries, clean_diag = run_pass("clean", 0.0, variant_ids)

    results_grid: dict[str, dict[str, list[dict[str, object]]]] = {
        c_key: {vid: [] for vid in variant_ids} for c_key in corruptions_cfg
    }
    table_rows: list[dict[str, object]] = []

    for c_key, c_meta in corruptions_cfg.items():
        aff = str(c_meta["affected_modality"])
        for sev in severities:
            if sev <= 0.0:
                cur_sum = clean_summaries
                cur_diag = clean_diag
            else:
                active = [
                    vid
                    for vid in variant_ids
                    if not (
                        (aff == "optical" and vid == "sar_only_v0_2")
                        or (aff == "sar" and vid == "optical_only_v0_2")
                    )
                ]
                print(f"[STRESS] {c_key:18s} | severity={sev:.2f} | active_models={len(active)}")
                sub_sum, cur_diag = run_pass(c_key, sev, active)
                cur_sum = dict(sub_sum)
                for vid in variant_ids:
                    if vid not in cur_sum:
                        cur_sum[vid] = clean_summaries[vid]

            for vid in variant_ids:
                s_v = cur_sum[vid]
                base_v = clean_summaries[vid]
                clean_miou = max(1e-7, float(base_v["mIoU_foreground3"]))
                clean_loc = max(1e-7, float(base_v["bright_loc_f1"]))
                miou_fg3 = float(s_v["mIoU_foreground3"])
                mf1_fg3 = float(s_v["macro_f1_foreground3"])
                loc_f1 = float(s_v["bright_loc_f1"])
                dest_f1 = float(s_v["per_class"]["destroyed"]["f1"])  # type: ignore[index]

                entry = {
                    "corruption": c_key,
                    "severity": sev,
                    "variant_id": vid,
                    "short_name": variant_names[vid],
                    "mIoU_foreground3": round(miou_fg3, 6),
                    "macro_f1_foreground3": round(mf1_fg3, 6),
                    "bright_loc_f1": round(loc_f1, 6),
                    "destroyed_f1": round(dest_f1, 6),
                    "retention_mIoU_fg3_pct": round(100.0 * miou_fg3 / clean_miou, 2),
                    "retention_loc_f1_pct": round(100.0 * loc_f1 / clean_loc, 2),
                    "mean_opt_mae_u8": cur_diag["mean_opt_mae_u8"],
                    "mean_sar_mae_u8": cur_diag["mean_sar_mae_u8"],
                    "lite_gate_stage1_bldg": (
                        cur_diag["lite_gate_stage1_building_mean"]
                        if vid == "lite_gated_fusion_v0_2"
                        else None
                    ),
                    "lite_gate_stage1_bg": (
                        cur_diag["lite_gate_stage1_bg_mean"]
                        if vid == "lite_gated_fusion_v0_2"
                        else None
                    ),
                }
                results_grid[c_key][vid].append(entry)
                table_rows.append(entry)

    # Compute trapezoidal AUDC across [0, 1] per corruption and overall mean AUDC
    audc_summary: dict[str, dict[str, float]] = {vid: {} for vid in variant_ids}
    sev_arr = np.array(severities, dtype=np.float64)
    span = max(1e-6, float(sev_arr[-1] - sev_arr[0]))
    for vid in variant_ids:
        miou_audcs = []
        loc_audcs = []
        for c_key in corruptions_cfg:
            pts = results_grid[c_key][vid]
            y_m = np.array([float(p["mIoU_foreground3"]) * 100.0 for p in pts], dtype=np.float64)
            y_l = np.array([float(p["bright_loc_f1"]) * 100.0 for p in pts], dtype=np.float64)
            am = float(np.trapezoid(y_m, sev_arr) / span) if hasattr(np, "trapezoid") else float(np.trapz(y_m, sev_arr) / span)
            al = float(np.trapezoid(y_l, sev_arr) / span) if hasattr(np, "trapezoid") else float(np.trapz(y_l, sev_arr) / span)
            audc_summary[vid][f"{c_key}_audc_mIoU_fg3_pct"] = round(am, 3)
            audc_summary[vid][f"{c_key}_audc_loc_f1_pct"] = round(al, 3)
            miou_audcs.append(am)
            loc_audcs.append(al)
        audc_summary[vid]["mean_audc_mIoU_fg3_pct"] = round(float(np.mean(miou_audcs)), 3)
        audc_summary[vid]["mean_audc_loc_f1_pct"] = round(float(np.mean(loc_audcs)), 3)

    elapsed = round(time.perf_counter() - t0, 2)
    metrics_dir = args.results_dir / "metrics"
    figures_dir = args.results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "stress_test_v0_2_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    save_stress_curves_figure(
        corruptions_cfg,
        severities,
        results_grid,
        audc_summary,
        variant_names,
        figures_dir / "stress_test_v0_2_curves.png",
    )
    save_qualitative_gate_panel(
        data_dir,
        test_rows,
        models,
        norm_mode,
        seed,
        device,
        figures_dir / "stress_test_v0_2_corruptions_and_gates.png",
    )

    summary_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": "stress_test_v0_2",
        "split": split_name,
        "num_tiles": len(test_rows),
        "seed": seed,
        "duration_seconds": elapsed,
        "learned_lite_gated_layerscale_gammas": learned_gammas,
        "audc_summary": audc_summary,
        "results_by_corruption": results_grid,
    }
    (metrics_dir / "stress_test_v0_2_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== MILESTONE 6 ALL-WEATHER AUDC SUMMARY ===")
    print(
        json.dumps(
            {
                "duration_sec": elapsed,
                "learned_gammas": learned_gammas,
                "audc_summary": audc_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
