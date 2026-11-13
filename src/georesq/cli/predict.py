"""Standalone Multimodal Tile Inference & Human-in-the-Loop Triage CLI (Milestone 8).

Runs a GeoResQ dual-stream checkpoint (compact `fp16` ~51 MB export or full training
`best_model.pth`) on a paired Pre-Event Optical RGB + Post-Event SAR GeoTIFF tile and
writes:
  1. `predicted_damage_mask.tif` (uint8 class IDs: 0=Bg, 1=Intact, 2=Damaged, 3=Destroyed)
  2. `uncertainty_entropy.png` (normalized [0, 1] pixel-wise Shannon entropy map)
  3. `triage_panel.png` (side-by-side inputs, damage map, entropy & priority sector boxes)
  4. `triage_report.json` (strict RFC-8259 JSON summary with human-review disclaimers)

Responsible-AI Notice:
  Outputs are research-grade decision-support estimates ONLY and MUST NOT be used for
  automated emergency dispatch, insurance adjudication, or conflict targeting without
  qualified human imagery-analyst verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from georesq.data.dataset import read_rgb_tif, read_single_band_tif
from georesq.evaluation.metrics import CLASS_NAMES, DamageEvaluator, decompose_confusion_errors
from georesq.preprocessing.corruptions import VALID_CORRUPTIONS, corrupt_triplet
from georesq.preprocessing.transforms import normalize_multimodal
from georesq.training.loop import load_yaml_config

DAMAGE_PALETTE = np.array(
    [
        [24, 28, 36],     # 0: Background
        [46, 204, 113],   # 1: Intact
        [243, 156, 18],   # 2: Damaged
        [231, 76, 60],    # 3: Destroyed
    ],
    dtype=np.uint8,
)

DISCLAIMER_TEXT = (
    "RESEARCH DECISION-SUPPORT PROTOTYPE ONLY — HUMAN IMAGERY-ANALYST VERIFICATION REQUIRED. "
    "Do not use for autonomous emergency dispatch, insurance claims, or non-humanitarian targeting."
)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(4):
        out[mask == cls_id] = DAMAGE_PALETTE[cls_id]
    out[mask == 255] = np.array([120, 120, 120], dtype=np.uint8)
    return out


def compute_normalized_entropy(probs_chw: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon entropy H(p) / ln(K) in [0.0, 1.0] from [K, H, W] probabilities."""
    p = np.asarray(probs_chw, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 2:
        raise ValueError(f"Expected probs_chw shape (K>=2, H, W), got {p.shape}")
    k = p.shape[0]
    p_safe = np.clip(p, 1e-12, 1.0)
    p_safe = p_safe / np.sum(p_safe, axis=0, keepdims=True)
    raw_h = -np.sum(p_safe * np.log(p_safe), axis=0)
    norm_h = raw_h / float(np.log(k))
    return np.clip(norm_h, 0.0, 1.0).astype(np.float32)


def build_sector_triage_table(
    pred_mask: np.ndarray,
    entropy_map: np.ndarray,
    grid_n: int = 4,
    min_building_pct: float = 0.5,
) -> list[dict[str, object]]:
    """Partition tile into `grid_n x grid_n` sectors and rank human-review priority."""
    if pred_mask.shape != entropy_map.shape:
        raise ValueError(f"Shape mismatch: pred={pred_mask.shape} vs entropy={entropy_map.shape}")
    h, w = pred_mask.shape
    ys = np.linspace(0, h, grid_n + 1, dtype=int)
    xs = np.linspace(0, w, grid_n + 1, dtype=int)

    sectors: list[dict[str, object]] = []
    for r_i in range(grid_n):
        for c_i in range(grid_n):
            y0, y1 = int(ys[r_i]), int(ys[r_i + 1])
            x0, x1 = int(xs[c_i]), int(xs[c_i + 1])
            sub_p = pred_mask[y0:y1, x0:x1]
            sub_h = entropy_map[y0:y1, x0:x1]
            area = max(1, sub_p.size)

            bldg_mask = (sub_p > 0) & (sub_p < 4)
            bldg_px = int(bldg_mask.sum())
            bldg_pct = 100.0 * bldg_px / area
            severe_px = int(((sub_p == 2) | (sub_p == 3)).sum())
            dest_px = int((sub_p == 3).sum())

            severe_ratio_bldg = (severe_px / bldg_px) if bldg_px > 0 else 0.0
            mean_ent = float(sub_h[bldg_mask].mean()) if bldg_px > 0 else float(sub_h.mean())

            if bldg_pct < min_building_pct:
                priority = 0.10 * mean_ent
                reason = "sparse_background_low_priority"
            else:
                priority = 0.65 * severe_ratio_bldg + 0.35 * mean_ent
                if severe_ratio_bldg >= 0.20:
                    reason = "high_predicted_severe_damage"
                elif mean_ent >= 0.45:
                    reason = "elevated_predictive_uncertainty"
                else:
                    reason = "predicted_intact_routine_check"

            sectors.append(
                {
                    "sector_id": f"R{r_i + 1}C{c_i + 1}",
                    "bbox_yx": [y0, y1, x0, x1],
                    "building_coverage_pct": round(bldg_pct, 2),
                    "destroyed_pixels": dest_px,
                    "severe_damage_ratio_among_buildings": round(severe_ratio_bldg, 4),
                    "mean_entropy": round(mean_ent, 4),
                    "triage_priority_score": round(float(priority), 4),
                    "review_reason": reason,
                    "human_review_required": True,
                }
            )

    sectors.sort(key=lambda d: float(d["triage_priority_score"]), reverse=True)  # type: ignore[arg-type]
    return sectors


def save_mask_geotiff_or_png(
    pred_mask: np.ndarray,
    out_path: Path,
    reference_tif: Path | None = None,
) -> str:
    """Write uint8 prediction raster, preserving CRS/transform if rasterio & reference exist."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr_u8 = np.asarray(pred_mask, dtype=np.uint8)

    if reference_tif is not None and reference_tif.exists() and out_path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore

            with rasterio.open(reference_tif) as src:
                meta = src.meta.copy()
            meta.update({"count": 1, "dtype": "uint8", "compress": "deflate"})
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(arr_u8, 1)
            return "geotiff_georeferenced"
        except Exception:
            pass

    from PIL import Image

    Image.fromarray(arr_u8).save(out_path)
    return "pil_raster"


def render_triage_panel(
    pre_opt_u8: np.ndarray,
    post_sar_u8: np.ndarray,
    pred_mask: np.ndarray,
    entropy_map: np.ndarray,
    top_sectors: list[dict[str, object]],
    out_path: Path,
    tile_label: str = "sample_tile",
    gt_mask: np.ndarray | None = None,
) -> None:
    """Render side-by-side visual triage card with uncertainty & top-priority sector boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    ncols = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.1 * ncols, 4.7), constrained_layout=True)

    idx = 0
    axes[idx].imshow(pre_opt_u8)
    axes[idx].set_title("1. Pre-Event Optical", fontsize=10.5, fontweight="bold")
    idx += 1

    axes[idx].imshow(post_sar_u8, cmap="gray", vmin=0, vmax=255)
    axes[idx].set_title("2. Post-Event SAR", fontsize=10.5, fontweight="bold")
    idx += 1

    if gt_mask is not None:
        axes[idx].imshow(mask_to_rgb(gt_mask))
        axes[idx].set_title("3. Ground Truth", fontsize=10.5, fontweight="bold")
        idx += 1

    pred_num = idx + 1
    axes[idx].imshow(mask_to_rgb(pred_mask))
    axes[idx].set_title(f"{pred_num}. Predicted Damage", fontsize=10.5, fontweight="bold")
    idx += 1

    im_ent = axes[idx].imshow(entropy_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[idx].set_title(f"{idx + 1}. Entropy & Top Sectors", fontsize=10.5, fontweight="bold")
    for rank, sec in enumerate(top_sectors[:3], start=1):
        y0, y1, x0, x1 = [int(v) for v in sec["bbox_yx"]]  # type: ignore[index]
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.6,
            edgecolor="#38bdf8",
            facecolor="none",
            linestyle="--",
        )
        axes[idx].add_patch(rect)
        axes[idx].text(
            x0 + 4,
            y0 + 14,
            f"#{rank} {sec['sector_id']}",
            color="#fde047",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "#0f172a", "alpha": 0.72, "pad": 1.5, "edgecolor": "none"},
        )
    fig.colorbar(im_ent, ax=axes[idx], fraction=0.046, pad=0.04, label="Norm. Entropy [0,1]")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"GeoResQ Multimodal Triage Card — {tile_label}\n"
        "[Green=Intact | Orange=Damaged | Red=Destroyed] — HUMAN ANALYST REVIEW REQUIRED",
        fontsize=11.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def _select_device(requested: str):
    import torch

    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def _make_synthetic_demo_triplet(size: int = 256, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic synthetic urban triplet for fast end-to-end CLI smoke tests."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(55, 115, size=(size, size, 3), dtype=np.uint8)
    post = rng.integers(30, 80, size=(size, size), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)

    for r0 in range(24, size - 32, 52):
        for c0 in range(24, size - 32, 56):
            pre[r0 : r0 + 28, c0 : c0 + 32] = np.array([175, 168, 155], dtype=np.uint8)
            if (r0 + c0) % 3 == 0:
                # Destroyed footprint (low specular ash / altered SAR)
                post[r0 : r0 + 28, c0 : c0 + 32] = rng.integers(15, 45, size=(28, 32), dtype=np.uint8)
                gt[r0 : r0 + 28, c0 : c0 + 32] = 3
            else:
                # Intact footprint (strong double-bounce corner reflection)
                post[r0 : r0 + 28, c0 : c0 + 32] = rng.integers(155, 225, size=(28, 32), dtype=np.uint8)
                gt[r0 : r0 + 28, c0 : c0 + 32] = 1
    return pre, post, gt


def run_tile_inference(
    pre_opt_u8: np.ndarray,
    post_sar_u8: np.ndarray,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    benchmark_config_path: Path | None = None,
    variant_id: str = "lite_gated_fusion_v0_2",
    target_mask: np.ndarray | None = None,
    reference_tif: Path | None = None,
    tile_label: str = "tile",
    device_name: str = "auto",
    corruption: str = "clean",
    severity: float = 0.0,
    grid_n: int = 4,
) -> dict[str, object]:
    """Execute end-to-end multimodal inference, uncertainty estimation, and triage export."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from georesq.models import build_model
    from georesq.training.loop import load_checkpoint

    if corruption != "clean" and severity > 0.0:
        pre_opt_u8, post_sar_u8, corr_diag = corrupt_triplet(
            pre_opt_u8, post_sar_u8, corruption, severity, seed=42, sample_idx=0
        )
    else:
        corr_diag = {"opt_mae_u8": 0.0, "sar_mae_u8": 0.0, "sar_enl": None}

    device = _select_device(device_name)
    norm_mode = "modality_specific_v0_2"
    stack_sar = True
    v_cfg: dict[str, object] | None = None
    weights_source = "untrained_synthetic_smoke_test"

    if checkpoint_path is not None:
        raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw_ckpt, dict) and "variant_config" in raw_ckpt:
            v_cfg = dict(raw_ckpt["variant_config"])
            norm_mode = str(raw_ckpt.get("norm_mode", norm_mode))
        weights_source = str(checkpoint_path)

    if v_cfg is None:
        if benchmark_config_path is not None and benchmark_config_path.exists():
            b_cfg = load_yaml_config(benchmark_config_path)
            v_cfg = dict(b_cfg["variants"][variant_id])  # type: ignore[index]
            norm_mode = str(b_cfg["shared_data"]["norm_mode"])  # type: ignore[index]
            stack_sar = bool(b_cfg["shared_data"]["stack_sar_3ch"])  # type: ignore[index]
        else:
            v_cfg = {
                "architecture": "dual_stream_unet",
                "backbone": "resnet18",
                "fusion_mode": "lite_gated_fusion",
                "pretrained": False,
                "dropout_p": 0.1,
                "use_fullres_skip": True,
                "aux_loc_head": True,
            }

    v_cfg["pretrained"] = False
    v_cfg["stack_sar_3ch"] = stack_sar
    model = build_model(v_cfg).to(device)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    pre_chw, post_chw = normalize_multimodal(
        pre_opt_u8, post_sar_u8, norm_mode=norm_mode, stack_sar_3ch=stack_sar
    )
    h, w = pre_opt_u8.shape[:2]
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32

    pre_t = torch.from_numpy(pre_chw).unsqueeze(0).to(device)
    post_t = torch.from_numpy(post_chw).unsqueeze(0).to(device)
    if pad_h > 0 or pad_w > 0:
        pre_t = F.pad(pre_t, (0, pad_w, 0, pad_h), mode="reflect")
        post_t = F.pad(post_t, (0, pad_w, 0, pad_h), mode="reflect")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(pre_t, post_t)
        logits = logits[:, :, :h, :w].float()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_ms = round(1000.0 * (time.perf_counter() - t0), 2)

    pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
    max_conf = np.max(probs, axis=0)
    entropy_map = compute_normalized_entropy(probs)
    sectors = build_sector_triage_table(pred_mask, entropy_map, grid_n=grid_n)

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "predicted_damage_mask.tif"
    ent_path = output_dir / "uncertainty_entropy.png"
    panel_path = output_dir / "triage_panel.png"
    report_path = output_dir / "triage_report.json"

    writer_backend = save_mask_geotiff_or_png(pred_mask, mask_path, reference_tif=reference_tif)
    Image.fromarray(np.clip(np.rint(entropy_map * 255.0), 0, 255).astype(np.uint8)).save(ent_path)
    render_triage_panel(
        pre_opt_u8=pre_opt_u8,
        post_sar_u8=post_sar_u8,
        pred_mask=pred_mask,
        entropy_map=entropy_map,
        top_sectors=sectors,
        out_path=panel_path,
        tile_label=tile_label,
        gt_mask=target_mask,
    )

    tot_px = max(1, pred_mask.size)
    class_breakdown = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cnt = int((pred_mask == cls_idx).sum())
        class_breakdown[cls_name] = {
            "pixels": cnt,
            "tile_fraction_pct": round(100.0 * cnt / tot_px, 3),
        }

    gt_eval = None
    if target_mask is not None:
        ev = DamageEvaluator(num_classes=4, ignore_index=255)
        ev.update(target_mask.astype(np.int64), pred_mask.astype(np.int64))
        gt_eval = ev.compute_summary()
        gt_eval["error_decomposition"] = decompose_confusion_errors(gt_eval["confusion_matrix"])  # type: ignore[arg-type]

    bldg_mask = pred_mask > 0
    report: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "responsible_ai_disclaimer": DISCLAIMER_TEXT,
        "human_review_required": True,
        "tile_label": tile_label,
        "spatial_shape_hw": [int(h), int(w)],
        "model": {
            "variant_id": variant_id,
            "fusion_mode": str(v_cfg.get("fusion_mode", "unknown")),
            "norm_mode": norm_mode,
            "weights_source": weights_source,
            "trained_checkpoint_loaded": checkpoint_path is not None,
            "device": str(device),
            "inference_latency_ms": infer_ms,
        },
        "input_condition": {
            "corruption": corruption,
            "severity": float(severity),
            "diagnostics": corr_diag,
        },
        "prediction_summary": {
            "class_breakdown": class_breakdown,
            "predicted_building_coverage_pct": round(100.0 * float(bldg_mask.sum()) / tot_px, 3),
            "mean_max_softmax_confidence": round(float(max_conf.mean()), 4),
            "mean_normalized_entropy_all": round(float(entropy_map.mean()), 4),
            "mean_normalized_entropy_buildings": (
                round(float(entropy_map[bldg_mask].mean()), 4) if bldg_mask.any() else 0.0
            ),
        },
        "top_priority_sectors": sectors[:5],
        "ground_truth_evaluation": gt_eval,
        "artifacts": {
            "predicted_damage_mask": str(mask_path),
            "mask_writer_backend": writer_backend,
            "uncertainty_entropy_png": str(ent_path),
            "triage_panel_png": str(panel_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    repo_default_cfg = Path(__file__).resolve().parents[3] / "configs/benchmark_v0_2.yaml"
    parser = argparse.ArgumentParser(
        description="GeoResQ Standalone Multimodal Satellite Damage Triage CLI."
    )
    parser.add_argument("--pre-optical", type=Path, default=None, help="Path to pre-event RGB GeoTIFF.")
    parser.add_argument("--post-sar", type=Path, default=None, help="Path to post-event SAR GeoTIFF.")
    parser.add_argument("--target", type=Path, default=None, help="Optional ground-truth mask GeoTIFF.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to trained .pth checkpoint.")
    parser.add_argument("--config", type=Path, default=repo_default_cfg, help="Benchmark YAML config.")
    parser.add_argument("--variant", type=str, default="lite_gated_fusion_v0_2", help="Model variant ID.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/triage_demo"), help="Output folder.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--corruption", type=str, default="clean", choices=list(VALID_CORRUPTIONS))
    parser.add_argument("--severity", type=float, default=0.0, help="Optional stress severity in [0, 1].")
    parser.add_argument("--grid-size", type=int, default=4, help="Sector grid dimension N for NxN triage.")
    parser.add_argument(
        "--demo-synthetic",
        action="store_true",
        help="Run a self-contained synthetic 256x256 smoke-test without external GeoTIFFs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo_synthetic:
        pre_u8, post_u8, gt_u8 = _make_synthetic_demo_triplet(size=256, seed=42)
        report = run_tile_inference(
            pre_opt_u8=pre_u8,
            post_sar_u8=post_u8,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            benchmark_config_path=args.config if args.config.exists() else None,
            variant_id=args.variant,
            target_mask=gt_u8,
            reference_tif=None,
            tile_label="synthetic_smoke_test_256",
            device_name=args.device,
            corruption=args.corruption,
            severity=args.severity,
            grid_n=args.grid_size,
        )
    else:
        if args.pre_optical is None or args.post_sar is None:
            parser.error("Provide both `--pre-optical` and `--post-sar`, or pass `--demo-synthetic`.")
        if args.checkpoint is None:
            parser.error("Provide `--checkpoint path/to/model.pth` for real tile inference.")

        pre_u8 = read_rgb_tif(args.pre_optical)
        post_u8 = read_single_band_tif(args.post_sar)
        gt_u8 = read_single_band_tif(args.target) if args.target is not None else None
        report = run_tile_inference(
            pre_opt_u8=pre_u8,
            post_sar_u8=post_u8,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            benchmark_config_path=args.config if args.config.exists() else None,
            variant_id=args.variant,
            target_mask=gt_u8,
            reference_tif=args.pre_optical,
            tile_label=args.pre_optical.stem.replace("_pre_disaster", ""),
            device_name=args.device,
            corruption=args.corruption,
            severity=args.severity,
            grid_n=args.grid_size,
        )

    print(
        json.dumps(
            {
                "status": "OK",
                "tile_label": report["tile_label"],
                "model": report["model"],
                "prediction_summary": report["prediction_summary"],
                "top_sector": report["top_priority_sectors"][0] if report["top_priority_sectors"] else None,  # type: ignore[index]
                "artifacts": report["artifacts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
