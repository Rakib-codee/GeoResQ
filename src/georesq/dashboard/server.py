"""Zero-bloat HTTP + JSON API server for the GeoResQ Interactive Visual Dashboard.

Supports:
- Real PyTorch `DualStreamDamageNet` inference on CUDA, Apple Silicon MPS, or CPU
  with automatic discovery & caching of `.pth` checkpoints in `exports/` and `runs/`.
- Graceful pure-NumPy physics-guided preview fallback when running in a lightweight
  container without PyTorch installed, so UI & corruption stress-testing work everywhere.
- Live discovery of real BRIGHT `.tif` triplets alongside built-in procedural disaster
  test scenes and custom browser `.tif`/`.png`/`.jpg` uploads.
- Serving verified Milestone 5, 6, and 7 JSON/CSV metrics and PNG figures.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from georesq import __version__
from georesq.cli.predict import (
    DISCLAIMER_TEXT,
    build_sector_triage_table,
    compute_normalized_entropy,
    mask_to_rgb,
)
from georesq.data.dataset import read_rgb_tif, read_single_band_tif
from georesq.evaluation.metrics import CLASS_NAMES, DamageEvaluator, decompose_confusion_errors
from georesq.preprocessing.corruptions import VALID_CORRUPTIONS, corrupt_triplet
from georesq.preprocessing.transforms import normalize_multimodal
from georesq.training.loop import load_yaml_config

try:
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _array_to_png_data_uri(arr: np.ndarray, mode: str | None = None) -> str:
    img = Image.fromarray(arr, mode=mode) if mode else Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _heatmap_to_rgb(norm_01: np.ndarray, palette: str = "magma") -> np.ndarray:
    """Fast pure-NumPy colormap (magma-like or viridis-like) mapping [0, 1] -> uint8 RGB."""
    x = np.clip(norm_01, 0.0, 1.0).astype(np.float32)
    if palette == "cyan_gold":
        # Gate activation palette: dark navy -> cyan -> warm yellow
        stops = np.array([0.0, 0.35, 0.70, 1.0], dtype=np.float32)
        cols = np.array(
            [
                [15, 23, 42],
                [14, 116, 144],
                [56, 189, 248],
                [250, 204, 21],
            ],
            dtype=np.float32,
        )
    else:
        # Entropy uncertainty palette: obsidian -> purple -> crimson -> amber-white
        stops = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
        cols = np.array(
            [
                [12, 10, 32],
                [109, 40, 217],
                [225, 29, 72],
                [253, 224, 71],
            ],
            dtype=np.float32,
        )

    out = np.zeros((*x.shape, 3), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.clip(np.interp(x, stops, cols[:, c]), 0, 255).astype(np.uint8)
    return out


def _decode_uploaded_image(b64_str: str, want_rgb: bool) -> np.ndarray:
    if "," in b64_str and b64_str.strip().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)
    if want_rgb:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr[:, :, :3].astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def generate_builtin_scene(scene_key: str, size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate realistic multi-block procedural test scenes for zero-file instant exploration."""
    rng = np.random.default_rng({"wildfire_suburb": 101, "port_blast": 202, "seismic_coastal": 303}.get(scene_key, 42))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # Textured terrain & vegetation background
    base_g = 72 + 16 * np.sin(xx / 45.0) * np.cos(yy / 52.0)
    pre = np.zeros((size, size, 3), dtype=np.float32)
    pre[..., 0] = base_g * 0.82 + rng.normal(0, 6, (size, size))
    pre[..., 1] = base_g * 1.05 + rng.normal(0, 6, (size, size))
    pre[..., 2] = base_g * 0.70 + rng.normal(0, 5, (size, size))

    post = rng.gamma(shape=3.0, scale=14.0, size=(size, size)).astype(np.float32)
    gt = np.zeros((size, size), dtype=np.uint8)

    # Road network (asphalt in optical, specular dark in SAR)
    for r_road in range(64, size, 112):
        pre[r_road : r_road + 12, :] = [92, 95, 100]
        post[r_road : r_road + 12, :] *= 0.38
    for c_road in range(72, size, 120):
        pre[:, c_road : c_road + 12] = [92, 95, 100]
        post[:, c_road : c_road + 12] *= 0.38

    if scene_key == "port_blast":
        title = "Built-in Scene: Port Industrial Blast Zone (512x512)"
        # Water basin in top-left corner
        water = (yy + 0.65 * xx) < (size * 0.36)
        pre[water] = [28, 58, 84]
        post[water] = rng.uniform(6, 20, size=(size, size))[water]
        # Large port warehouses near blast epicenter + residential blocks farther away
        blocks = []
        for r0 in range(24, size - 48, 56):
            for c0 in range(24, size - 48, 60):
                if (r0 + 28) + 0.65 * (c0 + 28) < size * 0.38:
                    continue
                h_b = 34 if r0 < 190 else 22
                w_b = 42 if c0 < 220 else 24
                blocks.append((r0, c0, h_b, w_b))
        for r0, c0, hb, wb in blocks:
            dist = np.hypot(r0 - 110, c0 - 150)
            roof = [205, 202, 196] if hb > 28 else [176, 152, 134]
            pre[r0 : r0 + hb, c0 : c0 + wb] = roof
            if dist < 145:
                gt[r0 : r0 + hb, c0 : c0 + wb] = 3
                post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(18, 46, (hb, wb))
            elif dist < 235:
                gt[r0 : r0 + hb, c0 : c0 + wb] = 2
                post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(135, 195, (hb, wb))
            else:
                gt[r0 : r0 + hb, c0 : c0 + wb] = 1
                post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(165, 232, (hb, wb))

    elif scene_key == "seismic_coastal":
        title = "Built-in Scene: Seismic Timber District (512x512)"
        for r0 in range(20, size - 30, 36):
            for c0 in range(20, size - 30, 40):
                if (r0 % 112 < 14) or (c0 % 120 < 14):
                    continue
                hb, wb = int(rng.integers(16, 24)), int(rng.integers(18, 28))
                pre[r0 : r0 + hb, c0 : c0 + wb] = [148 + (r0 % 35), 144, 140]
                fault_band = abs((r0 - 0.8 * c0) - 40) < 85
                if fault_band and rng.random() < 0.68:
                    gt[r0 : r0 + hb, c0 : c0 + wb] = 3
                    post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(35, 95, (hb, wb))
                else:
                    gt[r0 : r0 + hb, c0 : c0 + wb] = 1
                    post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(160, 228, (hb, wb))
    else:
        title = "Built-in Scene: Suburban Wildfire Burn Perimeter (512x512)"
        burn_front = (0.75 * xx + 0.65 * yy) > (size * 0.68)
        for r0 in range(22, size - 32, 42):
            for c0 in range(22, size - 32, 44):
                if (r0 % 112 < 14) or (c0 % 120 < 14):
                    continue
                hb, wb = 22, 26
                pre[r0 : r0 + hb, c0 : c0 + wb] = [184, 174, 162]
                if burn_front[r0 + hb // 2, c0 + wb // 2]:
                    gt[r0 : r0 + hb, c0 : c0 + wb] = 3
                    post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(14, 42, (hb, wb))
                else:
                    gt[r0 : r0 + hb, c0 : c0 + wb] = 1
                    post[r0 : r0 + hb, c0 : c0 + wb] = rng.uniform(170, 238, (hb, wb))

    return (
        np.clip(np.rint(pre), 0, 255).astype(np.uint8),
        np.clip(np.rint(post), 0, 255).astype(np.uint8),
        gt,
        title,
    )


class DashboardState:
    """Thread-safe model cache and dataset/checkpoint discovery state."""

    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root
        self.lock = threading.Lock()
        self.bench_cfg = load_yaml_config(repo_root / "configs/benchmark_v0_2.yaml")
        self.loaded_models: dict[str, tuple[object, str, str]] = {}
        self.active_checkpoint: Path | None = None
        self._auto_select_initial_checkpoint()

    def discover_checkpoints(self) -> list[dict[str, object]]:
        search_roots = [
            self.repo_root / "exports",
            self.repo_root / "runs",
            Path("/kaggle/working/exports"),
            Path("/kaggle/working/runs"),
            Path.home() / "Downloads",
        ]
        found: list[dict[str, object]] = []
        seen: set[str] = set()
        for root in search_roots:
            if not root.exists():
                continue
            patterns = ["*.pth"] if root.name in ("exports", "Downloads") else ["**/*.pth"]
            for pat in patterns:
                for p in sorted(root.glob(pat)):
                    rp = str(p.resolve())
                    if rp in seen or not p.is_file():
                        continue
                    if root.name == "Downloads" and not any(
                        k in p.name.lower() for k in ("georesq", "gated", "concat", "optical", "sar", "v0_2", "best_model")
                    ):
                        continue
                    seen.add(rp)
                    found.append(
                        {
                            "name": p.name if p.name != "best_model.pth" else f"{p.parent.name}/{p.name}",
                            "path": rp,
                            "size_mb": round(p.stat().st_size / (1024 ** 2), 2),
                        }
                    )
        return found

    def _auto_select_initial_checkpoint(self) -> None:
        ckpts = self.discover_checkpoints()
        for c in ckpts:
            if "lite_gated" in str(c["name"]).lower() or "lite_gated" in str(c["path"]).lower():
                self.active_checkpoint = Path(str(c["path"]))
                return
        if ckpts:
            self.active_checkpoint = Path(str(ckpts[0]["path"]))

    def discover_real_tiles(self) -> list[dict[str, object]]:
        search_roots = [
            self.repo_root / "exports",
            self.repo_root / "data",
            Path("/tmp/bright_subset_v0_2"),
            Path("/kaggle/working/data/bright_subset_v0_2"),
        ]
        tiles: list[dict[str, object]] = []
        seen: set[str] = set()
        for root in search_roots:
            if not root.exists():
                continue
            for pre_p in sorted(root.rglob("*_pre_disaster.tif")):
                stem_id = pre_p.name.replace("_pre_disaster.tif", "")
                if stem_id in seen:
                    continue
                # Check sibling post-event folder or same directory
                post_cands = [
                    pre_p.parent.parent / "post-event" / f"{stem_id}_post_disaster.tif",
                    pre_p.parent / f"{stem_id}_post_disaster.tif",
                ]
                tgt_cands = [
                    pre_p.parent.parent / "target" / f"{stem_id}_building_damage.tif",
                    pre_p.parent / f"{stem_id}_building_damage.tif",
                ]
                post_p = next((c for c in post_cands if c.exists()), None)
                tgt_p = next((c for c in tgt_cands if c.exists()), None)
                if post_p is not None:
                    seen.add(stem_id)
                    tiles.append(
                        {
                            "id": f"real:{stem_id}",
                            "sample_id": stem_id,
                            "pre_path": str(pre_p),
                            "post_path": str(post_p),
                            "target_path": str(tgt_p) if tgt_p else None,
                        }
                    )
                if len(tiles) >= 30:
                    return tiles
        return tiles

    def get_device_label(self) -> str:
        if not HAS_TORCH:
            return "numpy-sandbox-preview"
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon GPU)"
        return "cpu (PyTorch)"

    def _get_or_load_pytorch_model(self, variant_id: str, ckpt_path: Path | None):
        from georesq.models import build_model
        from georesq.training.loop import load_checkpoint

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        cache_key = f"{variant_id}::{ckpt_path or 'untrained'}::{device}"
        with self.lock:
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key], device

            variants_cfg = self.bench_cfg["variants"]  # type: ignore[index]
            v_cfg = dict(variants_cfg.get(variant_id, variants_cfg["lite_gated_fusion_v0_2"]))  # type: ignore[attr-defined,index]
            norm_mode = str(self.bench_cfg["shared_data"]["norm_mode"])  # type: ignore[index]

            if ckpt_path is not None and ckpt_path.exists():
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(raw, dict) and "variant_config" in raw:
                    v_cfg = dict(raw["variant_config"])
                    norm_mode = str(raw.get("norm_mode", norm_mode))

            v_cfg["pretrained"] = False
            v_cfg["stack_sar_3ch"] = True
            model = build_model(v_cfg).to(device)
            src_label = "untrained_architecture_weights"
            if ckpt_path is not None and ckpt_path.exists():
                load_checkpoint(ckpt_path, model, device=device)
                src_label = ckpt_path.name
            model.eval()
            entry = (model, norm_mode, src_label)
            self.loaded_models[cache_key] = entry
            return entry, device

    def run_dashboard_inference(self, payload: dict[str, object]) -> dict[str, object]:
        scene_id = str(payload.get("scene_id", "builtin:wildfire_suburb"))
        variant_id = str(payload.get("variant_id", "lite_gated_fusion_v0_2"))
        corruption = str(payload.get("corruption", "clean"))
        if corruption not in VALID_CORRUPTIONS:
            corruption = "clean"
        severity = float(np.clip(float(payload.get("severity", 0.0)), 0.0, 1.0))  # type: ignore[arg-type]
        grid_n = int(np.clip(int(payload.get("grid_size", 4)), 2, 8))  # type: ignore[arg-type]

        req_ckpt = payload.get("checkpoint_path")
        ckpt_path = Path(str(req_ckpt)) if req_ckpt else self.active_checkpoint

        gt_u8: np.ndarray | None = None
        if payload.get("pre_opt_b64") and payload.get("post_sar_b64"):
            pre_u8 = _decode_uploaded_image(str(payload["pre_opt_b64"]), want_rgb=True)
            post_u8 = _decode_uploaded_image(str(payload["post_sar_b64"]), want_rgb=False)
            if post_u8.shape != pre_u8.shape[:2]:
                post_u8 = np.array(
                    Image.fromarray(post_u8).resize((pre_u8.shape[1], pre_u8.shape[0]), Image.Resampling.BILINEAR)
                )
            if payload.get("target_b64"):
                gt_u8 = _decode_uploaded_image(str(payload["target_b64"]), want_rgb=False)
            tile_label = str(payload.get("custom_label") or "Custom Uploaded Tile")
        elif scene_id.startswith("real:"):
            stem_id = scene_id.split("real:", 1)[1]
            matched = next((t for t in self.discover_real_tiles() if t["sample_id"] == stem_id), None)
            if matched is None:
                raise FileNotFoundError(f"Real tile {stem_id!r} not found on disk.")
            pre_u8 = read_rgb_tif(str(matched["pre_path"]))
            post_u8 = read_single_band_tif(str(matched["post_path"]))
            gt_u8 = read_single_band_tif(str(matched["target_path"])) if matched.get("target_path") else None
            tile_label = f"{stem_id} (Real BRIGHT GeoTIFF)"
        else:
            key = scene_id.split(":", 1)[-1]
            pre_u8, post_u8, gt_u8, tile_label = generate_builtin_scene(key, size=512)

        pre_c, post_c, corr_diag = corrupt_triplet(
            pre_u8, post_u8, corruption, severity, seed=42, sample_idx=0
        )

        t0 = time.perf_counter()
        learned_gammas: list[float] = []

        if HAS_TORCH and ckpt_path is not None and ckpt_path.exists():
            import torch.nn.functional as F

            (model, norm_mode, src_label), device = self._get_or_load_pytorch_model(variant_id, ckpt_path)
            pre_chw, post_chw = normalize_multimodal(pre_c, post_c, norm_mode=norm_mode, stack_sar_3ch=True)
            h, w = pre_c.shape[:2]
            pad_h = (32 - (h % 32)) % 32
            pad_w = (32 - (w % 32)) % 32

            pre_t = torch.from_numpy(pre_chw).unsqueeze(0).to(device)
            post_t = torch.from_numpy(post_chw).unsqueeze(0).to(device)
            if pad_h or pad_w:
                pre_t = F.pad(pre_t, (0, pad_w, 0, pad_h), mode="reflect")
                post_t = F.pad(post_t, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.no_grad():
                use_amp = device.type == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if hasattr(model, "extract_fusion_diagnostics"):
                        logits, gates, learned_gammas = model.extract_fusion_diagnostics(pre_t, post_t)  # type: ignore[attr-defined]
                    else:
                        logits, gates = model(pre_t, post_t), []  # type: ignore[operator]
                logits = logits[:, :, :h, :w].float()
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                if gates:
                    g_up = F.interpolate(
                        gates[1].float(), size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze().cpu().numpy()
                else:
                    g_up = np.max(probs[1:], axis=0)

            engine_mode = f"Trained PyTorch Checkpoint ({src_label} on {device})"
            trained_loaded = True
        else:
            # Physics-guided multimodal preview estimator when no .pth checkpoint is loaded yet
            # (keeps interactive exploration meaningful in sandbox preview & pre-checkpoint setup)
            lum = pre_c.astype(np.float32).mean(axis=-1) / 255.0
            rgb_spread = (pre_c.astype(np.float32).max(axis=-1) - pre_c.astype(np.float32).min(axis=-1)) / 255.0
            sar_n = post_c.astype(np.float32) / 255.0

            # Roof prior: bright structured man-made surfaces (attenuated by thick white cloud > 0.91)
            cloud_veil = np.clip((lum - 0.84) / 0.14, 0.0, 1.0) * np.clip(0.12 - rgb_spread, 0.0, 0.12) / 0.12
            opt_roof_prior = np.clip((lum - 0.46) / 0.28, 0.0, 1.0) * (1.0 - 0.85 * cloud_veil)

            if variant_id == "optical_only_v0_2":
                p_bldg = opt_roof_prior * 0.82
                p_dest = p_bldg * 0.10
                p_intact = p_bldg * 0.88
            elif variant_id == "sar_only_v0_2":
                p_bldg = np.clip((sar_n - 0.45) / 0.40, 0.0, 1.0) * 0.72
                p_dest = np.zeros_like(p_bldg)
                p_intact = p_bldg
            else:
                sar_strong = np.clip((sar_n - 0.48) / 0.35, 0.0, 1.0)
                sar_drop = np.clip((0.28 - sar_n) / 0.22, 0.0, 1.0)
                gate_boost = 1.08 if variant_id == "lite_gated_fusion_v0_2" else 0.95
                p_bldg = np.clip(gate_boost * (0.65 * opt_roof_prior + 0.35 * sar_strong * (0.4 + 0.6 * opt_roof_prior)), 0.0, 0.94)
                p_dest = np.clip(p_bldg * sar_drop * 1.25, 0.0, p_bldg * 0.92)
                p_intact = np.clip(p_bldg - p_dest, 0.0, 0.94)

            p_dmg = np.clip(0.03 * p_bldg, 0.0, 0.08)
            p_bg = np.clip(1.0 - (p_intact + p_dmg + p_dest), 0.04, 0.98)
            probs = np.stack([p_bg, p_intact, p_dmg, p_dest], axis=0).astype(np.float32)
            probs /= np.sum(probs, axis=0, keepdims=True)
            g_up = np.clip(0.22 + 0.58 * opt_roof_prior, 0.0, 1.0).astype(np.float32)
            engine_mode = (
                "Heuristic Demo Preview (Load lite_gated_fusion_v0_2_fp16.pth in header for trained neural weights)"
            )
            trained_loaded = False

        infer_ms = round(1000.0 * (time.perf_counter() - t0), 2)
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
        entropy_map = compute_normalized_entropy(probs)
        sectors = build_sector_triage_table(pred_mask, entropy_map, grid_n=grid_n)

        # Downscale display PNGs cleanly to <= 512px max side for snappy browser rendering
        disp_h, disp_w = pre_c.shape[:2]
        max_dim = 512
        if max(disp_h, disp_w) > max_dim:
            scale = max_dim / float(max(disp_h, disp_w))
            nw, nh = max(1, int(round(disp_w * scale))), max(1, int(round(disp_h * scale)))
            pre_vis = np.array(Image.fromarray(pre_c).resize((nw, nh), Image.Resampling.BILINEAR))
            post_vis = np.array(Image.fromarray(post_c).resize((nw, nh), Image.Resampling.BILINEAR))
            pred_vis = np.array(Image.fromarray(mask_to_rgb(pred_mask)).resize((nw, nh), Image.Resampling.NEAREST))
            ent_vis = np.array(Image.fromarray(_heatmap_to_rgb(entropy_map, "magma")).resize((nw, nh), Image.Resampling.BILINEAR))
            gate_vis = np.array(Image.fromarray(_heatmap_to_rgb(g_up, "cyan_gold")).resize((nw, nh), Image.Resampling.BILINEAR))
            gt_vis = (
                np.array(Image.fromarray(mask_to_rgb(gt_u8)).resize((nw, nh), Image.Resampling.NEAREST))
                if gt_u8 is not None
                else None
            )
        else:
            pre_vis = pre_c
            post_vis = post_c
            pred_vis = mask_to_rgb(pred_mask)
            ent_vis = _heatmap_to_rgb(entropy_map, "magma")
            gate_vis = _heatmap_to_rgb(g_up, "cyan_gold")
            gt_vis = mask_to_rgb(gt_u8) if gt_u8 is not None else None

        tot_px = max(1, pred_mask.size)
        class_breakdown = {
            cls_name: {
                "pixels": int((pred_mask == i).sum()),
                "pct": round(100.0 * float((pred_mask == i).sum()) / tot_px, 2),
            }
            for i, cls_name in enumerate(CLASS_NAMES)
        }

        gt_eval = None
        if gt_u8 is not None and gt_u8.shape == pred_mask.shape:
            ev = DamageEvaluator(num_classes=4, ignore_index=255)
            ev.update(gt_u8.astype(np.int64), pred_mask.astype(np.int64))
            gt_eval = ev.compute_summary()
            gt_eval["error_decomposition"] = decompose_confusion_errors(gt_eval["confusion_matrix"])  # type: ignore[arg-type]

        bldg_mask = pred_mask > 0
        post_rgb = np.stack([post_vis, post_vis, post_vis], axis=-1)

        return {
            "status": "OK",
            "disclaimer": DISCLAIMER_TEXT,
            "human_review_required": True,
            "tile_label": tile_label,
            "shape_hw": [int(pre_c.shape[0]), int(pre_c.shape[1])],
            "engine_mode": engine_mode,
            "trained_checkpoint_loaded": trained_loaded,
            "variant_id": variant_id,
            "inference_ms": infer_ms,
            "corruption": corruption,
            "severity": severity,
            "corruption_diagnostics": corr_diag,
            "learned_gammas": [round(float(x), 5) for x in learned_gammas],
            "gate_mean": round(float(g_up.mean()), 4),
            "summary": {
                "class_breakdown": class_breakdown,
                "building_coverage_pct": round(100.0 * float(bldg_mask.sum()) / tot_px, 2),
                "destroyed_pixels": int((pred_mask == 3).sum()),
                "mean_entropy_all": round(float(entropy_map.mean()), 4),
                "mean_entropy_buildings": round(float(entropy_map[bldg_mask].mean()), 4) if bldg_mask.any() else 0.0,
            },
            "sectors": sectors,
            "ground_truth_evaluation": gt_eval,
            "layers": {
                "pre_optical": _array_to_png_data_uri(pre_vis),
                "post_sar": _array_to_png_data_uri(post_rgb),
                "predicted_damage": _array_to_png_data_uri(pred_vis),
                "ground_truth": _array_to_png_data_uri(gt_vis) if gt_vis is not None else None,
                "entropy": _array_to_png_data_uri(ent_vis),
                "prior_gate": _array_to_png_data_uri(gate_vis),
            },
        }


def load_benchmark_bundle(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    metrics_dir = repo_root / "results/metrics"
    out: dict[str, object] = {}
    json_files = {
        "ablation_v0_1": "ablation_v0_1_summary.json",
        "benchmark_v0_2": "benchmark_v0_2_summary.json",
        "stress_test_v0_2": "stress_test_v0_2_summary.json",
        "failure_analysis_v0_2": "failure_analysis_v0_2_summary.json",
        "dataset_audit_v0_2": "dataset_audit_v0_2.json",
    }
    for key, fname in json_files.items():
        p = metrics_dir / fname
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if key == "benchmark_v0_2":
                out[key] = {"comparison_table": data.get("comparison_table", [])}
            elif key == "ablation_v0_1":
                out[key] = {"comparison_table": data.get("comparison_table", [])}
            else:
                out[key] = data

    audit_csv = metrics_dir / "per_tile_audit_v0_2.csv"
    if audit_csv.exists():
        with audit_csv.open("r", encoding="utf-8") as f:
            out["per_tile_audit"] = list(csv.DictReader(f))
    else:
        out["per_tile_audit"] = []
    return out


def make_handler(state: DashboardState):
    benchmark_cache = load_benchmark_bundle(state.repo_root)

    class GeoResQHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"[GeoResQ-UI] {fmt % args}\n")

        def _send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, head_only: bool = False) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            mime, _ = mimetypes.guess_type(str(path))
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if not head_only:
                self.wfile.write(data)

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            if route in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", head_only=True)
                return
            if route.startswith("/results/figures/"):
                fname = Path(route).name
                fig_path = (state.repo_root / "results/figures" / fname).resolve()
                if fig_path.parent == (state.repo_root / "results/figures").resolve():
                    self._send_file(fig_path, head_only=True)
                    return
            self.send_response(HTTPStatus.OK)
            self.end_headers()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path

            if route in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html")
                return
            if route == "/api/status":
                self._send_json(
                    {
                        "version": __version__,
                        "pytorch_available": HAS_TORCH,
                        "device": state.get_device_label(),
                        "active_checkpoint": str(state.active_checkpoint) if state.active_checkpoint else None,
                        "discovered_checkpoints": state.discover_checkpoints(),
                        "discovered_real_tiles": state.discover_real_tiles(),
                        "builtin_scenes": [
                            {"id": "builtin:wildfire_suburb", "label": "Built-in: Suburban Wildfire Burn Perimeter"},
                            {"id": "builtin:port_blast", "label": "Built-in: Port Industrial Blast Zone"},
                            {"id": "builtin:seismic_coastal", "label": "Built-in: Seismic Timber District"},
                        ],
                    }
                )
                return
            if route == "/api/benchmarks":
                self._send_json(benchmark_cache)
                return
            if route.startswith("/results/figures/"):
                fname = Path(route).name
                fig_path = (state.repo_root / "results/figures" / fname).resolve()
                if fig_path.parent == (state.repo_root / "results/figures").resolve():
                    self._send_file(fig_path)
                    return

            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {route}")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"

            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self._send_json({"error": f"Invalid JSON body: {exc}"}, status=400)
                return

            if route == "/api/set-checkpoint":
                p_str = str(payload.get("path", "")).strip()
                if not p_str:
                    state.active_checkpoint = None
                    self._send_json({"status": "OK", "active_checkpoint": None})
                    return
                cand = Path(p_str).expanduser().resolve()
                if not cand.exists():
                    self._send_json({"error": f"Checkpoint file not found: {cand}"}, status=404)
                    return
                state.active_checkpoint = cand
                self._send_json({"status": "OK", "active_checkpoint": str(cand)})
                return

            if route == "/api/predict":
                try:
                    res = state.run_dashboard_inference(payload)
                    self._send_json(res)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=500)
                return

            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown POST route: {route}")

    return GeoResQHandler


def create_server(host: str = "0.0.0.0", port: int = 8000, repo_root: Path = REPO_ROOT) -> ThreadingHTTPServer:
    state = DashboardState(repo_root=repo_root)
    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the GeoResQ Interactive Visual Triage Dashboard.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind TCP port (default: 8000).")
    args = parser.parse_args(argv)

    srv = create_server(host=args.host, port=args.port)
    actual_port = srv.server_address[1]
    print(f"GeoResQ Interactive Dashboard listening on http://localhost:{actual_port} (bound {args.host}:{actual_port})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GeoResQ Dashboard...")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
