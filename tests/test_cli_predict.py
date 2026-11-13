"""Unit tests for Milestone 8 Standalone Inference & Uncertainty Triage CLI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from georesq.cli.predict import (
    build_sector_triage_table,
    compute_normalized_entropy,
    main as cli_main,
    render_triage_panel,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestCliPredictAndTriage(unittest.TestCase):
    def test_normalized_entropy_bounds(self) -> None:
        # Uniform 4-class distribution -> entropy = 1.0
        uniform = np.full((4, 16, 16), 0.25, dtype=np.float32)
        h_uni = compute_normalized_entropy(uniform)
        self.assertEqual(h_uni.shape, (16, 16))
        self.assertTrue(np.allclose(h_uni, 1.0, atol=1e-5))

        # Deterministic one-hot distribution -> entropy = 0.0
        onehot = np.zeros((4, 16, 16), dtype=np.float32)
        onehot[1] = 1.0
        h_det = compute_normalized_entropy(onehot)
        self.assertTrue(np.allclose(h_det, 0.0, atol=1e-5))

    def test_sector_triage_ranking_and_panel_render(self) -> None:
        pred = np.zeros((64, 64), dtype=np.uint8)
        ent = np.full((64, 64), 0.10, dtype=np.float32)

        # Sector R1C1 (0:32, 0:32): intact buildings, low entropy
        pred[4:28, 4:28] = 1
        # Sector R2C2 (32:64, 32:64): destroyed buildings + elevated entropy -> must rank #1
        pred[36:60, 36:60] = 3
        ent[36:60, 36:60] = 0.70

        sectors = build_sector_triage_table(pred, ent, grid_n=2)
        self.assertEqual(len(sectors), 4)
        self.assertEqual(sectors[0]["sector_id"], "R2C2")
        self.assertEqual(sectors[0]["review_reason"], "high_predicted_severe_damage")
        self.assertTrue(all(bool(s["human_review_required"]) for s in sectors))

        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.png"
            pre = np.full((64, 64, 3), 120, dtype=np.uint8)
            post = np.full((64, 64), 90, dtype=np.uint8)
            render_triage_panel(pre, post, pred, ent, sectors, panel_path, tile_label="unit_test")
            self.assertTrue(panel_path.exists())
            self.assertGreater(panel_path.stat().st_size, 5000)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in lightweight container")
    def test_cli_demo_synthetic_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "triage_out"
            rc = cli_main(["--demo-synthetic", "--output-dir", str(out_dir), "--device", "cpu"])
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "predicted_damage_mask.tif").exists())
            self.assertTrue((out_dir / "uncertainty_entropy.png").exists())
            self.assertTrue((out_dir / "triage_panel.png").exists())
            rep = json.loads((out_dir / "triage_report.json").read_text(encoding="utf-8"))
            self.assertTrue(rep["human_review_required"])
            self.assertFalse(rep["model"]["trained_checkpoint_loaded"])
            self.assertEqual(rep["spatial_shape_hw"], [256, 256])


if __name__ == "__main__":
    unittest.main()
