"""Unit tests for Milestone 6 sensor corruptions & Milestone 7 error taxonomy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from georesq.evaluation.metrics import decompose_confusion_errors
from georesq.preprocessing.corruptions import (
    VALID_CORRUPTIONS,
    corrupt_triplet,
    severity_to_sar_enl,
)
from georesq.training.loop import load_yaml_config

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestSensorCorruptionsAndAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.pre_u8 = rng.integers(30, 180, size=(128, 128, 3), dtype=np.uint8)
        self.post_u8 = rng.integers(25, 190, size=(128, 128), dtype=np.uint8)

    def test_zero_severity_is_exact_identity(self) -> None:
        for corr in VALID_CORRUPTIONS:
            pre_c, post_c, diag = corrupt_triplet(self.pre_u8, self.post_u8, corr, 0.0, seed=42, sample_idx=3)
            self.assertTrue(np.array_equal(pre_c, self.pre_u8), msg=corr)
            self.assertTrue(np.array_equal(post_c, self.post_u8), msg=corr)
            self.assertEqual(diag["opt_mae_u8"], 0.0)
            self.assertEqual(diag["sar_mae_u8"], 0.0)

    def test_determinism_and_monotonic_severity(self) -> None:
        sevs = [0.25, 0.50, 0.75, 1.00]
        for corr in ("cloud_occlusion", "smoke_haze", "optical_dropout", "sar_speckle", "compound_storm"):
            maes = []
            for s in sevs:
                p1, s1, d1 = corrupt_triplet(self.pre_u8, self.post_u8, corr, s, seed=42, sample_idx=7)
                p2, s2, _ = corrupt_triplet(self.pre_u8, self.post_u8, corr, s, seed=42, sample_idx=7)
                p_other, s_other, _ = corrupt_triplet(self.pre_u8, self.post_u8, corr, s, seed=42, sample_idx=8)

                self.assertEqual(p1.dtype, np.uint8)
                self.assertEqual(s1.dtype, np.uint8)
                self.assertEqual(p1.shape, self.pre_u8.shape)
                self.assertEqual(s1.shape, self.post_u8.shape)
                self.assertTrue(np.array_equal(p1, p2))
                self.assertTrue(np.array_equal(s1, s2))

                if s < 1.0:
                    diff_other = not (np.array_equal(p1, p_other) and np.array_equal(s1, s_other))
                    self.assertTrue(diff_other, msg=f"{corr} at s={s} should vary across sample_idx")

                if corr in ("cloud_occlusion", "smoke_haze", "optical_dropout"):
                    self.assertTrue(np.array_equal(s1, self.post_u8))
                    maes.append(d1["opt_mae_u8"])
                elif corr == "sar_speckle":
                    self.assertTrue(np.array_equal(p1, self.pre_u8))
                    maes.append(d1["sar_mae_u8"])
                else:
                    maes.append(d1["opt_mae_u8"] + d1["sar_mae_u8"])

            for a, b in zip(maes[:-1], maes[1:]):
                self.assertLess(a, b, msg=f"Expected monotonic MAE for {corr}, got {maes}")

    def test_optical_dropout_full_blackout_and_sar_enl(self) -> None:
        pre_blk, post_blk, _ = corrupt_triplet(self.pre_u8, self.post_u8, "optical_dropout", 1.0)
        self.assertTrue(np.all(pre_blk == 0))
        self.assertTrue(np.array_equal(post_blk, self.post_u8))

        self.assertAlmostEqual(severity_to_sar_enl(0.25), 8.0, places=5)
        self.assertAlmostEqual(severity_to_sar_enl(0.50), 4.0, places=5)
        self.assertAlmostEqual(severity_to_sar_enl(0.75), 2.0, places=5)
        self.assertAlmostEqual(severity_to_sar_enl(1.00), 1.0, places=5)

    def test_confusion_error_decomposition(self) -> None:
        cm = np.array(
            [
                [900, 60, 10, 30],   # 1000 bg -> 100 false alarm (10%)
                [100, 350, 20, 30],  # 500 intact
                [50, 100, 30, 20],   # 200 damaged -> 50% pred intact
                [50, 80, 20, 150],   # 300 destroyed -> 50% pred destroyed
            ],
            dtype=np.int64,
        )
        res = decompose_confusion_errors(cm)
        self.assertAlmostEqual(res["false_building_alarm_rate_pct"], 10.0, places=3)  # type: ignore[arg-type]
        self.assertAlmostEqual(res["missed_building_fn_rate_pct"], 20.0, places=3)  # type: ignore[arg-type]
        # Detected building pixels = 1000 - 200 = 800; correct severity trace = 350 + 30 + 150 = 530 -> 66.25%
        self.assertAlmostEqual(res["conditional_severity_acc_given_detected_pct"], 66.25, places=3)  # type: ignore[arg-type]
        self.assertAlmostEqual(res["damaged_routing_pct"]["pred_intact_pct"], 50.0, places=3)  # type: ignore[index]
        self.assertAlmostEqual(res["destroyed_routing_pct"]["pred_destroyed_pct"], 50.0, places=3)  # type: ignore[index]

    def test_stress_config_yaml_loads(self) -> None:
        cfg = load_yaml_config(REPO_ROOT / "configs/stress_test_v0_2.yaml")
        self.assertEqual(cfg["experiment"]["suite_name"], "stress_test_v0_2")  # type: ignore[index]
        self.assertEqual(len(cfg["severities"]), 5)  # type: ignore[arg-type]
        self.assertIn("sar_speckle", cfg["corruptions"])  # type: ignore[operator]

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in lightweight container")
    def test_extract_fusion_diagnostics(self) -> None:
        import torch
        from georesq.models import build_model

        m = build_model(
            {
                "architecture": "dual_stream_unet",
                "backbone": "resnet18",
                "fusion_mode": "lite_gated_fusion",
                "pretrained": False,
                "use_fullres_skip": True,
                "aux_loc_head": True,
            }
        )
        m.eval()
        x_opt = torch.randn(1, 3, 128, 128)
        x_sar = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            logits, gates, gammas = m.extract_fusion_diagnostics(x_opt, x_sar)
        self.assertEqual(tuple(logits.shape), (1, 4, 128, 128))
        self.assertEqual(len(gates), 5)
        self.assertEqual(len(gammas), 5)


if __name__ == "__main__":
    unittest.main()
