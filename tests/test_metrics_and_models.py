"""Unit Tests for Milestone 3 Evaluator, Lovasz Combo Loss, and Baseline U-Net."""

from __future__ import annotations

import unittest
import numpy as np

from georesq.evaluation.metrics import DamageEvaluator


class TestMilestone3Core(unittest.TestCase):
    def test_perfect_predictions_give_unit_scores(self) -> None:
        ev = DamageEvaluator(num_classes=4, ignore_index=255)
        gt = np.array([[0, 1, 2, 3], [255, 1, 2, 3]], dtype=np.int64)
        pr = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
        ev.update(gt, pr)
        summary = ev.compute_summary()

        self.assertAlmostEqual(float(summary["mIoU_all4"]), 1.0, places=5)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(summary["mIoU_foreground3"]), 1.0, places=5)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(summary["macro_f1_foreground3"]), 1.0, places=5)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(summary["bright_loc_f1"]), 1.0, places=5)  # type: ignore[arg-type]
        cm = np.array(summary["confusion_matrix"])
        self.assertEqual(int(cm.sum()), 7)

    def test_background_dominant_trap_detected_by_foreground_metrics(self) -> None:
        """Predicting all-background on an imbalanced tile yields high accuracy but ~0 foreground mIoU."""
        ev = DamageEvaluator(num_classes=4)
        gt = np.zeros((100, 100), dtype=np.int64)
        gt[0:5, 0:5] = 1
        gt[5:8, 5:8] = 2
        gt[8:10, 8:10] = 3
        pr = np.zeros_like(gt)

        ev.update(gt, pr)
        summary = ev.compute_summary()
        self.assertGreater(float(summary["overall_pixel_accuracy"]), 0.99)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(summary["mIoU_foreground3"]), 0.0, places=5)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(summary["bright_loc_f1"]), 0.0, places=5)  # type: ignore[arg-type]

    def test_unet_and_combo_loss_if_torch_installed(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed in local lightweight environment")

        from georesq.models.unet import BaselineUNet
        from georesq.training.losses import BrightComboLoss

        model = BaselineUNet(in_channels=6, num_classes=4, base_channels=16)
        criterion = BrightComboLoss(class_weights=[0.3, 0.6, 1.4, 1.7], lovasz_weight=0.75)

        pre = torch.randn(2, 3, 64, 64)
        post = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 4, (2, 64, 64))
        target[0, 0, 0] = 255

        logits = model(pre, post)
        self.assertEqual(tuple(logits.shape), (2, 4, 64, 64))
        loss = criterion(logits, target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss).item()))


if __name__ == "__main__":
    unittest.main()
