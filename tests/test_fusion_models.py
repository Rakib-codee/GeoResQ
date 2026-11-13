"""Unit tests for Milestone 4 & 5 DualStreamDamageNet, 1x skip, AuxLoc & losses."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class TestDualStreamFusionModels(unittest.TestCase):
    def test_all_ablation_modes_shapes_isolation_and_gradients(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed in local lightweight environment")

        from georesq.models import VALID_FUSION_MODES, build_model, get_parameter_groups
        from georesq.training.losses import BrightComboLoss, BuildingLocBCEDiceLoss

        criterion = BrightComboLoss(class_weights=[0.6, 0.85, 1.25, 1.3])
        loc_crit = BuildingLocBCEDiceLoss()
        pre_a = torch.randn(2, 3, 128, 128)
        pre_b = torch.randn(2, 3, 128, 128)
        sar_a = torch.randn(2, 3, 128, 128)
        sar_b = torch.randn(2, 3, 128, 128)
        target = torch.randint(0, 4, (2, 128, 128))
        loc_target = (target > 0).long()

        for mode in VALID_FUSION_MODES:
            use_v02_heads = mode in ("optical_only", "sar_only", "concat", "lite_gated_fusion")
            model = build_model(
                {
                    "architecture": "dual_stream_unet",
                    "backbone": "resnet18",
                    "fusion_mode": mode,
                    "num_classes": 4,
                    "pretrained": False,
                    "use_fullres_skip": use_v02_heads,
                    "aux_loc_head": use_v02_heads,
                }
            )
            groups = get_parameter_groups(model, base_lr=3e-4)
            grouped_count = sum(len(g["params"]) for g in groups)  # type: ignore[arg-type]
            total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            self.assertEqual(grouped_count, total_trainable)

            if mode == "lite_gated_fusion":
                total_params = sum(p.numel() for p in model.parameters())
                self.assertLess(total_params, 28_000_000)

            model.eval()
            with torch.no_grad():
                out_1 = model(pre_a, sar_a)
                self.assertEqual(tuple(out_1.shape), (2, 4, 128, 128))

                if mode == "optical_only":
                    out_diff_sar = model(pre_a, sar_b)
                    self.assertTrue(
                        torch.allclose(out_1, out_diff_sar, atol=1e-6),
                        "optical_only must ignore post_sar completely",
                    )
                elif mode == "sar_only":
                    out_diff_opt = model(pre_b, sar_a)
                    self.assertTrue(
                        torch.allclose(out_1, out_diff_opt, atol=1e-6),
                        "sar_only must ignore pre_opt completely",
                    )
                else:
                    out_diff_sar = model(pre_a, sar_b)
                    out_diff_opt = model(pre_b, sar_a)
                    self.assertFalse(torch.allclose(out_1, out_diff_sar, atol=1e-5))
                    self.assertFalse(torch.allclose(out_1, out_diff_opt, atol=1e-5))

            model.train()
            clf_logits, loc_logits = model.forward_with_aux(pre_a, sar_a)
            loss = criterion(clf_logits, target)
            if use_v02_heads:
                self.assertIsNotNone(loc_logits)
                self.assertEqual(tuple(loc_logits.shape), (2, 1, 128, 128))  # type: ignore[union-attr]
                loss = loss + 0.5 * loc_crit(loc_logits, loc_target)  # type: ignore[arg-type]
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()

            grad_norms = [
                float(p.grad.norm().item())
                for p in model.parameters()
                if p.grad is not None
            ]
            self.assertGreater(len(grad_norms), 0)
            self.assertGreater(max(grad_norms), 0.0)

    def test_resnet_encoder_state_dict_keys_match_torchvision(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed in local lightweight environment")

        from georesq.models.fusion import ResNetBackbone

        enc = ResNetBackbone(backbone_name="resnet18", in_channels=3, pretrained=False)
        keys = set(enc.state_dict().keys())
        expected_sample = {
            "conv1.weight",
            "bn1.weight",
            "layer1.0.conv1.weight",
            "layer2.0.downsample.0.weight",
            "layer2.0.downsample.1.weight",
            "layer3.1.bn2.bias",
            "layer4.1.conv2.weight",
        }
        self.assertTrue(expected_sample.issubset(keys))


if __name__ == "__main__":
    unittest.main()
