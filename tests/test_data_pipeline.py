"""Automated Unit & Integration Tests for GeoResQ Milestone 2 Data Pipeline."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from georesq.data.dataset import (
    BrightDamageDataset,
    compute_class_weights_from_manifest,
    load_manifest_rows,
    verify_zero_event_leakage,
)
from georesq.preprocessing.transforms import (
    joint_random_crop,
    joint_random_flip_rot,
    validate_raw_triplet,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = REPO_ROOT / "results" / "metrics" / "manifest_v0_1.csv"


class TestGeoResQDataPipeline(unittest.TestCase):
    def test_real_manifest_integrity_and_zero_leakage(self) -> None:
        """Verify real v0.1 manifest has 45 rows, legal events, and zero split overlap."""
        self.assertTrue(REAL_MANIFEST.exists(), f"Missing {REAL_MANIFEST}")
        all_rows = load_manifest_rows(REAL_MANIFEST)
        self.assertEqual(len(all_rows), 45)

        split_events = verify_zero_event_leakage(REAL_MANIFEST)
        self.assertEqual(len(split_events["train"]), 3)
        self.assertEqual(len(split_events["val"]), 2)
        self.assertEqual(split_events["test"], ["noto-earthquake"])

    def test_train_only_class_weights_monotonic(self) -> None:
        """Minority damage classes (2, 3) must receive higher weights than background (0)."""
        weights = compute_class_weights_from_manifest(REAL_MANIFEST, power=0.35)
        self.assertEqual(len(weights), 4)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertLess(weights[2], weights[3])

    def test_illegal_label_rejected(self) -> None:
        pre = np.zeros((64, 64, 3), dtype=np.uint8)
        post = np.zeros((64, 64), dtype=np.uint8)
        bad_mask = np.full((64, 64), 7, dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_raw_triplet(pre, post, bad_mask)

    def test_joint_transforms_preserve_pixel_correspondence(self) -> None:
        """A marker placed at the exact same (y, x) in Optical, SAR, and Mask must stay co-located."""
        pre = np.zeros((256, 256, 3), dtype=np.uint8)
        post = np.zeros((256, 256), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)

        pre[40:60, 110:140, 0] = 250
        post[40:60, 110:140] = 200
        mask[40:60, 110:140] = 3

        c_pre, c_post, c_mask = joint_random_crop(
            pre, post, mask, crop_size=128, rng=random.Random(7)
        )
        a_pre, a_post, a_mask = joint_random_flip_rot(
            c_pre, c_post, c_mask, rng=random.Random(13)
        )
        np.testing.assert_array_equal(a_pre[:, :, 0] == 250, a_post == 200)
        np.testing.assert_array_equal(a_post == 200, a_mask == 3)

    def test_dataset_loading_and_modality_modes(self) -> None:
        """Verify BrightDamageDataset on temporary GeoTIFFs across multimodal and single-modality modes."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed in local environment")

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            for sub in ("pre-event", "post-event", "target"):
                (tmp / sub).mkdir(parents=True)

            sid = "bata-explosion_00000001"
            pre = np.full((600, 600, 3), 120, dtype=np.uint8)
            post = np.full((600, 600), 80, dtype=np.uint8)
            tgt = np.zeros((600, 600), dtype=np.uint8)
            tgt[100:250, 100:250] = 2
            tgt[300:400, 300:400] = 3

            Image.fromarray(pre).save(tmp / f"pre-event/{sid}_pre_disaster.tif")
            Image.fromarray(post).save(tmp / f"post-event/{sid}_post_disaster.tif")
            Image.fromarray(tgt).save(tmp / f"target/{sid}_building_damage.tif")

            mf = tmp / "manifest.csv"
            with mf.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sample_id",
                        "disaster_event",
                        "region",
                        "pre_event_path",
                        "post_event_path",
                        "target_path",
                        "split",
                        "modality_information",
                        "label_information",
                    ],
                )
                w.writeheader()
                w.writerow(
                    {
                        "sample_id": sid,
                        "disaster_event": "bata-explosion",
                        "region": "Bata, Equatorial Guinea",
                        "pre_event_path": f"pre-event/{sid}_pre_disaster.tif",
                        "post_event_path": f"post-event/{sid}_post_disaster.tif",
                        "target_path": f"target/{sid}_building_damage.tif",
                        "split": "train",
                        "modality_information": "{}",
                        "label_information": json.dumps(
                            {"background": 327500, "intact": 0, "damaged": 22500, "destroyed": 10000, "ignore_255": 0}
                        ),
                    }
                )

            ds_mm = BrightDamageDataset(
                tmp, mf, split="train", crop_size=512, norm_mode="official_imagenet", stack_sar_3ch=True
            )
            sample = ds_mm[0]
            self.assertEqual(tuple(sample["pre_opt"].shape), (3, 512, 512))  # type: ignore[attr-defined]
            self.assertEqual(tuple(sample["post_sar"].shape), (3, 512, 512))  # type: ignore[attr-defined]
            self.assertEqual(tuple(sample["clf_mask"].shape), (512, 512))  # type: ignore[attr-defined]

            ds_native = BrightDamageDataset(
                tmp, mf, split="train", augment=False, norm_mode="modality_specific", stack_sar_3ch=False
            )
            s_nat = ds_native[0]
            self.assertEqual(tuple(s_nat["pre_opt"].shape), (3, 600, 600))  # type: ignore[attr-defined]
            self.assertEqual(tuple(s_nat["post_sar"].shape), (1, 600, 600))  # type: ignore[attr-defined]

            ds_opt = BrightDamageDataset(tmp, mf, split="train", crop_size=512, input_modality="optical_only")
            s_opt = ds_opt[0]
            self.assertEqual(float(np.abs(np.asarray(s_opt["post_sar"])).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
