"""Manifest-driven BRIGHT Multimodal Building Damage Dataset for GeoResQ.

Designed for:
- Stage B Baseline UNet (6-ch concatenated Optical+SAR with official ImageNet norm)
- Stage C Dual-Encoder Multimodal Fusion (separate Optical 3-ch & SAR 1-ch/3-ch streams)
- Stage D Modality Ablations ('multimodal', 'optical_only', 'sar_only')
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from georesq.preprocessing.transforms import (
    joint_random_crop,
    joint_random_flip_rot,
    normalize_multimodal,
    validate_raw_triplet,
)

EXCLUDED_RESTRICTED_EVENTS = frozenset(
    {"ukraine-conflict", "myanmar-hurricane", "mexico-hurricane"}
)


def load_manifest_rows(
    manifest_path: Path, split: str | None = None
) -> list[dict[str, str]]:
    """Read manifest CSV, enforce license rules, and optionally filter by split."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r["disaster_event"] in EXCLUDED_RESTRICTED_EVENTS:
            raise ValueError(
                f"Restricted event '{r['disaster_event']}' detected in manifest!"
            )

    if split is not None:
        rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")

    return rows


def verify_zero_event_leakage(manifest_path: Path) -> dict[str, list[str]]:
    """Verify that train, val, and test splits share zero disaster events."""
    rows = load_manifest_rows(manifest_path, split=None)
    events_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        sp = r["split"]
        if sp in events_by_split:
            events_by_split[sp].add(r["disaster_event"])

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for sp_a, sp_b in pairs:
        overlap = events_by_split[sp_a] & events_by_split[sp_b]
        if overlap:
            raise AssertionError(
                f"Event leakage between '{sp_a}' and '{sp_b}': {sorted(overlap)}"
            )

    return {k: sorted(v) for k, v in events_by_split.items()}


def compute_class_weights_from_manifest(
    manifest_path: Path, power: float = 0.35
) -> list[float]:
    """Compute smoothed inverse-frequency weights [w0, w1, w2, w3] from TRAIN split only.

    Using `power` in (0.25, 0.5) dampens extreme weights (since Damaged/Destroyed are <2%
    of pixels) so training remains stable while still penalizing minority-class errors.
    """
    train_rows = load_manifest_rows(manifest_path, split="train")
    totals = [0, 0, 0, 0]
    for r in train_rows:
        lbl = json.loads(r["label_information"])
        totals[0] += int(lbl["background"])
        totals[1] += int(lbl["intact"])
        totals[2] += int(lbl["damaged"])
        totals[3] += int(lbl["destroyed"])

    total_px = sum(totals)
    raw = [(total_px / max(1, c)) ** power for c in totals]
    mean_w = sum(raw) / len(raw)
    return [round(w / mean_w, 4) for w in raw]


try:
    from torch.utils.data import Dataset as _BaseDataset  # type: ignore
except ImportError:
    _BaseDataset = object  # type: ignore


class BrightDamageDataset(_BaseDataset):  # type: ignore[misc]
    """PyTorch-compatible Dataset over BRIGHT GeoTIFF triplets listed in manifest CSV."""

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        split: str = "train",
        crop_size: int = 512,
        augment: bool | None = None,
        norm_mode: str = "official_imagenet",
        stack_sar_3ch: bool = True,
        input_modality: str = "multimodal",
        seed: int = 42,
    ) -> None:
        if input_modality not in ("multimodal", "optical_only", "sar_only"):
            raise ValueError(f"Unsupported input_modality='{input_modality}'")

        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.crop_size = int(crop_size)
        self.augment = (split == "train") if augment is None else bool(augment)
        self.norm_mode = norm_mode
        self.stack_sar_3ch = stack_sar_3ch
        self.input_modality = input_modality
        self.seed = int(seed)
        self.epoch = 0
        self._call_count = 0

        self.rows = load_manifest_rows(self.manifest_path, split=self.split)

    def set_epoch(self, epoch: int) -> None:
        """Update epoch counter for reproducible yet varied crops across epochs."""
        self.epoch = int(epoch)
        self._call_count = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        import numpy as np
        from PIL import Image

        row = self.rows[index]
        pre_path = self.data_root / row["pre_event_path"]
        post_path = self.data_root / row["post_event_path"]
        tgt_path = self.data_root / row["target_path"]

        pre_opt = np.array(Image.open(pre_path))[:, :, :3]
        post_sar = np.array(Image.open(post_path))
        if post_sar.ndim == 3:
            post_sar = post_sar[:, :, 0]
        clf_mask = np.array(Image.open(tgt_path))

        validate_raw_triplet(pre_opt, post_sar, clf_mask)

        if self.augment:
            rng = random.Random(
                self.seed + self.epoch * 1_000_003 + index * 10_007 + self._call_count
            )
            self._call_count += 1
            pre_opt, post_sar, clf_mask = joint_random_crop(
                pre_opt,
                post_sar,
                clf_mask,
                crop_size=self.crop_size,
                rng=rng,
            )
            pre_opt, post_sar, clf_mask = joint_random_flip_rot(
                pre_opt, post_sar, clf_mask, rng=rng
            )

        pre_chw, post_chw = normalize_multimodal(
            pre_opt,
            post_sar,
            norm_mode=self.norm_mode,
            stack_sar_3ch=self.stack_sar_3ch,
        )

        # Binary building localization mask (1 = any building, 0 = background, 255 = ignore)
        loc_mask = clf_mask.copy()
        loc_mask[(loc_mask == 2) | (loc_mask == 3)] = 1

        if self.input_modality == "optical_only":
            post_chw = np.zeros_like(post_chw)
        elif self.input_modality == "sar_only":
            pre_chw = np.zeros_like(pre_chw)

        try:
            import torch  # type: ignore

            pre_out: object = torch.from_numpy(pre_chw)
            post_out: object = torch.from_numpy(post_chw)
            clf_out: object = torch.from_numpy(clf_mask.astype(np.int64))
            loc_out: object = torch.from_numpy(loc_mask.astype(np.int64))
        except ImportError:
            pre_out = pre_chw
            post_out = post_chw
            clf_out = clf_mask.astype(np.int64)
            loc_out = loc_mask.astype(np.int64)

        return {
            "pre_opt": pre_out,
            "post_sar": post_out,
            "clf_mask": clf_out,
            "loc_mask": loc_out,
            "sample_id": row["sample_id"],
            "disaster_event": row["disaster_event"],
        }
