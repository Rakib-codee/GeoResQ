"""Joint Optical-SAR-Mask preprocessing and spatial augmentations for GeoResQ.

Supports verified normalization regimes:
1. 'official_imagenet': Matches official BRIGHT `bda_benchmark/dataset/imutils.py`.
2. 'modality_specific': Train-split-only empirical stats from BRIGHT v0.1 (30 train tiles).
3. 'modality_specific_v0_2': Train-split-only empirical stats from BRIGHT v0.2 (150 train tiles).
"""

from __future__ import annotations

import random

VALID_LABELS = frozenset({0, 1, 2, 3, 255})

IMAGENET_MEAN_255 = (123.675, 116.28, 103.53)
IMAGENET_STD_255 = (58.395, 57.12, 57.375)

SUBSET_OPTICAL_MEAN_01 = (0.395894, 0.374892, 0.317233)
SUBSET_OPTICAL_STD_01 = (0.250682, 0.203664, 0.188265)
SUBSET_SAR_MEAN_01 = 0.237721
SUBSET_SAR_STD_01 = 0.172405

SUBSET_V02_OPTICAL_MEAN_01 = (0.374346, 0.367878, 0.324874)
SUBSET_V02_OPTICAL_STD_01 = (0.247753, 0.210499, 0.196245)
SUBSET_V02_SAR_MEAN_01 = 0.237165
SUBSET_V02_SAR_STD_01 = 0.166060


def validate_raw_triplet(pre_opt, post_sar, mask) -> None:
    """Verify shape alignment, channel counts, and legal label values before training."""
    import numpy as np

    if pre_opt.ndim != 3 or pre_opt.shape[2] < 3:
        raise ValueError(f"Expected pre_opt shape (H, W, >=3), got {pre_opt.shape}")
    if post_sar.ndim != 2:
        raise ValueError(f"Expected raw 2D post_sar shape (H, W), got {post_sar.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D target mask shape (H, W), got {mask.shape}")

    if not (pre_opt.shape[:2] == post_sar.shape == mask.shape):
        raise ValueError(
            f"Spatial mismatch: pre={pre_opt.shape[:2]}, post={post_sar.shape}, mask={mask.shape}"
        )

    unique_vals = set(int(v) for v in np.unique(mask))
    illegal = unique_vals - VALID_LABELS
    if illegal:
        raise ValueError(
            f"Mask contains illegal pixel labels {sorted(illegal)}; allowed={sorted(VALID_LABELS)}"
        )


def joint_random_crop(
    pre_opt,
    post_sar,
    mask,
    crop_size: int = 512,
    cat_max_ratio: float = 0.75,
    ignore_index: int = 255,
    rng: random.Random | None = None,
):
    """Foreground-aware joint crop matching BRIGHT benchmark protocol (up to 10 tries)."""
    import numpy as np

    r = rng or random
    h, w = mask.shape
    pad_h = max(crop_size, h)
    pad_w = max(crop_size, w)

    if pad_h > h or pad_w > w:
        padded_pre = np.zeros((pad_h, pad_w, 3), dtype=pre_opt.dtype)
        padded_post = np.zeros((pad_h, pad_w), dtype=post_sar.dtype)
        padded_mask = np.full((pad_h, pad_w), ignore_index, dtype=mask.dtype)
        padded_pre[:h, :w, :] = pre_opt[:, :, :3]
        padded_post[:h, :w] = post_sar
        padded_mask[:h, :w] = mask
    else:
        padded_pre = pre_opt[:, :, :3]
        padded_post = post_sar
        padded_mask = mask

    best_box = (0, crop_size, 0, crop_size)
    for _ in range(10):
        y0 = r.randrange(0, pad_h - crop_size + 1)
        x0 = r.randrange(0, pad_w - crop_size + 1)
        y1, x1 = y0 + crop_size, x0 + crop_size
        best_box = (y0, y1, x0, x1)

        patch_lbl = padded_mask[y0:y1, x0:x1]
        vals, cnts = np.unique(patch_lbl, return_counts=True)
        valid_cnts = cnts[vals != ignore_index]
        if len(valid_cnts) > 1 and (valid_cnts.max() / valid_cnts.sum()) < cat_max_ratio:
            break

    y0, y1, x0, x1 = best_box
    return (
        padded_pre[y0:y1, x0:x1, :].copy(),
        padded_post[y0:y1, x0:x1].copy(),
        padded_mask[y0:y1, x0:x1].copy(),
    )


def joint_random_flip_rot(pre_opt, post_sar, mask, rng: random.Random | None = None):
    """Apply identical horizontal/vertical flips and 90-degree rotations to all 3 arrays."""
    import numpy as np

    r = rng or random
    if r.random() > 0.5:
        pre_opt = np.fliplr(pre_opt)
        post_sar = np.fliplr(post_sar)
        mask = np.fliplr(mask)
    if r.random() > 0.5:
        pre_opt = np.flipud(pre_opt)
        post_sar = np.flipud(post_sar)
        mask = np.flipud(mask)

    k = r.randrange(4)
    if k > 0:
        pre_opt = np.rot90(pre_opt, k)
        post_sar = np.rot90(post_sar, k)
        mask = np.rot90(mask, k)

    return (
        np.ascontiguousarray(pre_opt),
        np.ascontiguousarray(post_sar),
        np.ascontiguousarray(mask),
    )


def normalize_multimodal(
    pre_opt,
    post_sar,
    norm_mode: str = "official_imagenet",
    stack_sar_3ch: bool = True,
):
    """Normalize Optical & SAR and transpose to CHW float32 arrays ready for PyTorch."""
    import numpy as np

    pre_f = pre_opt[:, :, :3].astype(np.float32)
    post_f = post_sar.astype(np.float32)

    if norm_mode == "official_imagenet":
        mean_arr = np.array(IMAGENET_MEAN_255, dtype=np.float32)
        std_arr = np.array(IMAGENET_STD_255, dtype=np.float32)
        pre_norm = (pre_f - mean_arr) / std_arr

        post_3 = np.stack((post_f, post_f, post_f), axis=-1)
        post_norm = (post_3 - mean_arr) / std_arr
        if not stack_sar_3ch:
            post_norm = post_norm[:, :, :1]

    elif norm_mode in ("modality_specific", "modality_specific_v0_2"):
        if norm_mode == "modality_specific_v0_2":
            o_m, o_s = SUBSET_V02_OPTICAL_MEAN_01, SUBSET_V02_OPTICAL_STD_01
            s_m, s_s = SUBSET_V02_SAR_MEAN_01, SUBSET_V02_SAR_STD_01
        else:
            o_m, o_s = SUBSET_OPTICAL_MEAN_01, SUBSET_OPTICAL_STD_01
            s_m, s_s = SUBSET_SAR_MEAN_01, SUBSET_SAR_STD_01

        opt_mean = np.array(o_m, dtype=np.float32)
        opt_std = np.array(o_s, dtype=np.float32)
        pre_norm = ((pre_f / 255.0) - opt_mean) / opt_std

        sar_1 = ((post_f / 255.0) - s_m) / s_s
        if stack_sar_3ch:
            post_norm = np.stack((sar_1, sar_1, sar_1), axis=-1)
        else:
            post_norm = sar_1[:, :, None]
    else:
        raise ValueError(f"Unsupported norm_mode='{norm_mode}'")

    pre_chw = np.transpose(pre_norm, (2, 0, 1)).astype(np.float32)
    post_chw = np.transpose(post_norm, (2, 0, 1)).astype(np.float32)
    return np.ascontiguousarray(pre_chw), np.ascontiguousarray(post_chw)
