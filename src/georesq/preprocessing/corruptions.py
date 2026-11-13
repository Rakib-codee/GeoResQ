"""Physically grounded all-weather & sensor-degradation corruptions for GeoResQ (Milestone 6).

Operates on raw uint8 GeoTIFF arrays BEFORE normalization (`pre_opt` shape [H, W, 3],
`post_sar` shape [H, W]) so downstream normalization experiences realistic sensor degradations:

1. `cloud_occlusion`: Multi-octave spatially correlated fractal cloud optical thickness
   with soft alpha blending (`A_cloud = [242, 245, 250]`) and subtle cloud-fringe shadow darkening.
2. `smoke_haze`: Koschmieder atmospheric scattering veil `I = I_0 * t(x,y) + A_smoke * (1 - t(x,y))`
   with heterogeneous brownish-gray smoke plume density (`A_smoke = [198, 188, 172]`).
3. `optical_dropout`: Contiguous archive swath / tile missing-data blackouts covering target
   area fraction `severity` (`s = 1.0` -> complete missing pre-event optical image).
4. `sar_speckle`: Multiplicative Nakagami-L / Rayleigh amplitude SAR speckle
   `A_noisy = A * sqrt(Gamma(L, 1/L))` for `L in {8.0, 4.0, 2.0, 1.0}` (`CV = 0.5227 / sqrt(L)`).
5. `compound_storm`: Simultaneous partial pre-optical cloud + smoke haze and post-SAR multi-look speckle.
"""

from __future__ import annotations

import zlib
import numpy as np

VALID_CORRUPTIONS = (
    "clean",
    "cloud_occlusion",
    "smoke_haze",
    "optical_dropout",
    "sar_speckle",
    "compound_storm",
)


def _make_tile_rng(seed: int, sample_idx: int, corruption: str, severity: float) -> np.random.Generator:
    """Create a deterministic per-(tile, corruption, severity) NumPy Generator."""
    tag = f"{seed}:{sample_idx}:{corruption}:{severity:.4f}".encode("utf-8")
    sub_seed = (zlib.crc32(tag) ^ (int(seed) * 2654435761 + int(sample_idx) * 97531)) & 0xFFFFFFFF
    return np.random.default_rng(sub_seed)


def _upsample_bilinear_2d(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Fast pure-NumPy separable 2D bilinear upsampling from (gh, gw) to (out_h, out_w)."""
    gh, gw = grid.shape
    if gh == out_h and gw == out_w:
        return grid.astype(np.float32, copy=False)

    ys = np.linspace(0.0, gh - 1.0, out_h, dtype=np.float32)
    xs = np.linspace(0.0, gw - 1.0, out_w, dtype=np.float32)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)

    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    horiz = grid[:, x0] * (1.0 - wx) + grid[:, x1] * wx
    return (horiz[y0, :] * (1.0 - wy) + horiz[y1, :] * wy).astype(np.float32)


def _fractal_field_2d(
    h: int,
    w: int,
    rng: np.random.Generator,
    octaves: tuple[tuple[int, float], ...] = ((6, 0.50), (14, 0.28), (30, 0.14), (62, 0.08)),
) -> np.ndarray:
    """Synthesize a smooth [0, 1]-normalized multi-scale random field."""
    acc = np.zeros((h, w), dtype=np.float32)
    for grid_n, weight in octaves:
        gh = max(2, min(h, grid_n))
        gw = max(2, min(w, grid_n))
        coarse = rng.standard_normal((gh, gw), dtype=np.float32)
        acc += float(weight) * _upsample_bilinear_2d(coarse, h, w)
    lo = float(acc.min())
    hi = float(acc.max())
    if hi - lo < 1e-6:
        return np.full((h, w), 0.5, dtype=np.float32)
    return (acc - lo) / (hi - lo)


def severity_to_sar_enl(severity: float) -> float:
    """Map normalized severity s in (0, 1] to physical SAR Equivalent Number of Looks (ENL)."""
    s = float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return float("inf")
    # Anchor points: s=0.25 -> L=8, s=0.50 -> L=4, s=0.75 -> L=2, s=1.00 -> L=1
    return float(2.0 ** (4.0 * (1.0 - s)))


def apply_cloud_occlusion(
    pre_opt_u8: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate realistic multi-scale cloud cover + cloud-fringe shadow on uint8 RGB."""
    s = float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return pre_opt_u8.copy()

    h, w, _ = pre_opt_u8.shape
    field = _fractal_field_2d(h, w, rng)

    if s >= 0.999:
        # Complete thick overcast deck with subtle natural cloud-top texture
        alpha = np.clip(0.94 + 0.06 * field, 0.94, 1.0).astype(np.float32)
    else:
        # Calibrate quantile threshold so cloud coverage grows smoothly with severity
        thresh = float(np.quantile(field, max(0.0, 1.0 - min(0.98, s * 1.12))))
        ramp = max(0.08, 0.24 * (1.0 - 0.5 * s))
        raw_alpha = np.clip((field - thresh) / ramp, 0.0, 1.0)
        # Scale peak opacity so s=0.25 has translucent/cumulus patches and s=0.75 has thick cores
        alpha = np.clip(raw_alpha * (0.65 + 0.35 * s), 0.0, 1.0).astype(np.float32)

    # Offset cloud shadow slightly southeast (dy=+14, dx=+14) outside thick cloud cores
    shift_y = min(16, max(2, h // 64))
    shift_x = min(16, max(2, w // 64))
    shifted_alpha = np.zeros_like(alpha)
    shifted_alpha[shift_y:, shift_x:] = alpha[:-shift_y, :-shift_x]
    shadow_mask = np.clip(shifted_alpha - alpha, 0.0, 1.0)
    shadow_factor = (1.0 - 0.32 * shadow_mask)[:, :, None]

    # Subtle internal brightness modulation across cloud tops
    cloud_tex = _fractal_field_2d(h, w, rng, octaves=((12, 0.6), (28, 0.4)))
    airlight = np.array([240.0, 243.0, 248.0], dtype=np.float32)[None, None, :]
    cloud_rgb = np.clip(airlight - 18.0 * (1.0 - cloud_tex[:, :, None]), 210.0, 255.0)

    img_f = pre_opt_u8.astype(np.float32) * shadow_factor
    a3 = alpha[:, :, None]
    out = (1.0 - a3) * img_f + a3 * cloud_rgb
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


def apply_smoke_haze(
    pre_opt_u8: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate wildfire/blast smoke & atmospheric haze via Koschmieder scattering."""
    s = float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return pre_opt_u8.copy()

    h, w, _ = pre_opt_u8.shape
    plume = 0.70 + 0.60 * _fractal_field_2d(
        h, w, rng, octaves=((5, 0.55), (11, 0.30), (23, 0.15))
    )
    beta = 3.0 * s
    transmission = np.exp(-beta * plume).astype(np.float32)[:, :, None]

    # Warm brownish-gray smoke scattering veil + slight blue-channel Rayleigh/aerosol loss
    smoke_airlight = np.array([198.0, 188.0, 172.0], dtype=np.float32)[None, None, :]
    img_f = pre_opt_u8.astype(np.float32)
    out = img_f * transmission + smoke_airlight * (1.0 - transmission)
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


def apply_optical_dropout(
    pre_opt_u8: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate missing archive sensor strips / tile blackout covering fraction `severity`."""
    s = float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return pre_opt_u8.copy()
    if s >= 0.999:
        return np.zeros_like(pre_opt_u8)

    h, w, _ = pre_opt_u8.shape
    out = pre_opt_u8.copy()
    mask = np.zeros((h, w), dtype=bool)
    target_px = int(round(s * h * w))

    # Primary contiguous swath band (either vertical or horizontal sensor strip)
    vertical = bool(rng.integers(0, 2) == 0)
    primary_frac = 0.75 * s
    if vertical:
        band_w = max(1, int(round(primary_frac * w)))
        x0 = int(rng.integers(0, max(1, w - band_w + 1)))
        mask[:, x0 : x0 + band_w] = True
    else:
        band_h = max(1, int(round(primary_frac * h)))
        y0 = int(rng.integers(0, max(1, h - band_h + 1)))
        mask[y0 : y0 + band_h, :] = True

    # Add deterministic rectangular tile dropouts until target missing fraction is reached
    for _ in range(32):
        cur_px = int(mask.sum())
        rem = target_px - cur_px
        if rem <= int(0.01 * h * w):
            break
        side = max(8, int(np.sqrt(rem)))
        bh = min(h, max(8, int(round(side * float(rng.uniform(0.7, 1.3))))))
        bw = min(w, max(8, max(1, rem // bh)))
        ry = int(rng.integers(0, max(1, h - bh + 1)))
        rx = int(rng.integers(0, max(1, w - bw + 1)))
        mask[ry : ry + bh, rx : rx + bw] = True

    out[mask] = 0
    return out


def apply_sar_speckle(
    post_sar_u8: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate multiplicative Nakagami-L amplitude SAR speckle `A * sqrt(Gamma(L, 1/L))`."""
    s = float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return post_sar_u8.copy()

    enl = severity_to_sar_enl(s)
    amp = post_sar_u8.astype(np.float32) / 255.0
    gamma_intensity = rng.gamma(shape=enl, scale=1.0 / enl, size=post_sar_u8.shape).astype(
        np.float32
    )
    noisy_amp = np.clip(amp * np.sqrt(gamma_intensity), 0.0, 1.0)
    return np.clip(np.rint(noisy_amp * 255.0), 0.0, 255.0).astype(np.uint8)


def corrupt_triplet(
    pre_opt_u8: np.ndarray,
    post_sar_u8: np.ndarray,
    corruption: str,
    severity: float,
    seed: int = 42,
    sample_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
    """Apply a named degradation at `severity in [0, 1]` deterministically to a raw triplet."""
    if corruption not in VALID_CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption={corruption!r}; expected one of {VALID_CORRUPTIONS}"
        )
    s = float(np.clip(severity, 0.0, 1.0))
    if corruption == "clean" or s <= 0.0:
        return (
            pre_opt_u8.copy(),
            post_sar_u8.copy(),
            {"opt_mae_u8": 0.0, "sar_mae_u8": 0.0, "sar_enl": None},
        )

    rng = _make_tile_rng(seed, sample_idx, corruption, s)
    pre_c = pre_opt_u8
    post_c = post_sar_u8
    enl: float | None = None

    if corruption == "cloud_occlusion":
        pre_c = apply_cloud_occlusion(pre_opt_u8, s, rng)
        post_c = post_sar_u8.copy()
    elif corruption == "smoke_haze":
        pre_c = apply_smoke_haze(pre_opt_u8, s, rng)
        post_c = post_sar_u8.copy()
    elif corruption == "optical_dropout":
        pre_c = apply_optical_dropout(pre_opt_u8, s, rng)
        post_c = post_sar_u8.copy()
    elif corruption == "sar_speckle":
        pre_c = pre_opt_u8.copy()
        post_c = apply_sar_speckle(post_sar_u8, s, rng)
        enl = round(severity_to_sar_enl(s), 4)
    elif corruption == "compound_storm":
        rng_cloud = _make_tile_rng(seed, sample_idx, "compound_cloud", s)
        rng_smoke = _make_tile_rng(seed, sample_idx, "compound_smoke", s)
        rng_sar = _make_tile_rng(seed, sample_idx, "compound_sar", s)
        pre_tmp = apply_smoke_haze(pre_opt_u8, 0.55 * s, rng_smoke)
        pre_c = apply_cloud_occlusion(pre_tmp, 0.65 * s, rng_cloud)
        post_c = apply_sar_speckle(post_sar_u8, 0.60 * s, rng_sar)
        enl = round(severity_to_sar_enl(0.60 * s), 4)

    opt_mae = float(np.mean(np.abs(pre_c.astype(np.float32) - pre_opt_u8.astype(np.float32))))
    sar_mae = float(np.mean(np.abs(post_c.astype(np.float32) - post_sar_u8.astype(np.float32))))
    return pre_c, post_c, {"opt_mae_u8": round(opt_mae, 3), "sar_mae_u8": round(sar_mae, 3), "sar_enl": enl}
