# Changelog

All notable changes and verified experimental milestones for **GeoResQ** are documented in this file.

## [0.2.0] - 2026-11-14

### Added
- **Multi-Hazard `BRIGHT v0.2` Cloud Subset (`configs/dataset.yaml`, `scripts/build_cloud_subset.py`):**
  - Expanded open-license benchmark coverage to **294 aligned GeoTIFF triplets (`308,281,344` pixels)** across all **11 open-redistribution disasters** (`train = 150` across 6 events, `val = 40` across 3 events, `test = 104` across `noto-earthquake` [79] and `marshall-wildfire` [25]) with strict zero event leakage and parallel HTTP `206` byte-range extraction.
- **Lightweight Cross-Modal Architecture (`src/georesq/models/fusion.py`, `configs/benchmark_v0_2.yaml`):**
  - Introduced `LitePriorGatedFusionBlock` (`1/4` channel bottleneck, spatial sigmoid optical prior gate, depthwise-separable change branch, Squeeze-and-Excitation, and learnable LayerScale $(\gamma_0 = 0.05)$ residual) adding only `+0.46M` parameters (`+1.7%`) over `Dual Concat`.
  - Added optional `1×` (`32`-ch) full-resolution boundary skip stem and `BuildingLocBCEDiceLoss` auxiliary head (`src/georesq/training/losses.py`), lifting validation background recall from `49.8%` (`v0.1`) to `92.7%–94.2%` and cutting unseen `Test-104` missed-building false negatives in half (`15.36%` vs. `32.04%`).
- **All-Weather & Modality Degradation Stress-Test Suite (`src/georesq/preprocessing/corruptions.py`, `scripts/run_stress_test.py`):**
  - Deterministic post-training stress evaluation across 5 severities of `cloud_occlusion`, `smoke_haze` (Koschmieder scattering), `optical_dropout`, `sar_speckle` (Nakagami speckle with $L \in \{8, 4, 2, 1\}$ looks), and `compound_storm`, plus LayerScale `|gamma|` and spatial `opt_prior_gate` diagnostics.
- **Cross-Hazard Generalization & Physical Failure Analysis (`scripts/run_failure_analysis.py`, `src/georesq/evaluation/metrics.py`):**
  - Disaggregated evaluation across all 5 held-out disaster mechanisms (`144` tiles), `4×4` confusion error routing, per-tile audit CSV (`results/metrics/per_tile_audit_v0_2.csv`), automated 5-case failure gallery, and compact `~51 MB` `fp16` inference checkpoint export.
- **Standalone Uncertainty Triage CLI & Packaging (`src/georesq/cli/predict.py`, `pyproject.toml`, `docs/`):**
  - Installable `georesq-predict` (`scripts/predict_tile.py`) CLI supporting `cuda`, Apple Silicon `mps`, and `cpu`, outputting predicted damage masks, normalized Shannon entropy heatmaps, ranked sector priority tables, and Responsible-AI disclaimers, backed by `MODEL_CARD.md`, `REPRODUCIBILITY.md`, and GitHub Actions CI.

## [0.1.0] - 2026-11-13

### Added
- Initial `BRIGHT v0.1` 45-tile zero-leakage pipeline, official 6-channel `BaselineUNet` reproduction, and controlled 5-model modality/fusion ablation (`results/metrics/ablation_v0_1_table.csv`) diagnosing dense `3×3` fusion overfitting and stride-2 rooftop merging.
