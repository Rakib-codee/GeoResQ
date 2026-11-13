# Reproducibility & Compute Specification (`GeoResQ v0.2`)

Every quantitative claim, table, and figure in this repository is generated deterministically (`seed = 42`) from version-controlled YAML configurations in `configs/` and split manifests in `results/metrics/`.

---

## 1. Environment & Hardware Profile

Verified cloud benchmark environment (`results/metrics/kaggle_env_audit.json`):
- **Accelerator:** NVIDIA Tesla T4 (`14.56 GB` VRAM, AMP `float16` enabled)
- **System RAM / CPU:** `31.35 GB` RAM, 4 vCPU cores
- **Software Stack:** Python `3.10–3.13`, PyTorch `2.1+` (`2.11.0+cu130` on Kaggle), `Pillow`, `numpy`, `matplotlib`, optional `rasterio` & `PyYAML`
- **Local Development / CLI Inference:** Verified on Apple M2 MacBook (`8 GB` unified RAM, macOS `mps` / `cpu`) with strict `.gitignore` enforcement so raw `.tif` imagery and `.pth` weights never enter Git history.

```bash
git clone https://github.com/Rakib-codee/GeoResQ.git
cd GeoResQ
pip install -e ".[gpu]"
python3 -m unittest discover -s tests -v
```

---

## 2. Deterministic Dataset Reconstruction (`BRIGHT v0.2` — 294 Triplets)

Selective HTTP `206` byte-range streaming extracts the exact `294` open-license GeoTIFF triplets (`150` train / `40` val / `104` unseen test across 11 events; `0` alignment mismatches) from Hugging Face `Kullervo/BRIGHT` in **~90 seconds** without downloading full multi-gigabyte archives:

```bash
python3 scripts/build_cloud_subset.py \
  --subset-key subset_v0_2 \
  --output-dir /tmp/bright_subset_v0_2 \
  --workers 8
```

Verify zero event overlap (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`):
- `results/metrics/manifest_v0_2.csv`
- `results/metrics/dataset_audit_v0_2.json`
- `results/metrics/band_stats_v0_2.json`

---

## 3. End-to-End Experiment Reproduction

### 3.1 Milestone 4 — Initial 5-Model Ablation (`v0.1`, 45 Tiles, ~10 min on T4)
```bash
python3 scripts/run_ablation.py \
  --config configs/ablation_v0_1.yaml \
  --data-dir /tmp/bright_subset_v0_1 \
  --output-dir runs/ablation_v0_1
```

### 3.2 Milestone 5 — Multi-Hazard 4-Model Benchmark (`v0.2`, 294 Tiles, ~20.2 min on T4)
```bash
python3 scripts/train_v0_2_benchmark.py \
  --config configs/benchmark_v0_2.yaml \
  --data-dir /tmp/bright_subset_v0_2 \
  --output-dir runs/benchmark_v0_2
```

| Variant ID | Total Params | Peak T4 VRAM | 22-Epoch Train Time | Full-Tile (`1024×1024`) Latency |
|---|---:|---:|---:|---:|
| `optical_only_v0_2` | 14,432,613 (`14.43M`) | 3,005 MB | 284.3 s | 179.8 ms |
| `sar_only_v0_2` | 14,432,613 (`14.43M`) | 3,172 MB | 284.7 s | 164.2 ms |
| `dual_concat_v0_2` | 26,320,229 (`26.32M`) | 3,579 MB | 305.0 s | 183.1 ms |
| `lite_gated_fusion_v0_2` | 26,781,226 (`26.78M`) | 3,824 MB | 340.1 s | 194.1 ms |

### 3.3 Milestone 6 — All-Weather & Modality Degradation Stress Test (Zero Retraining)
```bash
python3 scripts/run_stress_test.py \
  --config configs/stress_test_v0_2.yaml \
  --data-dir /tmp/bright_subset_v0_2 \
  --runs-dir runs/benchmark_v0_2 \
  --results-dir results
```

### 3.4 Milestone 7 — Cross-Event Generalization, Confusion Matrices & `fp16` Export
```bash
python3 scripts/run_failure_analysis.py \
  --benchmark-config configs/benchmark_v0_2.yaml \
  --data-dir /tmp/bright_subset_v0_2 \
  --runs-dir runs/benchmark_v0_2 \
  --results-dir results \
  --exports-dir exports
```

### 3.5 Milestone 8 — Standalone Tile Inference & Uncertainty Triage CLI
```bash
# Quick synthetic smoke-test (runs in <2s on CPU/MPS/CUDA without external files):
python3 scripts/predict_tile.py --demo-synthetic --output-dir outputs/demo_triage

# Real paired GeoTIFF inference with trained ~51 MB fp16 checkpoint:
python3 scripts/predict_tile.py \
  --pre-optical /path/to/pre_disaster.tif \
  --post-sar /path/to/post_disaster.tif \
  --checkpoint exports/lite_gated_fusion_v0_2_fp16.pth \
  --output-dir outputs/tile_triage
```
