# Model Card — GeoResQ Dual-Stream Optical–SAR Building Damage Networks (`v0.2`)

Following Mitchell et al. (2019), *Model Cards for Model Reporting*, this document specifies the architecture, training data governance, disaggregated cross-hazard evaluation, sensor-degradation robustness, physical failure modes, and ethical guardrails for the **GeoResQ `v0.2`** model family (`lite_gated_fusion_v0_2` and `dual_concat_v0_2`).

---

## 1. Model Details

- **Developer / Repository:** Rakib Islam (`https://github.com/Rakib-codee/GeoResQ`)
- **Model Version:** `v0.2.0` (`seed = 42`)
- **License:** Apache-2.0 (code & derived configurations)
- **Task:** Pixel-level multimodal building localization and 4-class damage segmentation (`0: Background`, `1: Intact`, `2: Damaged`, `3: Destroyed`; `255: Ignore`) from **paired Pre-Disaster VHR Optical RGB (`1024×1024×3`, ~0.30–0.50 m GSD)** and **Post-Disaster VHR X-Band SAR (`1024×1024×1`, Capella / Umbra VV/HH)**.
- **Primary Architectures (`configs/benchmark_v0_2.yaml`):**
  - **`lite_gated_fusion_v0_2` (`26.78M` params, `~51 MB` `fp16` export):** Asymmetric dual-stream `ResNet-18` encoders with 5-scale **`LitePriorGatedFusionBlock`** (`1/4`-channel bottleneck, spatial sigmoid optical prior gate $g_l = \sigma(\mathrm{Conv}_{3\times 3}(o_m))$ modulating SAR features, $\lvert o_m - s_m \rvert$ depthwise-separable change branch, Squeeze-and-Excitation, and learnable LayerScale $(\gamma_0 = 0.05)$ residual), a full-resolution `1×` (`32`-ch) optical boundary skip stem, and an auxiliary binary building localization `BCE + Soft Dice` head (`weight = 0.5`) alongside `CE + 0.75 × Lovasz-Softmax` (`class_weight_power = 0.15`).
  - **`dual_concat_v0_2` (`26.32M` params):** Identical encoders, `1×` boundary skip, loss, and schedule, replacing gated residual fusion with `1×1 Conv + BN + ReLU` concatenation.
  - **Unimodal Controls:** `optical_only_v0_2` (`14.43M` params) and `sar_only_v0_2` (`14.43M` params).

---

## 2. Intended Use & Out-of-Scope / Prohibited Uses

### Intended Use
- **Academic & Methodological Research** in cross-modal optical–SAR remote sensing, sensor-degradation robustness, and cross-hazard domain generalization.
- **Human-in-the-Loop Reconnaissance Triage (Research Prototype Only):** Prioritizing `~90m × 90m` tile sub-sectors (`georesq-predict`) for **mandatory inspection by trained human imagery analysts**, accompanied by pixel-wise normalized Shannon entropy maps.

### Out-of-Scope & Strictly Prohibited Uses
1. **Autonomous Emergency Dispatch or Search-and-Rescue Allocation:** Models miss `15.4%–32.0%` of building pixels on unseen geographies (`Test-104`) and misclassify collapsed seismic timber rubble (`noto-earthquake`) as intact; automated resource denial could endanger lives.
2. **Property Insurance, Tax, Financial, or Legal Adjudication:** Nadir/off-nadir satellite imagery cannot observe interior structural failure, foundation cracking, or floodwater contamination.
3. **Military, Conflict, Surveillance, or Dual-Use Targeting:** Armed-conflict imagery (`ukraine-conflict`) was explicitly excluded at dataset construction time; applying these models in conflict settings is strictly prohibited.

---

## 3. Training & Evaluation Data Governance (`BRIGHT v0.2`)

- **Upstream Source:** **BRIGHT** (*Chen et al., Earth System Science Data, 2025*, DOI: `10.5194/essd-17-6217-2025`; Zenodo DOI: `10.5281/zenodo.20072020`).
- **License & Governance Filtering:** Only the **11 open-redistribution natural/industrial disaster events** (`CC-BY-NC-SA 4.0` / open academic terms) are used. Three restricted events (`ukraine-conflict`, `myanmar-hurricane`, `mexico-hurricane`) are hard-blocked in `src/georesq/data/dataset.py` (`EXCLUDED_RESTRICTED_EVENTS`).
- **Zero Event Leakage (`294` Tiles, `308,281,344` Verified Pixels):**
  - **Train (`150` tiles across 6 events):** `turkey-earthquake` (35), `morocco-earthquake` (30), `la_palma-volcano` (25), `bata-explosion` (20), `haiti-earthquake` (20), `hawaii-wildfire` (20). Pixel distribution: `85.72%` Bg, `12.03%` Intact, `0.73%` Damaged, `1.52%` Destroyed.
  - **Validation (`40` tiles across 3 held-out events):** `beirut-explosion` (15), `libya-flood` (15), `congo-volcano` (10).
  - **Unseen Test (`104` tiles across 2 held-out events):** `noto-earthquake` (79 tiles — 100% of event) and `marshall-wildfire` (25 tiles).
  - Normalization statistics (`SUBSET_V02_*`) and inverse-frequency class weights are computed **strictly on the 150-tile `train` split**.

---

## 4. Quantitative Disaggregated Performance (`results/metrics/`)

### 4.1 Aggregate Split Performance (`benchmark_v0_2_table.csv`)

| Model | Params | Val Bg Recall | Val Loc-F1 | Val `mIoU_fg3` | Val `Destroyed` F1 | Unseen `Test-104` Loc-F1 | Unseen `Test-104` `mIoU_fg3` | Unseen Missed-Bldg FN Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `optical_only_v0_2` | 14.43M | 92.26% | 0.7468 | 0.1427 | 0.1247 | 0.4262 | 0.0756 | 36.48% |
| `sar_only_v0_2` | 14.43M | 81.34% | 0.4801 | 0.0748 | 0.0000 | 0.3629 | 0.0652 | 13.24% |
| `dual_concat_v0_2` | 26.32M | **94.18%** | 0.7391 | 0.1857 | 0.3218 | **0.6259** | **0.1502** | 32.04% |
| `lite_gated_fusion_v0_2` | 26.78M | 92.67% | **0.7487** | **0.1937** | **0.3563** | 0.6075 | 0.1332 | **15.36%** |

### 4.2 Disaggregated Performance Across All 5 Held-Out Hazard Mechanisms (`failure_analysis_v0_2_summary.json`)

| Held-Out Event (Split, Tiles) | Hazard Mechanism | `Optical-Only` `mIoU_fg3` / `Dest F1` | `Dual Concat` `mIoU_fg3` / `Dest F1` | `Lite Gated Fusion` `mIoU_fg3` / `Dest F1` | `Lite Gated` Loc-F1 |
|---|---|---:|---:|---:|---:|
| **`congo-volcano`** (`val`, 10) | Basaltic lava inundation | 0.1438 / 0.0775 | **0.3494** / **0.6705** | 0.3318 / 0.6642 | 0.8244 |
| **`marshall-wildfire`** (`test`, 25) | Suburban wind-driven wildfire | 0.1365 / 0.0371 | **0.1839** / **0.2395** | 0.1683 / 0.1445 | **0.6787** |
| **`beirut-explosion`** (`val`, 15) | Port shockwave blast | **0.1666** / **0.2127** | 0.1302 / 0.0793 | 0.1535 / 0.1680 | 0.6986 |
| **`noto-earthquake`** (`test`, 79) | M7.6 seismic timber collapse | 0.0658 / **0.0585** | **0.1409** / 0.0094 | 0.1206 / 0.0088 | 0.5908 |
| **`libya-flood`** (`val`, 15) | Flash flood / dam breach | 0.1117 / 0.0000 | **0.1187** / **0.0017** | 0.1132 / 0.0015 | **0.7845** |

### 4.3 Sensor-Degradation Stress Test (`stress_test_v0_2_summary.json`, Unseen `Test-104`)

- **5-Corruption Mean Building Localization AUDC:** `Lite Gated Fusion` ranks **#1 (`47.60%`)** ahead of `Dual Concat` (`46.21%`), `Optical-Only` (`37.80%`), and `SAR-Only` (`32.44%`).
- **Compound Storm (`Cloud + Smoke + SAR Speckle`):** `Lite Gated Fusion` achieves the highest `mIoU_fg3` AUDC (**`12.02%`** vs. `11.73%` `Dual Concat`) and retains **`85.8%`** of clean `mIoU_fg3` at `s = 0.75` (`0.1142` vs. `0.1030` `Dual Concat`) and **`+55.4%` higher `mIoU_fg3`** at `s = 1.00` (`0.0662` vs. `0.0426`).
- **Heavy Smoke & Haze (`s = 0.75`):** `Lite Gated Fusion` retains **`2.11×` higher `mIoU_fg3`** (`0.0573` vs. `0.0271`) and **`2.35×` higher `Loc-F1`** (`0.3170` vs. `0.1350`) than `Dual Concat`.
- **Isolated Single-Look SAR Speckle (`ENL = 1`):** `Dual Concat` (`0.1159` `mIoU_fg3`) outperforms `Lite Gated Fusion` (`0.0930`), while both vastly exceed `SAR-Only` (`0.0378`).
- **Complete Missing Pre-Event Optical (`optical_dropout`, `s = 1.00`):** Both dual-stream models collapse to `0.0000` when zero-blackout optical tensors are fed without modality-dropout training — a critical operational limitation documented below (`SAR-Only` maintains `0.0652`).

---

## 5. Known Physical Limitations & Failure Modes

1. **Moderate / Partial Damage (`Class 2: Damaged`) Near-Zero Recall:** Across all architectures, `Damaged` F1 is `~0.00` (`88.9%` of true `Damaged` pixels in `Test-104` route to `Intact` under `Lite Gated Fusion`). Class 2 represents only `0.73%` of training pixels and physically corresponds to blown-out facades (`beirut-explosion`) or floodwater silt (`libya-flood`) where roof geometry and X-band double-bounce reflections remain intact.
2. **Seismic Timber Rubble vs. Wildfire / Lava Destruction Asymmetry:** Multimodal fusion achieves strong `Destroyed` F1 when buildings are physically cleared to smooth ash or buried under homogeneous lava (`congo-volcano` `0.66–0.67` F1; `marshall-wildfire` `0.14–0.24` F1), because corner reflections disappear. Conversely, in `noto-earthquake`, pancaked Japanese `kawara` ceramic tile roofs remain rough radar scatterers on their exact footprints, causing `74.4%–94.1%` of `Destroyed` pixels to be predicted as `Intact`.
3. **Complete Pre-Optical Absence (`100%` Swath Dropout):** Because `v0.2` models were trained with clean paired pre-event optical imagery and rely on a `1×` optical boundary stem, zeroing out `100%` of pre-event optical channels causes complete recall failure; if no archive optical baseline exists, practitioners must fall back to `sar_only_v0_2` or train with stochastic modality dropout.
4. **Steep-Incidence Urban SAR Layover:** Off-nadir SAR double-bounce streaks adjacent to tall buildings produce `3.1%–6.0%` background false-positive building predictions.
