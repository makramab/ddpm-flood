# DDPM Flood — V3 Handover Document

**Date**: March 5, 2026
**Author**: Afrai (with Claude Code assistance)
**Purpose**: Comprehensive handover for colleagues picking up this project
**Status**: V2 complete (training + validation + figures). V3 options under consideration.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Has Been Done](#2-what-has-been-done)
3. [Current Results Summary](#3-current-results-summary)
4. [Key Limitations Discovered](#4-key-limitations-discovered)
5. [Available Datasets — Full Inventory](#5-available-datasets--full-inventory)
6. [Options for V3](#6-options-for-v3)
7. [Recommended Direction](#7-recommended-direction)
8. [Repository Map](#8-repository-map)
9. [Environment and Setup](#9-environment-and-setup)
10. [Document Index](#10-document-index)

---

## 1. Project Overview

### Goal

Adapt Wang et al. 2022's DDPM (Denoising Diffusion Probabilistic Model, originally for molecular dynamics) to predict urban flood surge patterns from ADCIRC simulations. This is a PhD application assessment for Prof. Yuki Miura's CERA Lab at NYU Tandon.

### The Wang et al. Framework

Wang et al. train a 1D U-Net DDPM on molecular dynamics trajectories. Their conditioning variable is **instantaneous kinetic temperature** (a naturally fluctuating scalar). The model learns P(molecular coordinates | temperature) via an inpainting mechanism — the temperature column is never noised during diffusion, while the coordinate columns are generated through reverse diffusion.

### Our Adaptation

We replace:
- **Temperature** → **θ** (peak surge at a reference tide gauge — a scalar severity index)
- **Molecular coordinates** → **Surge values at 20 mesh nodes** (within spatial patches)
- **MD trajectory frames** → **ADCIRC hurricane simulation outputs**

The model learns: **P(surge pattern | θ, spatial location)**

### Two-Region Study

| Region | Role | Reference Gauge | Training Source |
|--------|------|----------------|-----------------|
| **Outer Banks (OBX)** | Primary proof-of-method | Hatteras (lat 35.0-35.4) | 183 Danso + CERA scenarios |
| **NYC** | Application case | Battery (lat 40.69-40.72) | 191 Danso + CERA scenarios |

Full planning rationale: [CURRENT_PLAN.md](CURRENT_PLAN.md) (v3, Feb 27)

---

## 2. What Has Been Done

### V1: Baseline (Feb 27-28)

**Plan**: [CURRENT_PLAN.md](CURRENT_PLAN.md)

- Preprocessed 193 ADCIRC scenarios (180 Danso synthetic + 13 CERA historical) into training matrices
- Training row format: `[θ, surge₁, ..., surge₂₀]` (21 columns, `UNMASK_NUMBER=1`)
- Trained OBX model (200K steps, MPS) and NYC model (100K steps, MPS)
- **Result**: Model learned correct conditional mean E[surge|θ] but produced **spatially flat predictions** (R² negative everywhere). It couldn't distinguish between patches because it had no location information.

### V2: Spatial Conditioning (Feb 28 - Mar 1)

**Plan**: [CURRENT_PLAN_V2.md](CURRENT_PLAN_V2.md)
**Results**: [RESULTS_V2.md](RESULTS_V2.md)

- Added 3 spatial conditioning columns: patch centroid latitude, longitude, and mean bathymetric depth
- Training row format: `[θ, lat, lon, depth, surge₁, ..., surge₂₀]` (24 columns, `UNMASK_NUMBER=4`)
- Retrained OBX on Google Colab A100 (50K steps, batch_size=2048, ~45 min)
- **Result**: Massive improvement for in-distribution Danso storms (R² = 0.95), but **cross-storm generalization failed** — the model cannot predict CERA historical storms (Arthur) even when θ is within training range.

### Diagnostic Analysis (Mar 1)

**Script**: [diagnose_arthur.py](diagnose_arthur.py)

- Confirmed that two storms with identical θ values produce fundamentally different spatial surge patterns
- Arthur (CERA, θ=2.05m) vs closest Danso scenario (θ=2.04m): spatial correlation only **0.31**
- The model learned Danso-specific spatial relationships, not universal surge physics

---

## 3. Current Results Summary

### V2 Validation — OBX Dorian (Danso, held-out)

| θ (m) | R² | RMSE (m) | Category |
|-------|-----|----------|----------|
| 0.701 | **0.949** | 0.062 | In-distribution |
| 0.702 | **0.952** | 0.060 | In-distribution |
| 0.766 | **0.947** | 0.065 | In-distribution |
| 2.781 | -1.451 | 0.816 | Near/above training max |
| 3.357 | -1.256 | 0.958 | Extrapolation |

### V2 Validation — OBX Arthur (CERA, held-out)

| θ (m) | R² | RMSE (m) | Bias (m) |
|-------|-----|----------|----------|
| 2.049 | **-2.568** | 0.602 | +0.329 |

### V1 → V2 Improvement

| Scenario | V1 R² | V2 R² | V1 RMSE | V2 RMSE |
|----------|-------|-------|---------|---------|
| Dorian low-θ (0.70m) | -0.089 | **+0.950** | 0.289m | 0.062m |
| Arthur (2.05m) | -1.269 | -2.568 | 0.480m | 0.602m |
| Dorian high-θ (3.36m) | -0.076 | -1.530 | 0.661m | 0.960m |

**Takeaway**: V2 works excellently as a **within-ensemble surrogate** (Danso synthetic storms at in-distribution θ). It fails at cross-storm generalization and extrapolation.

Full metrics with all 10 validation scenarios: [results/obx/generated/validation_metrics.json](results/obx/generated/validation_metrics.json)

### Generated Figures

All in [results/obx/figures/](results/obx/figures/):

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig_dorian_surge_comparison.png` | 3-panel: ADCIRC truth vs DDPM prediction vs error (Dorian θ=0.70m) |
| 2 | `fig_dorian_scatter.png` | Scatter plot pred vs truth, R²=0.95 |
| 3 | `fig_metrics_summary.png` | Bar chart of R²/RMSE across all 10 scenarios |
| 4 | `fig_dorian_uncertainty.png` | Spatial uncertainty map (mean σ=0.059m) |
| 5 | `fig_arthur_surge_comparison.png` | 3-panel comparison (Arthur, failure case) |
| 6 | `fig_arthur_scatter.png` | Scatter showing systematic over-prediction |

---

## 4. Key Limitations Discovered

These are documented in detail in [RESULTS_V2.md §4](RESULTS_V2.md).

### 4.1 θ Is Insufficient as Sole Storm Descriptor

The scalar θ (peak surge at one gauge) does not uniquely determine the spatial flood pattern. Two storms with θ ≈ 2.05m produce spatial correlation of only 0.31. This is a fundamental issue — θ is a downstream point measurement, not a state variable like Wang et al.'s temperature.

### 4.2 Training Data Is 100% Danso Synthetic Storms

All training data comes from Danso's parametric Holland model storms. The model learns Danso-specific spatial patterns and cannot generalize to real historical storms (CERA). At θ ≈ 2.0m, Danso storms produce 63% of nodes > 1m surge, while Arthur (real) produces only 18%.

### 4.3 θ Distribution Is Skewed

82% of training scenarios have θ < 1.0m. Only 6% have θ > 2.0m. The model is poorly trained for high-surge events.

### 4.4 Extrapolation Worsened in V2

V1 failed gracefully (predicted flat/safe values, R² ≈ -0.08). V2 fails **confidently** (predicts high surge in the Danso spatial pattern, R² ≈ -1.5). Spatial conditioning made the model more confident at all θ, including where it shouldn't be.

---

## 5. Available Datasets — Full Inventory

### Currently Used

| Dataset | Location | What's Used | Scenarios |
|---------|----------|------------|-----------|
| Danso et al. ADCIRC | `data/adcirc/` (20 storms × 9 variants) | `maxele.63.nc` — peak surge per node | 180 |
| CERA HSOFS | `data/cera/` (13 of 31 files) | `maxele.63.nc` — peak surge per node | 13 |

Each `maxele.63.nc` file also contains:
- `depth` (bathymetric depth per node) — **used in V2** for spatial conditioning
- `time_of_zeta_max` (time of peak surge per node) — **not used**
- `x`, `y` (node coordinates) — **used** for region extraction
- Boundary node arrays (`nbvv`, `ibtype`) — **not used**

### Available But Unused (Already Downloaded)

| Dataset | Location | Size | Contents |
|---------|----------|------|----------|
| **DeepSurge RAFT** | `data/deepsurge/raft_cmip_max_surge_hist_EC-Earth3.nc` | 840 MB | 50,000 synthetic hurricane tracks × 4,199 coastal nodes. Max surge per track per node. ~40,000 valid tracks. Covers full US Atlantic coast but sparse: only 30 nodes in OBX, 3 in NYC. |
| **NACCS Geodatabase** | `data/naccs/NACCS.gdb` | ~3 MB | 1,050 synthetic storm tracks with physical parameters (central pressure, heading, radius of max winds, translation speed) + 16,326 save points with peak water levels. 1,388 save points in NYC bbox, only 3 in OBX. **Note**: Only stores worst-case surge per station (2 values), not per-storm results. |
| **Sandy Validation** | `data/validation/` | ~92 MB | Battery tide gauge time series (`battery_tide_sandy.csv`), 347 USGS high water marks (`sandy_hwm.csv`), 69 sensor records (`sandy_instruments.csv`), official inundation zone shapefile. |
| **CERA non-HSOFS** | `data/cera/` (18 of 31 files) | ~2 GB | Different ADCIRC meshes (not compatible with Danso HSOFS data). Cannot be combined with current training data. |

### Available on TACC (Not Downloaded)

The CERA dataset on TACC (~3.1 TB total) contains additional ADCIRC output files for each of the 13 HSOFS storms:

| File | Contents | Size (per storm) |
|------|----------|------------------|
| `fort.63.nc` | **Full time series** of water elevation at all nodes (every output timestep) | Multi-GB |
| `fort.64.nc` | Time series of depth-averaged water velocity | Multi-GB |
| `maxvel.63.nc` | Maximum velocity per node | ~50 MB |
| `maxwvel.63.nc` | Maximum wind velocity per node | ~50 MB |
| `swan_HS_max.63.nc` | Maximum significant wave height (ADCIRC+SWAN coupled) | ~50 MB |

Download requires TACC MFA credentials. See [HANDOVER_V2.md §6.3](HANDOVER_V2.md) for the download script and TACC access details (username `makramab` at `data.tacc.utexas.edu`).

### Danso fort.63.nc (Not Available)

The Danso simulations must have produced `fort.63.nc` time-series files during the original ADCIRC runs, but only `maxele.63.nc` was published on Zenodo. These files may still exist on the authors' compute cluster. Contacting Danso directly is the best path to obtaining them. If available, this would provide 180 storms × ~300 timesteps of time-series data.

---

## 6. Options for V3

### Option A: Time-Series Conditioning (fort.63.nc)

**Effort**: Medium (TACC download + new preprocessing mode)
**Impact**: Very High
**New data**: Yes — download ~30-100 GB from TACC

Instead of using peak surge (one θ per storm), use the full surge time series. At each timestep `t`, the reference gauge reads θ(t) and every other node has a surge value — producing one complete spatial snapshot per timestep.

This makes the Wang et al. analogy **structurally exact**: just as their instantaneous kinetic temperature fluctuates at every MD frame, θ(t) rises and falls throughout each storm, producing a naturally continuous conditioning distribution.

**The 13 CERA HSOFS storms provide:**

| Storm | Peak θ (m) | Category |
|-------|-----------|----------|
| DEBBY (2012) | 0.54 | Low |
| FIONA (2022) | 0.56 | Low |
| MARIA (2017) | 0.68 | Low |
| ALBERTO (2018) | 0.69 | Low |
| FRANCES (2004) | 0.70 | Low |
| OMAR (2008) | 0.75 | Low-Med |
| ANDREA (2013) | 0.80 | Low-Med |
| NICOLE (2022) | 0.86 | Medium |
| EARL (2010) | 1.18 | Medium |
| MICHAEL (2018) | 1.23 | Medium |
| SANDY (2012) | 1.62 | High |
| ARTHUR (2014) | 2.05 | High |
| IRENE (2011) | 2.19 | High |

With time-series, each storm contributes ~200-500 timesteps. The effective θ distribution covers 0 to 2.19m continuously with no gaps. Total training rows: 12 storms (holding one out) × ~300 timesteps × 1,974 patches ≈ **7M rows**.

**Storm diversity at each θ level:**
- θ = 0.5m: 13 storms contribute (all storms pass through this level)
- θ = 1.0m: 5 storms contribute
- θ = 1.5m: 3 storms (Sandy, Arthur, Irene)
- θ = 2.0m: 2 storms (Arthur, Irene)

**Key advantage over V2**: All training data comes from real historical storms (CERA), not synthetic parametric storms (Danso). The cross-storm generalization failure from V2 (Danso → CERA) is eliminated because both training and validation are real storms.

**Validation strategy considerations**: With only 13 storms, the choice of holdout matters:
- Holding out Arthur (2.05m) removes one of only 2 storms above θ > 2.0m from training
- **Alternative**: Use all 13 CERA storms for training, validate against the 180 Danso maxele scenarios. This tests "can a model trained on real storms predict synthetic storms?" — the reverse of V2, and you don't waste scarce CERA data
- **Alternative**: Leave-one-out cross-validation across all 13 storms (train 13 models, each holding out one storm)

**If Danso fort.63 files can be obtained** (by contacting the authors): 180 storms × ~300 timesteps would provide ~54,000 conditioning snapshots, yielding ~100M+ training rows with excellent storm diversity at all θ levels. This would be the strongest possible configuration.

**Pros:**
- Exact structural match to Wang et al. framework
- Naturally continuous θ (no noise injection needed)
- Trains on real historical storms, not synthetic ones
- 20-40x more training data than V2

**Cons:**
- Requires ~30-100 GB download from TACC
- Only 13 storms — limited storm diversity at high θ (mitigated by each storm producing hundreds of distinct spatial snapshots)
- θ ceiling at 2.19m (Irene peak) — cannot train on θ > 2.19m
- Preprocessing pipeline needs a new mode to iterate over timesteps

---

### Option B: `time_of_zeta_max` as Additional Conditioning

**Effort**: Very Low (no new data, minor code change)
**Impact**: Moderate
**New data**: No

Every `maxele.63.nc` file already contains `time_of_zeta_max` — the timestamp (in seconds) when peak surge occurred at each node. This is completely unused.

Add the **time of peak at the reference gauge** as a conditioning variable:
```
[θ, t_peak_ref, lat, lon, depth, surge₁...surge₂₀]   # UNMASK_NUMBER = 5
```

Two storms with the same θ but different timing produce different spatial patterns. A fast-moving storm peaks early and symmetrically; a slow one peaks late and asymmetrically. This gives the model partial information to distinguish storm types.

**Pros:**
- Zero effort — data already exists in every file
- Trivial code change (one more conditioning column)
- Could be combined with any other option

**Cons:**
- Unclear how much discriminative power one extra scalar provides
- Still fundamentally limited by having only 193 discrete scenarios
- Doesn't address the Danso-vs-CERA generalization gap

---

### Option C: Sea Level Rise as Explicit Conditioning

**Effort**: Very Low (no new data, minor preprocessing change)
**Impact**: Moderate
**New data**: No

The Danso dataset has 3 sea level scenarios per storm (`reference`, `+0.44m`, `+0.74m`). Currently these are treated as independent scenarios with different θ values. Instead, make SLR an explicit conditioning variable:
```
[θ, SLR, lat, lon, depth, surge₁...surge₂₀]   # UNMASK_NUMBER = 5
```

At generation time, you could ask: "what would this storm look like at +0.30m SLR?" — an interpolation the model has never seen explicitly but could learn.

**Pros:**
- Operationally relevant (climate adaptation planning)
- Trivial to implement
- Disentangles θ and SLR effects (currently conflated)

**Cons:**
- Only 3 discrete SLR values — limited range
- Doesn't address the fundamental θ-insufficiency problem
- Only applies to Danso data (CERA storms have one SLR level)

---

### Option D: DeepSurge Pre-Training / Multi-Point θ

**Effort**: Medium
**Impact**: High
**New data**: No (840 MB file already downloaded)

The DeepSurge RAFT dataset contains ~40,000 valid synthetic hurricane tracks with max surge at 4,199 coastal nodes. The nodes are sparse (30 in OBX, 3 in NYC) but the storm count is massive.

Possible approaches:
1. **Multi-point θ conditioning**: Use surge at 2-3 DeepSurge nodes as conditioning instead of one reference gauge. Three surge values at different locations implicitly encode storm direction and size.
2. **Coarse-resolution standalone DDPM**: Train a DDPM directly on 40K × 4,199. Each "patch" is a single node. With 40K storms the θ distribution is extremely rich.
3. **Pre-training**: Learn a coarse surge-conditioning relationship from DeepSurge, then fine-tune on the dense ADCIRC data.

**Pros:**
- 40,000 storms provides incredible conditioning diversity
- Directly addresses the θ-insufficiency problem (multi-point conditioning)
- Already downloaded

**Cons:**
- Very sparse spatial resolution (30 nodes for all of OBX)
- DeepSurge nodes don't map to ADCIRC HSOFS nodes
- The coarse-to-fine transfer is non-trivial
- DeepSurge uses a different storm generation method than Danso/CERA

---

### Option E: NACCS Storm Parameters as Conditioning

**Effort**: Medium-High
**Impact**: High (if Danso storm parameters can be extracted)
**New data**: Partially (need to extract parameters from Danso forcing files or literature)

The NACCS geodatabase contains 1,050 synthetic storm tracks with explicit physical parameters: central pressure (28-98 mb), heading (-60° to +40°), radius of max winds (25-174 nm), translation speed (12-88 kt).

The idea: if we can extract equivalent parameters for the Danso storms, we can add them as conditioning:
```
[θ, pressure, heading, Rmax, speed, lat, lon, depth, surge₁...surge₂₀]
```

This directly addresses the θ-insufficiency — two storms with the same θ but different heading/pressure/Rmax would produce different spatial patterns, and the model would have the information to distinguish them.

**Challenge**: Extracting storm parameters for the 180 Danso scenarios requires either:
- Reading ADCIRC forcing/input files (which may not be in the Zenodo download)
- Looking up parameters from Danso's paper or supplementary materials
- Cross-referencing with HURDAT2 for the 20 named storms

For OBX specifically: only 3 NACCS save points in the OBX bounding box, so NACCS cannot be used as a direct training data source for OBX.

**Pros:**
- Directly addresses the root cause (θ insufficiency)
- Physically principled conditioning
- NACCS validates which parameters matter

**Cons:**
- Extracting Danso storm parameters is a research task
- More conditioning dimensions may require more training data
- At generation time, users must specify storm parameters (less practical operationally)

---

### Option F: Multi-Target DDPM (Surge + Timing)

**Effort**: Low-Medium
**Impact**: Moderate
**New data**: No

Instead of predicting only 20 surge values per patch, predict 40 values: 20 surge values + 20 time-of-peak values. Uses the `time_of_zeta_max` field from maxele files.

```
[θ, lat, lon, depth, surge₁...surge₂₀, time₁...time₂₀]   # 44 columns
```

**Pros:**
- Richer output — "when does each area flood?" not just "how much"
- No new data needed
- Doubles output dimensionality at moderate compute cost

**Cons:**
- Doesn't address cross-storm generalization
- Timing and surge may have different noise characteristics
- Increases sequence length significantly

---

### Option G: Boundary-Aware Features

**Effort**: Low
**Impact**: Low
**New data**: No

The maxele files contain boundary node lists (`nbvv`) and types (`ibtype` — types 20, 21, 24 = different ADCIRC boundary conditions). Compute per-patch distance to nearest coastline boundary, or fraction of boundary nodes.

**Pros:** Free information. **Cons:** Minor improvement at best.

---

## 7. Recommended Direction

### Primary: Option A (Time-Series Conditioning)

This is the strongest next step because it:

1. **Eliminates the discrete-θ problem** — the core weakness since V1
2. **Makes the Wang et al. analogy exact** — strongest possible scientific framing
3. **Trains on real historical storms** — eliminates the Danso-vs-CERA generalization gap
4. **Provides 20-40x more training data** — from ~360K rows to ~7M rows

The 13 CERA storms span θ from 0.54m to 2.19m with continuous coverage when using time-series data. While storm diversity thins above θ=1.5m (only 3 storms), each storm contributes hundreds of distinct spatial snapshots at different phases (rising vs falling surge, different wavefront positions), so the effective diversity is much higher than 13.

**If you can contact Danso** and obtain fort.63.nc files for their 20 storms, this becomes dramatically more powerful (180 storms × ~300 timesteps).

### Secondary: Combine Options B + C (Low-Hanging Fruit)

While setting up Option A, Options B (timing) and C (SLR conditioning) can be tested on the existing V2 infrastructure with minimal code changes. They won't solve the fundamental problems but may provide incremental improvements that are worth reporting.

### Validation Strategy for Option A

We recommend **using all 13 CERA storms for time-series training and validating against the 180 Danso maxele scenarios**. This:
- Doesn't waste any scarce CERA time-series data on holdout
- Provides 180 validation targets at θ values from 0.45 to 2.88m
- Tests "can a model trained on real storms predict synthetic storms?" — the reverse of V2
- If it works, validates that the model learned general surge physics, not just CERA-specific patterns

---

## 8. Repository Map

### Scripts

| File | Purpose | Status |
|------|---------|--------|
| [preprocess.py](preprocess.py) | V2 preprocessing — reads maxele files, clusters nodes into patches, builds 24-column training matrices | Complete |
| [train_flood.py](train_flood.py) | Local DDPM training (`UNMASK_NUMBER=4`, MPS device) | Complete |
| [train_flood_colab.py](train_flood_colab.py) | Colab A100 training (batch_size=2048, 50K steps) | Complete |
| [generate_flood.py](generate_flood.py) | Sample generation and validation (validate/sweep/generate modes) | Complete |
| [generate_flood_colab.py](generate_flood_colab.py) | Colab A100 generation (batch_size=8192) | Complete |
| [visualize_results.py](visualize_results.py) | Publication-quality figures (runs local inference for missing scenarios) | Complete |
| [visualize_flood.py](visualize_flood.py) | Earlier visualization script (Miura-style flood maps) | Complete |
| [diagnose_arthur.py](diagnose_arthur.py) | Arthur failure diagnostic — spatial correlation analysis | Complete |
| [main.py](main.py) | Placeholder entry point | Minimal |

### DDPM Library

| Path | Contents |
|------|----------|
| `DDPM_REMD/` | Wang et al.'s original DDPM code (gitignored — clone separately) |
| `DDPM_REMD/denoising_diffusion_pytorch/` | Core library: `GaussianDiffusion`, `Unet1D`, etc. **Unmodified**. |
| `DDPM_REMD/run_training.py` | Wang et al.'s training script (reference, not used directly) |
| `DDPM_REMD/gen_sample.py` | Wang et al.'s generation script (reference) |
| `DDPM_REMD/traj_AIB9/` | Wang et al.'s example molecular data |

To set up the DDPM library:
```bash
git clone https://github.com/wangdingyan/DDPM_REMD.git
```

### Data (gitignored — large files)

```
data/
├── adcirc/                    # Danso et al. — 20 storms × 9 variants = 180 scenarios
│   ├── Delta/                 # Type A layout: {Storm}/{sea_level}/{forcing}/maxele.63.nc
│   ├── Dorian/                # (9 storms: Delta, Dorian, Florence, Ian, Ida, Irma, Laura, Matthew, Michael)
│   ├── Bonnie/                # Type B layout: {Storm}/hot_start/{sea_level}/{forcing}/maxele.63.nc
│   └── ...                    # (11 storms: Bonnie, Charley, Floyd, Fran, Frances, Harvey, Ike, Isabel, Jeanne, Lili, Opal)
├── cera/                      # CERA HSOFS (13) + non-HSOFS (18) — identify HSOFS by node count = 1,813,443
│   ├── 2014_01_ARTHUR_maxele.63.nc
│   ├── 2012_18_SANDY_maxele.63.nc
│   └── ...
├── deepsurge/                 # DeepSurge RAFT — 50K tracks × 4,199 nodes
│   └── raft_cmip_max_surge_hist_EC-Earth3.nc
├── naccs/                     # NACCS geodatabase — 1,050 storm tracks + 16,326 save points
│   └── NACCS.gdb
├── validation/                # Sandy validation data
│   ├── battery_tide_sandy.csv
│   ├── sandy_hwm.csv
│   ├── sandy_instruments.csv
│   ├── sandy_inundation_zone/
│   └── sandy_inundation_zone.geojson
└── processed/                 # Preprocessed .npy matrices (committed to git)
    ├── obx/
    │   ├── train_data.npy          # (361,242 × 24) — V2 format
    │   ├── val_dorian_data.npy     # (17,766 × 24) — 9 Danso Dorian scenarios
    │   ├── val_arthur_data.npy     # (1,974 × 24) — 1 CERA Arthur scenario
    │   ├── patch_assignments.pkl   # {patch_id: [20 node indices]}
    │   ├── patch_centroids.npy     # (1,974 × 3) — [lat, lon, depth] per patch
    │   ├── node_coords.npy         # (39,494 × 2) — valid node lat/lon
    │   ├── node_depths.npy         # (39,494,) — valid node depths
    │   ├── node_indices.npy        # HSOFS mesh indices for mapping
    │   └── metadata.json
    ├── nyc/                        # Same structure, 219 patches
    └── preprocessing_report.json
```

### Results

```
results/
├── obx/                           # V2 OBX model (current)
│   ├── model-0.pt ... model-20.pt # 21 checkpoints (50K steps, A100)
│   ├── loss_log.csv               # Training loss curve
│   ├── generated/
│   │   ├── validation_metrics.json
│   │   ├── pred_dorian_mean.npy   # (1,974 × 20) per-patch predictions
│   │   ├── pred_dorian_std.npy
│   │   ├── pred_dorian_nodes.npy  # (58,645,) node-level stitched
│   │   ├── truth_dorian_nodes.npy
│   │   ├── pred_arthur_mean.npy
│   │   ├── pred_arthur_std.npy
│   │   ├── pred_arthur_nodes.npy
│   │   └── truth_arthur_nodes.npy
│   └── figures/
│       ├── fig_dorian_surge_comparison.png
│       ├── fig_dorian_scatter.png
│       ├── fig_dorian_uncertainty.png
│       ├── fig_metrics_summary.png
│       ├── fig_arthur_surge_comparison.png
│       └── fig_arthur_scatter.png
├── obx_v1/                        # V1 OBX backup (41 checkpoints)
└── nyc_v1/                        # V1 NYC backup (20 checkpoints)
```

---

## 9. Environment and Setup

### Python Environment

```bash
# Python 3.11 via uv
uv run python script.py          # Run any script
uv sync                           # Install dependencies from pyproject.toml
```

### Key Dependencies

- `torch` (with MPS for Apple Silicon, or CUDA for Colab)
- `netCDF4` — for reading ADCIRC files (do NOT use xarray)
- `scikit-learn` — KMeans clustering
- `matplotlib` — visualization
- `numpy`

### Device Notes

- **Local (Apple Silicon)**: Use `torch.device("mps")`, set `pin_memory=False` in DataLoaders
- **Colab A100**: Use `torch.device("cuda")`, `pin_memory=True`
- MPS produces identical results to CUDA, just slower (~6x for training)

### Critical Technical Notes

- **ADCIRC fill values**: Filter with `(zeta > 1e10) | (zeta < -1e10) | (zeta <= -99999)`
- **netCDF4 masked arrays**: Always wrap with `np.asarray()` to convert
- **CERA HSOFS identification**: Check `ds.dimensions["node"].size == 1_813_443`
- **Type B sea levels**: Directory is `ref` (not `reference`), but SLR dirs keep names (`slr_0.44m`, `slr_0.74m`)
- **Harvey**: Exclude `old/` and `Vmax_10_old/` directories

### Running V2 Pipeline End-to-End

```bash
# 1. Preprocessing (both regions, ~5 min)
uv run python preprocess.py

# 2. Training — local (OBX ~3h on MPS)
uv run python train_flood.py --region obx

# 2. Training — Colab (OBX ~45min on A100)
# Upload train_flood_colab.py + data/processed/obx/ to Colab, then run

# 3. Generation/Validation
uv run python generate_flood.py --region obx --mode validate

# 4. Figures
uv run python visualize_results.py
```

---

## 10. Document Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **This file** ([HANDOVER_V3.md](HANDOVER_V3.md)) | Comprehensive overview and handover | Start here |
| [CURRENT_PLAN.md](CURRENT_PLAN.md) | V1 plan — two-region strategy, experiment design, Wang et al. framing | For understanding the original approach and scientific framing |
| [CURRENT_PLAN_V2.md](CURRENT_PLAN_V2.md) | V2 plan — spatial conditioning fix, architectural details | For understanding V2 implementation details |
| [RESULTS_V2.md](RESULTS_V2.md) | V2 results, limitations, and future options | For understanding what worked and what didn't |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Earlier experiment design (data inventory, validation strategy) | For full data source documentation |
| [HANDOVER_V2.md](HANDOVER_V2.md) | Previous handover (data download instructions, TACC access) | For TACC credentials and download scripts |
| [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) | Original project handoff (data sources, URLs, validation datasets) | For data source URLs and download links |
| [PROGRESS.md](PROGRESS.md) | Day-by-day progress log | For chronological history |

### Reading Order for a New Colleague

1. **This file** — understand where we are and where we're going
2. [RESULTS_V2.md](RESULTS_V2.md) — understand what the current model can and cannot do
3. [CURRENT_PLAN_V2.md](CURRENT_PLAN_V2.md) §3-5 — understand the root cause and V2 fix
4. [CURRENT_PLAN.md](CURRENT_PLAN.md) §12 — understand the time-series approach (fort.63.nc)
5. [HANDOVER_V2.md](HANDOVER_V2.md) §4-6 — TACC access and download instructions (needed for Option A)
