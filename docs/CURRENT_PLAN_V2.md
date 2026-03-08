# DDPM Flood — Current Plan (v2)

**Date**: February 28, 2026
**Status**: V1 baseline complete, spatial enrichment iteration planned
**Previous version**: [CURRENT_PLAN.md](CURRENT_PLAN.md) (v3, Feb 27 2026)
**Approach**: Two-region study — Outer Banks (primary) + NYC (application), now with spatial conditioning

---

## Table of Contents

1. [Summary of Changes from V3](#1-summary-of-changes-from-v3)
2. [V1 Baseline Results and Diagnosis](#2-v1-baseline-results-and-diagnosis)
3. [Root Cause: Missing Spatial Identity](#3-root-cause-missing-spatial-identity)
4. [The Fix: Spatial Conditioning (Option 2)](#4-the-fix-spatial-conditioning-option-2)
5. [Theoretical Justification: Wang et al. Framework](#5-theoretical-justification-wang-et-al-framework)
6. [Updated Data Pipeline](#6-updated-data-pipeline)
7. [Updated DDPM Architecture](#7-updated-ddpm-architecture)
8. [Implementation Plan](#8-implementation-plan)
9. [Experiment Design (Unchanged)](#9-experiment-design-unchanged)
10. [Expected Figures (Updated)](#10-expected-figures-updated)
11. [Risk Assessment](#11-risk-assessment)
12. [Future Options if V2 is Insufficient](#12-future-options-if-v2-is-insufficient)
13. [Full V1 Results Reference](#13-full-v1-results-reference)
14. [File and Data Reference](#14-file-and-data-reference)

---

## 1. Summary of Changes from V3

### What was completed (V1 baseline)

| Phase | Status | Details |
|-------|--------|---------|
| Preprocessing | Complete | `preprocess.py` — 193 scenarios processed, two regions |
| OBX Training | Complete | 200K steps, ~3h9m on MPS, loss 1.12 → 0.01 |
| NYC Training | Complete | 100K steps, ~74m on MPS, loss 1.14 → 0.01 |
| OBX Validation | Complete | Dorian (9 θ values) + Arthur (1 θ value) evaluated |
| NYC Validation | Complete | Sandy + Irene evaluated |
| Generation script | Complete | `generate_flood.py` — validate/sweep/generate modes |

### What this plan changes

The V1 baseline revealed that the DDPM learns θ-conditional mean surge correctly but produces **spatially flat predictions** — all patches get the same statistical output regardless of location. V4 adds three spatial conditioning columns (latitude, longitude, bathymetric depth) to give the model location awareness.

| Property | V1 (baseline) | V2 (this plan) |
|----------|--------------|----------------|
| Training columns | 21: [θ, surge₁...surge₂₀] | 24: [θ, lat, lon, depth, surge₁...surge₂₀] |
| `unmask_number` | 1 (θ only) | 4 (θ + lat + lon + depth) |
| `OP_NUM` | 20 | 20 (unchanged — surge columns) |
| Sequence length | 21 | 24 |
| Spatial identity | None — model can't distinguish patches | Yes — model knows each patch's location and bathymetry |

---

## 2. V1 Baseline Results and Diagnosis

### Training convergence (both models trained successfully)

**OBX model** (200K steps, batch_size=128, lr=1e-5):
```
Step      0: loss = 1.1215
Step  1,000: loss = 0.0734
Step 10,000: loss = 0.0094
Step 50,000: loss = 0.0102
Step100,000: loss = 0.0056
Step200,000: loss = 0.0110 (final)
```
- 41 checkpoints saved (model-0.pt through model-40.pt, 8.6MB each)
- Converged by ~10K steps, stable around 0.005–0.015

**NYC model** (100K steps, batch_size=64, lr=1e-5):
```
Step      0: loss = 1.1375
Step100,000: loss = 0.0142 (final)
```
- 20 checkpoints saved (model-1.pt through model-20.pt)
- Similar convergence pattern

Both models trained without issues on Apple Silicon (MPS). A benign `malloc: Failed to allocate segment from range group` warning appeared around step ~67K (OBX) and ~67K (NYC) but did not affect training.

### Validation results — OBX

Validated against two held-out storms using the EMA model (milestone 40, step 200K), n_samples=10:

**Dorian** (9 Danso forcing variants, θ ranging from 0.70m to 3.36m):

| θ (m) | R² | RMSE (m) | MAE (m) | Bias (m) | Notes |
|-------|-----|----------|---------|----------|-------|
| 0.701 | -0.089 | 0.289 | 0.264 | -0.031 | Within training range |
| 0.702 | -0.086 | 0.288 | 0.263 | -0.030 | Within training range |
| 0.766 | -0.082 | 0.293 | 0.272 | -0.015 | Within training range |
| 2.781 | -0.197 | 0.570 | 0.463 | +0.175 | Near edge of training range (max=2.88m) |
| 2.899 | -0.166 | 0.579 | 0.464 | +0.159 | Slight extrapolation |
| 2.984 | -0.149 | 0.592 | 0.474 | +0.153 | Extrapolation |
| 3.097 | -0.076 | 0.626 | 0.492 | +0.093 | Extrapolation |
| 3.237 | -0.073 | 0.640 | 0.501 | +0.097 | Extrapolation |
| 3.357 | -0.076 | 0.661 | 0.510 | +0.084 | Extrapolation (highest) |

**Arthur** (1 CERA scenario, θ=2.05m — within training range):

| θ (m) | R² | RMSE (m) | MAE (m) | Bias (m) |
|-------|-----|----------|---------|----------|
| 2.049 | -1.269 | 0.480 | 0.406 | +0.316 |

### Validation results — NYC

Validated against two held-out storms using the EMA model (milestone 20, step 100K), n_samples=10:

| Storm | θ (m) | R² | RMSE (m) | MAE (m) | Bias (m) | Notes |
|-------|-------|-----|----------|---------|----------|-------|
| Sandy | 3.179 | -34.159 | 1.905 | 1.875 | -1.875 | Far outside training range (max=1.17m) |
| Irene | 2.353 | -7.394 | 0.991 | 0.927 | -0.927 | Far outside training range |

### Key observations

1. **Bias is well-calibrated for in-range θ values**: Dorian's low-θ scenarios (θ≈0.7m) have bias near zero (-0.03m), showing the model correctly predicts average surge magnitude given θ.

2. **R² is negative everywhere**: R² measures spatial pattern accuracy. Negative R² means the model's spatial predictions are worse than just predicting the mean at every location. This indicates the model produces **spatially flat** predictions.

3. **Pearson correlation near zero**: Arthur: r=0.05, Dorian: r=-0.01. No spatial correlation between predicted and actual surge patterns.

4. **Spatial variance severely compressed**: Arthur truth has std=0.33m across nodes; predictions have std=0.19m. Dorian truth std=0.67m vs predictions std=0.15m. The model predicts roughly the same surge everywhere.

5. **NYC results are dominated by extrapolation failure**: Sandy and Irene θ values (3.18m, 2.35m) are 2-3x beyond the training range (max 1.17m). The model has never seen surge that high and massively underpredicts. This is a separate issue from the spatial problem.

6. **RMSE is reasonable in absolute terms for in-range θ**: 0.29m RMSE at θ=0.7m is acceptable for surge prediction. The issue is spatial pattern, not magnitude.

---

## 3. Root Cause: Missing Spatial Identity

### Why the model produces spatially flat predictions

In V1, each training row is:
```
[θ, surge₁, surge₂, ..., surge₂₀]    # 21 columns
```

Every patch shares the same θ but has different surge values depending on its location (coastal patches get high surge, offshore/deep-water patches get lower surge, etc.). However, the model has **no way to know which patch it's generating for**. It learns P(surge|θ) — the marginal distribution of surge conditioned only on θ — and all patches with the same θ get drawn from the same distribution.

When we average over multiple DDPM samples (n_samples=10), we get approximately E[surge|θ] — the expected surge across all patches. This is a single number, producing a spatially flat prediction.

### Contrast with Wang et al.

In Wang et al.'s molecular dynamics application, the 18 internal coordinates (bond distances, angles) are **exchangeable** — there's no concept of "location" because the molecule is a single isolated entity. Every coordinate participates equally in defining the molecular state.

In our flood application, the 20 surge nodes within each patch are **not exchangeable** — node 1 might be at the waterfront (high surge) while node 15 is in deep water (lower surge). And critically, different patches occupy different geographic locations with fundamentally different surge characteristics.

### What the model actually learned

Despite the spatial failure, the model did learn something useful:
- The conditional mean E[surge|θ] tracks correctly with θ
- Bias is near-zero for in-range θ, showing the θ-surge relationship is captured
- The model generates physically reasonable surge magnitudes (0.02–2.65m for θ=2.0m)
- Loss converged normally, indicating the U-Net successfully learned the denoising task

The model successfully learned the **first moment** of P(surge|θ) but not the **spatial structure**. Adding location conditioning should give it the information needed to learn P(surge|θ, location).

---

## 4. The Fix: Spatial Conditioning (Option 2)

### What we're adding

Three new conditioning columns extracted from the ADCIRC mesh, placed before the surge columns:

| Column | Content | Source | Physical meaning |
|--------|---------|--------|-----------------|
| 0 | θ (peak surge at reference gauge) | Computed per scenario | Storm severity index |
| 1 | **Patch centroid latitude** | Mean of 20 node latitudes | North-south position |
| 2 | **Patch centroid longitude** | Mean of 20 node longitudes | East-west position |
| 3 | **Patch mean bathymetric depth** | Mean of 20 node depths from ADCIRC `depth` variable | Water depth (positive = underwater, negative = above sea level) |
| 4–23 | Surge at 20 nodes | zeta_max from ADCIRC | The values to predict |

New training row format:
```
[θ, lat, lon, depth, surge₁, surge₂, ..., surge₂₀]    # 24 columns
```

With `unmask_number=4`, columns 0–3 are **never noised** during training and **held fixed** during generation. The model learns:

**P(surge₁...surge₂₀ | θ, latitude, longitude, depth)**

### Why these three features

**Latitude and longitude** — provide coarse spatial identity. Our correlation analysis shows:

| θ range | r(lat, surge) | r(lon, surge) | Pattern |
|---------|---------------|---------------|---------|
| θ ≈ 0.5m | -0.384 | +0.333 | Southern/eastern patches get more surge |
| θ ≈ 1.0m | -0.474 | -0.385 | Southern/western patches get more surge |
| θ ≈ 1.5m | -0.618 | -0.478 | Strong southern/western pattern |
| θ ≈ 2.0m | -0.227 | -0.239 | Weaker, more uniform |
| θ ≈ 2.5m | +0.263 | -0.023 | Pattern flips — northern patches get more |

The sign flip across θ values means the spatial surge pattern is **nonlinear and θ-dependent**. A linear model with lat/lon achieves only R²=0.09, but the DDPM is nonlinear and could learn these shifting patterns.

**Bathymetric depth** — from the ADCIRC `depth` variable (available in every maxele.63.nc file). This encodes the local water environment:
- Shallow water (depth 0–5m): surge amplification through shoaling
- Deep water (depth > 20m): surge attenuated
- Negative depth (above sea level): land areas, only flood if surge exceeds elevation

Linear correlation with surge is weak (r ≈ -0.13 to +0.16), but the relationship is physically nonlinear — surge amplification scales as ~1/√depth in shallow water.

### Spatial feature statistics

**OBX patches (1,974 total):**

| Feature | Min | Max | Range |
|---------|-----|-----|-------|
| Centroid latitude | 34.510 | 35.996 | 1.486° |
| Centroid longitude | -76.493 | -75.308 | 1.185° |
| Mean depth | 0.54m | 2586.26m | Wide range |

Depth distribution: 714 shallow (<5m), 780 medium (5–20m), 480 deep (>20m)

**NYC patches (219 total):**

| Feature | Min | Max | Range |
|---------|-----|-----|-------|
| Centroid latitude | 40.452 | 40.942 | 0.490° |
| Centroid longitude | -74.242 | -73.707 | 0.535° |
| Mean depth | -33.43m | 19.52m | Includes land (negative) |

### What the ADCIRC `depth` variable is

The `depth` variable in ADCIRC maxele.63.nc files contains the **distance below geoid** in meters for every mesh node:
- **Positive values** = below water (ocean floor depth)
- **Negative values** = above water (land elevation above sea level)
- **Near zero** = near the waterline (most flood-sensitive)

This is a static property of the mesh — identical across all scenarios. It is read once from any maxele.63.nc file (we use the first Danso file encountered) and applied to all patches.

### Why not just lat+lon (Option 1)?

We considered three options:

| Option | Columns | unmask | Rationale |
|--------|---------|--------|-----------|
| 1. lat+lon | 23 | 3 | Minimal spatial identity |
| **2. lat+lon+depth** | **24** | **4** | **Adds physically meaningful bathymetry at negligible cost** |
| 3. Per-node depth | 43 | 23 | Maximum spatial info, but doubles sequence length |

Option 2 is chosen because:
- Adding 1 extra column (depth) has zero meaningful compute cost vs Option 1
- Depth provides physically meaningful information that lat/lon cannot capture (bathymetry governs surge dynamics)
- Option 3 is kept as a fallback if Option 2 proves insufficient

---

## 5. Theoretical Justification: Wang et al. Framework

### How conditioning works in Wang et al.'s DDPM

Wang et al.'s DDPM learns the conditional distribution P(x₁|x₂) where:
- x₂ = conditioning variable (temperature, kept fixed during diffusion via inpainting)
- x₁ = target variables (internal molecular coordinates, generated through reverse diffusion)

The conditioning is implemented through **inpainting**: during both forward noising and reverse denoising, the first `unmask_number` columns are never modified. The model learns to denoise the remaining columns given fixed values in the conditioning columns.

### Our extension is theoretically identical

We expand x₂ from a single scalar to a vector:

| | Wang et al. | V1 (baseline) | V2 (this plan) |
|--|-------------|---------------|----------------|
| x₂ (conditioning) | [T] (temperature) | [θ] (surge at reference gauge) | [θ, lat, lon, depth] |
| x₁ (generated) | [d₁...d₁₈] (interatomic distances) | [s₁...s₂₀] (surge at 20 nodes) | [s₁...s₂₀] (surge at 20 nodes) |
| `unmask_number` | 1 | 1 | 4 |
| Distribution learned | P(distances \| temperature) | P(surge \| θ) | P(surge \| θ, location, depth) |

This is a valid use of the inpainting mechanism. The GaussianDiffusion class already supports arbitrary `unmask_number` — it generates a mask that keeps columns `[0, 1, ..., unmask_number-1]` fixed. No architectural changes are needed.

### Scientific framing

> "Following Wang et al., we condition the DDPM on known variables using the inpainting mechanism. While Wang et al. condition on a single scalar (temperature), we extend this to a vector of known quantities: the severity index θ, and the spatial coordinates (latitude, longitude, bathymetric depth) of each prediction patch. This allows the model to learn the location-dependent conditional distribution P(surge | θ, location), capturing how storm severity translates differently to surge at different geographic locations."

---

## 6. Updated Data Pipeline

### Changes to preprocessing

The `preprocess.py` script needs to be updated to:

1. **Extract bathymetric depth** from the ADCIRC `depth` variable (read once from any maxele.63.nc file)
2. **Compute patch centroid coordinates** (mean lat, lon of 20 nodes in each patch)
3. **Compute patch mean depth** (mean depth of 20 nodes in each patch)
4. **Build 24-column training matrix** instead of 21-column:

Old: `[θ, surge₁, ..., surge₂₀]` (21 columns)
New: `[θ, lat_centroid, lon_centroid, mean_depth, surge₁, ..., surge₂₀]` (24 columns)

### What stays the same

- All data discovery, scenario handling, region extraction — unchanged
- KMeans clustering — unchanged (same patches, same node assignments)
- Train/validation splits — unchanged (same held-out storms)
- Noise injection on θ — unchanged (still applied during training, not baked into .npy)

### Updated output shapes

| Region | V1 shape | V2 shape |
|--------|----------|----------|
| OBX train | (361,242 × 21) | (361,242 × 24) |
| OBX val Dorian | (17,766 × 21) | (17,766 × 24) |
| OBX val Arthur | (1,974 × 21) | (1,974 × 24) |
| NYC train | (41,829 × 21) | (41,829 × 24) |
| NYC val Sandy | (219 × 24) | (219 × 24) |
| NYC val Irene | (219 × 24) | (219 × 24) |

### New output files

In addition to existing files, preprocessing will save:
- `patch_centroids.npy` — (n_patches, 3) array of [lat, lon, depth] per patch, for use during generation

---

## 7. Updated DDPM Architecture

### Model (unchanged from V1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | 1D U-Net | Identical to Wang et al. |
| Base dimension | 32 | |
| Channel multipliers | (1, 2, 2, 4) | |
| Group normalization | 8 groups | |
| Diffusion timesteps | 1,000 | |
| Beta schedule | Linear, 0.0001 → 0.02 | |
| Loss | L2 (MSE) on predicted noise | |

The U-Net architecture is sequence-length agnostic — it processes the input through 1D convolutions that work with any length. Changing from 21 to 24 columns requires no architectural changes.

### Training parameters (updated)

| Parameter | V1 | V2 | Notes |
|-----------|-----|-----|-------|
| `OP_NUM` | 20 | 20 | Still 20 surge values per patch |
| Sequence length | 21 (OP_NUM+1) | 24 (OP_NUM+4) | 4 conditioning cols + 20 surge cols |
| `UNMASK_NUMBER` | 1 | 4 | θ + lat + lon + depth never noised |
| Batch size | 128 (OBX) / 64 (NYC) | 128 (OBX) / 64 (NYC) | Unchanged |
| Learning rate | 1e-5 | 1e-5 | Unchanged |
| Training steps | 200K (OBX) / 100K (NYC) | 200K (OBX) / 100K (NYC) | Unchanged |
| EMA decay | 0.995 | 0.995 | Unchanged |
| θ noise injection | σ=0.15m | σ=0.15m | Unchanged, applied to col 0 only |

### Conditioning during generation

During generation, columns 0–3 are set to the target values:
- Column 0: θ (desired surge severity)
- Column 1: patch centroid latitude
- Column 2: patch centroid longitude
- Column 3: patch mean depth

The model then generates columns 4–23 (the 20 surge values) through 1000-step reverse diffusion, with columns 0–3 held fixed at every step via the inpainting mask.

This means each patch gets a **unique conditioning vector** [θ, lat_i, lon_i, depth_i], so the model can produce **location-specific** surge predictions.

---

## 8. Implementation Plan

### Files to modify

| File | Change | Effort |
|------|--------|--------|
| `preprocess.py` | Add depth extraction, centroid computation, build 24-col matrix | Medium |
| `train_flood.py` | Change `OP_NUM=20` to seq_len=24, `UNMASK_NUMBER=4` | Small |
| `generate_flood.py` | Set lat/lon/depth from patch centroids during generation | Medium |

### Step-by-step

1. **Update `preprocess.py`**:
   - Read `depth` variable from first ADCIRC file encountered
   - Extract depth values for region nodes (same indices as lat/lon)
   - After KMeans clustering, compute per-patch: centroid lat, centroid lon, mean depth
   - Build training matrix as `[θ, lat_c, lon_c, depth_c, surge₁...surge₂₀]`
   - Save `patch_centroids.npy` for use during generation
   - Re-run preprocessing for both regions

2. **Update `train_flood.py`**:
   - Change `UNMASK_NUMBER = 4`
   - Change column indexing: θ noise injection still on column 0
   - No change to FloodDataset normalization (min-max on all columns works)

3. **Re-train both models**:
   - OBX: `uv run python train_flood.py --region obx` (~3 hours on MPS, or ~30 min on A100)
   - NYC: `uv run python train_flood.py --region nyc` (~75 min on MPS, or ~10 min on A100)

4. **Update `generate_flood.py`**:
   - Load `patch_centroids.npy` to get per-patch [lat, lon, depth]
   - When generating, set columns 0–3 instead of just column 0
   - Each batch now has patch-specific conditioning (not all identical)
   - This means we can't generate all patches in one batch with the same conditioning — need to batch by groups or generate individually

5. **Re-run validation**:
   - Same commands as V1
   - Compare R²/RMSE/MAE against V1 baseline

### Expected timeline

| Task | MPS (Apple Silicon) | A100 (Colab) |
|------|-------------------|--------------|
| Preprocessing | ~5 min | ~5 min |
| OBX training | ~3 hours | ~30 min |
| NYC training | ~75 min | ~10 min |
| OBX validation | ~200 min | ~20 min |
| NYC validation | ~4 min | ~1 min |

---

## 9. Experiment Design (Unchanged)

The experiment design from V3 Section 7 remains the same. The three tiers of experiments are:

1. **Hold-out validation** — Dorian+Arthur (OBX), Sandy+Irene (NYC)
2. **Extreme event extrapolation** — Dorian at θ=3.36m beyond training max of 2.88m
3. **Tipping-point θ sweep** — Dense θ sweep from 0.5m to 3.5m

The only difference is that we now expect the spatial pattern (R²) to improve significantly, while the extrapolation limitations for NYC (Sandy/Irene far outside training range) will likely persist.

---

## 10. Expected Figures (Updated)

All figures from V3 Section 8 remain planned. Additional figures for V2:

### Figure 0 (new): V1 vs V2 Comparison
- Scatter plot of predicted vs actual surge for Arthur (θ=2.05m)
- Left panel: V1 (spatially flat), Right panel: V2 (with spatial conditioning)
- Demonstrates the improvement from adding location features
- **Purpose**: Justifies the spatial conditioning approach

### Figure 2 (updated): Hold-Out Validation
- Now expected to show positive R² values (instead of negative)
- Include both V1 and V2 results for comparison

---

## 11. Risk Assessment

### What should work well

- **Each unique (lat, lon, depth) appears in all 183 training scenarios**: the model sees enough examples per location to learn the location-surge relationship
- **No spatial interpolation needed at generation time**: the exact same (lat, lon, depth) values from training are used during generation (same patch centroids)
- **The θ × location interaction is learnable**: the DDPM is nonlinear and can capture the sign-flipping correlation patterns observed in the data

### Legitimate risks

1. **Within-patch structure still unidentified**: Even with patch-level lat/lon/depth, the 20 nodes *inside* each patch have no individual identity. Node 5 could be at the waterfront while node 12 is 200m inland. However, the ordering is **consistent across training** (patch 42 always has the same nodes in the same order), so the model *can* implicitly learn "position 5 in this location tends to have a particular surge pattern."

2. **Extrapolation for θ outside training range will NOT improve**: Adding spatial features only helps the spatial pattern, not the θ generalization. Dorian's high-θ scenarios (2.8–3.4m, beyond training max 2.88m) and NYC's Sandy/Irene will still show poor magnitude predictions.

3. **Overfitting risk**: With 183 scenarios per unique (lat, lon, depth), the model might memorize surge patterns rather than learning smooth physical relationships. Mitigated by θ noise injection (σ=0.15m) which acts as regularization, and the fact that each unique θ value slightly alters the surge pattern.

4. **Three coordinates may not be enough**: Lat/lon/depth are a crude spatial proxy. Actual surge depends on detailed coastline geometry, barrier island topology, inlet dynamics, etc. We might get R² of 0.3–0.6 (a meaningful improvement from -0.1) but not necessarily 0.9. This is still a significant result worth reporting.

5. **Min-max normalization range**: Depth ranges from 0.54m to 2586m (OBX), which is a very wide range. The min-max normalization will compress most depth values near -1 in normalized space. Consider whether log-transforming depth before normalization would help (future investigation).

### Success criteria

| Metric | V1 baseline | Minimum acceptable | Good result |
|--------|-------------|-------------------|-------------|
| R² (Arthur, θ=2.05m) | -1.27 | > 0.0 | > 0.5 |
| R² (Dorian low-θ) | -0.09 | > 0.0 | > 0.3 |
| RMSE (in-range θ) | 0.29m | < 0.25m | < 0.15m |
| Pearson r (Arthur) | 0.05 | > 0.3 | > 0.7 |

If R² moves from negative to positive (even if only 0.2–0.3), this confirms that spatial conditioning is working and the approach is sound. Further improvements can come from Option 3 (per-node depth features) or other enhancements.

---

## 12. Future Options if V2 is Insufficient

If adding lat/lon/depth (Option 2) does not produce satisfactory R² values, the following options are available in order of increasing complexity:

### Option 3: Per-node depth features

Add individual depth for each of the 20 nodes within a patch:
```
[θ, lat, lon, d₁, d₂, ..., d₂₀, surge₁, surge₂, ..., surge₂₀]   # 43 columns, unmask=23
```

This gives each node its own depth identity, allowing the model to learn that shallow nodes get higher surge than deep nodes within the same patch. Increases sequence length from 24 to 43.

### Option 4: Additional physical features

Add more per-node features beyond depth:
- Distance to nearest coastline
- Local mesh element size (proxy for ADCIRC resolution)
- Mean elevation in surrounding area

### Option 5: Time series approach

Use `fort.63.nc` time series data instead of peak surge (maxele). This provides:
- Naturally continuous θ distribution (no noise injection needed)
- 200-500x more training data per storm
- Exact structural match to Wang et al.'s framework

This is detailed in V3 Section 12 and requires downloading ~30-100GB from TACC.

### Option 6: Different architecture

If the 1D U-Net with inpainting fundamentally cannot capture spatial patterns:
- 2D DDPM treating the surge field as an image (requires mesh-to-grid interpolation)
- Graph neural network DDPM operating directly on the ADCIRC mesh
- Conditional normalizing flows instead of diffusion

These are larger architectural departures and would be Phase 2 work.

---

## 13. Full V1 Results Reference

### OBX spatial analysis

| Metric | Arthur (θ=2.05m) | Dorian (θ=3.36m, last scenario) |
|--------|-------------------|-------------------------------|
| Prediction node range | 0.378 – 1.746m | 0.579 – 1.745m |
| Truth node range | 0.168 – 2.188m | 0.097 – 3.588m |
| Prediction std | 0.186m | 0.148m |
| Truth std | 0.333m | 0.670m |
| Pearson r | 0.051 | -0.008 |

Key finding: prediction spatial variance (std) is 2-4x lower than truth, confirming spatially flat output.

### OBX prediction distribution (Arthur, θ=2.05m)

| Percentile | Prediction | Truth |
|------------|-----------|-------|
| p5 | 0.718m | 0.285m |
| p25 | 0.893m | 0.457m |
| p50 | 1.016m | 0.648m |
| p75 | 1.140m | 0.936m |
| p95 | 1.327m | 1.304m |

The prediction distribution is compressed — the tails are pulled toward the center. At p95, predictions roughly match truth, but at p5 they overshoot by 0.43m. The prediction acts as a "mean field" — roughly correct on average but without spatial variation.

### Correlation of spatial features with surge (OBX)

| θ (m) | r(lat) | r(lon) | r(depth) |
|-------|--------|--------|----------|
| 0.50 | -0.384 | +0.333 | +0.161 |
| 0.70 | -0.384 | +0.330 | +0.162 |
| 1.01 | -0.474 | -0.385 | +0.040 |
| 1.50 | -0.618 | -0.478 | +0.010 |
| 2.01 | -0.227 | -0.239 | +0.002 |
| 2.52 | +0.263 | -0.023 | -0.126 |

Linear regression R² (lat+lon): 0.092
Linear regression R² (lat+lon+depth): 0.093

Note: Linear R² is low (9%), but the DDPM is nonlinear and the θ-dependent sign flips suggest learnable patterns that a linear model cannot capture.

### Files generated by V1

```
results/
├── obx/
│   ├── model-0.pt through model-40.pt    # 41 checkpoints (8.6MB each)
│   ├── loss_log.csv                       # 1001 entries
│   └── generated/
│       ├── validation_metrics.json        # All OBX validation metrics
│       ├── pred_dorian_mean.npy           # (1974, 20) last Dorian scenario
│       ├── pred_dorian_std.npy            # (1974, 20) uncertainty
│       ├── pred_dorian_nodes.npy          # (58645,) stitched node-level
│       ├── truth_dorian_nodes.npy         # (58645,) stitched truth
│       ├── pred_arthur_mean.npy           # (1974, 20)
│       ├── pred_arthur_std.npy            # (1974, 20)
│       ├── pred_arthur_nodes.npy          # (58645,) stitched
│       └── truth_arthur_nodes.npy         # (58645,) stitched
├── nyc/
│   ├── model-1.pt through model-20.pt    # 20 checkpoints
│   ├── loss_log.csv
│   └── generated/
│       ├── validation_metrics.json
│       ├── pred_sandy_mean.npy
│       ├── pred_sandy_std.npy
│       ├── pred_irene_mean.npy
│       └── pred_irene_std.npy
```

---

## 14. File and Data Reference

### Code files

| File | Purpose | Status |
|------|---------|--------|
| `preprocess.py` | Data preprocessing (scenario discovery, node extraction, KMeans, matrix building) | Needs V2 update |
| `train_flood.py` | DDPM training (FloodDataset, training loop, checkpointing) | Needs V2 update |
| `generate_flood.py` | Sample generation and evaluation (validate, sweep, generate modes) | Needs V2 update |
| `DDPM_REMD/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py` | Wang et al.'s original DDPM library (unmodified) | No changes needed |

### Data files

| Path | Contents | Size |
|------|----------|------|
| `data/adcirc/` | Raw ADCIRC maxele.63.nc files (Danso + CERA) | ~10GB |
| `data/processed/obx/` | OBX preprocessed data | ~30MB |
| `data/processed/nyc/` | NYC preprocessed data | ~4MB |
| `results/obx/` | OBX model checkpoints + V1 validation results | ~360MB |
| `results/nyc/` | NYC model checkpoints + V1 validation results | ~180MB |

### ADCIRC `depth` variable

Available in every maxele.63.nc file. Read using:
```python
import netCDF4 as nc
ds = nc.Dataset('path/to/maxele.63.nc')
depth = np.asarray(ds.variables['depth'][:])  # (1813443,) for HSOFS mesh
# Positive = below water, negative = above sea level
```

This is a static mesh property — identical across all scenarios. Read once from any file.

### Key technical notes

- **ADCIRC fill values**: Filter with `(zeta > 1e10) | (zeta < -1e10) | (zeta <= -99999)`
- **netCDF4 masked arrays**: Always use `np.asarray()` to convert
- **MPS device**: Use `pin_memory=False` (only enable for CUDA)
- **CERA HSOFS identification**: Node count = 1,813,443
- **Isabel (CERA)**: NOT HSOFS (6,056,968 nodes) — excluded
- **Nicole (CERA)**: IS HSOFS — included
- **Type B sea levels**: `ref`, `slr_0.44m`, `slr_0.74m` (not `slr_044`)
- **Harvey**: Exclude `old/` and `Vmax_10_old/` directories
