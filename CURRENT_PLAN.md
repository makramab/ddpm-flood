# DDPM Flood — Current Plan (v3)

**Date**: February 27, 2026
**Status**: Planning complete, ready to begin preprocessing
**Approach**: Two-region study — Outer Banks (primary proof-of-method) + NYC (application case)

---

## Table of Contents

1. [What Changed and Why](#1-what-changed-and-why)
2. [The Two-Region Strategy](#2-the-two-region-strategy)
3. [Region A: Outer Banks, North Carolina (Primary)](#3-region-a-outer-banks-north-carolina-primary)
4. [Region B: New York City (Application)](#4-region-b-new-york-city-application)
5. [The θ Noise Injection Fix](#5-the-θ-noise-injection-fix)
6. [DDPM Architecture (Unchanged from Wang et al.)](#6-ddpm-architecture-unchanged-from-wang-et-al)
7. [Experiment Design](#7-experiment-design)
8. [Expected Figures](#8-expected-figures)
9. [Implementation Phases](#9-implementation-phases)
10. [Known Limitations and Honest Framing](#10-known-limitations-and-honest-framing)
11. [File and Data Reference](#11-file-and-data-reference)
12. [Phase 2: Time Series Approach (After Current Plan)](#12-phase-2-time-series-approach-after-current-plan)

---

## 1. What Changed and Why

### Problem identified: the θ distribution

After reviewing Wang et al.'s training data, we found a fundamental mismatch with our NYC-only approach:

| Property | Wang et al. (temperature) | NYC-only (Battery surge) |
|---|---|---|
| Training rows | 500,000 | ~42,460 |
| Unique conditioning values | ~15,726 (continuous) | ~41 (discrete, clustered) |
| Distribution shape | Smooth, quasi-Gaussian across full range | 180 scenarios crammed into 0.7–1.2m, two outliers at 2.35m and 3.18m |

Wang et al.'s method works because their conditioning variable (instantaneous kinetic temperature) naturally fluctuates — every molecular frame has a slightly different temperature, creating a smooth, continuous distribution. Our θ (Battery surge) is one fixed number per storm scenario, and almost all our scenarios produce similar low values at NYC because these storms mostly hit the Gulf Coast and Carolinas.

### The solution: use a region that was actually hit hard

The Danso dataset simulates 20 hurricanes across the **entire US Atlantic coast** (1.8 million ADCIRC nodes). While NYC barely feels these storms, the **Outer Banks of North Carolina** gets direct hits from several of them, with surge ranging from 0.4m to 4.0m — a much wider and better-distributed range.

By running the DDPM on Outer Banks first (where data is abundant), we prove the method works. Then we apply it to NYC (where data is scarce but the application matters) as the target case.

### How we are framing θ

We are using θ (peak surge at a reference tide gauge) as a **scalar severity index** — a single number that summarizes "how bad is this storm for this region." This is a simplification compared to full storm parameterization (pressure, wind speed, track angle, etc.), but it:

- Keeps the DDPM architecture identical to Wang et al. (`unmask_number=1`, no code changes)
- Has clear physical meaning (tide gauge readings drive evacuation decisions)
- Is directly comparable to historical records

We are honest that this is **conditional spatial prediction given a severity index**, not a direct replication of Wang et al.'s thermodynamic framework.

---

## 2. The Two-Region Strategy

### Why two regions?

| | Outer Banks (Region A) | NYC (Region B) |
|---|---|---|
| **Purpose** | Prove the method works | Apply to the region Prof. Miura cares about |
| **θ range (Danso)** | 0.40 – 4.00m | 1.10 – 2.41m |
| **θ distribution** | Spread across low, medium, high | Clustered in 1.0–1.2m |
| **Storms with θ > 2m** | 14 scenarios (Danso) + 3 CERA | 18 scenarios (Danso, all SLR) + 2 CERA |
| **Storms with θ > 3m** | 3 scenarios | 0 scenarios (only Sandy from CERA at 3.18m) |
| **Region nodes (HSOFS)** | 58,645 | 9,728 |
| **Validation** | Hold out Dorian (θ=3.36m), Isabel (θ=2.04m) | Hold out Sandy (θ=3.18m), Irene (θ=2.35m) |
| **DEM for visualization** | Not available (use ADCIRC mesh viz) | Miura's 18 DEM tiles (30m resolution) |

### Two separate models, same architecture

Each region gets its own:
- Reference gauge location (for computing θ)
- Set of extracted mesh nodes
- Spatial patch clustering
- Training matrix
- Trained DDPM model

The DDPM architecture and hyperparameters are identical for both. Only the data changes.

### Presentation narrative

> "We demonstrate our DDPM-based flood prediction method on two Atlantic coast regions. At the Outer Banks — where diverse hurricane impacts provide a rich training distribution — the model accurately predicts unseen storms and discovers flood tipping points. We then apply the same method to New York City, where training data is scarcer, and test whether it can extrapolate to Sandy-level events it has never seen."

---

## 3. Region A: Outer Banks, North Carolina (Primary)

### Why Outer Banks?

The Outer Banks is a chain of barrier islands along the North Carolina coast. It is one of the most hurricane-prone areas on the US Atlantic seaboard. Of the 20 storms in the Danso dataset, several made direct or near-direct landfall here:

- **Dorian (2019)**: θ = 3.36m (Category 1, stalled over OBX)
- **Bonnie (1998)**: θ = 2.88m (Category 2, hit Wilmington NC)
- **Isabel (2003)**: θ = 2.04m (Category 2, hit OBX directly)
- **Floyd (1999)**: θ = 1.71m (Category 2, hit Cape Fear)
- **Florence (2018)**: θ = 1.50m (Category 1, hit Wilmington)
- **Matthew (2016)**: θ = 1.01m (Category 1, brushed OBX)

This gives a well-spread θ distribution from 0.4m to 4.0m.

### Geographic definition

| Property | Value |
|---|---|
| Bounding box | lat 34.50–36.00, lon -76.50 to -75.30 |
| Region description | Outer Banks barrier islands, Pamlico Sound, and adjacent mainland coast |
| Total HSOFS mesh nodes | 58,645 |
| Land nodes | 19,111 |
| Water nodes | 39,534 |

### Reference gauge (θ definition)

| Property | Value |
|---|---|
| Location name | Cape Hatteras / Pamlico Sound region |
| Bounding box | lat 35.00–35.40, lon -76.10 to -75.50 |
| Nodes in reference area | 9,788 |
| θ computation | Peak `zeta_max` across all reference area nodes per scenario |

Note: Unlike NYC's Battery (a single tide gauge), the Outer Banks reference area is broader because there is no single dominant gauge equivalent. The Hatteras region was chosen because it sits at the crux of the barrier islands where surge funnels into Pamlico Sound.

### Training data: θ distribution

**Danso et al. (180 scenarios):**
```
0.0–0.5m:  21 scenarios  ████████████████████
0.5–1.0m: 128 scenarios  ████████████████████████████████████████████████████████████████
1.0–1.5m:   9 scenarios  █████████
1.5–2.0m:   8 scenarios  ████████
2.0–2.5m:   2 scenarios  ██
2.5–3.0m:   9 scenarios  █████████
3.0–3.5m:   3 scenarios  ███
                          ─────────
Total > 1m: 31 scenarios
Total > 2m: 14 scenarios
Total > 3m:  3 scenarios
```

**CERA HSOFS (13 storms):**
- Isabel: 2.07m
- Irene: 2.19m
- Arthur: 2.05m
- Sandy: 1.62m
- Others: 0.54 – 1.23m

**Combined: 193 scenarios, θ range 0.40 – 4.00m**

This is dramatically better than NYC's distribution. The model will see data across the full 0–4m range, with 14+ scenarios above 2m.

### Spatial patches

| Property | Actual value (from preprocessing) |
|---|---|
| Nodes to extract | 58,645 total, 39,494 with valid surge (90% threshold) |
| Nodes per patch | 20 |
| Number of patches | 1,974 |
| Training rows (183 scenarios × 1,974 patches) | 361,242 |
| Training matrix columns | 21 (1 θ + 20 surge values) |

This is close to Wang et al.'s 500,000 rows — a significant improvement over NYC's ~42,000.

### Held-out validation storms

| Storm | θ (Hatteras) | Why hold out |
|---|---|---|
| **Dorian** (Danso) | 3.36m | Highest surge — extrapolation test if held out of training |
| **Arthur** (CERA) | 2.05m | Strong OBX storm, good for within-range validation |

**Note**: Isabel (CERA) was originally planned as validation but has 6,056,968 nodes (not HSOFS mesh), making it incompatible. Arthur (CERA HSOFS, θ=2.05m) replaces it.

**Training set**: 171 Danso (excluding Dorian's 9 variants) + 12 CERA HSOFS (excluding Arthur) = **183 scenarios**

---

## 4. Region B: New York City (Application)

### Geographic definition

| Property | Value |
|---|---|
| Bounding box | lat 40.45–40.95, lon -74.30 to -73.70 |
| Total HSOFS mesh nodes | 9,728 |
| Nodes with valid surge | ~4,438 |
| Nodes near Battery | 169 |

### Reference gauge (θ definition)

| Property | Value |
|---|---|
| Location | The Battery, southern tip of Manhattan |
| NOAA Station | 8518750 |
| Bounding box | lat 40.69–40.72, lon -74.02 to -74.00 |
| θ computation | Peak `zeta_max` across Battery-area nodes per scenario |

### Training data: θ distribution

**Danso (180 scenarios):** θ = 0.70 – 1.17m (all clustered in a narrow band)
**CERA HSOFS (11 storms, excluding Sandy + Irene):** θ = 0.86 – 1.10m

**Combined training: ~180 Danso + 11 CERA = 191 scenarios, θ = 0.70 – 1.17m**

This is the weak distribution we discussed. The noise injection fix (Section 5) partially addresses this.

### Held-out validation storms

| Storm | Battery θ | Why hold out |
|---|---|---|
| **Sandy** (CERA) | 3.18m | Headline result — can DDPM predict an extreme event from weak training data? |
| **Irene** (CERA) | 2.35m | Intermediate extrapolation test |

### Spatial patches

| Property | Actual value (from preprocessing) |
|---|---|
| Nodes to extract | 9,728 total, 4,393 with valid surge (90% threshold) |
| Nodes per patch | 20 |
| Number of patches | 219 |
| Training rows (191 × 219) | 41,829 |
| Training matrix columns | 21 |

### Visualization

NYC has Miura's DEM (18 tiles, 30m resolution, EPSG:2263) for pixel-level flood maps. This produces presentation-quality maps matching Miura et al. 2021 Figures 8, 9, 12.

---

## 5. The θ Noise Injection Fix

### The problem

θ is a deterministic value per scenario. After spatial patching, the ~220 (NYC) or ~1,000 (OBX) patches from a single scenario all share the exact same θ. The DDPM sees only ~41 distinct θ values (NYC) or ~100 distinct values (OBX), not a continuous distribution.

### The fix

Add small Gaussian noise to θ during training:

```
θ_noisy = θ + ε,  where ε ~ N(0, σ²)
```

Each training row gets a slightly different θ, even rows from the same scenario. This converts discrete θ values into a smooth continuous distribution.

### Choice of σ

**σ = 0.15m**, justified by:

1. **Physical basis**: NOAA's operational storm surge forecast error is typically 0.3–0.5m (1-sigma). Our σ = 0.15m is conservative — well within real measurement and forecast uncertainty.
2. **Practical effect**: A scenario with θ = 1.0m will contribute training rows with θ values ranging from roughly 0.55m to 1.45m (±3σ), creating overlap between adjacent scenarios.
3. **Not too large**: σ = 0.15m preserves the signal. The model can still distinguish a 0.7m scenario from a 1.2m scenario.

### Scientific framing

> "Following Wang et al., we require the conditioning variable to vary continuously across training samples. Since θ represents storm surge height — an inherently uncertain quantity subject to tidal phase, measurement error, and sub-grid-scale variability — we model this uncertainty as Gaussian perturbation with σ = 0.15m, consistent with operational surge forecast error margins (NWS, 2024)."

### Implementation

This is a one-line change during data loading — add noise to column 0 of each training batch. No architecture changes needed.

### Note for Outer Banks

The Outer Banks data already has a wider natural spread of θ values (~100 distinct values vs. NYC's ~41). The noise injection is still applied for consistency between the two experiments, but its effect is less critical for OBX.

---

## 6. DDPM Architecture (Unchanged from Wang et al.)

### Model

| Parameter | Value | Notes |
|---|---|---|
| Architecture | 1D U-Net | Identical to Wang et al. |
| Base dimension | 32 | |
| Channel multipliers | (1, 2, 2, 4) | |
| Group normalization | 8 groups | |
| Attention | Linear attention in middle block | |
| Diffusion timesteps | 1,000 | |
| Beta schedule | Linear, 0.0001 → 0.02 | |
| Loss | L2 (MSE) on predicted noise | |

### Training

| Parameter | Value | Notes |
|---|---|---|
| `op_num` | 20 | Number of surge values per patch |
| `unmask_number` | 1 | θ is one scalar (column 0), never noised |
| Batch size | 128 (OBX) / 64 (NYC) | OBX has more data |
| Learning rate | 1e-5 | Same as Wang et al. |
| Training steps | 200,000 (OBX) / 100,000 (NYC) | Scaled to dataset size |
| EMA decay | 0.995 | |
| Optimizer | Adam | |

### Conditioning (inpainting)

θ sits in column 0 of the training data. During training, it is never noised (masked via `unmask_index = [0]`). During generation, column 0 is fixed to the desired θ value and the model generates columns 1–20 (surge at the 20 patch nodes).

This is identical to how Wang et al. handles temperature — no code changes needed.

### Key code files

| File | What to modify |
|---|---|
| `DDPM_REMD/run_training.py` | Change `op_num`, data path, training steps, batch size |
| `DDPM_REMD/gen_sample.py` | Change sample generation to sweep θ values |
| `DDPM_REMD/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py` | Minimal changes — add MPS device support, noise injection in `Dataset_traj` |

---

## 7. Experiment Design

### Overview

We run three tiers of experiments, first on Outer Banks, then on NYC:

```
Tier 1: Hold-out validation    → Does the DDPM predict unseen storms accurately?
Tier 2: Extreme extrapolation  → Can it predict rare events beyond its training range?
Tier 3: Tipping-point discovery → Can it find critical surge thresholds?
```

### Experiment 1: Hold-Out Validation (Quantitative)

**Outer Banks:**
- Train on 183 scenarios (exclude Dorian 9 variants + Arthur)
- Generate surge for held-out storms at their actual θ values
- Compare to actual ADCIRC surge at every node
- Metrics: R², RMSE per patch and overall

**NYC:**
- Train on 191 scenarios (exclude Sandy + Irene)
- Generate surge for held-out storms
- Compare to actual ADCIRC

### Experiment 2: Extreme Event Extrapolation (Qualitative + Quantitative)

**Outer Banks:**
- Hold out Dorian (θ = 3.36m, the highest in dataset)
- Train on remaining data (max θ ≈ 2.88m from Bonnie)
- Generate at θ = 3.36m → compare to actual Dorian ADCIRC output
- This tests extrapolation ~17% beyond training range

**NYC (staged):**

| Experiment | Training data | Max training θ | Test | Extrapolation gap |
|---|---|---|---|---|
| **2a** | Danso only (180 scenarios) | 1.17m | Irene (θ=2.35m) | 2× beyond range |
| **2b** | Danso + CERA excl. Sandy & Irene | 1.17m | Sandy (θ=3.18m) | 2.7× beyond range |
| **2c** | Danso + CERA excl. Sandy (Irene in training) | 2.35m | Sandy (θ=3.18m) | 35% beyond range |

Experiment 2c is the most likely to succeed and the most directly comparable to Wang et al.'s extrapolation (they went ~3% beyond range; we go ~35%).

**Validation data for Sandy (NYC):**
- 347 USGS high water marks (`data/validation/sandy_hwm.csv`)
- Official Sandy inundation zone (`data/validation/sandy_inundation_zone/`)
- Battery tide gauge peak record (`data/validation/battery_tide_sandy.csv`)

### Experiment 3: Tipping-Point Discovery

**Both regions:**
- Generate at dense θ sweep: θ = 0.5, 0.6, 0.7, ... 5.0m (46 values)
- For each θ, generate all patches → stitch into complete surge map
- Calculate % of region flooded at each θ
- Plot response curves: surge at key locations vs. θ
- Identify sharp jumps = tipping points

**NYC specific:**
- Render each θ on Miura's DEM → pixel-level flood maps
- Create 4-panel progression figure (θ = 1.0, 2.0, 3.0, 4.0m)
- Compare to PoC bathtub model outputs already generated

---

## 8. Expected Figures

### Figure 1: θ Distribution Comparison
- Side-by-side histograms: Wang et al. temperature distribution vs. our Outer Banks θ distribution vs. NYC θ distribution
- Shows why Outer Banks is a better test case and why noise injection is needed for NYC
- **Purpose**: Motivates the two-region approach

### Figure 2: Hold-Out Validation Scatter Plot
- X: actual ADCIRC surge, Y: DDPM-predicted surge
- One panel for Outer Banks, one for NYC
- R², RMSE annotated
- **Purpose**: Proves the model works

### Figure 3: Dorian Extrapolation (Outer Banks)
- DDPM-predicted surge map at θ = 3.36m vs. actual ADCIRC Dorian output
- Side-by-side comparison
- **Purpose**: Proves extrapolation works where data is good

### Figure 4: Sandy Extrapolation (NYC, Miura-style)
- Left: DDPM-predicted flood map at θ = 3.18m, rendered on Miura's DEM
- Right: Actual Sandy inundation zone + USGS high water marks
- **Purpose**: The headline NYC result — can DDPM predict Sandy from weak training data?

### Figure 5: Tipping-Point Response Curves
- X: θ (0.5 – 5.0m), Y: predicted surge at key locations
- Multiple curves for different neighborhoods
- Sharp jumps annotated as tipping points
- One panel for each region
- **Purpose**: Demonstrates tipping-point discovery capability

### Figure 6: Flood Map Progression (NYC, Miura-style)
- 4-panel: θ = 1.0, 2.0, 3.0, 4.0m
- Pixel-level flood boundaries on Miura's DEM
- Progressive flooding visible
- **Purpose**: Visual demonstration of DDPM output quality

### Figure 7: Method Comparison Table
```
                | ADCIRC       | DDPM (ours)       | GISSR (Miura)
Accuracy        | Baseline     | ~90%+ of ADCIRC   | ~83% (from paper)
Speed/scenario  | Hours        | < 1 second        | 0.01 second
Extrapolation   | No (new run) | Yes               | Yes but simplified
Distributional  | No           | Yes (probabilistic)| No
Tipping points  | Not feasible | Yes               | Not feasible
```

---

## 9. Implementation Phases

### Phase 1: Preprocessing (both regions) — COMPLETE

**Goal**: Convert raw ADCIRC files into two training matrices.

**Status**: Complete. Run `uv run python preprocess.py` to regenerate.

**Script**: `preprocess.py` — reads all 193 maxele.63.nc files (handles both Danso directory layouts + CERA HSOFS identification by node count), extracts region nodes, clusters via KMeans, builds training matrices. Noise injection is NOT baked in — applied during DDPM training for flexibility.

**Outputs:**
```
data/processed/
├── obx/
│   ├── train_data.npy           # (361,242 × 21) — 183 scenarios × 1,974 patches
│   ├── val_dorian_data.npy      # (17,766 × 21) — 9 Danso Dorian scenarios
│   ├── val_arthur_data.npy      # (1,974 × 21) — 1 CERA Arthur scenario
│   ├── patch_assignments.pkl    # {patch_id: [20 node indices]}
│   ├── node_coords.npy          # (39,494 × 2) lat/lon of valid nodes
│   ├── node_indices.npy         # HSOFS indices for mapping back to full mesh
│   └── metadata.json            # Region config, θ stats, scenario lists
├── nyc/
│   ├── train_data.npy           # (41,829 × 21) — 191 scenarios × 219 patches
│   ├── val_sandy_data.npy       # (219 × 21) — Sandy θ=3.18m
│   ├── val_irene_data.npy       # (219 × 21) — Irene θ=2.35m
│   ├── patch_assignments.pkl
│   ├── node_coords.npy          # (4,393 × 2)
│   ├── node_indices.npy         # For DEM visualization mapping
│   └── metadata.json
└── preprocessing_report.json
```

### Phase 2: DDPM Training

**Goal**: Train two DDPM models (one per region).

**Steps:**
1. Smoke-test: run Wang et al.'s AIB9 example for ~1,000 steps to verify code works on MPS
2. Adapt `run_training.py` for flood data (change `op_num`, data path, device)
3. Train Outer Banks model first (~200K steps)
4. Train NYC model (~100K steps)
5. Monitor loss convergence, save checkpoints

**Compute notes:**
- MPS (Apple Silicon) should work with minor code changes (`cuda` → `mps`)
- If too slow, use Google Colab (free T4 GPU)
- OBX training may take longer due to larger dataset

### Phase 3: Experiments and Figures

**Goal**: Run all experiments from Section 7, generate all figures from Section 8.

**Steps:**
1. Adapt `gen_sample.py` for flood data
2. Run hold-out validation (Experiment 1) for both regions
3. Run extrapolation experiments (Experiment 2) — Dorian for OBX, Sandy for NYC
4. Run tipping-point sweep (Experiment 3) for both regions
5. Generate Miura-style flood maps for NYC results
6. Create all publication-quality figures

### Phase 4: Presentation

- 12–15 slides
- Narrative from Section 2 of this document
- Key figures from experiments
- Honest discussion of limitations (Section 10)

---

## 10. Known Limitations and Honest Framing

### What this project IS

- A demonstration that DDPM (a generative AI method from molecular dynamics) can be adapted to coastal flood prediction
- A conditional spatial prediction model: given surge at a reference gauge, predict the spatial flood pattern
- A tool for rapid scenario exploration (tipping points, what-if analysis) that would be prohibitively expensive with ADCIRC

### What this project IS NOT

- A direct replication of Wang et al.'s thermodynamic framework (θ is a spatial observation, not an external control parameter like temperature)
- A replacement for ADCIRC in operational forecasting
- A physically-based model (it learns statistical patterns, not fluid dynamics)

### Specific limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| θ is a spatial measurement, not an external control parameter | Weaker analogy to Wang et al. | Frame honestly as "conditional spatial prediction"; the practical utility is the same |
| NYC θ distribution is narrow (0.7–1.2m) | Model poorly trained for extreme NYC events | Noise injection + Outer Banks experiment proves method works with good data |
| Spatial patches are generated independently | Possible discontinuities at patch boundaries | Smoothing during stitching; overlapping patches as future work |
| Bathtub model for DEM visualization | Less accurate than GISSR's Manning's equation | Sufficient for demonstration; integrate Manning's as future work |
| No temporal dynamics (max surge only) | Cannot predict timing or duration of flooding | Addressed in Phase 2 plan (see Section 12) |
| θ is discrete (~41 unique values), not continuous like Wang et al.'s temperature | Model may struggle to interpolate smoothly | Noise injection (σ=0.15m) partially mitigates; fully addressed in Phase 2 plan |
| Training on 20 specific historical storms | May not generalize to fundamentally different storm types | Acknowledge as data limitation; more ADCIRC runs would help |

### How to handle tough questions

**Q: "Why not just use GISSR?"**
A: GISSR is fast (0.01s) but uses simplified physics (Manning's equation on divisions). Our approach learns from ADCIRC, the gold-standard simulator. Future work could combine both — use DDPM for surge prediction and GISSR for inland redistribution.

**Q: "How is this different from a simple regression model?"**
A: DDPM generates **distributions**, not point predictions. Run it 100 times at the same θ and you get 100 different plausible flood patterns — this captures the inherent uncertainty. A regression gives one answer.

**Q: "Can this work for other cities?"**
A: Yes — any region covered by the HSOFS mesh (entire US Atlantic coast). We demonstrate this with two regions (Outer Banks and NYC) using the same method.

---

## 11. File and Data Reference

### Data inventory

| Dataset | Location | Scenarios | Role |
|---|---|---|---|
| Danso et al. ADCIRC | `data/adcirc/` (20 storm dirs) | 180 | Primary training data (both regions) |
| CERA HSOFS storms | `data/cera/` (13 HSOFS files) | 13 | Supplementary training + validation |
| CERA non-HSOFS storms | `data/cera/` (18 other files) | 18 | Not used (incompatible mesh) |
| Miura DEM tiles | `GIS_FloodSimulation/Data/LM_div18/` | 18 tiles | NYC visualization only |
| Sandy validation | `data/validation/` | Various | NYC Sandy comparison |
| Wang et al. DDPM code | `DDPM_REMD/` | — | Base architecture to adapt |

### Danso directory layouts

**Type A (9 storms):** Delta, Dorian, Florence, Ian, Ida, Irma, Laura, Matthew, Michael
```
data/adcirc/{Storm}/{sea_level}/{forcing}/maxele.63.nc
```

**Type B (11 storms):** Bonnie, Charley, Floyd, Fran, Frances, Harvey, Ike, Isabel, Jeanne, Lili, Opal
```
data/adcirc/{Storm}/hot_start/{sea_level_mapped}/{forcing}/maxele.63.nc
```

Sea level mapping for Type B: `reference` → `ref` (slr directories keep their names: `slr_0.44m`, `slr_0.74m`)

**Note:** Harvey has 8 extra files in `old/` and `Vmax_10_old/` directories — exclude these.

### CERA HSOFS storms (13 compatible files)

**Identified by node count = 1,813,443 (matching HSOFS mesh used by Danso).**

| Storm | File | OBX θ | NYC θ | Notes |
|---|---|---|---|---|
| Frances (2004) | `2004_06_FRANCES_maxele.63.nc` | 0.70m | 1.03m | |
| Omar (2008) | `2008_15_OMAR_maxele.63.nc` | 0.75m | 1.10m | |
| Earl (2010) | `2010_07_EARL_maxele.63.nc` | 1.18m | 1.00m | |
| **Irene (2011)** | `2011_09_IRENE_maxele.63.nc` | 2.19m | **2.35m** | NYC validation |
| Debby (2012) | `2012_04_DEBBY_maxele.63.nc` | 0.55m | 0.89m | |
| **Sandy (2012)** | `2012_18_SANDY_maxele.63.nc` | 1.62m | **3.18m** | NYC validation |
| Andrea (2013) | `2013_01_ANDREA_maxele.63.nc` | 0.80m | 0.93m | |
| **Arthur (2014)** | `2014_01_ARTHUR_maxele.63.nc` | **2.05m** | 0.87m | OBX validation |
| Maria (2017) | `2017_15_MARIA_maxele.63.nc` | 0.68m | 1.00m | |
| Alberto (2018) | `2018_01_ALBERTO_maxele.63.nc` | 0.69m | 0.99m | |
| Michael (2018) | `2018_14_MICHAEL_maxele.63.nc` | 1.23m | 1.06m | |
| Fiona (2022) | `2022_07_FIONA_maxele.63.nc` | 0.56m | 0.86m | |
| Nicole (2022) | `2022_17_NICOLE_maxele.63.nc` | 0.86m | 1.01m | |

**Important**: Isabel (2003), Matthew (2016), Dorian (2019), Florence (2018) in CERA have different meshes (5.6M or 6.1M nodes) and are NOT HSOFS-compatible. The Danso versions of these storms ARE HSOFS.

### Technical gotchas

- **Always use `netCDF4.Dataset()`**, never xarray, for ADCIRC files
- **Filter zeta_max fill values**: `np.where((zeta > 1e10) | (zeta < -1e10) | (zeta <= -99999), np.nan, zeta)` — ADCIRC uses multiple fill conventions
- **Python environment**: `uv run python script.py`, packages managed via `pyproject.toml`
- **PyTorch device**: MPS (Apple Silicon), change `torch.device("cuda")` to `torch.device("mps")`
- **Miura DEM CRS**: EPSG:2263 (NY State Plane) — transform to EPSG:3857 for web basemaps

---

## 12. Phase 2: Time Series Approach (After Current Plan)

### Why this section exists

The current plan (Sections 1–11) uses **peak surge only** (`maxele.63.nc`) — one θ value per scenario. This is simpler, faster to implement, and uses smaller data. We execute this first to get results quickly.

However, additional ADCIRC time series data (`fort.63.nc`) is available on TACC for the 13 CERA HSOFS storms. This data would enable a fundamentally stronger approach that we plan to pursue after evaluating results from the current plan.

### What is the time series data?

The CERA dataset on TACC contains more than just peak surge. Key files per storm:

| File | Contents | Size |
|---|---|---|
| `maxele.63.nc` | Peak water elevation per node (one value) — **what we have now** | ~50MB |
| `fort.63.nc` | **Full time series** of water elevation at all nodes (every output timestep) | Multiple GB |
| `fort.64.nc` | Time series of depth-averaged water velocity | Multiple GB |
| `maxvel.63.nc` | Maximum velocity per node | ~50MB |
| `maxwvel.63.nc` | Maximum wind velocity | ~50MB |
| `swan_HS_max.63.nc` | Maximum significant wave height (ADCIRC+SWAN coupled) | ~50MB |

The `fort.63.nc` files are the important ones. They contain the full rising-and-falling surge curve at every mesh node throughout the storm — typically ~200–500 timesteps over 24–72 hours.

### Why time series data is a game-changer

In Wang et al., the conditioning variable (temperature) is the **instantaneous kinetic temperature** — it fluctuates naturally at every time frame. This gives a smooth, continuous distribution across thousands of unique values.

The `fort.63.nc` time series provides the **exact same thing for surge**:

- At each timestep, the reference gauge reads a different surge value (the water rises, peaks, and falls)
- Each timestep also gives the spatial surge pattern across all nodes at that exact moment
- This creates a naturally continuous θ distribution — no noise injection needed

| | Current plan (maxele) | Time series plan (fort.63) |
|---|---|---|
| θ values per storm | 1 (peak) | ~200–500 (one per timestep) |
| θ distribution | Discrete, needs noise injection | Naturally continuous |
| Training rows (13 CERA × OBX patches) | ~13K | ~3–6 million |
| Analogy to Wang et al. | Approximate | **Exact structural match** |
| Data download | Already done | ~30–100 GB from TACC |
| What model learns | Peak surge → peak flood pattern | Instantaneous surge → instantaneous flood pattern |

### How to access the data

The CERA time series data is on TACC (Texas Advanced Computing Center). The existing `download_cera.sh` script can be adapted — change `maxele.63.nc` to `fort.63.nc` in the SCP paths. TACC access requires MFA authentication (see HANDOVER_V2.md Section 4.1 for credentials and path details).

Only the **13 HSOFS storms** need to be downloaded (same ones we already use).

### When to pursue this

**After completing the current plan.** Specifically:

1. Finish Phases 1–3 of current plan (preprocessing, training, experiments with maxele data)
2. Evaluate results — does the DDPM produce reasonable flood predictions?
3. If results are promising but limited by discrete θ → pursue time series approach
4. If results already work well → time series becomes a "strengthening" improvement
5. If results fail entirely → diagnose whether the issue is θ distribution or something else before investing in the larger download

### What changes in the pipeline

The preprocessing script would need a new mode to handle `fort.63.nc`:
- Read time series at each timestep (not just peak)
- Extract spatial surge pattern at each timestep across all region nodes
- Compute θ = surge at reference gauge at that timestep
- Produce training rows: [θ_t, surge_node1_t, ..., surge_node20_t] per timestep per patch
- Filter out timesteps with very low surge (baseline/calm conditions add noise, not signal)

The DDPM training code stays identical — it just receives a much larger `.npy` file.

### Scientific framing

If we reach this phase, the narrative upgrades from:

> "We use peak surge as a scalar severity index with noise injection to approximate continuity"

to:

> "Following Wang et al., we treat the conditioning variable as a naturally fluctuating quantity. Just as Wang et al. use instantaneous kinetic temperature (which varies at every simulation frame), we use instantaneous storm surge at the reference gauge (which varies at every ADCIRC timestep). This provides a structurally identical framework — the DDPM learns the joint distribution of instantaneous surge observations across space, conditioned on the reference gauge reading."

This eliminates the need to justify noise injection and makes the Wang et al. analogy exact rather than approximate.
