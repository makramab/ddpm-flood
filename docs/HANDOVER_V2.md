# DDPM Flood — Handover Document (v2)

**Date**: February 27, 2026
**From**: Initial implementation team (data acquisition, PoC visualization, experiment design)
**To**: Domain expert for preprocessing, DDPM training, experiments, and presentation
**Status**: Data acquired, visualization PoC done, preprocessing + training not yet started

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Key Reference Documents](#2-key-reference-documents)
3. [How We Got Here — The Journey](#3-how-we-got-here--the-journey)
4. [Complete Data Inventory](#4-complete-data-inventory)
5. [The Pipeline — From Raw Data to Flood Map](#5-the-pipeline--from-raw-data-to-flood-map)
6. [What We've Built So Far](#6-what-weve-built-so-far)
7. [What Still Needs to Be Done](#7-what-still-needs-to-be-done)
8. [Technical Notes & Gotchas](#8-technical-notes--gotchas)
9. [File & Directory Map](#9-file--directory-map)

---

## 1. Project Context

### What is this project?

This is a **PhD application assessment project** for Prof. Yuki Miura at the CERA Lab (Climate, Energy, and Risk Analytics), NYU Tandon School of Engineering. The goal is to demonstrate the ability to produce research-quality results, not just a proposal.

### The core idea

Adapt the **DDPM (Denoising Diffusion Probabilistic Model)** framework from Wang et al. 2022 (originally used for molecular dynamics) to **urban flood prediction** in New York City. The DDPM replaces expensive ADCIRC storm surge simulations with instant predictions, enabling:

1. **Tipping-point discovery** — find critical surge thresholds where neighborhoods transition from safe to flooded
2. **Extrapolation to unprecedented storms** — predict what would happen during events worse than anything simulated
3. **Probabilistic flood risk** — generate distributions of outcomes, not single deterministic predictions

### The two key papers

1. **Wang et al. 2022** — "From Data to Noise to Data for Mixing Physics Across Temperatures with Generative Artificial Intelligence" (PNAS). This is the DDPM method we're adapting. The key insight: treat a control parameter (temperature) as a random variable, train DDPM on data from multiple temperatures, and generate at any temperature — including ones never simulated. Code: https://github.com/tiwarylab/DDPM_REMD

2. **Miura et al. 2021** — "High-Speed GIS-Based Simulation of Storm Surge–Induced Flooding Accounting for Sea Level Rise" (Natural Hazards Review). This is Prof. Miura's own work — a GIS-based flood simulator called GISSR. We use her DEM data and visual style for our output maps. Code: https://github.com/ym2540/GIS_FloodSimulation

### The analogy between the two

| Wang et al. (molecular dynamics) | Our project (flood prediction) |
|---|---|
| Temperature T (one number) | Storm surge θ at the Battery (one number) |
| Molecular configuration (18 dihedral angles) | Flood response (surge at ~20 nearby nodes per patch) |
| REMD simulations at T₁...Tₖ (expensive, hours each) | ADCIRC simulations at θ₁...θₖ (expensive, hours each) |
| Generate at unsimulated temperatures | Generate at unsimulated surge levels |
| Discover transition states between molecular conformations | Discover tipping points where neighborhoods go from dry to flooded |

### Timeline pressure

Prof. Miura expects **actual results with a presentation**, not a theoretical proposal. The target is approximately 1 week for a working demo with real figures.

---

## 2. Key Reference Documents

These files are in the repo and contain important context:

| File | What it contains | When to read it |
|---|---|---|
| `PROJECT_HANDOFF.md` | The **original** project brief written before any data work. Contains detailed background on both papers, the conceptual mapping, DDPM code structure (Section 8 is essential for the training phase), compute setup, and the 7-day execution plan. **Some sections are now outdated** (e.g., NACCS as primary data source) but the DDPM architecture details and code adaptation guide remain fully relevant. | Read Section 8 (DDPM code structure) before starting the training phase. Skim Sections 1-6 for background. |
| `EXPERIMENT_PLAN.md` | The **current** experiment plan, updated to reflect what we actually found during data investigation. Contains the spatial patch approach, θ explanation, full pipeline description, visualization methodology, and all expected figures. **This is the authoritative plan.** | Read this end-to-end before starting any work. |
| `PROGRESS.md` | Day-by-day progress tracker. Shows what's been checked off and what's pending. | Quick status check. |
| `refs/wang-et-al-2022-...pdf` | The Wang et al. DDPM paper (PDF). | Read if you need to understand the DDPM methodology deeply. |
| `refs/miura-et-al-2021-...pdf` | Miura's GISSR paper (PDF). Figures 8, 9, 12 are the visual style we're targeting. | Look at Figures 8, 9, 12 to understand the target output format. |

---

## 3. How We Got Here — The Journey

This section explains the decisions we made and why. Understanding this will save you from re-exploring dead ends.

### Starting point: the original plan (PROJECT_HANDOFF.md)

The original plan called for using **NACCS** (North Atlantic Comprehensive Coastal Study) as the primary training data — 1,050 synthetic storms simulated with ADCIRC, with surge values at ~19,000 save points along the Atlantic coast. This would have been ideal: many storms, systematic parameter variation, real ADCIRC.

### What went wrong with NACCS

We downloaded the NACCS geodatabase and explored it thoroughly. It contains:
- 16,326 save points with coordinates
- 1,050 storm tracks with parameters
- **BUT only summary statistics** (ARI return periods at 13 levels per point)

The actual per-storm surge matrix (1,050 storms × 16,326 points) is locked in the **Coastal Hazards System (CHS)** at `chs.erdc.dren.mil`, which is completely inaccessible. The NACCS REST API documented at `naccs.docs.apiary.io` has a dead server. We tried every available access method — all failed.

**Result**: NACCS cannot provide per-storm training data. Only aggregate statistics.

### DeepSurge — explored and rejected

DeepSurge (Zenodo) offers 900,000 synthetic storms with surge at 4,199 coastal nodes. We downloaded a sample and found:
- Only **3-7 nodes** in the NYC metro area (extremely sparse)
- The Battery tide gauge location is 22-34km from the nearest DeepSurge node
- DeepSurge is itself an ML model (LSTM trained on ADCIRC) — training our DDPM on ML-generated data creates an "ML-on-ML" chain that undermines the scientific argument ("ADCIRC is expensive, so we use DDPM" doesn't hold if we're not actually using ADCIRC data)

**Result**: DeepSurge rejected due to sparse NYC coverage and novelty concern.

### Finding Danso et al. — the breakthrough

The DeepSurge authors (including Danso) must have had actual ADCIRC data to train their ML model. We traced this back to:

**Danso et al. 2025** — "U.S. Atlantic Hurricane Storm Surge Events Simulations with Sea Level Rise and Storm Intensification" on Zenodo (https://zenodo.org/records/17601768). This contains:
- 20 real Atlantic hurricanes simulated with actual ADCIRC
- 9 scenario variants per storm (3 sea levels × 3 forcings) = 180 scenarios
- Full `maxele.63.nc` files with surge at every mesh node (1,813,443 nodes)
- Freely available, CC-BY-4.0 license

We downloaded all 20 storms (11.3 GB) and verified:
- 9,728 NYC metro nodes, 4,438 with surge data, 169 near the Battery
- **Problem**: Battery surge range is only 0.69m to 1.12m across all 180 scenarios. These storms mostly hit the Gulf Coast and Carolinas — they barely affect NYC.

### CERA/DesignSafe — extending the range

We got approved for the DesignSafe CERA dataset (PRJ-3932): 70 real historical storms with ADCIRC hindcasts. We downloaded 31 storms (~3.6 GB) using an SSH multiplexing script.

The CERA data has **4 different ADCIRC meshes** (HSOFS, STOFSatl, NAC2014, SABv20a). Only the **HSOFS mesh** (13 storms) is the same mesh as Danso's data, making them compatible for combined training.

Critically, the CERA HSOFS storms include:
- **Sandy (2012)**: Battery θ = 3.18m
- **Irene (2011)**: Battery θ = 2.14m
- Other storms ranging from 0.70m to 1.24m

### The final training dataset: 193 scenarios

| Source | Scenarios | Battery θ range |
|---|---|---|
| Danso et al. | 180 | 0.69m – 1.12m |
| CERA HSOFS | 13 | 0.70m – 3.18m |
| **Total** | **193** | **0.69m – 3.18m** |

All 193 scenarios use the identical HSOFS mesh (1,813,443 nodes), so they can be combined directly.

### The spatial patch approach — making 193 scenarios enough

193 scenarios is far less than Wang et al.'s 500,000 samples. But DDPM doesn't see "scenarios" — it sees rows of data. By splitting the 4,438 NYC nodes into ~220 spatial patches of ~20 nodes each, every scenario produces ~220 training rows:

```
193 scenarios × ~220 patches = ~42,460 training rows × 21 columns
```

This is ~2,000:1 row-to-column ratio, compared to Wang et al.'s 26,000:1. Not as generous but within a workable range — Wang et al. used 19 columns (AIB9 peptide), we use ~21.

### The visualization breakthrough — Miura's DEM

Our early visualizations of ADCIRC data looked nothing like Miura's maps. The reason: ADCIRC gives surge at scattered coastal mesh nodes, while Miura's maps show crisp, block-level flood boundaries on land.

The fix: use Miura's **DEM (Digital Elevation Model)** — a 30m-resolution grid of ground elevation covering all of Lower Manhattan — which is freely available in her GitHub repo (`GIS_FloodSimulation/Data/LM_div18/`). The visualization becomes:

```
For each 30m DEM cell:
    if surge_height > ground_elevation → FLOODED (blue)
    else → DRY (show street map)
```

This produces maps visually near-identical to Miura's published Figures 8, 9, and 12.

---

## 4. Complete Data Inventory

### 4.1 Data we ARE using for training

#### Danso et al. ADCIRC Simulations — `data/adcirc/`

| Property | Value |
|---|---|
| Location | `data/adcirc/{StormName}/{sea_level}/{forcing}/maxele.63.nc` |
| # Storms | 20 |
| # Scenarios | 180 (20 × 9 variants) |
| File format | NetCDF4 (`maxele.63.nc`) |
| Mesh | HSOFS, 1,813,443 nodes |
| Key variables | `x` (lon), `y` (lat), `depth` (bathymetry), `zeta_max` (max water elevation), `element` (triangle connectivity) |
| Total size | ~11.3 GB |
| Battery θ range | 0.686m – 1.115m |
| How to read | `netCDF4.Dataset()` — do NOT use xarray (see Section 8) |

**The 20 storms**: Bonnie, Charley, Delta, Dorian, Florence, Floyd, Fran, Frances, Harvey, Ian, Ida, Ike, Irma, Isabel, Jeanne, Laura, Lili, Matthew, Michael, Opal

**Directory structure note**: 9 of the 20 storms have a different directory layout. Some use `reference/historical/maxele.63.nc`, others use `hot_start/ref/historical/maxele.63.nc`. A path resolution function that tries both structures is needed during preprocessing.

#### CERA HSOFS Storms — `data/cera/`

| Property | Value |
|---|---|
| Location | `data/cera/{YEAR}_{NUM}_{NAME}_maxele.63.nc` |
| # Storms (HSOFS mesh) | 13 |
| # Storms (total downloaded) | 31 (but only 13 are on HSOFS mesh) |
| File format | NetCDF4 (`maxele.63.nc`) |
| Mesh | HSOFS (same as Danso), 1,813,443 nodes |
| Total size | ~3.6 GB |
| Battery θ range | 0.70m – 3.18m |

**The 13 HSOFS storms** (the ones compatible with Danso data):
- 2003_13_ISABEL, 2004_06_FRANCES, 2008_15_OMAR, 2010_07_EARL
- **2011_09_IRENE** (θ = 2.14m), 2012_04_DEBBY, **2012_18_SANDY** (θ = 3.18m)
- 2013_01_ANDREA, 2014_01_ARTHUR, 2017_15_MARIA, 2018_01_ALBERTO
- 2018_14_MICHAEL, 2022_07_FIONA

**Non-HSOFS storms** (downloaded but NOT used for training — different meshes):
- 9 storms on SABv20a mesh (no Battery nodes at all)
- 8 storms on STOFSatl mesh
- 1 storm on NAC2014 mesh

**Additional CERA data available on TACC (not downloaded)**: We only downloaded `maxele.63.nc` (peak surge) from each storm. The full CERA dataset on TACC (~3.1 TB total) also contains other ADCIRC output files for each storm, including:

| File | What it contains | Typical size |
|---|---|---|
| `maxele.63.nc` | Max water elevation at each node (one value per node) — **this is what we downloaded** | 100-200 MB |
| `fort.63.nc` | **Full time series** of water elevation at all nodes (every output timestep throughout the storm — the rising and falling surge curve) | GBs per storm |
| `fort.64.nc` | Time series of depth-averaged water velocity | GBs per storm |
| `maxvel.63.nc` | Maximum velocity at each node | 100-200 MB |
| `maxwvel.63.nc` | Maximum wind velocity | 100-200 MB |
| `minpr.63.nc` | Minimum barometric pressure | Smaller |
| `swan_HS_max.63.nc` | Maximum significant wave height (ADCIRC+SWAN coupled) | 100-200 MB |

The time series data (`fort.63.nc`) is what Miura uses in GISSR to compute the storm surge + tide time history — the rising and falling curves shown in her Figure 10. For our current approach we only need peak values (`maxele.63.nc`), but the time series could be useful for:
- A more sophisticated conditioning variable (e.g., storm duration, rise rate, not just peak)
- Validating against Miura's time history plots (Figure 10)
- Future work on temporal flood dynamics

The same `download_cera.sh` script can be adapted to download these files — just change `maxele.63.nc` to `fort.63.nc` in the SCP path. TACC access credentials: username `makramab` at `data.tacc.utexas.edu`, base path is in the script. Access requires TACC MFA (password + token).

#### Miura's DEM — `GIS_FloodSimulation/Data/LM_div18/`

| Property | Value |
|---|---|
| Location | `GIS_FloodSimulation/Data/LM_div18/dem_lm_z35_{0-17}.TIF` |
| # Tiles | 18 |
| Format | GeoTIFF |
| CRS | EPSG:2263 (NY State Plane, Long Island) |
| Resolution | 30m × 30m per cell |
| Coverage | Lower Manhattan (from Battery to ~Midtown) |
| Merged size | 703 × 445 cells, ~199,000 valid pixels |
| Elevation range | 0.0m – 21.0m |
| Purpose | Visualization only — converting surge height to pixel-level flood map |

### 4.2 Validation data — `data/validation/`

| File | What it contains | Format | Records |
|---|---|---|---|
| `sandy_hwm.csv` | USGS High Water Marks from Hurricane Sandy. Each record has lat, lon, and peak water elevation in feet (NAVD88). These are actual field measurements of how high water reached during Sandy. | CSV | 347 |
| `sandy_inundation_zone/` | Official Sandy inundation zone polygons from NYC Open Data. Shows the actual area that flooded during Sandy. | Shapefile (+ .geojson) | 492 polygons |
| `battery_tide_sandy.csv` | NOAA Battery tide gauge readings during Sandy. Peak: 14.06 ft = 4.29m at 2012/10/30 01:24 GMT. This is the most-cited Sandy surge number. | CSV | Time series |
| `sandy_instruments.csv` | USGS sensor instrument records during Sandy. | CSV | 69 |

**How validation will work**: Hold Sandy out of training (or use Danso-only training which doesn't include Sandy). Set θ = 3.18m or 4.29m. Generate a flood map. Compare to HWMs and inundation zone.

### 4.3 Data we explored but are NOT using

| Dataset | Location | Why not used |
|---|---|---|
| NACCS geodatabase | `data/naccs/NACCS.gdb` | Only has summary statistics, no per-storm surge matrix |
| DeepSurge | `data/deepsurge/raft_cmip_max_surge_hist_EC-Earth3.nc` | Only 3-7 NYC nodes; ML-on-ML novelty concern |
| CERA non-HSOFS storms | `data/cera/*` (18 of 31 files) | Different ADCIRC meshes, incompatible with Danso data |

### 4.4 Reference repositories (cloned, gitignored)

| Repo | Location | Purpose |
|---|---|---|
| Wang et al. DDPM_REMD | `DDPM_REMD/` | The DDPM code to adapt for training. See `PROJECT_HANDOFF.md` Section 8 for detailed code walkthrough. |
| Miura GISSR | `GIS_FloodSimulation/` | DEM data for visualization + reference for how Miura's flood maps are generated. |

---

## 5. The Pipeline — From Raw Data to Flood Map

### 5.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING (done once)                       │
│                                                                 │
│  193 ADCIRC files                                               │
│       ↓                                                         │
│  Extract ~4,438 NYC nodes + compute Battery θ per scenario      │
│       ↓                                                         │
│  Cluster into ~220 spatial patches of ~20 nodes                 │
│       ↓                                                         │
│  Build .npy matrix: (42,460 rows × 21 columns)                 │
│       ↓                                                         │
│  Train DDPM (Wang et al. architecture)                          │
│       ↓                                                         │
│  Trained model checkpoint                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (instant, repeatable)               │
│                                                                 │
│  User provides θ (one number, e.g. 3.0)                         │
│       ↓                                                         │
│  DDPM generates surge at ~220 patches × 20 nodes each           │
│       ↓                                                         │
│  Stitch patches → surge at all ~4,438 NYC nodes                 │
│       ↓                                                         │
│  Compare surge to Miura's DEM (surge > elevation = flooded)     │
│       ↓                                                         │
│  Pixel-level flood map (Miura style)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 What is θ?

θ (theta) is a **single number**: the storm surge height at "The Battery" in meters.

**The Battery** is the southern tip of Manhattan — next to Battery Park and the Staten Island Ferry terminal. NOAA operates tide gauge Station 8518750 there. It's the most important sea-level measurement point in NYC, with 100+ years of data.

θ tells you "how bad is the storm?" in one number:
- θ = 1.0m → minor storm, some waterfront flooding
- θ = 2.0m → moderate storm (roughly Irene-level)
- θ = 3.18m → Sandy-level (ADCIRC measurement)
- θ = 4.29m → Sandy actual tide gauge reading (including tidal component)
- θ = 5.0m → extreme unprecedented event

In the DDPM, θ sits in **column 0** of the training data and is **never noised** during diffusion (the "inpainting" approach from Wang et al.). During generation, you fix column 0 to your desired θ value and the DDPM fills in the remaining columns (surge at the 20 patch nodes). No code changes are needed from Wang et al.'s implementation — just set `unmask_number = 1`.

### 5.3 The spatial patch approach

**The problem**: 193 scenarios × 4,438 nodes = 193 rows × 4,438 columns. DDPM can't train on this — too few rows, too many columns.

**The solution**: Split the 4,438 nodes into ~220 groups of ~20 nearby nodes (using spatial clustering, e.g., KMeans on lat/lon). Each group is a "patch."

For each scenario and each patch, create one training row:
```
[θ, surge_node_1, surge_node_2, ... surge_node_20]
```

This gives: 193 scenarios × 220 patches = **42,460 rows × 21 columns**

Wang et al. trained on 500,000 rows × 19 columns (AIB9 peptide). Our 42,460 × 21 is smaller but similar in column count. The architecture is designed for this scale.

**During inference**: generate all ~220 patches independently for a given θ, then stitch them back together to get surge at all 4,438 nodes.

### 5.4 The DEM visualization step

The DDPM outputs surge at scattered ADCIRC mesh nodes. To create the crisp, block-level flood maps that Miura's paper shows:

1. Load Miura's 18 DEM tiles (30m resolution grid of ground elevation for Lower Manhattan)
2. For each 30m cell: if `surge_height > ground_elevation` → cell is FLOODED
3. Color flooded cells in cyan/blue, show street basemap underneath
4. Add cartographic elements (scale bar, north arrow, legend, simulation boundary)

This is a simplified "bathtub" model. Miura's full GISSR uses Manning's equation for water flow redistribution, which is more physically accurate. For the presentation, the bathtub model is sufficient — the DEM does the heavy lifting in determining flood boundaries.

---

## 6. What We've Built So Far

### 6.1 Visualization proof-of-concept

**File**: `visualize_flood.py`

This script generates flood maps from ADCIRC data. It has two approaches:

1. **ADCIRC mesh-based** (v1/v2): Reads ADCIRC mesh nodes, interpolates surge onto a grid, masks water bodies using the `depth` variable, overlays on a basemap. Produces smooth but less crisp maps.

2. **Miura DEM-based** (final): Uses Miura's 18 DEM tiles. Given a surge height, classifies each 30m cell as flooded or dry. Produces maps visually near-identical to Miura's published figures.

### 6.2 Sample outputs (in `data/`)

| File | What it shows | Style |
|---|---|---|
| `viz_miura_style_sandy.png` | Sandy flood extent on Lower Manhattan. Single cyan layer showing all cells below 2.864m elevation. | Miura-style (DEM-based, crisp boundaries) |
| `viz_miura_style_sandy_slr.png` | Sandy + SLR scenarios. Three overlapping layers: Sandy (light cyan), Sandy+SLR 2050 (medium cyan), Sandy+SLR 2100 (dark blue). Directly replicates Miura Fig 9/12 visual style. | Miura-style (3-layer) |
| `viz_miura_style_theta_sweep.png` | 4-panel showing progressive flooding at θ = 1.0, 2.0, 3.0, 4.0m. Shows tipping-point concept: at 1.0m only piers flood; at 4.0m large areas are submerged. | Miura-style (4-panel) |
| `viz_sandy_lower_manhattan_v2.png` | Sandy surge on Lower Manhattan using ADCIRC mesh interpolation. Smooth contours, land-masked. | ADCIRC mesh-based |
| `viz_sandy_nyc_wide_v2.png` | NYC-wide Sandy surge map. Shows flooding in Jamaica Bay, Rockaway, Coney Island, Red Hook. | ADCIRC mesh-based |
| `viz_sandy_comparison_v2.png` | Side-by-side: ADCIRC flood vs. USGS High Water Mark overlay. 34 HWMs plotted in Lower Manhattan. | ADCIRC mesh-based + validation |
| `viz_theta_sweep_v2.png` | 4-panel theta sweep using ADCIRC mesh interpolation. | ADCIRC mesh-based |

**The Miura-style outputs are the ones to use for the presentation.** The ADCIRC mesh-based outputs are useful for understanding the data but are not presentation-quality.

### 6.3 Download script

**File**: `download_cera.sh`

Bash script that downloads all 31 CERA maxele files from TACC (Texas Advanced Computing Center) via SCP with SSH multiplexing. Enter password/token once, downloads all files automatically. Skips already-downloaded files.

### 6.4 TACC helper script

**File**: `tacc_download_script.sh`

An earlier attempt to copy files on TACC and tar them. Failed due to disk quota. Kept for reference but not needed.

---

## 7. What Still Needs to Be Done

### Phase 1: Preprocessing (estimated: 1 day)

**Goal**: Convert 193 raw ADCIRC files into a single `flood_training_data.npy` that the DDPM code can consume.

Steps:
1. **Write a preprocessing script** that:
   - Reads each of the 193 `maxele.63.nc` files using `netCDF4.Dataset()`
   - Extracts the ~4,438 NYC nodes (bbox: lat 40.45–40.95, lon -74.30 to -73.70) that have valid surge
   - Computes θ = peak surge at Battery nodes (lat ~40.7, lon ~-74.01) for each scenario
   - Note: Danso storms have two directory layouts — need a path resolver (see Section 8.2)

2. **Cluster NYC nodes into ~220 spatial patches** of ~20 nodes each:
   - Use KMeans or similar on (lat, lon) coordinates
   - Save patch assignments as `patch_assignments.pkl`

3. **Build the training matrix**:
   - For each scenario × each patch: one row = [θ, surge₁, surge₂, ..., surge₂₀]
   - Shape: (~42,460, 21)
   - Normalize with `sklearn.preprocessing.StandardScaler`
   - Save as `flood_training_data.npy` and `scaler.pkl`

4. **Hold out validation data**:
   - Reserve 3-4 storms for validation (don't include their patches in training)
   - Reserve Sandy (CERA) for extrapolation test
   - Save held-out data separately

### Phase 2: DDPM Training (estimated: 1-2 days)

**Goal**: Train the DDPM on the flood data and get a working model.

Steps:
1. **Verify DDPM_REMD works**: Run the AIB9 example first (`cd DDPM_REMD && python run_training.py`) for ~1000 steps to confirm no errors.

2. **Adapt `run_training.py`**:
   - `op_num = 20` (number of response variables per patch)
   - `unmask_number = 1` (θ is one scalar in column 0)
   - `train_num_steps = 50000–100000` (we have ~42K samples, not 500K)
   - `batch_size = 64` (adjust based on memory)
   - Data path → your `flood_training_data.npy`
   - See `PROJECT_HANDOFF.md` Section 8 for full code structure details

3. **Train and monitor**: Loss should converge. If it plateaus early, try reducing `dim` from 32 to 16.

4. **Note on compute**: The code was written for CUDA GPUs. MPS (Apple Silicon) may work with minor modifications but is untested. Google Colab (free tier, T4 GPU) is recommended if no CUDA GPU is available locally. See `PROJECT_HANDOFF.md` Section 9 for environment details.

### Phase 3: Experiments (estimated: 1-2 days)

**Goal**: Run the three key experiments and generate figures.

#### Experiment 1: Hold-out validation (quantitative)
- Use `gen_sample.py` to generate surge for held-out storms
- Compare DDPM predictions to actual ADCIRC values
- Metrics: R², RMSE
- Output: scatter plot (Figure 1 in EXPERIMENT_PLAN.md)

#### Experiment 2: Sandy extrapolation (qualitative)
- Set θ = 3.18m (or 4.29m including tidal component)
- Generate DDPM flood prediction
- Render on Miura's DEM
- Compare to Sandy HWMs and inundation zone
- Output: side-by-side Miura-style map (Figure 2)

#### Experiment 3: Tipping-point discovery (the headline result)
- Generate at θ = 0.5, 0.6, 0.7, ... 5.0m (dense sweep)
- For each θ, stitch patches into full NYC surge map
- Render each on Miura's DEM → calculate % of Lower Manhattan flooded
- Plot response curves: surge at key locations vs. θ
- Identify sharp jumps → these are the tipping points
- Output: response curves (Figure 3) + 4-panel flood map progression (Figure 4)

### Phase 4: Presentation (estimated: 1 day)

- 12-15 slides
- Key figures from experiments
- Narrative: see `EXPERIMENT_PLAN.md` Section 8
- Slide structure: see `PROJECT_HANDOFF.md` Section 13
- Prepare for questions from Prof. Miura about:
  - Why not use GISSR as training data? (GISSR is too fast/simple — no computational bottleneck for DDPM to solve)
  - How does this compare to Microsoft Aurora? (Aurora does weather at 28km resolution, not urban-scale flooding)
  - What about the bathtub model simplification? (Future work: integrate Manning's equation)

---

## 8. Technical Notes & Gotchas

### 8.1 ADCIRC files must use netCDF4, NOT xarray

```python
# THIS FAILS:
import xarray as xr
ds = xr.open_dataset('maxele.63.nc')  # ValueError: dimension 'neta' already exists as a scalar variable

# THIS WORKS:
import netCDF4 as nc
ds = nc.Dataset('maxele.63.nc')
x = ds.variables['x'][:]       # longitude
y = ds.variables['y'][:]       # latitude
depth = ds.variables['depth'][:] # bathymetry (>0 = underwater, ≤0 = land)
zeta_max = ds.variables['zeta_max'][:]  # max water surface elevation
elements = ds.variables['element'][:] - 1  # triangle connectivity (0-indexed)
ds.close()
```

The `neta` dimension in ADCIRC files conflicts with xarray's internal handling. Always use `netCDF4.Dataset()` directly.

### 8.2 Danso storms have two directory structures

9 of the 20 Danso storms use a different directory layout:

**Type A** (11 storms): `data/adcirc/{Storm}/reference/historical/maxele.63.nc`
**Type B** (9 storms): `data/adcirc/{Storm}/hot_start/ref/historical/maxele.63.nc`

You need a path resolution function that tries both:

```python
def find_maxele(storm, sea_level, forcing):
    # Type A path
    path_a = f"data/adcirc/{storm}/{sea_level}/{forcing}/maxele.63.nc"
    if os.path.exists(path_a):
        return path_a
    # Type B path
    sl_map = {"reference": "ref", "slr_0.44m": "slr_044", "slr_0.74m": "slr_074"}
    path_b = f"data/adcirc/{storm}/hot_start/{sl_map.get(sea_level, sea_level)}/{forcing}/maxele.63.nc"
    if os.path.exists(path_b):
        return path_b
    return None
```

### 8.3 CERA has 4 different ADCIRC meshes

Only 13 of the 31 downloaded CERA storms use the HSOFS mesh (same as Danso). The others use different meshes with different node positions:

| Mesh | # Storms | # Nodes | Battery nodes? |
|---|---|---|---|
| HSOFS | 13 | 1,813,443 | Yes (169) |
| STOFSatl | 8 | 6,056,968 | Yes |
| NAC2014 | 1 | 3,110,470 | Yes |
| SABv20a | 9 | 5,584,241 | **No** |

**Only use the 13 HSOFS storms for training.** Mixing meshes would require spatial interpolation between different node grids, which adds complexity without much benefit for a PoC.

### 8.4 The `depth` variable distinguishes land from water

In ADCIRC files:
- `depth > 0` = underwater (ocean, river)
- `depth ≤ 0` = land (elevation above sea level = `-depth`)
- Inundation on land = `zeta_max + depth` (since depth is negative on land, this subtracts ground elevation)

This is useful for masking water bodies in ADCIRC-based visualizations (not needed for DEM-based Miura-style maps).

### 8.5 zeta_max fill values

ADCIRC uses `-99999` or very large negative values as fill/nodata for `zeta_max` at dry nodes. Always filter:

```python
zeta_max = np.where((zeta_max > 1e10) | (zeta_max < -1e10), np.nan, zeta_max)
```

### 8.6 Python environment

- Python 3.11, managed by **uv** (not conda, not pip directly)
- `uv run python script.py` to run scripts
- `uv add package-name` to install packages
- PyTorch 2.10.0 with MPS backend (Apple Silicon) — CUDA not available on this machine
- Key packages installed: netCDF4, numpy, matplotlib, contextily, rasterio, scipy, pyproj, geopandas, matplotlib-scalebar

### 8.7 Miura's DEM CRS

The DEM tiles are in **EPSG:2263** (NY State Plane, Long Island, US Survey Feet). When overlaying on web basemaps, transform to EPSG:3857 (Web Mercator). The `pyproj.Transformer` handles this.

---

## 9. File & Directory Map

```
ddpm-flood/
│
├── PROJECT_HANDOFF.md          ← Original project brief (background, DDPM code guide, 7-day plan)
├── EXPERIMENT_PLAN.md          ← Current experiment plan (updated with actual findings)
├── PROGRESS.md                 ← Day-by-day progress tracker
├── HANDOVER_V2.md              ← THIS FILE
│
├── main.py                     ← (Placeholder / entry point — not yet implemented)
├── visualize_flood.py          ← Visualization script (ADCIRC mesh + Miura DEM approaches)
├── download_cera.sh            ← Script to download CERA data from TACC
├── tacc_download_script.sh     ← Earlier TACC helper (failed, kept for reference)
│
├── pyproject.toml              ← Python project config (uv)
├── uv.lock                     ← Dependency lock file
├── .python-version             ← Python 3.11
├── .gitignore                  ← Ignores data/, .venv/, cloned repos, binary files
│
├── refs/                       ← Reference papers (PDF)
│   ├── wang-et-al-2022-...pdf
│   └── miura-et-al-2021-...pdf
│
├── DDPM_REMD/                  ← [gitignored] Wang et al. DDPM code (cloned)
│   ├── denoising_diffusion_pytorch/
│   │   └── denoising_diffusion_pytorch.py   ← Core DDPM architecture (611 lines)
│   ├── run_training.py                      ← Training script (adapt this)
│   ├── gen_sample.py                        ← Generation script (adapt this)
│   └── traj_AIB9/                           ← Example data (500K × 19)
│
├── GIS_FloodSimulation/        ← [gitignored] Miura GISSR code (cloned)
│   ├── Data/
│   │   ├── LM_div18/                       ← 18 DEM tiles (THE key data for visualization)
│   │   ├── LM_div18_grouped/               ← Grouped DEM tiles
│   │   ├── NewSurfaceVolumeCombined/        ← Height-volume lookup curves
│   │   ├── SurgeData/                       ← Storm surge parameters
│   │   ├── LMN_div_low19.csv               ← Division elevations
│   │   ├── LMN_Roughness.csv               ← Manning's roughness coefficients
│   │   └── LMN_Slope.csv                   ← Division slopes
│   ├── Flood_Estimate_from_Surge_Height.ipynb  ← Main GISSR notebook
│   └── Sample_Output_Data/                  ← Example GISSR outputs
│
└── data/                       ← [gitignored] All downloaded data
    ├── adcirc/                 ← Danso et al. — 20 storms × 9 variants (180 scenarios)
    │   ├── Bonnie/
    │   ├── Charley/
    │   ├── ... (20 storm directories)
    │   └── Opal/
    ├── cera/                   ← CERA — 31 maxele files (13 HSOFS + 18 other meshes)
    │   ├── 2012_18_SANDY_maxele.63.nc    ← KEY FILE (θ = 3.18m)
    │   └── ... (30 more files)
    ├── naccs/                  ← NACCS geodatabase (explored, not used for training)
    │   └── NACCS.gdb
    ├── deepsurge/              ← DeepSurge sample (explored, not used)
    │   └── raft_cmip_max_surge_hist_EC-Earth3.nc
    ├── validation/             ← Sandy validation data
    │   ├── sandy_hwm.csv
    │   ├── sandy_inundation_zone/
    │   ├── sandy_inundation_zone.geojson
    │   ├── battery_tide_sandy.csv
    │   └── sandy_instruments.csv
    ├── viz_miura_style_sandy.png         ← PoC output: Miura-style Sandy flood
    ├── viz_miura_style_sandy_slr.png     ← PoC output: Sandy + SLR (3-layer)
    ├── viz_miura_style_theta_sweep.png   ← PoC output: θ = 1.0, 2.0, 3.0, 4.0m
    ├── viz_sandy_lower_manhattan_v2.png  ← ADCIRC mesh visualization
    ├── viz_sandy_nyc_wide_v2.png         ← ADCIRC mesh NYC wide
    ├── viz_sandy_comparison_v2.png       ← ADCIRC vs HWM comparison
    └── viz_theta_sweep_v2.png            ← ADCIRC mesh θ sweep
```

---

## Quick Start Checklist

If you're picking this up fresh, here's the recommended reading order:

1. Read this file (`HANDOVER_V2.md`) end to end — you're doing this now
2. Skim `EXPERIMENT_PLAN.md` — the authoritative experiment plan with all details
3. Look at the PoC images in `data/viz_miura_style_*.png` — this is the target output format
4. Look at Miura's paper Figures 8, 9, 12 (`refs/miura-et-al-2021-...pdf`, pages 8-12) — this is what we're matching
5. Read `PROJECT_HANDOFF.md` Section 8 — the DDPM code structure and adaptation guide
6. Start with Phase 1 (preprocessing) from Section 7 above
