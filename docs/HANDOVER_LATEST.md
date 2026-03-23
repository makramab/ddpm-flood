# DDPM Flood — Latest Handover Document

**Date**: March 7, 2026
**Author**: Afra Izzati Kamili (with Claude Code assistance)
**Audience**: Domain expert preparing a presentation for Prof. Yuki Miura (CERA Lab, NYU Tandon)
**Status**: V2 model complete. Interactive visualization app complete. Presentation deck drafted. Deep-dive review needed before presenting.

---

## Table of Contents

1. [Purpose of This Document](#1-purpose-of-this-document)
2. [Project Summary](#2-project-summary)
3. [Document Trail — Chronological Index](#3-document-trail--chronological-index)
4. [Current State of the Work](#4-current-state-of-the-work)
5. [The Data — Sources, Processing, and Pipeline](#5-the-data--sources-processing-and-pipeline)
6. [The Model — Architecture, Training, and Generation](#6-the-model--architecture-training-and-generation)
7. [The Results — What Worked, What Failed, and Why](#7-the-results--what-worked-what-failed-and-why)
8. [The Visualization App](#8-the-visualization-app)
9. [The Primer — Deep Technical Explainer](#9-the-primer--deep-technical-explainer)
10. [The Presentation Deck](#10-the-presentation-deck)
11. [What the Presenter Needs to Know](#11-what-the-presenter-needs-to-know)
12. [Recommended Next Step: Time-Series Conditioning](#12-recommended-next-step-time-series-conditioning)
13. [Repository Map](#13-repository-map)
14. [Environment and Setup](#14-environment-and-setup)

---

## 1. Purpose of This Document

This is the **single entry point** for someone picking up this project to prepare a presentation for Prof. Yuki Miura. It assumes you have domain knowledge in coastal engineering, storm surge modeling, or related fields.

What you should do with this document:

1. **Read it end-to-end** to understand the full project state
2. **Cross-reference the primer and presentation** (Section 9 and 10) to verify the scientific content is accurate and well-framed
3. **Explore the actual data** — the scripts, preprocessed files, and raw ADCIRC outputs are all available (Section 5)
4. **Review the model architecture and training** — understand the choices made and their justification (Section 6)
5. **Understand the limitations deeply** — these are the most likely points of questioning (Section 7 and 11)
6. **Run the visualization app** to see the interactive results (Section 8)

All previous documentation has been moved to `docs/` and is indexed chronologically in Section 3. You do not need to read them all — this document synthesizes everything — but they contain detailed context if you need to trace a specific decision.

---

## 2. Project Summary

### The assignment

Prof. Yuki Miura assigned this as a PhD application assessment. The task: apply the DDPM framework from Wang et al. 2022 (originally for molecular dynamics) to coastal storm surge prediction. She expects **actual results with a presentation**, not just a proposal.

### The core idea

Wang et al. trained a 1D U-Net DDPM on molecular dynamics trajectories, conditioning on **instantaneous kinetic temperature** (a naturally fluctuating scalar). The model learns P(molecular coordinates | temperature) and can generate at any temperature — including ones never simulated.

This project adapts that framework:

| Wang et al. (Molecular Dynamics) | This Project (Storm Surge) |
|---|---|
| Temperature T (1 scalar) | Peak surge θ at reference gauge (1 scalar) |
| Molecular coordinates (18 values) | Surge at 20 nearby mesh nodes (20 values) |
| MD simulation frames | ADCIRC hurricane simulation outputs |
| P(coordinates \| T) | P(surge pattern \| θ, latitude, longitude, depth) |

### The key result

The model works excellently as a **within-ensemble surrogate** for the training storm family (Danso synthetic storms at in-distribution θ values: R² = 0.95, RMSE = 0.06m). It fails at cross-storm generalization (real historical storms) and extrapolation (θ values beyond training range).

### The study region

**Outer Banks, North Carolina** — not NYC. NYC was the original target but had insufficient data coverage (82% of training scenarios produced < 1m surge at the Battery). The Outer Banks gets direct hurricane landfalls, providing θ values from 0.4m to 4.0m.

---

## 3. Document Trail — Chronological Index

All documents are in `docs/`. Read them in order only if you need to trace the full decision history. Otherwise, this handover document is sufficient.

### Timeline

```
Feb 24  PROJECT_HANDOFF.md     Original brief
   │    ↓ Data investigation: NACCS inaccessible, found Danso et al.
Feb 27  EXPERIMENT_PLAN.md     Updated plan with actual data sources
   │    ↓ NYC θ distribution too narrow → two-region strategy
Feb 27  CURRENT_PLAN.md        Two-region approach (OBX + NYC)
   │
Feb 27  PROGRESS.md            Day-by-day task tracker
   │
Feb 27  HANDOVER_V2.md         Full data inventory, TACC access, file map
   │    ↓ V1 trained → spatially flat predictions
Feb 28  CURRENT_PLAN_V2.md     Spatial conditioning fix (lat/lon/depth)
   │    ↓ V2 trained → in-distribution success, cross-storm failure
Mar 1   RESULTS_V2.md          V2 results, limitation analysis, diagnostic
   │    ↓ Synthesized findings for handover
Mar 5   HANDOVER_V3.md         Comprehensive handover with V3 options
   │    ↓ Viz app + primer + presentation built
Mar 7   HANDOVER_LATEST.md     THIS FILE — final handover for presenter
   │    ↓ Literature review of 10 papers, explored alternatives
Mar 23  HANDOVER_LIT_REVIEW.md Literature review, landscape comparison, time-series confirmed
```

### Per-document details

| # | File | Date | What it covers | Still valid? |
|---|------|------|---------------|--------------|
| 1 | `PROJECT_HANDOFF.md` | Feb 24 | Original project brief. Background on Wang et al. and Miura et al. papers. **Section 8 (DDPM code structure)** remains the best guide to the Wang et al. codebase. Section 7 (data sources) is outdated — NACCS turned out to be inaccessible. | Section 8: Yes. Sections 1-6: Background only. Section 7+: Superseded. |
| 2 | `EXPERIMENT_PLAN.md` | Feb 27 | Updated after data investigation. Documents the Danso + CERA combined dataset. Explains the spatial patch approach, θ definition, and visualization strategy. | Mostly yes. Training format superseded by V2 (21→24 columns). |
| 3 | `CURRENT_PLAN.md` | Feb 27 | Introduces the two-region strategy (OBX primary, NYC application). Explains why OBX was chosen over NYC. Contains the θ noise injection design (σ=0.15m). Has the Phase 2 time-series plan in Section 12. | Yes for strategy rationale. Training details superseded by V2. |
| 4 | `PROGRESS.md` | Feb 27+ | Task tracker. Only Day 1 items are checked off. Reflects the original 7-day plan before the project evolved. | Historical interest only. |
| 5 | `HANDOVER_V2.md` | Feb 27 | The data journey — how we went from NACCS → DeepSurge → Danso → CERA. Contains TACC download credentials (username `makramab`), the download script, and the complete file/directory map. | Yes for data access and file structure. |
| 6 | `CURRENT_PLAN_V2.md` | Feb 28 | Diagnoses V1's spatial flatness. Designs the V2 fix (add lat/lon/depth conditioning). Contains V1 baseline results, correlation analysis, and theoretical justification. | Yes — essential for understanding the V1→V2 evolution. |
| 7 | `RESULTS_V2.md` | Mar 1 | V2 validation results. Documents the three failure modes (extrapolation, cross-storm, skewed θ). Contains the Arthur diagnostic (spatial correlation = 0.31). Lists future options. | Yes — the authoritative results reference. |
| 8 | `HANDOVER_V3.md` | Mar 5 | Comprehensive handover. Full data inventory, V3 options analysis (A through G), recommended direction (time-series conditioning). | Yes — the most complete single overview before this document. |
| 9 | `HANDOVER_LIT_REVIEW.md` | Mar 23 | Literature review of 10 papers. Landscape comparison showing DDPM+surge is novel. Explored 4 alternatives (enriched conditioning, multi-gauge, more data, time-series). Confirmed time-series as best path with detailed justification. | Yes — authoritative literature positioning. |

### Key cross-references between documents

- `CURRENT_PLAN_V2.md` §2-3 explains **why V1 failed** (spatially flat predictions) — read this before looking at V2 results
- `RESULTS_V2.md` §4 explains **why V2 partially failed** (θ insufficiency, Danso-vs-CERA gap) — read this before the presentation
- `HANDOVER_V3.md` §6 lists **all V3 options** with effort/impact analysis — read this if considering future work
- `PROJECT_HANDOFF.md` §8 has the **DDPM code walkthrough** — read this if you need to understand the Wang et al. codebase

---

## 4. Current State of the Work

### What is complete

| Component | Status | Key files |
|-----------|--------|-----------|
| Data acquisition | Complete | `data/adcirc/` (180 Danso), `data/cera/` (13 HSOFS) |
| Preprocessing | Complete | `preprocess.py` → `data/processed/obx/` |
| V1 model (θ-only) | Complete, archived | `results/obx_v1/` |
| V2 model (spatial conditioning) | Complete | `results/obx/model-*.pt` |
| Validation & figures | Complete | `results/obx/figures/`, `results/obx/generated/` |
| Arthur failure diagnostic | Complete | `diagnose_arthur.py` |
| Viz data export | Complete | `viz_data/*.json` (4.7 MB) |
| Interactive viz app | Complete | `results/obx/generated/ddpm-flood-vis/` |
| Primer (11 sections) | Complete | `…/src/components/primer/` |
| Presentation deck (8 slides) | Complete | `…/src/components/presentation/` |

### What is NOT complete

- NYC model was trained in V1 but not updated for V2 (OBX is the focus)
- Time-series conditioning (fort.63.nc) — recommended next step, not implemented
- Miura-style DEM flood maps for NYC — proof-of-concept exists (`visualize_flood.py`) but not integrated into the viz app
- The presentation has not been reviewed by someone with domain expertise

---

## 5. The Data — Sources, Processing, and Pipeline

### 5.1 Training data sources

**Source 1: Danso et al. 2025** (primary, 180 scenarios)
- 20 real Atlantic hurricanes simulated with ADCIRC
- 9 variants per storm: 3 sea levels (reference, +0.44m, +0.74m) × 3 forcing types (control/tidal, historical, +10% wind)
- HSOFS mesh: 1,813,443 nodes covering US Atlantic coast
- Published on Zenodo: record 17601768, CC-BY-4.0
- Location: `data/adcirc/{StormName}/{sea_level}/{forcing}/maxele.63.nc`
- **Key characteristic**: Uses the **Holland parametric wind model** — idealized, symmetric wind fields. This is what "synthetic" or "same storm family" means throughout the project. The wind forcing is a mathematical approximation, not actual meteorological data.

**Source 2: CERA Historical Hindcasts** (supplementary, 13 scenarios)
- 13 storms on the HSOFS mesh from DesignSafe PRJ-3932
- **Real meteorological forcing** — actual recorded wind and pressure fields, not parametric
- Includes Sandy (θ=1.62m at OBX, 3.18m at NYC) and Irene (θ=2.19m at OBX)
- Location: `data/cera/{YEAR}_{NUM}_{NAME}_maxele.63.nc`
- **Key characteristic**: These are "different storm family" — real physics forcing. The spatial surge patterns differ fundamentally from Danso's parametric storms even at the same θ.

**Why "storm family" matters**: The Danso parametric Holland model produces characteristic spatial patterns — surge tends to be high everywhere simultaneously due to the idealized symmetric wind field. Real storms (CERA) produce more complex, asymmetric patterns. At θ ≈ 2.0m, Danso storms flood 63% of nodes above 1m; Arthur (real) floods only 18%. This is the root cause of the cross-storm generalization failure.

### 5.2 Validation data (held out from training)

| Storm | Source | θ (m) | Role |
|-------|--------|-------|------|
| Dorian (9 variants) | Danso | 0.70 – 3.36 | In-distribution + extrapolation test |
| Arthur (1 scenario) | CERA | 2.05 | Cross-storm generalization test |

### 5.3 The preprocessing pipeline

**Script**: `preprocess.py`

```
193 ADCIRC maxele.63.nc files
    ↓ Read with netCDF4 (NOT xarray — xarray fails on ADCIRC's neta dimension)
    ↓ Extract OBX region nodes (lat 34.5-36.0, lon -76.5 to -75.3)
    ↓ Filter to nodes with valid surge in ≥90% of scenarios → 39,494 nodes
    ↓ Compute θ = max(zeta_max) across Hatteras reference nodes per scenario
    ↓ KMeans cluster 39,494 nodes → 1,974 patches of 20 nodes each
    ↓ Extract bathymetric depth from ADCIRC depth variable (static, read once)
    ↓ Build training matrix: each row = [θ, lat_centroid, lon_centroid, mean_depth, surge₁...surge₂₀]
    ↓ 183 scenarios × 1,974 patches = 361,242 rows × 24 columns
    ↓ Save as train_data.npy + patch metadata
```

**Preprocessed output** (`data/processed/obx/`):

| File | Shape | Contents |
|------|-------|----------|
| `train_data.npy` | (361,242, 24) | Training matrix |
| `val_dorian_data.npy` | (17,766, 24) | 9 Dorian variants × 1,974 patches |
| `val_arthur_data.npy` | (1,974, 24) | 1 Arthur scenario × 1,974 patches |
| `patch_assignments.pkl` | dict | {patch_id: [20 node indices]} |
| `patch_centroids.npy` | (1,974, 3) | [lat, lon, depth] per patch |
| `node_coords.npy` | (39,494, 2) | lat/lon of valid nodes |
| `node_depths.npy` | (39,494,) | Bathymetric depth per node |
| `node_indices.npy` | (39,494,) | HSOFS mesh indices |
| `metadata.json` | — | Region config, θ stats, scenario lists |

### 5.4 Key data facts for presentation

- ADCIRC mesh: HSOFS, 1,813,443 nodes, covers entire US Atlantic coast
- OBX region: 39,494 valid nodes, 1,974 patches of 20 nodes
- Training θ range: 0.45m – 2.88m (training max is Bonnie at 2.88m)
- θ distribution: 82% below 1.0m, 12% at 1.0–2.0m, 6% at 2.0–2.88m (heavily skewed)
- Danso directory layouts: 9 storms are "Type A" (`{Storm}/{sea_level}/{forcing}/`), 11 are "Type B" (`{Storm}/hot_start/{sea_level}/{forcing}/`) — the preprocessing script handles both
- CERA HSOFS identification: check `ds.dimensions["node"].size == 1_813_443`
- ADCIRC fill values: filter with `(zeta > 1e10) | (zeta < -1e10) | (zeta <= -99999)`

### 5.5 Exported visualization data

**Script**: `export_viz_data.py` → `viz_data/` (4.7 MB total, all JSON)

Three scenarios exported at node-level (34,854 nodes) and patch-level (1,974 centroids):
- `scenario_dorian_low.json` — success case (θ=0.701m)
- `scenario_dorian_high.json` — extrapolation failure (θ=3.357m)
- `scenario_arthur.json` — cross-storm failure (θ=2.049m)

Plus: `node_coords.json`, `patches.json`, `theta_distribution.json`, `all_metrics.json`, `scenarios.json`

---

## 6. The Model — Architecture, Training, and Generation

### 6.1 Architecture (identical to Wang et al.)

The DDPM uses the Wang et al. codebase **unmodified**: `DDPM_REMD/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py` (611 lines).

| Parameter | Value |
|-----------|-------|
| Architecture | 1D U-Net |
| Base dimension | 32 channels |
| Channel multipliers | (1, 2, 2, 4) → 32, 64, 64, 128 channels |
| Group normalization | 8 groups |
| Attention | Linear attention in middle block |
| Diffusion timesteps | 1,000 |
| Beta schedule | Linear, β₁=0.0001 → β₁₀₀₀=0.02 |
| Loss | L2 (MSE) on predicted noise, columns 4–23 only |

### 6.2 Conditioning mechanism (inpainting)

Columns 0–3 are **never noised** during training and **held fixed** during generation:

| Column | Content | Source |
|--------|---------|--------|
| 0 | θ (peak surge at Hatteras reference area) | Computed per scenario |
| 1 | Patch centroid latitude | Mean of 20 node latitudes |
| 2 | Patch centroid longitude | Mean of 20 node longitudes |
| 3 | Patch mean bathymetric depth | Mean of 20 node ADCIRC depths |
| 4–23 | Surge at 20 nodes (target) | zeta_max from ADCIRC |

`UNMASK_NUMBER = 4`. During generation, each patch gets a unique conditioning vector [θ, lat_i, lon_i, depth_i].

### 6.3 Data normalization

Min-max normalization to [-1, 1] per column (matching Wang et al.'s approach). The scaler is fitted on training data and applied to validation data.

### 6.4 θ noise injection

Because θ is a discrete value per scenario (only ~100 unique values across 183 scenarios), Gaussian noise is added during training: θ_noisy = θ + ε, where ε ~ N(0, 0.15²). This converts discrete θ values into a smooth continuous distribution. σ=0.15m is justified as within NOAA operational surge forecast uncertainty.

### 6.5 Training configuration

| Parameter | Value |
|-----------|-------|
| Platform | Google Colab, single A100 80GB |
| Batch size | 2,048 |
| Learning rate | 4e-5 |
| Training steps | 50,000 |
| Total rows processed | ~102M (50K steps × 2048 batch) |
| Training time | ~45 minutes |
| Final loss | ~0.006 |
| VRAM usage | 7.72 GB / 85 GB |
| EMA decay | 0.995 |
| Checkpoints | 21 (model-0.pt through model-20.pt, 8.6 MB each) |

**Scripts**: `train_flood.py` (local MPS), `train_flood_colab.py` (Colab A100)

### 6.6 Generation

To generate a flood map for a given θ:

1. For each of the 1,974 patches, set conditioning = [θ, lat_i, lon_i, depth_i]
2. Start from pure random noise (20 values)
3. Run 1,000 reverse diffusion steps with conditioning held fixed
4. Repeat 10 times per patch for uncertainty estimation
5. Take mean of 10 samples as prediction, std as uncertainty
6. Stitch all patches together → surge at 34,854 nodes

Generation time: 2–5 minutes on A100 for all patches.

**Scripts**: `generate_flood.py` (local), `generate_flood_colab.py` (Colab)

---

## 7. The Results — What Worked, What Failed, and Why

### 7.1 V1 → V2 evolution

| Version | Conditioning | Key result |
|---------|-------------|------------|
| V1 | θ only (UNMASK=1, 21 cols) | Correct mean surge E[surge\|θ] but **spatially flat** — all patches get same prediction. R² negative everywhere. |
| V2 | θ + lat + lon + depth (UNMASK=4, 24 cols) | Location-specific predictions. In-distribution R²=0.95. Cross-storm R²=-2.57. |

V1 failed because the model had no way to know *where* it was generating — every patch with the same θ got identical predictions. V2 fixed this by adding spatial identity.

### 7.2 V2 validation results

**Dorian (Danso, held-out) — 9 variants:**

| θ (m) | R² | RMSE (m) | Category |
|-------|-----|----------|----------|
| 0.701 | **0.950** | 0.062 | In-distribution |
| 0.702 | **0.947** | 0.063 | In-distribution |
| 0.766 | **0.943** | 0.066 | In-distribution |
| 2.781 | -1.263 | 0.821 | Near training max |
| 2.899 | -1.390 | 0.853 | Extrapolation |
| 2.984 | -1.465 | 0.876 | Extrapolation |
| 3.097 | -1.372 | 0.882 | Extrapolation |
| 3.237 | -1.280 | 0.929 | Extrapolation |
| 3.357 | **-1.530** | 0.960 | Extrapolation |

**Arthur (CERA, held-out):**

| θ (m) | R² | RMSE (m) | Bias (m) |
|-------|-----|----------|----------|
| 2.049 | **-2.568** | 0.602 | +0.329 |

### 7.3 The three scenarios that tell the story

These three scenarios are the backbone of the presentation and the viz app:

**1. Dorian low-θ (0.701m) — "It works"**
- Same storm family (Danso synthetic), θ well within training range
- R² = 0.95, RMSE = 0.06m
- The model accurately reproduces spatial surge patterns
- Caveat: at θ=0.70m, surge is mostly flat (max 0.76m), so the test is relatively easy

**2. Dorian high-θ (3.357m) — "Extrapolation fails"**
- Same storm family (Danso), but θ=3.36m exceeds training max (2.88m)
- R² = -1.26, RMSE = 0.96m
- The model **confidently predicts the wrong pattern** — it produces the characteristic Danso spatial signature at an intensity it has never seen
- "Extrapolation" = asking the model for a θ value beyond any it saw during training

**3. Arthur (2.049m) — "Cross-storm fails" (the key scientific finding)**
- Different storm family (CERA real storm), θ=2.05m is *within* training range
- R² = -2.62, RMSE = 0.60m
- Despite θ being within range, the spatial pattern is fundamentally different from any Danso storm
- Arthur at θ≈2.0m: 18% of nodes > 1m surge. Closest Danso at θ≈2.0m: 63% of nodes > 1m surge. Spatial correlation: 0.31.
- This proves that θ does not uniquely determine the spatial pattern — the model learned Danso-specific patterns, not universal surge physics

### 7.4 Four core limitations

| # | Limitation | Evidence | Root cause |
|---|-----------|----------|------------|
| 1 | θ does not uniquely determine spatial pattern | Arthur vs Danso at same θ: correlation 0.31 | θ is a point measurement, not a state variable like Wang et al.'s temperature |
| 2 | Synthetic-to-real generalization gap | Danso 63% vs Arthur 18% nodes > 1m at same θ | Training is 100% Danso parametric Holland model storms |
| 3 | Skewed θ distribution | 82% of training θ < 1.0m | Most Danso storms are Gulf/Carolina hurricanes that barely affect OBX |
| 4 | Confident extrapolation failures | V1 R²≈-0.08 (graceful) → V2 R²≈-1.5 (confident) | Spatial conditioning makes model confident everywhere, including out-of-distribution |

### 7.5 What the model actually learned

Despite the limitations, the model **did learn something real**:
- Within-ensemble P(surge | θ, location) at in-distribution θ: R² > 0.94
- Correct conditional mean E[surge|θ] — bias near zero for in-range θ
- Physically reasonable uncertainty estimates (mean σ = 0.059m for Dorian low-θ)
- The model is a valid **fast surrogate for ADCIRC** within the Danso synthetic storm family at θ < ~1m

---

## 8. The Visualization App

### Location and setup

```
results/obx/generated/ddpm-flood-vis/
```

```bash
cd results/obx/generated/ddpm-flood-vis
npm install
npm run dev
# Opens at http://localhost:5173
```

### Tech stack

React 19 + TypeScript + Vite + TanStack Router + Tailwind CSS + shadcn/ui + Framer Motion (`motion@12.35.0`) + Deck.gl + Mapbox GL

### Routes / pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Hero + scenario explorer with live Deck.gl map |
| `/metrics` | Metrics Dashboard | Scatter plots, bar charts, θ distribution histogram |
| `/about` | About | Background, data scope, limitations |
| `/0x-primer` | Primer | 11-section deep technical explainer with animated diagrams |
| `/presentation` | Presentation Deck | 8 full-viewport slides for presenting to Prof. Miura |

### Key interactive features

- **Live Deck.gl map** with 34,854 colored dots showing surge predictions on OBX coastline
- **Layer toggle**: Truth / Prediction / Error / Uncertainty views
- **3 clickable scenario cards** switching the map between Dorian low, Dorian high, and Arthur
- **Animated diffusion diagrams** using real data from Patch 1030 / Hurricane Arthur
- **Clickable U-Net architecture diagram** with expandable detail panels

---

## 9. The Primer — Deep Technical Explainer

**Route**: `/0x-primer`
**Files**: `results/obx/generated/ddpm-flood-vis/src/components/primer/`

The primer is an 11-section standalone educational resource. It explains the entire project from scratch with animated diagrams and real data. This is the most comprehensive technical reference within the app.

### Section map

| # | Section | File | What it covers | Key data/values used |
|---|---------|------|---------------|---------------------|
| 1 | The Problem | `IntroSections.tsx` | Storm surge, ADCIRC, why surrogates are needed | Sandy $19B damage, ADCIRC 1.8M nodes |
| 2 | The Idea | `IntroSections.tsx` | Core question, why generative over regression | P(surge \| θ) framing |
| 3 | DDPMs | `DDPMSection.tsx` | Forward/reverse diffusion, 1D U-Net, inpainting, noise schedule | Real data from Patch 1030/Arthur (20 surge values), animated diagrams |
| 4 | Wang et al. | `WangSection.tsx` | Reference paper, structural analogy, **critical difference** (θ vs temperature) | Mapping table |
| 5 | Adaptation | `AdaptationSection.tsx` | Molecules→floods mapping, θ definition, spatial patches, conditioning columns | 1,974 patches, 24-column format, UNMASK=4 |
| 6 | Data | `DataSection.tsx` | NYC limitation → OBX pivot, Danso vs CERA, preprocessing pipeline | 180 Danso, 13 CERA, Zenodo/DesignSafe links |
| 7 | Model | `ModelSection.tsx` | Architecture details, loss, normalization, θ noise injection, training config, generation process | A100 training, 50K steps, 45min, loss 0.006 |
| 8 | Results | `ResultsSection.tsx` | Three scenarios, metrics table, R² breakdown | R²=0.95 / -1.26 / -2.62 |
| 9 | Limitations | `OutlookSection.tsx` | 4 core limitations | Correlation 0.31, 82% skew, V1→V2 confidence |
| 10 | What's Next | `OutlookSection.tsx` | Time-series conditioning (fort.63.nc) | 7M rows, 13 CERA storms, continuous θ |
| 11 | Glossary | `GlossarySection.tsx` | 25 term definitions | All technical terms grounded in project context |

### Diagram components

| Component | File | What it shows |
|-----------|------|-------------|
| Diffusion filmstrip | `DiffusionProcessDiagram.tsx` | 6-panel SVG showing surge grid at t=0,200,400,600,800,1000 |
| Animated diffusion grid | `DiffusionAnimatedDiagram.tsx` | Real-time 5×4 grid animating from noise → clean surge |
| U-Net flow diagram | `UNetFlowDiagram.tsx` | Animated encoder→bottleneck→decoder with skip connections |
| U-Net architecture | `UNetArchitectureDiagram.tsx` | Vertical pipeline with clickable detail panels (Input/Encoder/Bottleneck/Decoder/Output) |
| U-Net output | `UNetOutputDiagram.tsx` | One reverse step: noised input → predicted noise → less noisy output |

### Utilities

| File | Purpose |
|------|---------|
| `diffusion-utils.ts` | All math: alpha schedule, noised profiles, reverse steps, color mapping. Contains `RAW_SURGE` (real Patch 1030 data), `CONDITIONING` values (θ=2.049m, lat=35.74°, lon=-75.62°, depth=3.61m) |
| `useSyncedDiffusion.ts` | Animation hook: loops through keyframes [1000→800→600→400→200→0], 1.5s per segment, syncs filmstrip + animated diagram. Respects `prefers-reduced-motion` |
| `shared.tsx` | Reusable UI: Section, Prose, InfoCard, DataTable, StatGrid, NumberedCard, StepList |

### What to verify in the primer

If you are reviewing for scientific accuracy, check:

1. **Section 4 (Wang et al.)**: The "important difference" paragraph correctly states that θ is a downstream point measurement, not a thermodynamic state variable. This is the most important scientific caveat.
2. **Section 6 (Data)**: The Danso Holland model description — confirm it accurately represents parametric wind forcing vs real meteorological forcing.
3. **Section 8 (Results)**: The three-scenario narrative — verify the R²/RMSE values match `results/obx/generated/validation_metrics.json`.
4. **Section 9 (Limitations)**: The "correlation 0.31" finding — this comes from `diagnose_arthur.py`.
5. **Section 10 (What's Next)**: The time-series conditioning proposal — verify the claim that it makes the Wang et al. analogy "structurally exact."

---

## 10. The Presentation Deck

**Route**: `/presentation`
**Files**: `results/obx/generated/ddpm-flood-vis/src/components/presentation/`

### Architecture

| File | Purpose |
|------|---------|
| `Presentation.tsx` | Orchestrates slides, manages navigation direction for transitions |
| `SlideShell.tsx` | Full-viewport container with AnimatePresence horizontal slide transitions |
| `SlideNavigation.tsx` | Fixed bottom bar: prev/next arrows, dot indicators, "1 / 8" counter |
| `slides/*.tsx` | 8 individual slide components |

**Navigation**: Arrow keys, Space, URL hash sync (#0 through #7)

### Slide order and content

| # | Slide | File | Content summary | Primer link |
|---|-------|------|----------------|-------------|
| 0 | Intro | `IntroSlide.tsx` | Title, subtitle, author name (Afra Izzati Kamili) | `/0x-primer` |
| 1 | Research Question | `ResearchQuestionSlide.tsx` | "Can a generative model learn spatial surge patterns?" Formula: P(surge \| θ, location). Wang et al. mapping cards. | `/0x-primer#the-idea` |
| 2 | Diffusion | `DiffusionSlide.tsx` | How DDPMs work. Embeds `DiffusionProcessDiagram` and `DiffusionAnimatedDiagram` from primer. | `/0x-primer#ddpms` |
| 3 | Model | `ModelSlide.tsx` | 1D U-Net. 4 stat cards. Inpainting callout. Embeds `UNetArchitectureDiagram` from primer. | `/0x-primer#model` |
| 4 | Dataset | `DatasetSlide.tsx` | ADCIRC definition. 4 stats (183 scenarios, 1,974 patches, 361K rows, 24 cols). Data sources (Danso + CERA). Sample training row from Patch 1030/Arthur. Stitching pipeline visual. | `/0x-primer#data` |
| 5 | Results | `ResultsSlide.tsx` | **Live Deck.gl map** with 3 clickable scenario cards (Dorian low/high + Arthur). Layer toggle. | `/0x-primer#results` |
| 6 | Discussion | `DiscussionSlide.tsx` | "Why the Model Falls Short" — 4 limitation cards matching Section 7.4 of this document. | `/0x-primer#limitations` |
| 7 | Next Steps | `NextStepsSlide.tsx` | Time-series conditioning. Replace maxele → fort.63.nc. Comparison table: current vs time-series. | `/0x-primer#whats-next` |

### Wording constraints

- **No possessive language**: no "our", "my". Use "the model", "this adaptation"
- **Terminology**: "storm surge" not "flood surge" — the data is water surface elevation (ADCIRC maxele.63.nc), not inland flooding depth

### What to verify in the slides

1. **Slide 4 (Dataset)**: The sample training row shows real values from Patch 1030 / Hurricane Arthur. Verify these match the data files.
2. **Slide 5 (Results)**: The scenario cards show θ, R², RMSE values. Verify against `validation_metrics.json`.
3. **Slide 6 (Discussion)**: The 4 limitation points must be defensible. The presenter should be able to elaborate on each with evidence from the diagnostic analysis.
4. **Slide 7 (Next Steps)**: The claim that time-series conditioning "fixes" the four problems. The presenter should understand *why* for each.

---

## 11. What the Presenter Needs to Know

### Likely questions from Prof. Miura

**Q: "How is this different from Microsoft Aurora?"**
A: Aurora predicts weather at 0.25° (~28km) resolution. This predicts urban-scale flood inundation at mesh-node level (~10-500m). Aurora cannot model seawalls, barrier islands, or infrastructure-level flooding.

**Q: "Why not just use GISSR?"**
A: GISSR is fast (0.01s) but uses simplified physics (Manning's equation). This approach learns from ADCIRC (gold standard, full shallow water equations). Future work could combine both — DDPM for surge prediction, GISSR for inland redistribution.

**Q: "θ is not like temperature — temperature determines the equilibrium distribution. θ doesn't."**
A: Correct. This is acknowledged as a fundamental limitation. θ is a downstream point measurement that does not uniquely determine the spatial pattern (correlation 0.31 between storms at same θ). The time-series approach partially addresses this by using instantaneous θ(t) which, like temperature, fluctuates naturally.

**Q: "Why does adding spatial conditioning make extrapolation worse?"**
A: Without spatial information, the model could only predict the global mean — a safe, non-committal prediction (R²≈-0.08, near zero). With spatial conditioning, the model learned the Danso-specific spatial signature and applies it confidently at all θ values, even where it shouldn't. It produces the "Danso pattern" at intensities it's never seen — actively wrong rather than passively uninformative.

**Q: "82% of training data has θ < 1m. Isn't the success case (θ=0.70m) trivially easy?"**
A: Partially. At θ=0.70m, surge is mostly flat (max 0.76m), so "predict low surge everywhere" gets close. The model does correctly predict the spatial variation within that range (R²=0.95, not just low RMSE), but a more demanding test would be a held-out Danso storm at moderate θ (1.5-2.5m). The skewed distribution is a known limitation.

**Q: "Can this work for NYC?"**
A: The data exists but the θ distribution is terrible (0.7–1.2m, essentially flat). NYC would require either (a) time-series data from fort.63.nc giving continuous θ coverage, or (b) many more storm simulations that actually affect NYC.

### Key terms the presenter must understand

- **"Storm family"**: Whether a storm simulation used Danso's parametric Holland wind model (synthetic) or real meteorological forcing (CERA historical). Same family = same data generation method.
- **"Extrapolation"**: Generating at a θ value above the training maximum (2.88m). The model has never seen surge this severe.
- **"In-distribution"**: θ within the training range AND from the same storm family as training data. Only this combination succeeds.
- **"Inpainting"**: The conditioning mechanism — columns 0-3 are never corrupted with noise during training and held fixed during generation. No separate conditioning input to the network.
- **"Spatial patch"**: A cluster of 20 nearby ADCIRC mesh nodes. The model processes one patch at a time and patches are stitched together for the full map.

---

## 12. Recommended Next Step: Time-Series Conditioning

### The idea

Replace `maxele.63.nc` (one peak value per storm) with `fort.63.nc` (full time series, 200-500 timesteps per storm). At each timestep t, the reference gauge reads θ(t) and every other node has a surge value — producing one complete spatial snapshot per timestep.

### Why this fixes all four limitations

| Limitation | How time-series fixes it |
|-----------|------------------------|
| θ insufficient | θ(t) fluctuates naturally at every timestep — exact analog to Wang et al.'s temperature |
| Synthetic-to-real gap | Train on CERA historical storms (real forcing), not Danso synthetic |
| Skewed θ distribution | Every storm contributes at every θ level between 0 and its peak — continuous, uniform coverage |
| Insufficient data | 13 storms × ~300 timesteps × 1,974 patches ≈ **7M training rows** (vs 361K) |

### Data availability

The 13 CERA HSOFS storms have `fort.63.nc` files on TACC (~30-100 GB total). Download via the existing `download_cera.sh` script (change filename from `maxele.63.nc` to `fort.63.nc`). TACC credentials: username `makramab` at `data.tacc.utexas.edu`. See `docs/HANDOVER_V2.md` for full access details.

### What changes in the pipeline

- `preprocess.py` needs a new mode to iterate over timesteps within each `fort.63.nc`
- The DDPM training code stays identical — it just receives a much larger `.npy` file
- Generation is unchanged — specify θ, get surge pattern

---

## 13. Repository Map

```
ddpm-flood/
│
├── docs/                              ← All project documentation (moved here)
│   ├── PROJECT_HANDOFF.md             Feb 24 — Original brief
│   ├── EXPERIMENT_PLAN.md             Feb 27 — Updated plan
│   ├── CURRENT_PLAN.md                Feb 27 — Two-region strategy
│   ├── PROGRESS.md                    Feb 27 — Task tracker
│   ├── HANDOVER_V2.md                 Feb 27 — Data journey + TACC access
│   ├── CURRENT_PLAN_V2.md             Feb 28 — Spatial conditioning fix
│   ├── RESULTS_V2.md                  Mar 1  — V2 results + diagnostics
│   ├── HANDOVER_V3.md                 Mar 5  — Comprehensive handover
│   └── HANDOVER_LATEST.md             Mar 7  — THIS FILE
│
├── preprocess.py                      ← Preprocessing pipeline
├── train_flood.py                     ← Local training (MPS)
├── train_flood_colab.py               ← Colab A100 training
├── generate_flood.py                  ← Generation + validation (local)
├── generate_flood_colab.py            ← Generation (Colab)
├── visualize_results.py               ← Publication figures
├── diagnose_arthur.py                 ← Arthur failure diagnostic
├── export_viz_data.py                 ← Export JSON for viz app
├── visualize_flood.py                 ← Earlier Miura-style flood maps (PoC)
│
├── data/                              ← [gitignored] Raw + processed data
│   ├── adcirc/                        180 Danso scenarios (20 storms × 9 variants)
│   ├── cera/                          31 CERA storms (13 HSOFS-compatible)
│   ├── processed/obx/                 Preprocessed training/validation matrices
│   ├── validation/                    Sandy HWMs, inundation zone, tide gauge
│   ├── naccs/                         NACCS geodatabase (explored, not used)
│   └── deepsurge/                     DeepSurge sample (explored, not used)
│
├── viz_data/                          ← Exported JSON for viz app (4.7 MB)
│
├── results/
│   ├── obx/                           V2 model checkpoints + results
│   │   ├── model-0.pt ... model-20.pt
│   │   ├── loss_log.csv
│   │   ├── generated/                 Validation predictions + metrics
│   │   │   ├── validation_metrics.json
│   │   │   ├── pred_*.npy / truth_*.npy
│   │   │   └── ddpm-flood-vis/        ← THE VISUALIZATION APP
│   │   │       ├── src/
│   │   │       │   ├── components/
│   │   │       │   │   ├── primer/            11-section technical explainer
│   │   │       │   │   │   ├── IntroSections.tsx        Sections 1-2
│   │   │       │   │   │   ├── DDPMSection.tsx          Section 3
│   │   │       │   │   │   ├── WangSection.tsx          Section 4
│   │   │       │   │   │   ├── AdaptationSection.tsx    Section 5
│   │   │       │   │   │   ├── DataSection.tsx          Section 6
│   │   │       │   │   │   ├── ModelSection.tsx         Section 7
│   │   │       │   │   │   ├── ResultsSection.tsx       Section 8
│   │   │       │   │   │   ├── OutlookSection.tsx       Sections 9-10
│   │   │       │   │   │   ├── GlossarySection.tsx      Section 11
│   │   │       │   │   │   ├── DiffusionProcessDiagram.tsx
│   │   │       │   │   │   ├── DiffusionAnimatedDiagram.tsx
│   │   │       │   │   │   ├── UNetFlowDiagram.tsx
│   │   │       │   │   │   ├── UNetArchitectureDiagram.tsx
│   │   │       │   │   │   ├── UNetOutputDiagram.tsx
│   │   │       │   │   │   ├── diffusion-utils.ts
│   │   │       │   │   │   ├── useSyncedDiffusion.ts
│   │   │       │   │   │   └── shared.tsx
│   │   │       │   │   └── presentation/      8-slide presentation deck
│   │   │       │   │       ├── Presentation.tsx
│   │   │       │   │       ├── SlideShell.tsx
│   │   │       │   │       ├── SlideNavigation.tsx
│   │   │       │   │       └── slides/
│   │   │       │   │           ├── IntroSlide.tsx
│   │   │       │   │           ├── ResearchQuestionSlide.tsx
│   │   │       │   │           ├── DiffusionSlide.tsx
│   │   │       │   │           ├── ModelSlide.tsx
│   │   │       │   │           ├── DatasetSlide.tsx
│   │   │       │   │           ├── ResultsSlide.tsx
│   │   │       │   │           ├── DiscussionSlide.tsx
│   │   │       │   │           └── NextStepsSlide.tsx
│   │   │       │   └── hooks/presentation/
│   │   │       │       └── useSlideNavigation.ts
│   │   │       └── package.json
│   │   └── figures/                   Publication-quality PNG figures
│   ├── obx_v1/                        V1 model backup
│   └── nyc_v1/                        V1 NYC model backup
│
├── DDPM_REMD/                         ← [gitignored] Wang et al. code (clone separately)
├── GIS_FloodSimulation/               ← [gitignored] Miura GISSR code + DEM data
│
├── refs/                              Reference papers (PDF)
│   ├── wang-et-al-2022-...pdf
│   └── miura-et-al-2021-...pdf
│
├── pyproject.toml                     Python project config (uv)
├── uv.lock                           Dependency lock
└── .python-version                   Python 3.11
```

---

## 14. Environment and Setup

### Python (for model scripts)

```bash
# Python 3.11 via uv
uv sync                              # Install dependencies
uv run python preprocess.py           # Run preprocessing
uv run python train_flood.py --region obx   # Train (local, MPS)
uv run python generate_flood.py --region obx --mode validate  # Validate
```

Key packages: `torch` (MPS), `netCDF4`, `scikit-learn`, `matplotlib`, `numpy`

### Visualization app

```bash
cd results/obx/generated/ddpm-flood-vis
npm install
npm run dev                           # http://localhost:5173
```

### DDPM library (must clone separately)

```bash
git clone https://github.com/tiwarylab/DDPM_REMD.git
```

The training scripts import from `DDPM_REMD/denoising_diffusion_pytorch/`. The library is used **unmodified**.

### Critical technical notes

- **ADCIRC files**: Use `netCDF4.Dataset()`, never `xarray` (fails on `neta` dimension)
- **MPS (Apple Silicon)**: Set `pin_memory=False` in DataLoaders
- **CERA HSOFS identification**: Node count = 1,813,443
- **Danso directory layouts**: Two types — preprocessing script handles both automatically
- **Fill values**: Filter with `(zeta > 1e10) | (zeta < -1e10) | (zeta <= -99999)`
