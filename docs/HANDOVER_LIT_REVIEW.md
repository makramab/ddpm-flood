# DDPM Flood — Literature Review & Exploration Handover

**Date**: March 23, 2026
**Author**: Afra Izzati Kamili (with Claude Code assistance)
**Status**: Literature review complete. Time-series conditioning confirmed as next step.

---

## Table of Contents

1. [Purpose of This Document](#1-purpose-of-this-document)
2. [Literature Review Summary](#2-literature-review-summary)
3. [Paper-by-Paper Summaries](#3-paper-by-paper-summaries)
4. [Landscape Comparison — Where Our DDPM Sits](#4-landscape-comparison--where-our-ddpm-sits)
5. [Exploration of Alternative Approaches](#5-exploration-of-alternative-approaches)
6. [Why Time-Series Conditioning Remains the Best Path](#6-why-time-series-conditioning-remains-the-best-path)
7. [References](#7-references)

---

## 1. Purpose of This Document

This document captures the literature review conducted on March 23, 2026, covering 10 papers spanning DDPM applications in geosciences, storm surge surrogate modeling, and ML-based surge prediction. The review had two goals:

1. **Understand the existing landscape** — what methods exist for spatial storm surge prediction and where generative diffusion models have been applied in related domains
2. **Inform the next step** — given that our V2 model showed θ (single scalar peak surge) is insufficient for cross-storm generalization, determine the best path forward

The review confirmed that **no existing work applies DDPM to spatial storm surge prediction**, validating our contribution's novelty, and that **time-series conditioning using CERA fort.63.nc** remains the strongest next step — over alternatives like multi-gauge conditioning or enriched storm parameterization.

---

## 2. Literature Review Summary

The 10 papers divide into three groups:

### Group A: DDPM in Geosciences (Papers 1, 3, 5)
These prove DDPM works for geophysical spatial fields (thunderstorms, dust storms, tropical cyclone intensity) but none apply it to storm surge. They provide **methodological precedent** for our work.

### Group B: Storm Surge Surrogates (Papers 7, 8, 9)
These are our **direct competitors and domain context**. Papers 7–8 use Kriging+PCA surrogates with 5–6D storm parameterization on large synthetic storm databases (446–1,050 storms). Paper 9 provides physics-based validation for our Outer Banks study region. This group defines the state-of-the-art we aim to improve upon.

### Group C: Adjacent Methods (Papers 2, 4, 6, 10)
These use "diffusion" in non-DDPM senses (network diffusion, information diffusion, branching diffusion) or apply ML to surge at single points (LSTM). They provide contrast and framing but are not direct competitors.

---

## 3. Paper-by-Paper Summaries

### Paper 1 — Luo et al. (2024), "Probabilistic Forecasting Model for Tropical Cyclone Intensity Based on Diffusion Model"

Luo et al. apply a conditional DDPM to probabilistic TC intensity forecasting, predicting future intensity (Vmax) given current storm state (intensity, latitude, longitude, SST, vertical wind shear, etc.). The model generates an ensemble of intensity trajectories, outperforming traditional statistical models (SHIPS) and deterministic deep learning approaches in CRPS. The key insight is using DDPM's generative nature for uncertainty quantification — sampling multiple trajectories to produce probabilistic forecasts rather than point predictions.

**Relevance**: Demonstrates DDPM for a hurricane-related variable, but predicts a scalar (intensity) over time rather than spatial fields. Their use of DDPM for ensemble-based uncertainty quantification directly supports our argument that DDPM surrogates can provide native UQ — something Kriging surrogates (papers 7–8) cannot.

### Paper 2 — Du et al. (2022), "High-accuracy estimation method of typhoon storm surge disaster loss under small sample conditions by information diffusion model coupled with machine learning models"

Du et al. use information diffusion theory (not DDPM) to handle small-sample estimation of economic losses from typhoon storm surge. They convert sparse historical loss records into fuzzy probability distributions, then couple with ML models (RF, SVR, XGBoost) for loss prediction. Tested on Guangdong Province typhoon loss data.

**Relevance**: Minimal. Different meaning of "diffusion" (information-theoretic, not denoising). The small-sample handling is conceptually interesting but methodologically unrelated to our generative modeling approach.

### Paper 3 — Hermes et al. (2025), "Nowcasting of dust and convective storms via diffusion-model predictions of SEVIRI RGB imagery"

Hermes et al. use a conditional DDPM to nowcast dust storms and convective systems from SEVIRI satellite RGB imagery over the Sahel. The model takes current satellite frames and generates future frames (1–4 hour lead times), producing probabilistic forecasts that capture storm initiation and propagation. Evaluated against persistence, optical flow, and deep learning baselines.

**Relevance**: Strong methodological parallel — this is DDPM generating spatial geophysical fields conditioned on prior state. Their approach to conditioning (feeding current spatial fields to predict future ones) is analogous to what we could do with time-series surge fields. Their evaluation metrics and probabilistic framing are useful references.

### Paper 4 — Gabrielov et al. (2007), "Predictability of extreme events in a branching diffusion model"

Gabrielov et al. study extreme event predictability in a theoretical branching diffusion model applied to seismicity patterns. Purely mathematical — analyzes how branching and diffusion processes generate heavy-tailed event distributions and whether precursors can predict extreme events.

**Relevance**: None for our methodology. The "diffusion" here is a stochastic process model, not a generative neural network. Only relevant as general background on extreme event statistics.

### Paper 5 — Dai et al. (2023), "Four-hour thunderstorm nowcasting using a deep diffusion model for satellite data"

Dai et al. apply a conditional diffusion model to 4-hour thunderstorm nowcasting using Fengyun-4A satellite data. The model generates future infrared brightness temperature fields conditioned on past observations, using a U-Net backbone with temporal attention. Achieves superior CSI/POD scores compared to ConvLSTM and optical flow methods, particularly for longer lead times.

**Relevance**: Another example of DDPM for spatial weather field prediction. Their temporal conditioning approach (past frames → future frames) is relevant to our time-series direction. Their U-Net architecture choices and evaluation methodology provide useful reference points.

### Paper 6 — Toski et al. (2023), "Simulating flood situations in urban hydrology using a diffusion model in complex networks"

Toski et al. model flood propagation through urban drainage networks using a network diffusion model. Water flow is modeled as a diffusion process on a graph (pipe network), not using neural networks. They simulate how flooding spreads through connected infrastructure under different rainfall scenarios.

**Relevance**: Minimal. Different meaning of "diffusion" (physical process on networks). The urban flooding domain is adjacent but the methodology is unrelated to DDPM.

### Paper 7 — Jia et al. (2016), "Surrogate modeling for peak or time-dependent storm surge prediction over an extended coastal region using an existing database of synthetic storms"

Jia et al. build a Kriging+PCA surrogate for storm surge using 446 synthetic storms simulated with ADCIRC+STWAVE over the Louisiana coast. Each storm is parameterized by 5 inputs (landfall longitude, heading angle, central pressure deficit, forward speed, radius of maximum winds). The high-dimensional surge output (545,635 nodes for peak surge; 30 locations × 92 timesteps for time series) is compressed via PCA before fitting Gaussian Process regressors. The surrogate achieves an average R² of 0.942 for peak surge and handles dry/wet node transitions through a data augmentation technique.

**Relevance**: **Primary competing methodology.** Solves essentially the same problem (predicting spatial surge fields from storm parameters) but with Kriging+PCA instead of DDPM. Their 5D storm parameterization (x_lon, θ_heading, c_p, v_f, R_m) is much richer than our single scalar θ. Key limitation: Kriging produces smooth mean-field predictions and cannot capture multimodal uncertainty or generate ensembles. Their dataset of 446 synthetic storms is also far larger than our ~20 storms. We should benchmark against this approach and argue that DDPM captures non-Gaussian spatial structure that Kriging smooths out.

### Paper 8 — Zhang et al. (2018), "Advances in surrogate modeling for storm surge prediction: storm selection and addressing characteristics related to climate change"

Zhang et al. extend the Kriging+PCA framework (paper 7) to the NACCS database — 1,050 synthetic storms covering the US East Coast from Virginia to Maine. Key advances: adaptive Design of Experiments (DoE) for cost-efficient storm selection, cross-validation hyperparameter tuning, and explicit climate change treatment (storm intensification via modified Holland B parameter, sea level rise as linear offset). Their 6D parameterization adds landfall latitude to account for the longer coastline.

**Relevance**: **Closest competitor for our NYC case.** They include the Battery, NY (40.70°N, 74.01°W) as a validation point — the exact same reference location we use for our NYC conditioning variable θ. Their NACCS storm data is the same database we originally explored but found inaccessible (full ADCIRC spatial outputs not publicly available — only storm parameters and save-point peak surges are published). Their climate change methodology is relevant for framing future applications of our model.

### Paper 9 — Funakoshi et al. (2010), "Coupling a Finite Element Storm Surge Model of the North Carolina Sounds with Operational Ocean and Weather Prediction Models"

Funakoshi et al. develop the North Carolina Coastal Flooding Model (CFM), an ADCIRC-based system for the Pamlico and Albemarle Sounds. They validate against Hurricane Isabel (2003), testing three meteorological forcing products (H*Wind, GFDL, GFS) and two boundary condition types (basin-scale ADCIRC, RTOFS). H*Wind with ADCIRC boundaries gives the best results (~20 cm RMSE at Hatteras). The paper details the physics of wind-driven setup through narrow Outer Banks inlets.

**Relevance**: **Domain context for our primary study region.** Covers the exact Outer Banks and Hatteras area where we compute θ. Validates the ADCIRC model framework underlying our training data (Danso and CERA). Provides ground truth understanding of surge physics in our domain: wind-driven setup through inlets, Pamlico Sound dynamics, sensitivity to forcing quality. Useful for building domain credibility in the writeup.

### Paper 10 — Chen et al. (2022), "Storm Surge Prediction Based on Long Short-Term Memory Neural Network in the East China Sea"

Chen et al. apply LSTM to predict storm surge water levels at the Lusi tidal station in the Yangtze River Delta. Inputs are meteorological variables (atmospheric pressure, wind speed, wind direction) and historical water level observations from 12 typhoons (1986–2016). The model achieves strong short-term accuracy (1h lead: MAE=13.4cm, RMSE=16.3cm, ACC=0.95) but degrades at longer horizons (15h: ACC=0.75). Outperforms BRR, GBDT, LR, SVR by ~27% in accuracy.

**Relevance**: **Contrast methodology.** LSTM predicts surge at a single station over time — temporal but not spatial. Our DDPM produces spatial fields at a single time — spatial but not temporal. These are complementary approaches. We can cite LSTM methods as effective for time-series forecasting at individual locations while arguing our approach fills the gap of producing full spatial surge maps needed for emergency planning and flood mapping.

---

## 4. Landscape Comparison — Where Our DDPM Sits

### Method × capability matrix

| | Paper | Method | Output | Spatial? | Temporal? | Surge? |
|---|---|---|---|---|---|---|
| 1 | Luo et al. | DDPM | TC intensity (scalar) | No | Yes | No |
| 2 | Du et al. | Info diffusion | Economic loss | No | No | Partial |
| 3 | Hermes et al. | DDPM | Dust/convection fields | Yes | No | No |
| 4 | Gabrielov et al. | Branching diffusion | Seismicity patterns | Theoretical | Theoretical | No |
| 5 | Dai et al. | DDPM | Thunderstorm images | Yes | No | No |
| 6 | Toski et al. | Network diffusion | Flood propagation | Yes (graph) | Yes | Partial |
| 7 | Jia et al. | Kriging+PCA | Surge fields + time series | Yes | Yes | **Yes** |
| 8 | Zhang et al. | Kriging+PCA | Surge fields | Yes | No | **Yes** |
| 9 | Funakoshi et al. | ADCIRC (physics) | Full hydrodynamics | Yes | Yes | **Yes** |
| 10 | Chen et al. | LSTM | Water level at 1 station | No | Yes | **Yes** |
| **Ours** | **This project** | **DDPM+PCA** | **Surge spatial fields** | **Yes** | **No** | **Yes** |

### The gap we fill

No existing work applies DDPM to spatial storm surge prediction. Our contribution sits at the intersection of:

- **Generative diffusion models** (papers 1, 3, 5) — we take the method
- **Surge surrogate modeling** (papers 7, 8) — we take the problem
- **ADCIRC-based simulation** (paper 9) — we take the training data

### Advantages over existing approaches

1. **vs. Kriging surrogates (papers 7–8):** DDPM generates full distributions, not point estimates. Each sample is a plausible surge field, giving natural uncertainty quantification without assuming Gaussianity. Kriging produces smooth mean predictions.
2. **vs. LSTM (paper 10):** We produce entire spatial fields, not single-station predictions.
3. **vs. physics models (paper 9):** Orders of magnitude faster at inference time.
4. **vs. other DDPM work (papers 1, 3, 5):** First application to the storm surge domain.

### Honest weaknesses relative to existing work

- Our conditioning (scalar θ) is simpler than Kriging's 5–6D storm parameterization (papers 7–8)
- We don't handle time series yet (papers 7, 9, 10 do)
- Our training set is much smaller (~20 storms) compared to the synthetic databases of 446–1,050 storms in papers 7–8
- V2 results show θ alone is insufficient for cross-storm generalization (R² = -2.62 on Arthur)

---

## 5. Exploration of Alternative Approaches

Given that V2 demonstrated scalar θ is insufficient, we evaluated four paths forward during this review session:

### Option A: Enrich conditioning to 5–6D (match Kriging papers)

Add storm parameters (heading angle, forward speed, Rmax, central pressure, landfall location) to the conditioning vector, matching the parameterization used by Jia et al. and Zhang et al.

**Problem**: We only have ~20 unique storms. Papers 7–8 needed 446–1,050 storms for 5–6D conditioning to work. With 20 storms in 5–6D space, any model will overfit. This path requires **more data**, not just more parameters.

### Option B: Multi-gauge spatial conditioning

Instead of a single θ at one reference gauge, condition on peak surge at 5–10 well-distributed gauge locations. This carries more spatial information per sample and is operationally realistic (NHC forecasts surge at multiple stations).

**Assessment**: Addresses the spatial information deficit but still suffers from discrete θ per storm (only ~20 distinct conditioning vectors). Doesn't fix the fundamental mismatch with Wang et al.'s framework where the conditioning variable (temperature) varies continuously.

### Option C: Additional data sources

Seek access to the NACCS ADCIRC outputs (1,050 storms) or generate new synthetic storms.

**Assessment**: NACCS full spatial outputs are not publicly available (paper 8 had them through USACE collaboration). Generating new ADCIRC runs requires HPC access and weeks of computation. Not practical within the PhD assessment timeline.

### Option D: Time-series conditioning with CERA fort.63.nc (previous recommendation)

Replace `maxele.63.nc` (one peak value per storm) with `fort.63.nc` (full time series, ~300 timesteps per storm). Each timestep provides an instantaneous θ(t) at the reference gauge paired with the complete spatial surge snapshot.

**Assessment**: See Section 6 below for detailed comparison. This is the selected path.

---

## 6. Why Time-Series Conditioning Remains the Best Path

### Head-to-head: Time-Series vs. Multi-Gauge (the two strongest alternatives)

| Dimension | Time-Series (fort.63.nc) | Multi-Gauge (Option B) |
|---|---|---|
| **Conditioning input** | θ(t) at 1 gauge across ~300 timesteps | θ₁, θ₂, ... θₖ at k gauges, peak values |
| **What it adds** | Temporal richness | Spatial richness |
| **Training rows** | ~7M (13 storms × 300 steps × 1,974 patches) | Same as current (~361K) |
| **Wang et al. analogy** | **Structurally identical** — θ(t) fluctuates naturally like temperature | Departs further — multi-scalar conditioning |
| **Data needed** | Download fort.63.nc from TACC (~30–100 GB) | Already available in maxele.63.nc |
| **Preprocessing** | New mode in preprocess.py for timestep iteration | Pick k reference nodes, extract peak values |
| **Fixes discrete θ?** | Yes — continuous, naturally varying | No — still discrete per storm |
| **Operationally realistic?** | Less — requires real-time hydrograph | More — NHC forecasts at multiple stations |

### Why time-series wins

1. **Fixes the fundamental problem.** The core failure isn't just that θ lacks spatial information — it's that θ as a single discrete value per storm doesn't match Wang et al.'s framework. In Wang et al., temperature fluctuates naturally across simulation frames, and each frame is a valid training sample at that temperature. Time-series θ(t) achieves exactly this: at every timestep during a hurricane, the reference gauge reads a different θ(t), and the spatial field at that instant is a valid training pair.

2. **20× more training data.** 13 CERA storms × ~300 timesteps × 1,974 patches ≈ 7M training rows, versus the current 361K. This addresses the data scarcity problem without needing new storms or external databases.

3. **No architectural changes.** The DDPM training code stays identical — it just receives a larger `.npy` file. Each row is still [θ(t), lat, lon, depth, surge₁...surge₂₀]. Multi-gauge conditioning would require modifying how θ is injected into the model.

4. **Trains on real storms only.** By using only CERA (real meteorological forcing), we eliminate the Danso synthetic-to-real gap that caused cross-storm failure in V2. No more Holland parametric wind model artifacts.

5. **Continuous θ coverage.** Every storm contributes training samples at every θ level between 0 and its peak. The skewed θ distribution problem (82% below 1m in V2) disappears because the rising and falling limbs of each storm hydrograph sweep through the full range.

### What time-series does NOT fix

- **Spatial information in conditioning**: θ(t) at one gauge still doesn't capture storm geometry (heading, size, landfall location). Two different storms passing through at different angles could produce the same θ(t) at the gauge but different spatial patterns. However, with 7M training samples, the DDPM should learn the average spatial response at each θ level and capture the variance through its generative sampling.
- **Operational deployment framing**: Conditioning on a real-time hydrograph is less clean operationally than conditioning on a surge forecast at one gauge. This is a framing issue, not a technical one — for this assessment, demonstrating the method works is more important.

### Implementation path

The implementation was already specified in `HANDOVER_LATEST.md` Section 12:

1. Download `fort.63.nc` from TACC for the 13 CERA HSOFS storms (~30–100 GB)
2. Modify `preprocess.py` to iterate over timesteps within each storm file
3. Build training matrix: each row = [θ(t), lat, lon, depth, surge₁...surge₂₀]
4. Train with identical DDPM architecture and hyperparameters
5. Validate on held-out storms (Arthur, Sandy) using peak-surge timesteps

---

## 7. References

All papers are stored in `readings/`:

| # | Citation | File |
|---|----------|------|
| 1 | Dai et al. (2023). Four-hour thunderstorm nowcasting using a deep diffusion model for satellite data. | `readings/1. Four-hour thunderstorm...pdf` |
| 2 | Du et al. (2022). High-accuracy estimation method of typhoon storm surge disaster loss under small sample conditions by information diffusion model coupled with machine learning models. | `readings/2. High-accuracy estimation...pdf` |
| 3 | Hermes et al. (2025). Nowcasting of dust and convective storms via diffusion-model predictions of SEVIRI RGB imagery. | `readings/3. Nowcasting of dust...pdf` |
| 4 | Gabrielov et al. (2007). Predictability of extreme events in a branching diffusion model. | `readings/4. Predictability of extreme...pdf` |
| 5 | Luo et al. (2024). Probabilistic Forecasting Model for Tropical Cyclone Intensity Based on Diffusion Model. | `readings/5. Probabilistic Forecasting...pdf` |
| 6 | Toski et al. (2023). Simulating flood situations in urban hydrology using a diffusion model in complex networks. | `readings/6. Simulating flood situations...pdf` |
| 7 | Jia et al. (2016). Surrogate modeling for peak or time-dependent storm surge prediction over an extended coastal region using an existing database of synthetic storms. | `readings/7. Surrogate modeling...pdf` |
| 8 | Zhang et al. (2018). Advances in surrogate modeling for storm surge prediction: storm selection and addressing characteristics related to climate change. | `readings/8. Advances in surrogate...pdf` |
| 9 | Funakoshi et al. (2010). Coupling a finite element storm surge model of the North Carolina sounds with operational ocean and weather prediction models. | `readings/9. COUPLING A FINITE ELEMENT...pdf` |
| 10 | Chen et al. (2022). Storm Surge Prediction Based on Long Short-Term Memory Neural Network in the East China Sea. | `readings/10. Storm Surge Prediction...pdf` |

### Key reference not in readings folder

- **Wang et al. (2022)** — The foundational DDPM framework this project adapts. Code at `DDPM_REMD/`. Paper in `refs/wang-et-al-2022-...pdf`.
