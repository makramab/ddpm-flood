# Primer Section — Diagram Implementation Details

This document describes the interactive diagrams built for the `/0x-primer` route, which serves as an educational walkthrough of the DDPM flood surge prediction model.

## Overview

The primer page contains interactive diagram components across multiple sections. All diagrams use **real data from Patch 1030, Hurricane Arthur** (θ=2.049m, Hatteras Island, Outer Banks) for consistency.

## Diagrams

### 1. U-Net Architecture Diagram (`UNetArchitectureDiagram.tsx`)

**Purpose:** Explain the U-Net's compress → reason → reconstruct pipeline in plain English.

**Design:** A vertical flowchart with 5 annotated boxes. Each box has:
- A **plain English title** as the primary label (e.g., "Compress (Encoder)", "Bottleneck: most compressed")
- The **tensor shapes** as secondary detail (e.g., "24 → 12 → 6 → 3 positions", "32ch → 64ch → 64ch")

**Key annotations:**
- **Skip connections** explained on the right side with a dashed bracket from Encoder to Decoder: "Encoder saves a snapshot at each level. Decoder uses these to recover fine detail that compression lost."
- **Timestep `t`** centered vertically on the left between the 3 middle boxes (Encoder, Bottleneck, Decoder) with symmetric dashed lines fanning out to each box: "Timestep tells model how noisy the input is"

**Animation:**
- **Staggered entrance (plays once):** Boxes appear top-to-bottom with fade+slide, mirroring data flow direction. Arrows appear with their destination box. Skip connections fade in after all boxes, then timestep annotation last. Uses `whileInView` with `viewport={{ once: true }}`.
- **Looping data-flow pulse (while in view):** A subtle primary-colored glow sweeps down through all 5 boxes sequentially (CSS `@keyframes` with staggered `animation-delay`). Reinforces the top-to-bottom data flow. Pauses when scrolled out of view. Disabled if `useReducedMotion` is true.

**Interactive detail panels:**
- Clicking any box opens an HTML panel below the SVG with a detailed explanation of that pipeline stage
- Descriptions are sourced from ModelSection, DDPMSection, and PRIMER_DIAGRAMS verified facts
- "Click any box to learn more" hint fades away after first interaction
- Keyboard accessible (Tab + Enter/Space)
- Content per box:
  - **Input:** Shape (batch, 1, 24), 4 conditioning columns (never noised), 20 surge columns, min-max normalization, inpainting trick
  - **Encoder:** Conv1d stride=2 halving spatial dimensions, channel path 32→64→64, skip connection saves at each level
  - **Bottleneck:** 3 positions × 128 features, linear attention, timestep injection, different behavior at high vs low noise
  - **Decoder:** ConvTranspose1d doubling, channels 64→64→32, skip connections inject encoder detail at each level
  - **Output:** Predicts noise ε (not clean data), fixed formula removes noise, loss computed only on columns 4-23

**Tensor shapes verified from source:** `DDPM_REMD/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py`
- Downsample: `Conv1d(dim, dim, kernel=3, stride=2, padding=1)` — halves spatial dimension
- Upsample: `ConvTranspose1d(dim, dim, kernel=4, stride=2, padding=1)` — doubles spatial dimension
- Spatial path: 24 → 12 → 6 → 3 (bottleneck) → 6 → 12 → 24
- Channel path: 1 → 32 → 64 → 64 → 128 (bottleneck) → 64 → 64 → 32 → 1
- Output shape: (batch, 1, 24) — model outputs all 24 values; GaussianDiffusion masks columns 0-3 back

**Design iterations:**
1. v1: Proportionally-sized blocks in U-shape with tiny data cell strips at input/output. Too abstract.
2. v2: "Data hourglass" — colored cell strips at each resolution. Intermediate cells were fabricated. Didn't communicate clearly.
3. v3: Vertical flowchart with plain English annotations. Clear and self-explanatory.
4. v4 (current): Added animations (staggered entrance + looping pulse), centered timestep `t`, and clickable detail panels.

### 2. U-Net Output Diagram (`UNetOutputDiagram.tsx`)

**Purpose:** Show what one reverse diffusion step actually produces, using the same 5×4 grid format as the Section 3 diffusion diagrams.

**Design:** Three 5×4 grids in a horizontal row, connected by labeled arrows:
- **Grid 1 (x_t):** Noised input at t=600 — uses `noisedProfile(600)` with `noiseFraction = 3/5` to exactly match the t=600 panel in Section 3's filmstrip
- **Grid 2 (ε):** Predicted noise — uses `rawNoise(600)` with a diverging blue/red colormap (`noiseColor`)
- **Grid 3 (x_{t-1}):** One step less noisy — computed via `reverseStep()` using the actual posterior mean formula from the DDPM source code. Uses `noiseFraction = 0.56` (vs 0.60 for x_t) so the color difference is perceptible.

**Animation:**
- **Staggered entrance (plays once):** Elements appear left-to-right mirroring computation flow: x_t grid → "U-Net" arrow → ε grid → "formula" arrow → x_{t-1} grid. Labels appear with their grids. Loop arrow and annotations fade in last.
- **Looping grid pulse (while in view):** Subtle highlight sweeps left-to-right across the three grids, reinforcing data flow direction.
- **Loop arrow flow:** The dashed loop-back arrow has a continuously animated `stroke-dashoffset`, making it visually "flow" from x_{t-1} back to x_t. Uses an SVG `<marker>` with `orient="auto"` for a seamless arrowhead aligned to the curve's tangent.

**Key design decisions:**
- **t=600 chosen deliberately** to match one of the Section 3 filmstrip panels [0, 200, 400, **600**, 800, 1000]. Same values, same color formula — the reader recognizes the grid instantly.
- **Grid 3 uses NF_PREV = 0.56** (vs NF = 0.60 for grid 1) so x_{t-1} has slightly more saturated colors, making "one step less noisy" visible. The actual `reverseStep()` values also differ numerically, but the coloring makes the difference perceptible.
- **Reverse step uses the actual formula** from `GaussianDiffusion.p_sample`: posterior mean = blend of estimated x₀ and current x_t, using the beta schedule coefficients.

**Annotations:**
- Loop arrow from x_{t-1} back to x_t: "Repeat 1,000 times (our config): x_{t-1} becomes the new x_t"
- Note: "After all 1,000 steps: pure noise → clean surge data (the t=0 grid from Section 3)"
- Conditioning row showing all 4 fixed values: θ=2.049m, lat=35.74°, lon=75.62°, d=3.61m

### 3. Reverse-Step Formula (collapsible in `ModelSection.tsx`)

**Purpose:** Let curious readers see the exact math behind the "fixed formula" mentioned in the prose.

**Design:** A `<details>` collapsible labeled "View the exact reverse-step formula". Expands to show three steps:

1. **Estimate clean data:** x̂₀ = √(1/ᾱₜ) · xₜ − √(1/ᾱₜ − 1) · ε
2. **Posterior mean:** μₜ = (βₜ · √ᾱₜ₋₁)/(1 − ᾱₜ) · x̂₀ + ((1 − ᾱₜ₋₁) · √αₜ)/(1 − ᾱₜ) · xₜ
3. **Add noise:** x_{t-1} = μₜ + σₜ · z

Formulas verified against `GaussianDiffusion.__init__` and `p_sample` in the DDPM source.

### 4. Diffusion Process Filmstrip (`DiffusionProcessDiagram.tsx`)

**Fix:** The "Generative reverse denoising process" bottom arrow's `arrow-left` marker had an incorrect path direction. With `orient="auto"` on a right-to-left line, the marker path must point in the +x direction (same as arrow-right) because the auto-orientation handles the rotation. Fixed from `M8,0 L0,3 L8,6` to `M0,0 L8,3 L0,6`.

### 5. Training Row Diagram (`AdaptationSection.tsx`)

**Purpose:** Replace the ASCII art representation of the training row format in Section 5 with a visual diagram.

**Design:** A horizontal strip of 24 colored cells showing one complete training row:
- **Cells 0–3:** Conditioning columns with primary-colored backgrounds and θ/lat/lon/d labels. Visually distinct from surge cells.
- **Cells 4–23:** Surge values colored with the same blue→red surge colormap used in Section 3 diagrams
- **Bracket labels** above: "Conditioning (fixed)" and "Target (generated by DDPM)"
- **Column indices** below: "0–3" and "4–23"
- **Value legend:** θ=2.049m, lat=35.74°, lon=−75.62°, d=3.61m + "20 surge values (0.235–0.965 m)"
- **Source note:** "Patch 1030 / Hurricane Arthur — same data as the diffusion diagrams in Section 3"

Uses real values from `diffusion-utils.ts` (`CONDITIONING`, `RAW_SURGE`, `CLEAN_PROFILE`, `cellColor`).

## ModelSection Prose Updates

The "What one forward pass produces" subsection was rewritten to clearly explain:
- The U-Net **only predicts noise** — it never directly outputs clean data
- A separate **fixed math formula** (not learned) removes a tiny amount of noise each step
- One step barely changes anything visible — you need 1,000 steps
- Analogy: U-Net = expert eye identifying dirt, formula = hand wiping it away

The "Generation process" subsection was simplified from 7 procedural steps to 5 clearer ones, with step 3 now explaining what happens **inside** each of the 1,000 reverse steps. An InfoCard was added explaining why 10 samples per patch (uncertainty estimation).

## Shared Utilities (`diffusion-utils.ts`)

Functions added during this work:
- `CONDITIONING` — theta, lat, lon, depth values for Patch 1030
- `COND_LABELS` — display labels ['θ', 'lat', 'lon', 'd']
- `rawNoise(t)` — seeded noise values at timestep t (what the model predicts)
- `reverseStep(xt, noise, t)` — one reverse diffusion step using the actual posterior mean formula
- `noiseColor(value)` — diverging blue↔red colormap for noise values in [-1, 1]

## Files

### Created
- `UNetArchitectureDiagram.tsx` — Vertical flowchart architecture diagram with animations and detail panels
- `UNetOutputDiagram.tsx` — Three-grid "one step" diagram with loop annotation and animations
- `PRIMER_DIAGRAMS.md` — This file

### Modified
- `diffusion-utils.ts` — Added conditioning data, noise utilities, reverse step formula
- `ModelSection.tsx` — New imports, rewritten prose, collapsible formula, improved generation process
- `DiffusionProcessDiagram.tsx` — Fixed reverse denoising arrow direction
- `AdaptationSection.tsx` — Replaced ASCII art with visual training row diagram

### Superseded (still on disk, no longer imported)
- `UNetFlowDiagram.tsx` — Original animated U-shape schematic (replaced by UNetArchitectureDiagram)

## Data Consistency

All diagrams use **Patch 1030 / Hurricane Arthur** data, consistent with the Section 3 diffusion diagrams:
- 20 surge values from `scenario_arthur.json` (node_data.truth mapped via patches.json assignments[1030])
- θ=2.049m, lat=35.7402, lon=-75.6237, depth=3.61m
- Surge range: 0.235m–0.965m (all 20 values unique)
- The output diagram at t=600 uses the exact same `noisedProfile(600)` values and `noiseFraction = 3/5` color formula as the Section 3 filmstrip
