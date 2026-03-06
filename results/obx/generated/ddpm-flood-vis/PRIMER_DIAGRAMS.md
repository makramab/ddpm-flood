# Primer Section — Diagram Implementation Details

This document describes the interactive diagrams built for the `/0x-primer` route, which serves as an educational walkthrough of the DDPM flood surge prediction model.

## Overview

The primer page (Section 7: "The Model") contains three diagram components that explain the U-Net architecture and the reverse diffusion process. All diagrams use **real data from Patch 1030, Hurricane Arthur** (θ=2.049m, Hatteras Island, Outer Banks) for consistency with the Section 3 diffusion diagrams.

## Diagrams

### 1. U-Net Architecture Diagram (`UNetArchitectureDiagram.tsx`)

**Purpose:** Explain the U-Net's compress → reason → reconstruct pipeline in plain English.

**Design:** A vertical flowchart with 5 annotated boxes, replacing the traditional U-shape schematic (which proved too abstract for a primer audience). Each box has:
- A **plain English title** as the primary label (e.g., "Compress (Encoder)", "Bottleneck: most compressed")
- The **tensor shapes** as secondary detail (e.g., "24 → 12 → 6 → 3 positions", "32ch → 64ch → 64ch")
- A description of what happens at that stage

**Key annotations:**
- **Skip connections** explained on the right side with a dashed bracket from Encoder to Decoder: "Encoder saves a snapshot at each level. Decoder uses these to recover fine detail that compression lost."
- **Timestep `t`** on the left with dashed lines pointing to the 3 middle boxes: "Timestep tells model how noisy the input is"

**Tensor shapes verified from source:** `DDPM_REMD/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py`
- Downsample: `Conv1d(dim, dim, kernel=3, stride=2, padding=1)` — halves spatial dimension
- Upsample: `ConvTranspose1d(dim, dim, kernel=4, stride=2, padding=1)` — doubles spatial dimension
- Spatial path: 24 → 12 → 6 → 3 (bottleneck) → 6 → 12 → 24
- Channel path: 1 → 32 → 64 → 64 → 128 (bottleneck) → 64 → 64 → 32 → 1
- Output shape: (batch, 1, 24) — model outputs all 24 values; GaussianDiffusion masks columns 0-3 back

**Design iterations:**
1. v1: Proportionally-sized blocks in U-shape with tiny data cell strips at input/output. Too abstract — data strips were 5.5px cells crammed at edges, disconnected from the architecture blocks.
2. v2: "Data hourglass" — colored cell strips at each resolution (24, 12, 6, 3 cells). Intermediate cells were fabricated averages with no real mapping. Didn't communicate clearly.
3. v3 (final): Vertical flowchart with plain English annotations. Dropped fake data cells entirely. Architecture details as supporting text, not headlines. Clear and self-explanatory.

### 2. U-Net Output Diagram (`UNetOutputDiagram.tsx`)

**Purpose:** Show what one reverse diffusion step actually produces, using the same 5×4 grid format as the Section 3 diffusion diagrams.

**Design:** Three 5×4 grids in a horizontal row, connected by labeled arrows:
- **Grid 1 (x_t):** Noised input at t=600 — uses `noisedProfile(600)` with `noiseFraction = 3/5` to exactly match the t=600 panel in Section 3's filmstrip
- **Grid 2 (ε):** Predicted noise — uses `rawNoise(600)` with a diverging blue/red colormap (`noiseColor`)
- **Grid 3 (x_{t-1}):** One step less noisy — computed via `reverseStep()` using the actual posterior mean formula from the DDPM source code

**Key design decisions:**
- **t=600 chosen deliberately** to match one of the Section 3 filmstrip panels [0, 200, 400, **600**, 800, 1000]. Same values, same color formula — the reader recognizes the grid instantly.
- **Grid 3 shows x_{t-1}, not x̂₀.** An earlier version showed the fully clean result, which was misleading — it implied one U-Net call produces clean data. In reality, one step barely changes anything. Showing the near-identical x_{t-1} honestly communicates why 1,000 steps are needed.
- **Reverse step uses the actual formula** from `GaussianDiffusion.p_sample`: posterior mean = blend of estimated x₀ and current x_t, using the beta schedule coefficients.

**Annotations:**
- Loop arrow from x_{t-1} back to x_t: "Repeat 1,000 times (our config): x_{t-1} becomes the new x_t"
- Note: "After all 1,000 steps: pure noise → clean surge data (the t=0 grid from Section 3)"
- Conditioning row showing all 4 fixed values: θ=2.049m, lat=35.74°, lon=75.62°, d=3.61m

### 3. Reverse-Step Formula (collapsible in `ModelSection.tsx`)

**Purpose:** Let curious readers see the exact math behind the "fixed formula" mentioned in the prose.

**Design:** A `<details>` collapsible (same pattern as DDPMSection's "View actual data row") labeled "View the exact reverse-step formula". Expands to show three steps:

1. **Estimate clean data:** x̂₀ = √(1/ᾱₜ) · xₜ − √(1/ᾱₜ − 1) · ε
2. **Posterior mean:** μₜ = (βₜ · √ᾱₜ₋₁)/(1 − ᾱₜ) · x̂₀ + ((1 − ᾱₜ₋₁) · √αₜ)/(1 − ᾱₜ) · xₜ
3. **Add noise:** x_{t-1} = μₜ + σₜ · z

Formulas verified against `GaussianDiffusion.__init__` (posterior_mean_coef1, posterior_mean_coef2, posterior_variance) and `p_sample` in the DDPM source.

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
- `UNetArchitectureDiagram.tsx` — Vertical flowchart architecture diagram
- `UNetOutputDiagram.tsx` — Three-grid "one step" diagram with loop annotation
- `PRIMER_DIAGRAMS.md` — This file

### Modified
- `diffusion-utils.ts` — Added conditioning data, noise utilities, reverse step formula
- `ModelSection.tsx` — New imports, rewritten prose, collapsible formula, improved generation process

### Superseded (still on disk, no longer imported)
- `UNetFlowDiagram.tsx` — Original animated U-shape schematic (replaced by UNetArchitectureDiagram)

## Data Consistency

All diagrams use **Patch 1030 / Hurricane Arthur** data, consistent with the Section 3 diffusion diagrams:
- 20 surge values from `scenario_arthur.json` (node_data.truth mapped via patches.json assignments[1030])
- θ=2.049m, lat=35.7402, lon=-75.6237, depth=3.61m
- Surge range: 0.235m–0.965m (all 20 values unique)
- The output diagram at t=600 uses the exact same `noisedProfile(600)` values and `noiseFraction = 3/5` color formula as the Section 3 filmstrip
