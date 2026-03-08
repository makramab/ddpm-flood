# DDPM Flood — V2 Results

**Date**: March 1, 2026
**Predecessor**: [CURRENT_PLAN_V2.md](CURRENT_PLAN_V2.md)
**Status**: V2 training and validation complete, visualization complete

---

## 1. What Was Done

### V2 Spatial Conditioning (as planned in [CURRENT_PLAN_V2.md §4](CURRENT_PLAN_V2.md#4-the-fix-spatial-conditioning-option-2))

Expanded the conditioning vector from 1 to 4 columns:

| Column | V1 | V2 |
|--------|----|----|
| 0 | θ | θ |
| 1 | surge₁ | **lat** (patch centroid) |
| 2 | surge₂ | **lon** (patch centroid) |
| 3 | surge₃ | **depth** (patch mean bathymetry) |
| 4–23 | — | surge₁ ... surge₂₀ |

`UNMASK_NUMBER` changed from 1 → 4. Training matrix widened from 21 → 24 columns.

### Training

- Region: Outer Banks (OBX)
- Platform: Google Colab, single A100 80GB
- Optimized config: batch_size=2048, LR=4e-5, 50K steps (linear scaling from local config)
- Training time: ~45 minutes
- Final loss: ~0.006
- VRAM usage: 7.72GB / 85GB

### Validation

Validated against two held-out storms:
- **Dorian** — 9 Danso forcing variants (θ = 0.70m to 3.36m)
- **Arthur** — 1 CERA historical scenario (θ = 2.05m)

---

## 2. Results

### OBX Validation — Dorian

| θ (m) | R² | RMSE (m) | MAE (m) | Bias (m) | Category |
|-------|-----|----------|---------|----------|----------|
| 0.701 | **0.9505** | 0.062 | 0.048 | -0.031 | In-distribution |
| 0.702 | **0.9472** | 0.063 | 0.049 | -0.031 | In-distribution |
| 0.766 | **0.9434** | 0.066 | 0.051 | -0.023 | In-distribution |
| 2.781 | -1.263 | 0.821 | 0.692 | +0.610 | Near/above training max |
| 2.899 | -1.390 | 0.853 | 0.720 | +0.639 | Extrapolation |
| 2.984 | -1.465 | 0.876 | 0.739 | +0.659 | Extrapolation |
| 3.097 | -1.372 | 0.882 | 0.744 | +0.659 | Extrapolation |
| 3.237 | -1.280 | 0.929 | 0.780 | +0.692 | Extrapolation |
| 3.357 | -1.530 | 0.960 | 0.815 | +0.732 | Extrapolation |

### OBX Validation — Arthur

| θ (m) | R² | RMSE (m) | MAE (m) | Bias (m) |
|-------|-----|----------|---------|----------|
| 2.049 | **-2.568** | 0.602 | 0.488 | +0.329 |

### V1 → V2 Comparison

| Scenario | V1 R² | V2 R² | V1 RMSE | V2 RMSE | Verdict |
|----------|-------|-------|---------|---------|---------|
| Dorian low-θ (0.70m) | -0.089 | **+0.950** | 0.289m | 0.062m | Massive improvement |
| Arthur (2.05m) | -1.269 | -2.568 | 0.480m | 0.602m | Worsened |
| Dorian high-θ (3.36m) | -0.076 | -1.530 | 0.661m | 0.960m | Worsened |

### Against Success Criteria ([CURRENT_PLAN_V2.md §11](CURRENT_PLAN_V2.md#11-risk-assessment))

| Metric | V1 | Minimum Acceptable | Good Result | V2 Actual |
|--------|-----|-------------------|-------------|-----------|
| R² (Arthur, θ=2.05m) | -1.27 | > 0.0 | > 0.5 | -2.57 (failed) |
| R² (Dorian low-θ) | -0.09 | > 0.0 | > 0.3 | **+0.95 (exceeded)** |
| RMSE (in-range θ) | 0.29m | < 0.25m | < 0.15m | **0.06m (exceeded)** |
| Pearson r (Arthur) | 0.05 | > 0.3 | > 0.7 | ~0.31 (partial) |

---

## 3. Figures Generated

All figures saved to `results/obx/figures/`:

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig_dorian_surge_comparison.png` | 3-panel: ADCIRC truth vs DDPM prediction vs error (Dorian θ=0.70m) |
| 2 | `fig_dorian_scatter.png` | Scatter plot pred vs truth with R²=0.95 (Dorian θ=0.70m) |
| 3 | `fig_metrics_summary.png` | Bar chart of R² and RMSE across all 10 validation scenarios |
| 4 | `fig_dorian_uncertainty.png` | Spatial uncertainty map (mean σ = 0.059m) |
| 5 | `fig_arthur_surge_comparison.png` | 3-panel comparison (Arthur θ=2.05m, failure case) |
| 6 | `fig_arthur_scatter.png` | Scatter plot showing systematic over-prediction (Arthur) |

---

## 4. Limitations Found

### 4.1 θ is Theoretically Insufficient as Sole Storm Descriptor

The scalar conditioning variable θ (peak surge at one reference gauge) does not uniquely determine the spatial flood pattern. Two storms with the same θ can produce very different flood maps depending on track, size, speed, and wind field structure.

This contrasts with Wang et al.'s use of temperature, which is a thermodynamic state variable that genuinely determines the equilibrium distribution of molecular configurations. θ is a downstream point measurement, not a state variable.

**Evidence**: Arthur (θ=2.05m) and the closest Danso training scenario (θ=2.04m) have a spatial correlation of only **0.31** — essentially unrelated surge patterns despite nearly identical θ values.

### 4.2 Training Data Imbalance

The θ distribution is heavily skewed toward low values:

| θ Range | Training Scenarios | % of Total |
|---------|-------------------|------------|
| 0.45 – 1.00m | 135 | 82% |
| 1.00 – 2.00m | 20 | 12% |
| 2.00 – 2.88m | 9 | 6% |

Only 3 out of 164 training scenarios have θ near Arthur's 2.05m.

### 4.3 Cross-Storm Generalization Failure

The model was trained on 100% Danso synthetic scenarios (parametric Holland model storms). It fails to generalize to CERA historical storms (Arthur) even when θ is within the training range.

At θ≈2.0m:
- **Danso training**: mean surge 1.09m, 63% of nodes > 1m
- **Arthur truth**: mean surge 0.71m, 18% of nodes > 1m
- **DDPM prediction**: resembles the Danso pattern (over-predicts everywhere)

The model learned Danso-specific spatial relationships, not universal surge physics.

### 4.4 Dorian Low-θ Success is Partially Trivial

Dorian low-θ (0.70m) has max surge of only 0.76m — nearly flat everywhere. Predicting "low surge everywhere" is correct regardless of storm type, so the spatial ambiguity doesn't hurt. The more demanding test would be a held-out storm with moderate-to-high θ from the Danso ensemble itself.

### 4.5 Extrapolation Worsened in V2

V1 failed gracefully on out-of-range θ (predicted flat/safe values → R² ≈ -0.08). V2 fails confidently (predicts high surge in the Danso spatial pattern → R² ≈ -1.5). Spatial conditioning made the model more confident at all θ values, including ones where it shouldn't be.

---

## 5. Key Takeaway

V2 spatial conditioning **works as a within-ensemble surrogate**: for Danso synthetic storms at in-distribution θ values, the DDPM achieves R² > 0.94 as a fast alternative to ADCIRC. However, the approach cannot generalize across storm types because θ alone does not characterize storm spatial patterns.

**Framing for presentation:**

> The DDPM successfully learns the conditional spatial surge distribution P(surge | θ, location) within the training storm ensemble (R² > 0.94). Cross-storm generalization remains an open challenge — storms with identical reference gauge readings can produce fundamentally different flood patterns depending on track, size, and wind field structure.

---

## 6. Future Considerations

Refer to [CURRENT_PLAN_V2.md §12](CURRENT_PLAN_V2.md#12-future-options-if-v2-is-insufficient) for the full list of future options. Below are the most relevant given our findings.

### 6.1 Multi-Point Surge Conditioning (Recommended Next Step)

**Effort**: Low | **Expected Impact**: Moderate

Instead of one θ at Hatteras, use surge readings at 2–3 geographically spread reference points. Three surge values at different locations implicitly encode storm direction and size — if the northern gauge reads high but the southern reads low, the storm came from the north.

- Architecturally trivial: increase `UNMASK_NUMBER` from 4 to 6–7
- No new data sources needed — same ADCIRC output files
- At generation time: user specifies 3 surge values instead of 1

### 6.2 Include CERA Storms in Training

**Effort**: Low | **Expected Impact**: Low-Moderate

Add some CERA HSOFS storms to training data alongside Danso. This exposes the model to real hurricane patterns, not just parametric synthetic ones. Limited by the small number of available CERA HSOFS storms (~13 total, minus held-out validation).

### 6.3 Balance θ Distribution

**Effort**: Low | **Expected Impact**: Low

Oversample high-θ training scenarios or use importance weighting so the model sees more high-surge examples. Addresses §4.2 but not the fundamental issue in §4.1.

### 6.4 Storm Track Parameters as Conditioning

**Effort**: High | **Expected Impact**: High (if data available)

Add heading angle, forward speed, radius of max wind, central pressure deficit as conditioning variables. Directly addresses §4.1 by giving the model information to distinguish storm types.

Drawbacks:
- Requires extracting storm parameters from Danso forcing files and HURDAT2 for CERA storms
- More conditioning dimensions → need more training data to maintain coverage
- At generation time, user must specify these values (less practical operationally)

### 6.5 Time Series Approach (from [CURRENT_PLAN_V2.md §12, Option 5](CURRENT_PLAN_V2.md#12-future-options-if-v2-is-insufficient))

Use fort.63.nc time series data instead of peak surge. Provides 200–500x more training data per storm and a naturally continuous θ distribution. This is the most structurally faithful adaptation of Wang et al.'s framework but requires downloading ~30–100GB from TACC.

---

## 7. File Reference

### V2 Model and Results

```
results/obx/
├── model-0.pt through model-20.pt   # V2 checkpoints (50K steps, A100)
├── loss_log.csv                      # V2 training loss
├── generated/
│   ├── validation_metrics.json       # All V2 validation metrics
│   ├── pred_dorian_mean.npy          # (1974, 20) last Dorian scenario
│   ├── pred_dorian_std.npy
│   ├── pred_dorian_nodes.npy         # (58645,) last Dorian scenario (high-θ)
│   ├── truth_dorian_nodes.npy
│   ├── pred_dorian_theta_0.701_mean.npy  # (1974, 20) cached Dorian low-θ
│   ├── pred_dorian_theta_0.701_std.npy
│   ├── pred_arthur_mean.npy          # (1974, 20) Arthur
│   ├── pred_arthur_std.npy
│   ├── pred_arthur_nodes.npy         # (58645,) Arthur
│   └── truth_arthur_nodes.npy
└── figures/
    ├── fig_dorian_surge_comparison.png
    ├── fig_dorian_scatter.png
    ├── fig_dorian_uncertainty.png
    ├── fig_metrics_summary.png
    ├── fig_arthur_surge_comparison.png
    └── fig_arthur_scatter.png
```

### V1 Backup

```
results/obx_v1/                       # V1 checkpoints and results (untouched)
```

### Scripts

| File | Purpose |
|------|---------|
| `preprocess.py` | V2 preprocessing (24-col matrix with spatial features) |
| `train_flood.py` | Local training (UNMASK_NUMBER=4) |
| `train_flood_colab.py` | A100 training (batch_size=2048, 50K steps) |
| `generate_flood.py` | Local generation/validation |
| `generate_flood_colab.py` | A100 generation (batch_size=8192) |
| `visualize_results.py` | Result figures (runs local inference for missing scenarios) |
| `diagnose_arthur.py` | Arthur failure diagnostic analysis |
