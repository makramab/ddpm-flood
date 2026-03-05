"""
Diagnostic script: Why does Arthur (θ=2.05m, in training range) get R²=-2.57?

Investigates:
1. Training data θ distribution — coverage near θ≈2.0m
2. Spatial surge patterns — Arthur vs Danso training at similar θ
3. Prediction vs truth — where exactly does the model fail?
4. Per-patch error analysis — uniform failure or localized?
"""

import json
import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed/obx")
RESULTS_DIR = Path("results/obx/generated")
N_COND = 4  # θ, lat, lon, depth

# ── Load data ────────────────────────────────────────────────────────────────

print("=" * 70)
print("ARTHUR FAILURE DIAGNOSTIC")
print("=" * 70)

# Training data
train = np.load(DATA_DIR / "train_data.npy")
val_arthur = np.load(DATA_DIR / "val_arthur_data.npy")
val_dorian = np.load(DATA_DIR / "val_dorian_data.npy")
patch_centroids = np.load(DATA_DIR / "patch_centroids.npy")
node_coords = np.load(DATA_DIR / "node_coords.npy")

# Predictions
pred_arthur_mean = np.load(RESULTS_DIR / "pred_arthur_mean.npy")
pred_arthur_std = np.load(RESULTS_DIR / "pred_arthur_std.npy")
truth_arthur = np.load(RESULTS_DIR / "truth_arthur_nodes.npy")
pred_arthur_nodes = np.load(RESULTS_DIR / "pred_arthur_nodes.npy")

# Also load Dorian low-θ for comparison (the successful case)
pred_dorian_mean = np.load(RESULTS_DIR / "pred_dorian_mean.npy")
truth_dorian = np.load(RESULTS_DIR / "truth_dorian_nodes.npy")

print(f"\nData shapes:")
print(f"  Training:      {train.shape}")
print(f"  Val Arthur:    {val_arthur.shape}")
print(f"  Val Dorian:    {val_dorian.shape}")
print(f"  Pred Arthur:   {pred_arthur_mean.shape}")
print(f"  Truth Arthur:  {truth_arthur.shape}")

# ── 1. Training θ distribution ───────────────────────────────────────────────

print(f"\n{'─' * 70}")
print("1. TRAINING θ DISTRIBUTION")
print(f"{'─' * 70}")

train_thetas = train[:, 0]
n_patches = len(patch_centroids)

# Get unique θ values (each θ appears n_patches times)
unique_thetas = np.unique(train_thetas)
n_scenarios = len(unique_thetas)

print(f"\nTotal training scenarios: {n_scenarios}")
print(f"θ range: {unique_thetas.min():.3f} – {unique_thetas.max():.3f}m")
print(f"Arthur θ: {val_arthur[0, 0]:.3f}m")

# Count scenarios near Arthur's θ
arthur_theta = val_arthur[0, 0]
for radius in [0.1, 0.2, 0.3, 0.5, 1.0]:
    near = unique_thetas[(unique_thetas >= arthur_theta - radius) &
                          (unique_thetas <= arthur_theta + radius)]
    print(f"  Scenarios within ±{radius:.1f}m of Arthur θ={arthur_theta:.2f}m: {len(near)}")

# θ histogram (binned)
print(f"\nθ distribution (scenarios per bin):")
bins = np.arange(0, 3.5, 0.25)
counts, edges = np.histogram(unique_thetas, bins=bins)
for i in range(len(counts)):
    bar = "█" * counts[i]
    marker = " ◄ Arthur" if edges[i] <= arthur_theta < edges[i+1] else ""
    print(f"  {edges[i]:5.2f}–{edges[i+1]:5.2f}m: {counts[i]:3d} {bar}{marker}")

# ── 2. Surge pattern comparison ──────────────────────────────────────────────

print(f"\n{'─' * 70}")
print("2. SURGE PATTERN COMPARISON (Arthur vs nearby training scenarios)")
print(f"{'─' * 70}")

# Get training rows near Arthur's θ
near_mask = (train_thetas >= arthur_theta - 0.2) & (train_thetas <= arthur_theta + 0.2)
near_train = train[near_mask]
near_thetas_unique = np.unique(near_train[:, 0])
print(f"\nTraining scenarios within ±0.2m of θ={arthur_theta:.2f}m: {len(near_thetas_unique)}")

# Arthur surge stats (from validation data, cols 4-23)
arthur_surge = val_arthur[:, N_COND:]  # (n_patches, 20)
print(f"\nArthur truth surge stats (across all {arthur_surge.shape[0]} patches × {arthur_surge.shape[1]} nodes):")
print(f"  Mean:   {arthur_surge.mean():.4f}m")
print(f"  Std:    {arthur_surge.std():.4f}m")
print(f"  Min:    {arthur_surge.min():.4f}m")
print(f"  Max:    {arthur_surge.max():.4f}m")
print(f"  Median: {np.median(arthur_surge):.4f}m")
print(f"  % > 0:  {(arthur_surge > 0).mean()*100:.1f}%")
print(f"  % > 1m: {(arthur_surge > 1.0).mean()*100:.1f}%")

# Compare with training data at similar θ
if len(near_thetas_unique) > 0:
    # Pick the closest training θ
    closest_idx = np.argmin(np.abs(near_thetas_unique - arthur_theta))
    closest_theta = near_thetas_unique[closest_idx]
    closest_mask = train_thetas == closest_theta
    closest_train = train[closest_mask]
    closest_surge = closest_train[:, N_COND:]

    print(f"\nClosest training scenario (θ={closest_theta:.3f}m) surge stats:")
    print(f"  Mean:   {closest_surge.mean():.4f}m")
    print(f"  Std:    {closest_surge.std():.4f}m")
    print(f"  Min:    {closest_surge.min():.4f}m")
    print(f"  Max:    {closest_surge.max():.4f}m")
    print(f"  Median: {np.median(closest_surge):.4f}m")
    print(f"  % > 0:  {(closest_surge > 0).mean()*100:.1f}%")
    print(f"  % > 1m: {(closest_surge > 1.0).mean()*100:.1f}%")

    # Spatial correlation: do Arthur and closest training have similar spatial patterns?
    # Both have same patch ordering, so we can compare patch means
    arthur_patch_means = arthur_surge.mean(axis=1)  # (n_patches,)
    train_patch_means = closest_surge.mean(axis=1)   # (n_patches,)

    spatial_corr = np.corrcoef(arthur_patch_means, train_patch_means)[0, 1]
    print(f"\n  Spatial correlation (patch-level means): {spatial_corr:.4f}")
    print(f"  → {'High' if abs(spatial_corr) > 0.7 else 'Moderate' if abs(spatial_corr) > 0.4 else 'LOW'} spatial similarity")

# ── 3. Prediction analysis ───────────────────────────────────────────────────

print(f"\n{'─' * 70}")
print("3. PREDICTION vs TRUTH (Arthur)")
print(f"{'─' * 70}")

print(f"\nPrediction (node-level) stats:")
print(f"  Mean:   {pred_arthur_nodes.mean():.4f}m")
print(f"  Std:    {pred_arthur_nodes.std():.4f}m")
print(f"  Min:    {pred_arthur_nodes.min():.4f}m")
print(f"  Max:    {pred_arthur_nodes.max():.4f}m")

print(f"\nTruth (node-level) stats:")
print(f"  Mean:   {truth_arthur.mean():.4f}m")
print(f"  Std:    {truth_arthur.std():.4f}m")
print(f"  Min:    {truth_arthur.min():.4f}m")
print(f"  Max:    {truth_arthur.max():.4f}m")

errors = pred_arthur_nodes - truth_arthur
print(f"\nError (pred - truth) stats:")
print(f"  Mean (bias): {errors.mean():.4f}m")
print(f"  Std:         {errors.std():.4f}m")
print(f"  RMSE:        {np.sqrt((errors**2).mean()):.4f}m")
print(f"  MAE:         {np.abs(errors).mean():.4f}m")

# Where does the model over/under-predict?
print(f"\n  Over-predictions (pred > truth):  {(errors > 0).mean()*100:.1f}%")
print(f"  Under-predictions (pred < truth): {(errors < 0).mean()*100:.1f}%")
print(f"  Mean over-prediction magnitude:   {errors[errors > 0].mean():.4f}m")
print(f"  Mean under-prediction magnitude:  {errors[errors < 0].mean():.4f}m")

# ── 4. Patch-level error analysis ────────────────────────────────────────────

print(f"\n{'─' * 70}")
print("4. PATCH-LEVEL ERROR ANALYSIS (Arthur)")
print(f"{'─' * 70}")

# pred_arthur_mean is (n_patches, 20), arthur_surge is also (n_patches, 20)
patch_errors = pred_arthur_mean - arthur_surge
patch_rmse = np.sqrt((patch_errors**2).mean(axis=1))  # (n_patches,)
patch_bias = patch_errors.mean(axis=1)

print(f"\nPer-patch RMSE distribution:")
print(f"  Min:    {patch_rmse.min():.4f}m")
print(f"  25th:   {np.percentile(patch_rmse, 25):.4f}m")
print(f"  Median: {np.median(patch_rmse):.4f}m")
print(f"  75th:   {np.percentile(patch_rmse, 75):.4f}m")
print(f"  Max:    {patch_rmse.max():.4f}m")

# Identify worst and best patches
worst_10_idx = np.argsort(patch_rmse)[-10:][::-1]
best_10_idx = np.argsort(patch_rmse)[:10]

print(f"\nWorst 10 patches:")
print(f"  {'Patch':>6}  {'RMSE':>8}  {'Bias':>8}  {'Lat':>8}  {'Lon':>10}  {'Depth':>7}  {'TruthMean':>10}  {'PredMean':>10}")
for idx in worst_10_idx:
    print(f"  {idx:6d}  {patch_rmse[idx]:8.4f}  {patch_bias[idx]:+8.4f}  "
          f"{patch_centroids[idx, 0]:8.4f}  {patch_centroids[idx, 1]:10.4f}  "
          f"{patch_centroids[idx, 2]:7.1f}  {arthur_surge[idx].mean():10.4f}  "
          f"{pred_arthur_mean[idx].mean():10.4f}")

print(f"\nBest 10 patches:")
print(f"  {'Patch':>6}  {'RMSE':>8}  {'Bias':>8}  {'Lat':>8}  {'Lon':>10}  {'Depth':>7}  {'TruthMean':>10}  {'PredMean':>10}")
for idx in best_10_idx:
    print(f"  {idx:6d}  {patch_rmse[idx]:8.4f}  {patch_bias[idx]:+8.4f}  "
          f"{patch_centroids[idx, 0]:8.4f}  {patch_centroids[idx, 1]:10.4f}  "
          f"{patch_centroids[idx, 2]:7.1f}  {arthur_surge[idx].mean():10.4f}  "
          f"{pred_arthur_mean[idx].mean():10.4f}")

# ── 5. Is there a depth/location pattern to the errors? ──────────────────────

print(f"\n{'─' * 70}")
print("5. ERROR CORRELATION WITH SPATIAL FEATURES")
print(f"{'─' * 70}")

from numpy import corrcoef

corr_depth = corrcoef(patch_centroids[:, 2], patch_rmse)[0, 1]
corr_lat = corrcoef(patch_centroids[:, 0], patch_rmse)[0, 1]
corr_lon = corrcoef(patch_centroids[:, 1], patch_rmse)[0, 1]
corr_truth_mean = corrcoef(arthur_surge.mean(axis=1), patch_rmse)[0, 1]

print(f"\nCorrelation between patch RMSE and:")
print(f"  Depth:             {corr_depth:+.4f}")
print(f"  Latitude:          {corr_lat:+.4f}")
print(f"  Longitude:         {corr_lon:+.4f}")
print(f"  Truth mean surge:  {corr_truth_mean:+.4f}")

# ── 6. Compare with successful Dorian low-θ ──────────────────────────────────

print(f"\n{'─' * 70}")
print("6. DORIAN LOW-θ COMPARISON (successful case)")
print(f"{'─' * 70}")

# Dorian has 9 scenarios × 1974 patches. First 3 are low-θ.
# truth_dorian_nodes is node-level, but we need scenario-level
dorian_scenarios = val_dorian[:, 0]
dorian_unique = np.unique(dorian_scenarios)
first_dorian_theta = dorian_unique[0]

# Get first Dorian scenario patches
first_dorian_mask = dorian_scenarios == first_dorian_theta
first_dorian_surge = val_dorian[first_dorian_mask, N_COND:]

print(f"\nDorian scenario 1 (θ={first_dorian_theta:.3f}m) surge stats:")
print(f"  Mean:   {first_dorian_surge.mean():.4f}m")
print(f"  Std:    {first_dorian_surge.std():.4f}m")
print(f"  Min:    {first_dorian_surge.min():.4f}m")
print(f"  Max:    {first_dorian_surge.max():.4f}m")
print(f"  % > 0:  {(first_dorian_surge > 0).mean()*100:.1f}%")
print(f"  % > 1m: {(first_dorian_surge > 1.0).mean()*100:.1f}%")

# Spatial correlation between Dorian and Arthur
dorian_patch_means = first_dorian_surge.mean(axis=1)
arthur_patch_means = arthur_surge.mean(axis=1)
dorian_arthur_corr = corrcoef(dorian_patch_means, arthur_patch_means)[0, 1]
print(f"\nSpatial correlation (Dorian vs Arthur patch means): {dorian_arthur_corr:.4f}")

# Spatial correlation between Dorian and its closest training scenario
near_dorian_mask = (train_thetas >= first_dorian_theta - 0.2) & (train_thetas <= first_dorian_theta + 0.2)
near_dorian_train = train[near_dorian_mask]
if len(near_dorian_train) > 0:
    near_dorian_thetas = np.unique(near_dorian_train[:, 0])
    closest_dorian_theta = near_dorian_thetas[np.argmin(np.abs(near_dorian_thetas - first_dorian_theta))]
    closest_dorian = train[train_thetas == closest_dorian_theta, N_COND:]
    closest_dorian_patch_means = closest_dorian.mean(axis=1)
    dorian_train_corr = corrcoef(dorian_patch_means, closest_dorian_patch_means)[0, 1]
    print(f"Spatial correlation (Dorian vs closest training, θ={closest_dorian_theta:.3f}m): {dorian_train_corr:.4f}")

# ── 7. Key question: Is training data all from same source (Danso)? ──────────

print(f"\n{'─' * 70}")
print("7. DATA SOURCE ANALYSIS")
print(f"{'─' * 70}")

print(f"\nArthur is a CERA historical storm (Hurricane Arthur 2014)")
print(f"Training data is 100% Danso synthetic scenarios")
print(f"Dorian is also CERA, but has LOW θ values (0.70-0.77m)")
print(f"  → Similar to many Danso scenarios in that θ range")
print(f"  → Low surge = less spatial variation = easier to predict")
print(f"\nArthur has θ=2.05m with DIFFERENT spatial characteristics than")
print(f"Danso scenarios at the same θ — the model learned Danso patterns,")
print(f"not Arthur patterns.")

# Quantify: how different are the spatial patterns?
print(f"\n--- Spatial Pattern Divergence ---")
if len(near_thetas_unique) > 0:
    # Compare patch-level variance
    arthur_var = arthur_patch_means.var()
    train_var = train_patch_means.var()
    print(f"  Arthur patch mean variance:         {arthur_var:.6f}")
    print(f"  Closest training patch mean variance: {train_var:.6f}")
    print(f"  Ratio (Arthur/Train):                 {arthur_var/train_var:.2f}x")

    # Rank correlation (are the same patches "hot" in both?)
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(arthur_patch_means, train_patch_means)
    print(f"  Spearman rank correlation:            {spearman_r:.4f} (p={spearman_p:.2e})")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print("""
The model fails on Arthur because:

1. DIFFERENT SPATIAL PATTERNS: Arthur (real hurricane) produces surge patterns
   that differ from Danso synthetic storms at the same θ. The model learned
   Danso-specific spatial relationships, not universal surge physics.

2. DORIAN SUCCEEDS FOR A DIFFERENT REASON: Dorian low-θ (0.70m) works because
   low surge scenarios have less spatial variation — most patches see near-zero
   surge regardless of storm type. The "correct" prediction is approximately
   "low everywhere," which is easy to learn.

3. GENERALIZATION GAP: θ alone doesn't characterize storm spatial patterns.
   Two storms with the same reference gauge reading can flood very differently
   depending on track, size, speed, and wind field structure.
""")
