"""
Visualization of DDPM flood prediction results for OBX.

Generates publication-quality figures:
  1. Spatial surge comparison: DDPM prediction vs ADCIRC truth (Dorian low-θ)
  2. Scatter plot: predicted vs truth surge (with R² line)
  3. Validation metrics summary: R² and RMSE across all scenarios
  4. Arthur failure case comparison

Usage:
    uv run python visualize_results.py
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/processed/obx")
RESULTS_DIR = Path("results/obx")
GEN_DIR = RESULTS_DIR / "generated"
OUTPUT_DIR = RESULTS_DIR / "figures"

N_COND = 4  # θ, lat, lon, depth


# ── Surge colormap (warm: low surge → cool: high surge) ──────────────────────

def make_surge_cmap():
    colors = [
        (0.0,  "#ffffb2"),   # pale yellow (low surge)
        (0.15, "#fed976"),   # warm yellow
        (0.30, "#feb24c"),   # orange
        (0.45, "#fd8d3c"),   # dark orange
        (0.60, "#fc4e2a"),   # red-orange
        (0.75, "#e31a1c"),   # red
        (0.90, "#b10026"),   # dark red
        (1.0,  "#67000d"),   # very dark red (high surge)
    ]
    return mcolors.LinearSegmentedColormap.from_list("surge", colors, N=256)


SURGE_CMAP = make_surge_cmap()


# ── Helper: extract truth for a specific θ from validation data ──────────────

def extract_scenario(val_data, target_theta_idx=0):
    """Extract patches for a specific scenario (by index) from validation data.

    Returns:
        theta: scalar θ value
        centroids: (n_patches, 3) lat, lon, depth
        surge: (n_patches, 20) surge values
    """
    unique_thetas = np.unique(val_data[:, 0])
    theta = unique_thetas[target_theta_idx]
    mask = val_data[:, 0] == theta
    rows = val_data[mask]
    centroids = rows[:, 1:N_COND]  # (n_patches, 3)
    surge = rows[:, N_COND:]       # (n_patches, 20)
    return float(theta), centroids, surge


# ── Helper: generate predictions for one θ ───────────────────────────────────

def generate_predictions(theta, patch_centroids, model_dir, milestone=None):
    """Run DDPM inference for a single θ value. Returns (mean_preds, std_preds)."""
    import torch

    sys.path.insert(0, "DDPM_REMD/denoising_diffusion_pytorch")
    from denoising_diffusion_pytorch import GaussianDiffusion, Unet

    # Suppress internal tqdm
    import denoising_diffusion_pytorch as _ddpm_mod
    _ddpm_mod.tqdm = lambda iterable, **kwargs: iterable

    DIM, DIM_MULTS, GROUPS = 32, (1, 2, 2, 4), 8
    TIMESTEPS, UNMASK_NUMBER = 1000, 4
    N_SAMPLES, BATCH_SIZE = 10, 2048
    OP_NUM = 20

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"  Device: {device}")

    # Find latest checkpoint
    if milestone is None:
        milestones = [int(p.stem.split("-")[1]) for p in Path(model_dir).glob("model-*.pt")]
        milestone = max(milestones)

    model = Unet(dim=DIM, dim_mults=DIM_MULTS, groups=GROUPS).to(device)
    diffusion = GaussianDiffusion(model, timesteps=TIMESTEPS,
                                  unmask_number=UNMASK_NUMBER, loss_type="l2").to(device)

    ckpt = torch.load(str(Path(model_dir) / f"model-{milestone}.pt"),
                       map_location=device, weights_only=False)
    diffusion.load_state_dict(ckpt["ema"])
    diffusion.eval()
    min_data, max_data = ckpt["data_range"]
    print(f"  Loaded model-{milestone}.pt (step {ckpt['step']})")

    # Normalize conditioning
    def normalize(x, mn, mx):
        return 2 * x / (mx - mn) - 2 * mn / (mx - mn) - 1

    def denormalize(x, mn, mx):
        return (x + 1) / 2.0 * (mx - mn) + mn

    seq_len = OP_NUM + N_COND
    n_patches = patch_centroids.shape[0]

    theta_norm = float(normalize(np.array([theta]), min_data[0:1], max_data[0:1])[0])
    lat_norms = normalize(patch_centroids[:, 0], min_data[1], max_data[1])
    lon_norms = normalize(patch_centroids[:, 1], min_data[2], max_data[2])
    depth_norms = normalize(patch_centroids[:, 2], min_data[3], max_data[3])

    cond = np.column_stack([
        np.full(n_patches, theta_norm), lat_norms, lon_norms, depth_norms,
    ]).astype(np.float32)

    cond_repeated = np.repeat(cond, N_SAMPLES, axis=0)
    total = n_patches * N_SAMPLES

    all_predictions = []
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        offset = 0
        batch_idx = 0
        while offset < total:
            bs = min(BATCH_SIZE, total - offset)
            batch_idx += 1
            batch_cond = cond_repeated[offset:offset + bs]

            samples = torch.randn(bs, 1, seq_len, device=device)
            cond_tensor = torch.from_numpy(batch_cond).to(device)
            for c in range(N_COND):
                samples[:, 0, c] = cond_tensor[:, c]

            print(f"    Batch {batch_idx}/{n_batches} ({bs} samples)...", end="", flush=True)
            generated = diffusion.sample(op_number=seq_len, batch_size=bs, samples=samples)
            print(" done")

            all_predictions.append(generated.cpu())
            offset += bs

    all_preds = torch.cat(all_predictions, dim=0).numpy()
    all_preds_raw = denormalize(all_preds[:, 0, :], min_data, max_data)
    all_preds_raw = all_preds_raw[:total].reshape(n_patches, N_SAMPLES, seq_len)
    surge_preds = all_preds_raw[:, :, N_COND:]

    return surge_preds.mean(axis=1), surge_preds.std(axis=1)


# ── Figure 1: Spatial surge comparison (DDPM vs Truth) ───────────────────────

def figure_surge_comparison(pred_surge, truth_surge, centroids, theta, storm_name,
                            output_path, vmin=0, vmax=1.0):
    """Side-by-side spatial surge maps at patch centroid locations."""
    lats = centroids[:, 0]
    lons = centroids[:, 1]

    # Use mean surge per patch for coloring
    pred_mean = pred_surge.mean(axis=1)
    truth_mean = truth_surge.mean(axis=1)
    error = pred_mean - truth_mean

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Panel 1: ADCIRC Truth
    sc1 = axes[0].scatter(lons, lats, c=truth_mean, cmap=SURGE_CMAP,
                          vmin=vmin, vmax=vmax, s=4, alpha=0.8)
    axes[0].set_title(f"ADCIRC Truth — {storm_name}\n(θ = {theta:.2f}m)", fontsize=13, fontweight="bold")
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label="Surge (m)")

    # Panel 2: DDPM Prediction
    sc2 = axes[1].scatter(lons, lats, c=pred_mean, cmap=SURGE_CMAP,
                          vmin=vmin, vmax=vmax, s=4, alpha=0.8)
    axes[1].set_title(f"DDPM Prediction — {storm_name}\n(θ = {theta:.2f}m)", fontsize=13, fontweight="bold")
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label="Surge (m)")

    # Panel 3: Error (pred - truth)
    err_max = max(abs(error.min()), abs(error.max()), 0.2)
    sc3 = axes[2].scatter(lons, lats, c=error, cmap="RdBu_r",
                          vmin=-err_max, vmax=err_max, s=4, alpha=0.8)
    axes[2].set_title(f"Error (Pred - Truth)\nRMSE = {np.sqrt((error**2).mean()):.3f}m",
                      fontsize=13, fontweight="bold")
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label="Error (m)")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"DDPM Flood Surge Prediction — Outer Banks, NC\n"
                 f"Hurricane {storm_name} (θ = {theta:.2f}m)",
                 fontsize=15, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Figure 2: Scatter plot (pred vs truth) ───────────────────────────────────

def figure_scatter(pred_surge, truth_surge, theta, storm_name, output_path):
    """Scatter plot of predicted vs truth surge (all patch-node values)."""
    pred_flat = pred_surge.flatten()
    truth_flat = truth_surge.flatten()

    # R² and RMSE
    ss_res = np.sum((pred_flat - truth_flat) ** 2)
    ss_tot = np.sum((truth_flat - truth_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred_flat - truth_flat) ** 2))
    bias = np.mean(pred_flat - truth_flat)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Hexbin for density
    hb = ax.hexbin(truth_flat, pred_flat, gridsize=50, cmap="YlOrRd",
                   mincnt=1, linewidths=0.2)
    plt.colorbar(hb, ax=ax, shrink=0.7, label="Count")

    # 1:1 line
    lims = [min(truth_flat.min(), pred_flat.min()) - 0.05,
            max(truth_flat.max(), pred_flat.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.7, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("ADCIRC Truth Surge (m)", fontsize=12)
    ax.set_ylabel("DDPM Predicted Surge (m)", fontsize=12)
    ax.set_title(f"DDPM vs ADCIRC — {storm_name} (θ = {theta:.2f}m)",
                 fontsize=13, fontweight="bold")

    textstr = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}m\nBias = {bias:+.4f}m\nn = {len(pred_flat)}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Figure 3: Validation metrics summary ─────────────────────────────────────

def figure_metrics_summary(output_path):
    """Bar/line chart of R² and RMSE across all validation scenarios."""
    metrics_path = GEN_DIR / "validation_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Collect all scenarios
    thetas, r2s, rmses, labels, colors_list = [], [], [], [], []
    storm_colors = {"dorian": "#2166ac", "arthur": "#b2182b"}

    for storm_name, scenarios in metrics.items():
        for sc in scenarios:
            thetas.append(sc["theta"])
            r2s.append(sc["r2"])
            rmses.append(sc["rmse"])
            labels.append(f"{storm_name.title()}\nθ={sc['theta']:.2f}m")
            colors_list.append(storm_colors.get(storm_name, "#666666"))

    x = np.arange(len(thetas))

    # R² bars
    bar_colors = ["#2166ac" if r > 0 else "#ef8a62" for r in r2s]
    ax1.bar(x, r2s, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.axhline(y=0.9, color="green", linewidth=1, linestyle="--", alpha=0.5, label="R² = 0.9")
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("Validation Performance Across All Scenarios", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # RMSE bars
    ax2.bar(x, rmses, color=colors_list, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("RMSE (m)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#2166ac", lw=8, label="Dorian"),
        Line2D([0], [0], color="#b2182b", lw=8, label="Arthur"),
    ]
    ax2.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Figure 4: Uncertainty map ─────────────────────────────────────────────────

def figure_uncertainty(std_preds, centroids, theta, storm_name, output_path):
    """Spatial map of prediction uncertainty (std across DDPM samples)."""
    lats = centroids[:, 0]
    lons = centroids[:, 1]
    mean_std = std_preds.mean(axis=1)  # mean std across 20 nodes per patch

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(lons, lats, c=mean_std, cmap="YlOrRd", s=4, alpha=0.8)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="Mean Std (m)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"DDPM Prediction Uncertainty — {storm_name} (θ = {theta:.2f}m)\n"
                 f"Mean σ = {mean_std.mean():.4f}m", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation data
    val_dorian = np.load(DATA_DIR / "val_dorian_data.npy")
    val_arthur = np.load(DATA_DIR / "val_arthur_data.npy")

    print("=" * 60)
    print("DDPM Flood Prediction — Visualization")
    print("=" * 60)

    # ── Figure 3: Metrics summary (no inference needed) ───────────────
    print("\n[Figure 3] Validation metrics summary...")
    figure_metrics_summary(OUTPUT_DIR / "fig_metrics_summary.png")

    # ── Dorian low-θ (success case): need to generate predictions ────
    theta_dorian, centroids_dorian, truth_dorian = extract_scenario(val_dorian, target_theta_idx=0)
    print(f"\n[Dorian] Scenario 1: θ = {theta_dorian:.3f}m, {centroids_dorian.shape[0]} patches")
    print(f"  Truth surge range: {truth_dorian.min():.3f} – {truth_dorian.max():.3f}m")

    # Check if cached predictions exist
    cache_path = GEN_DIR / f"pred_dorian_theta_{theta_dorian:.3f}_mean.npy"
    cache_std_path = GEN_DIR / f"pred_dorian_theta_{theta_dorian:.3f}_std.npy"

    if cache_path.exists() and cache_std_path.exists():
        print(f"  Loading cached predictions from {cache_path.name}")
        pred_dorian = np.load(cache_path)
        std_dorian = np.load(cache_std_path)
    else:
        print(f"  Running DDPM inference for θ = {theta_dorian:.3f}m...")
        pred_dorian, std_dorian = generate_predictions(
            theta_dorian, centroids_dorian, str(RESULTS_DIR)
        )
        np.save(cache_path, pred_dorian)
        np.save(cache_std_path, std_dorian)
        print(f"  Cached predictions to {cache_path.name}")

    # ── Figure 1: Dorian spatial comparison ───────────────────────────
    print("\n[Figure 1] Dorian spatial surge comparison...")
    figure_surge_comparison(
        pred_dorian, truth_dorian, centroids_dorian, theta_dorian,
        "Dorian", OUTPUT_DIR / "fig_dorian_surge_comparison.png",
        vmin=0, vmax=max(0.8, truth_dorian.mean(axis=1).max() * 1.1),
    )

    # ── Figure 2: Dorian scatter plot ─────────────────────────────────
    print("\n[Figure 2] Dorian scatter plot...")
    figure_scatter(
        pred_dorian, truth_dorian, theta_dorian,
        "Dorian", OUTPUT_DIR / "fig_dorian_scatter.png",
    )

    # ── Figure 4: Dorian uncertainty map ──────────────────────────────
    print("\n[Figure 4] Dorian uncertainty map...")
    figure_uncertainty(
        std_dorian, centroids_dorian, theta_dorian,
        "Dorian", OUTPUT_DIR / "fig_dorian_uncertainty.png",
    )

    # ── Arthur (failure case) ─────────────────────────────────────────
    theta_arthur, centroids_arthur, truth_arthur = extract_scenario(val_arthur, target_theta_idx=0)
    print(f"\n[Arthur] θ = {theta_arthur:.3f}m, {centroids_arthur.shape[0]} patches")

    # Arthur predictions already exist from validation
    pred_arthur = np.load(GEN_DIR / "pred_arthur_mean.npy")

    # ── Figure 5: Arthur spatial comparison ───────────────────────────
    print("\n[Figure 5] Arthur spatial surge comparison...")
    vmax_arthur = max(truth_arthur.mean(axis=1).max(), pred_arthur.mean(axis=1).max()) * 1.1
    figure_surge_comparison(
        pred_arthur, truth_arthur, centroids_arthur, theta_arthur,
        "Arthur", OUTPUT_DIR / "fig_arthur_surge_comparison.png",
        vmin=0, vmax=vmax_arthur,
    )

    # ── Figure 6: Arthur scatter plot ─────────────────────────────────
    print("\n[Figure 6] Arthur scatter plot...")
    figure_scatter(
        pred_arthur, truth_arthur, theta_arthur,
        "Arthur", OUTPUT_DIR / "fig_arthur_scatter.png",
    )

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
