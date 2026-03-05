"""
DDPM sample generation and evaluation — A100 Colab version.

Optimized for single A100 GPU (80GB):
- Larger generation batch size (8192 vs 2048) — reverse diffusion is inference-only
- TF32 tensor cores enabled
- cuDNN benchmark mode

Usage (on Colab):
    !python generate_flood_colab.py --region obx --mode validate
    !python generate_flood_colab.py --region obx --mode generate --theta 2.0
    !python generate_flood_colab.py --region obx --mode sweep --theta-min 0.5 --theta-max 3.5 --theta-step 0.1
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── A100 optimizations ───────────────────────────────────────────────────────

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Import from Wang et al.'s DDPM library (unmodified)
sys.path.insert(0, "DDPM_REMD/denoising_diffusion_pytorch")
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

# Suppress Wang et al.'s internal tqdm (prints 1000 lines per batch)
import denoising_diffusion_pytorch as _ddpm_mod
_ddpm_mod.tqdm = lambda iterable, **kwargs: iterable

# ── Config ───────────────────────────────────────────────────────────────────

CONFIGS = {
    "obx": {
        "data_dir": "data/processed/obx",
        "results_dir": "results/obx",
        "val_storms": {
            "dorian": "val_dorian_data.npy",
            "arthur": "val_arthur_data.npy",
        },
    },
    "nyc": {
        "data_dir": "data/processed/nyc",
        "results_dir": "results/nyc",
        "val_storms": {
            "sandy": "val_sandy_data.npy",
            "irene": "val_irene_data.npy",
        },
    },
}

# Architecture (must match training)
DIM = 32
DIM_MULTS = (1, 2, 2, 4)
GROUPS = 8
TIMESTEPS = 1000
UNMASK_NUMBER = 4
LOSS_TYPE = "l2"
OP_NUM = 20  # 20 surge nodes per patch (columns 0-3 = conditioning, columns 4-23 = surge)
N_COND = 4   # Number of conditioning columns (θ, lat, lon, depth)

# Generation — A100 optimized
N_SAMPLES = 10   # Number of DDPM samples per (θ, patch) to average over
BATCH_SIZE = 8192  # A100: 4x larger than local (reverse diffusion is inference-only)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(results_dir, milestone, device):
    """Load EMA model from checkpoint."""
    model = Unet(dim=DIM, dim_mults=DIM_MULTS, groups=GROUPS)
    model.to(device)

    diffusion = GaussianDiffusion(
        model,
        timesteps=TIMESTEPS,
        unmask_number=UNMASK_NUMBER,
        loss_type=LOSS_TYPE,
    ).to(device)

    ckpt_path = Path(results_dir) / f"model-{milestone}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Load EMA weights (the better model) into diffusion
    diffusion.load_state_dict(ckpt["ema"])
    diffusion.eval()

    min_data = ckpt["data_range"][0]
    max_data = ckpt["data_range"][1]

    print(f"  Loaded step {ckpt['step']}")
    return diffusion, min_data, max_data


def find_latest_milestone(results_dir):
    """Find the highest milestone number in results directory."""
    results_path = Path(results_dir)
    milestones = []
    for p in results_path.glob("model-*.pt"):
        try:
            m = int(p.stem.split("-")[1])
            milestones.append(m)
        except (IndexError, ValueError):
            continue
    if not milestones:
        raise FileNotFoundError(f"No checkpoints found in {results_dir}")
    return max(milestones)


# ── Normalization ────────────────────────────────────────────────────────────

def normalize(x, min_data, max_data):
    """Min-max normalize to [-1, 1] (same formula as training)."""
    return 2 * x / (max_data - min_data) - 2 * min_data / (max_data - min_data) - 1


def denormalize(x, min_data, max_data):
    """Inverse of normalize: [-1, 1] back to original scale."""
    return (x + 1) / 2.0 * (max_data - min_data) + min_data


# ── Generation ───────────────────────────────────────────────────────────────

def generate_for_patches(diffusion, theta, patch_centroids, min_data, max_data,
                         device, n_samples=N_SAMPLES, batch_size=BATCH_SIZE):
    """Generate surge predictions for all patches at a given θ,
    with per-patch spatial conditioning (lat, lon, depth).
    """
    seq_len = OP_NUM + N_COND  # 24
    n_patches = patch_centroids.shape[0]

    # Normalize conditioning values
    theta_norm = float(normalize(np.array([theta]), min_data[0:1], max_data[0:1])[0])
    lat_norms = normalize(patch_centroids[:, 0], min_data[1], max_data[1])
    lon_norms = normalize(patch_centroids[:, 1], min_data[2], max_data[2])
    depth_norms = normalize(patch_centroids[:, 2], min_data[3], max_data[3])

    cond = np.column_stack([
        np.full(n_patches, theta_norm),
        lat_norms,
        lon_norms,
        depth_norms,
    ]).astype(np.float32)

    cond_repeated = np.repeat(cond, n_samples, axis=0)
    total = n_patches * n_samples

    all_predictions = []
    n_batches = (total + batch_size - 1) // batch_size

    with torch.no_grad():
        offset = 0
        batch_idx = 0
        while offset < total:
            bs = min(batch_size, total - offset)
            batch_idx += 1
            batch_cond = cond_repeated[offset:offset + bs]

            samples = torch.randn(bs, 1, seq_len, device=device)
            cond_tensor = torch.from_numpy(batch_cond).to(device)
            samples[:, 0, 0] = cond_tensor[:, 0]
            samples[:, 0, 1] = cond_tensor[:, 1]
            samples[:, 0, 2] = cond_tensor[:, 2]
            samples[:, 0, 3] = cond_tensor[:, 3]

            print(f"    Batch {batch_idx}/{n_batches} ({bs} samples, 1000 diffusion steps)...", end="", flush=True)
            generated_batch = diffusion.sample(
                op_number=seq_len, batch_size=bs, samples=samples
            )
            print(" done")

            all_predictions.append(generated_batch.cpu())
            offset += bs

    all_preds = torch.cat(all_predictions, dim=0).numpy()
    all_preds_raw = denormalize(all_preds[:, 0, :], min_data, max_data)
    all_preds_raw = all_preds_raw[:total].reshape(n_patches, n_samples, seq_len)
    surge_preds = all_preds_raw[:, :, N_COND:]

    mean_preds = surge_preds.mean(axis=1)
    std_preds = surge_preds.std(axis=1)

    return mean_preds, std_preds


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_metrics(predictions, truth):
    """Compute R² and RMSE between predictions and truth."""
    residuals = predictions - truth
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias}


def stitch_patches_to_nodes(patch_predictions, patch_assignments):
    """Map patch-level predictions back to node-level surge values."""
    max_idx = max(idx for nodes in patch_assignments.values() for idx in nodes)
    node_surge = np.full(max_idx + 1, np.nan)
    for patch_id, node_indices in patch_assignments.items():
        for j, node_idx in enumerate(node_indices):
            node_surge[node_idx] = patch_predictions[patch_id, j]
    return node_surge


# ── Mode: Validate ───────────────────────────────────────────────────────────

def run_validate(region, diffusion, min_data, max_data, device, output_dir,
                 n_samples=N_SAMPLES, batch_size=BATCH_SIZE):
    """Validate against held-out storms."""
    cfg = CONFIGS[region]
    data_dir = Path(cfg["data_dir"])

    patch_assignments = pickle.load(open(data_dir / "patch_assignments.pkl", "rb"))
    node_coords = np.load(data_dir / "node_coords.npy")
    patch_centroids = np.load(data_dir / "patch_centroids.npy")
    n_patches = len(patch_assignments)
    n_nodes = node_coords.shape[0]

    print(f"\nValidation: {n_patches} patches, {n_nodes} nodes")

    results = {}

    for storm_name, val_file in cfg["val_storms"].items():
        val_path = data_dir / val_file
        if not val_path.exists():
            print(f"  Skipping {storm_name}: {val_path} not found")
            continue

        val_data = np.load(val_path)
        unique_thetas = np.unique(val_data[:, 0])
        print(f"\n  Storm: {storm_name} ({len(unique_thetas)} scenario(s))")

        storm_results = []

        for theta in unique_thetas:
            mask = val_data[:, 0] == theta
            truth_rows = val_data[mask]
            truth_surge_patches = truth_rows[:, N_COND:]
            n_val_patches = truth_rows.shape[0]

            print(f"    θ = {theta:.3f}m ({n_val_patches} patches)")

            val_centroids = truth_rows[:, 1:N_COND]

            mean_preds, std_preds = generate_for_patches(
                diffusion, theta, val_centroids, min_data, max_data, device,
                n_samples=n_samples, batch_size=batch_size,
            )

            pred_flat = mean_preds.flatten()
            truth_flat = truth_surge_patches.flatten()

            metrics = {k: float(v) for k, v in compute_metrics(pred_flat, truth_flat).items()}
            metrics["theta"] = float(theta)
            metrics["n_patches"] = int(n_val_patches)
            metrics["mean_std"] = float(std_preds.mean())

            storm_results.append(metrics)
            print(f"      R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}m, "
                  f"MAE = {metrics['mae']:.4f}m, bias = {metrics['bias']:.4f}m")

        results[storm_name] = storm_results

        np.save(output_dir / f"pred_{storm_name}_mean.npy", mean_preds)
        np.save(output_dir / f"pred_{storm_name}_std.npy", std_preds)

        if n_val_patches == n_patches:
            node_pred = stitch_patches_to_nodes(mean_preds, patch_assignments)
            np.save(output_dir / f"pred_{storm_name}_nodes.npy", node_pred)
            node_truth = stitch_patches_to_nodes(truth_surge_patches, patch_assignments)
            np.save(output_dir / f"truth_{storm_name}_nodes.npy", node_truth)

    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    return results


# ── Mode: Sweep ──────────────────────────────────────────────────────────────

def run_sweep(region, diffusion, min_data, max_data, device, output_dir,
              theta_min, theta_max, theta_step,
              n_samples=N_SAMPLES, batch_size=BATCH_SIZE):
    """Generate predictions for a range of θ values (tipping-point discovery)."""
    cfg = CONFIGS[region]
    data_dir = Path(cfg["data_dir"])

    patch_assignments = pickle.load(open(data_dir / "patch_assignments.pkl", "rb"))
    node_coords = np.load(data_dir / "node_coords.npy")
    patch_centroids = np.load(data_dir / "patch_centroids.npy")
    n_patches = len(patch_assignments)
    n_nodes = node_coords.shape[0]

    thetas = np.arange(theta_min, theta_max + theta_step / 2, theta_step)
    print(f"\nSweep: {len(thetas)} θ values from {theta_min:.2f}m to {theta_max:.2f}m")

    sweep_results = []

    for theta in thetas:
        mean_preds, std_preds = generate_for_patches(
            diffusion, theta, patch_centroids, min_data, max_data, device,
            n_samples=n_samples, batch_size=batch_size,
        )

        node_pred = stitch_patches_to_nodes(mean_preds, patch_assignments)

        result = {
            "theta": float(theta),
            "surge_mean": float(np.nanmean(node_pred)),
            "surge_median": float(np.nanmedian(node_pred)),
            "surge_max": float(np.nanmax(node_pred)),
            "surge_p95": float(np.nanpercentile(node_pred, 95)),
            "surge_p99": float(np.nanpercentile(node_pred, 99)),
            "pct_above_1m": float(np.nanmean(node_pred > 1.0) * 100),
            "pct_above_2m": float(np.nanmean(node_pred > 2.0) * 100),
            "pct_above_3m": float(np.nanmean(node_pred > 3.0) * 100),
            "mean_uncertainty": float(std_preds.mean()),
        }
        sweep_results.append(result)

        print(f"  θ={theta:.2f}m: mean={result['surge_mean']:.3f}m, "
              f"max={result['surge_max']:.3f}m, >2m={result['pct_above_2m']:.1f}%")

        np.save(output_dir / f"sweep_theta_{theta:.3f}_nodes.npy", node_pred)

    sweep_path = output_dir / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to {sweep_path}")

    return sweep_results


# ── Mode: Generate ───────────────────────────────────────────────────────────

def run_generate(region, diffusion, min_data, max_data, device, output_dir, theta,
                 n_samples=N_SAMPLES, batch_size=BATCH_SIZE):
    """Generate predictions for a single θ value."""
    cfg = CONFIGS[region]
    data_dir = Path(cfg["data_dir"])

    patch_assignments = pickle.load(open(data_dir / "patch_assignments.pkl", "rb"))
    node_coords = np.load(data_dir / "node_coords.npy")
    patch_centroids = np.load(data_dir / "patch_centroids.npy")
    n_patches = len(patch_assignments)
    n_nodes = node_coords.shape[0]

    print(f"\nGenerating for θ = {theta:.3f}m ({n_patches} patches)")

    mean_preds, std_preds = generate_for_patches(
        diffusion, theta, patch_centroids, min_data, max_data, device,
        n_samples=n_samples, batch_size=batch_size,
    )

    node_pred = stitch_patches_to_nodes(mean_preds, patch_assignments)
    node_std = stitch_patches_to_nodes(std_preds, patch_assignments)

    np.save(output_dir / f"pred_theta_{theta:.3f}_nodes.npy", node_pred)
    np.save(output_dir / f"pred_theta_{theta:.3f}_std.npy", node_std)
    np.save(output_dir / f"pred_theta_{theta:.3f}_coords.npy", node_coords)

    node_depths_path = data_dir / "node_depths.npy"
    if node_depths_path.exists():
        import shutil
        shutil.copy2(node_depths_path, output_dir / "node_depths.npy")

    print(f"  Surge stats: mean={np.nanmean(node_pred):.3f}m, "
          f"max={np.nanmax(node_pred):.3f}m, "
          f"median={np.nanmedian(node_pred):.3f}m")
    print(f"  Uncertainty: mean σ = {np.nanmean(node_std):.4f}m")
    print(f"  Saved to {output_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate DDPM flood predictions (A100)")
    parser.add_argument("--region", required=True, choices=["obx", "nyc"])
    parser.add_argument("--mode", required=True, choices=["validate", "sweep", "generate"])
    parser.add_argument("--milestone", type=int, default=None,
                        help="Checkpoint milestone (default: latest)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    # Sweep args
    parser.add_argument("--theta-min", type=float, default=0.5)
    parser.add_argument("--theta-max", type=float, default=3.5)
    parser.add_argument("--theta-step", type=float, default=0.1)

    # Generate args
    parser.add_argument("--theta", type=float, default=None)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — this script requires a GPU"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using device: {device} ({gpu_name}, {gpu_mem:.0f}GB)")

    cfg = CONFIGS[args.region]
    milestone = args.milestone or find_latest_milestone(cfg["results_dir"])
    diffusion, min_data, max_data = load_model(cfg["results_dir"], milestone, device)

    output_dir = Path(cfg["results_dir"]) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = {"n_samples": args.n_samples, "batch_size": args.batch_size}

    if args.mode == "validate":
        run_validate(args.region, diffusion, min_data, max_data, device, output_dir,
                     **gen_kwargs)
    elif args.mode == "sweep":
        run_sweep(args.region, diffusion, min_data, max_data, device, output_dir,
                  args.theta_min, args.theta_max, args.theta_step, **gen_kwargs)
    elif args.mode == "generate":
        if args.theta is None:
            parser.error("--theta is required for generate mode")
        run_generate(args.region, diffusion, min_data, max_data, device, output_dir,
                     args.theta, **gen_kwargs)

    print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB / {gpu_mem:.0f}GB")


if __name__ == "__main__":
    main()
