"""
DDPM training script for flood surge prediction — A100 Colab version.

Optimized for single A100 GPU (80GB):
- Batch size 512 (4x local) — A100 easily handles this for (1, 24) data
- TF32 tensor cores enabled (automatic 3x matmul speedup on A100)
- cuDNN benchmark mode for optimal convolution kernels
- Multi-worker DataLoader with persistent workers
- torch.compile for additional speedup (PyTorch 2.x)

Usage (on Colab):
    !python train_flood_colab.py --region obx
    !python train_flood_colab.py --region obx --steps 100   # Smoke test
"""

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm

# ── A100 optimizations (set BEFORE model creation) ───────────────────────────

# Enable TF32 on A100 — uses Tensor Cores for float32 matmuls
# ~3x speedup for matrix operations, negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# cuDNN benchmark: auto-tunes convolution algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True

# Import from Wang et al.'s DDPM library (unmodified)
sys.path.insert(0, "DDPM_REMD/denoising_diffusion_pytorch")
from denoising_diffusion_pytorch import (
    EMA,
    GaussianDiffusion,
    Unet,
    cycle,
    loss_backwards,
    num_to_groups,
)

# ── Config ───────────────────────────────────────────────────────────────────

CONFIGS = {
    "obx": {
        "data_path": "data/processed/obx/train_data.npy",
        "batch_size": 2048,  # A100 80GB: 16x local — only uses ~8GB VRAM
        "train_steps": 50_000,  # 4x fewer steps (2048/512=4x more data per step)
        "results_dir": "results/obx",
    },
    "nyc": {
        "data_path": "data/processed/nyc/train_data.npy",
        "batch_size": 1024,  # A100 80GB: 16x local
        "train_steps": 25_000,  # 4x fewer steps
        "results_dir": "results/nyc",
    },
}

# Architecture (identical to Wang et al.)
DIM = 32
DIM_MULTS = (1, 2, 2, 4)
GROUPS = 8
TIMESTEPS = 1000
UNMASK_NUMBER = 4  # θ + lat + lon + depth (columns 0-3), never noised
LOSS_TYPE = "l2"
OP_NUM = 20  # 20 surge nodes per patch (columns 0-3 = conditioning, columns 4-23 = surge)

# Training
LR = 4e-5  # Linear scaling: 1e-5 × (2048/512) = 4e-5
EMA_DECAY = 0.995
EMA_START_STEP = 500  # Scaled down from 2000 (2000 / 4)
THETA_NOISE_SIGMA = 0.15  # meters, applied to raw θ before normalization

# Logging and checkpointing
SAVE_EVERY = 2_500  # Scaled down from 5000 (checkpoints at same data intervals)
PRINT_EVERY = 50  # More frequent logging (fewer total steps)
UPDATE_EMA_EVERY = 10

# DataLoader workers (Colab typically has 2 CPU cores)
NUM_WORKERS = 2


# ── Dataset ──────────────────────────────────────────────────────────────────

class FloodDataset(data.Dataset):
    """Dataset for flood surge training data with θ noise injection."""

    def __init__(self, npy_path, theta_noise_sigma=0.0):
        super().__init__()
        self.data = np.load(npy_path)
        self.max_data = np.max(self.data, axis=0)
        self.min_data = np.min(self.data, axis=0)
        self.theta_noise_sigma = theta_noise_sigma

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index : index + 1, :].copy()  # (1, n_cols), copy to avoid mutating

        # Add Gaussian noise to θ (column 0) BEFORE normalization
        if self.theta_noise_sigma > 0:
            x[0, 0] += np.random.normal(0, self.theta_noise_sigma)

        # Min-max normalize to [-1, 1] (same formula as Wang et al.)
        x = 2 * x / (self.max_data - self.min_data)
        x = x - 2 * self.min_data / (self.max_data - self.min_data) - 1

        return torch.from_numpy(x).float()


# ── Training ─────────────────────────────────────────────────────────────────

def train(region, steps_override=None, resume_milestone=None):
    cfg = CONFIGS[region]

    assert torch.cuda.is_available(), "CUDA not available — this script requires a GPU"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using device: {device} ({gpu_name}, {gpu_mem:.0f}GB)")
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_steps = steps_override or cfg["train_steps"]

    # ── Model ────────────────────────────────────────────────────────────
    model = Unet(dim=DIM, dim_mults=DIM_MULTS, groups=GROUPS)
    model.to(device)

    diffusion = GaussianDiffusion(
        model,
        timesteps=TIMESTEPS,
        unmask_number=UNMASK_NUMBER,
        loss_type=LOSS_TYPE,
    ).to(device)

    ema = EMA(EMA_DECAY)
    ema_model = copy.deepcopy(diffusion)

    # ── Data ─────────────────────────────────────────────────────────────
    ds = FloodDataset(cfg["data_path"], theta_noise_sigma=THETA_NOISE_SIGMA)
    dl = cycle(data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    ))

    print(f"Dataset: {len(ds)} rows, {ds.data.shape[1]} columns")
    print(f"θ range (raw): {ds.data[:, 0].min():.2f} – {ds.data[:, 0].max():.2f}m")
    print(f"Lat range: {ds.data[:, 1].min():.4f} – {ds.data[:, 1].max():.4f}")
    print(f"Lon range: {ds.data[:, 2].min():.4f} – {ds.data[:, 2].max():.4f}")
    print(f"Depth range: {ds.data[:, 3].min():.2f} – {ds.data[:, 3].max():.2f}m")
    print(f"Training for {train_steps} steps, batch size {cfg['batch_size']}")

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = Adam(diffusion.parameters(), lr=LR)

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_step = 0
    if resume_milestone is not None:
        ckpt_path = results_dir / f"model-{resume_milestone}.pt"
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        diffusion.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        # Restore data range from checkpoint
        ds.min_data = ckpt["data_range"][0]
        ds.max_data = ckpt["data_range"][1]
        dl = cycle(data.DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True,
            num_workers=NUM_WORKERS, persistent_workers=True,
        ))
        print(f"Resumed at step {start_step}")

    # ── Loss log ─────────────────────────────────────────────────────────
    loss_log_path = results_dir / "loss_log.csv"
    loss_log_exists = loss_log_path.exists() and resume_milestone is not None
    loss_file = open(loss_log_path, "a" if loss_log_exists else "w", newline="")
    loss_writer = csv.writer(loss_file)
    if not loss_log_exists:
        loss_writer.writerow(["step", "loss"])

    # ── Training loop ────────────────────────────────────────────────────
    pbar = tqdm(range(start_step, train_steps), desc="Training", initial=start_step, total=train_steps)

    for step in pbar:
        diffusion.train()
        batch = next(dl).to(device)
        loss = diffusion(batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # EMA update
        if step % UPDATE_EMA_EVERY == 0:
            if step < EMA_START_STEP:
                ema_model.load_state_dict(diffusion.state_dict())
            else:
                ema.update_model_average(ema_model, diffusion)

        # Logging
        if step % PRINT_EVERY == 0:
            loss_val = loss.item()
            pbar.set_postfix(loss=f"{loss_val:.6f}")
            loss_writer.writerow([step, f"{loss_val:.6f}"])
            loss_file.flush()

        # Checkpoint
        if step != 0 and step % SAVE_EVERY == 0:
            milestone = step // SAVE_EVERY
            ckpt = {
                "step": step,
                "model": diffusion.state_dict(),
                "ema": ema_model.state_dict(),
                "data_range": [ds.min_data, ds.max_data],
            }
            torch.save(ckpt, str(results_dir / f"model-{milestone}.pt"))
            print(f"\n  Checkpoint saved: model-{milestone}.pt (step {step})")

    # Save final checkpoint
    final_milestone = train_steps // SAVE_EVERY
    ckpt = {
        "step": train_steps,
        "model": diffusion.state_dict(),
        "ema": ema_model.state_dict(),
        "data_range": [ds.min_data, ds.max_data],
    }
    torch.save(ckpt, str(results_dir / f"model-{final_milestone}.pt"))
    print(f"\nTraining complete. Final checkpoint: model-{final_milestone}.pt")

    loss_file.close()

    # Print GPU memory summary
    print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB / {gpu_mem:.0f}GB")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DDPM flood model (A100)")
    parser.add_argument("--region", required=True, choices=["obx", "nyc"],
                        help="Region to train on")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override number of training steps")
    parser.add_argument("--resume", type=int, default=None,
                        help="Resume from checkpoint milestone number")
    args = parser.parse_args()

    train(args.region, steps_override=args.steps, resume_milestone=args.resume)


if __name__ == "__main__":
    main()
