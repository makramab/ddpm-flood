"""
Export DDPM flood prediction results to JSON for interactive visualization.

Reads the preprocessed spatial data + model prediction arrays and outputs
lightweight JSON files consumable by a web frontend.

Usage:
    uv run python export_viz_data.py
"""

import json
import pickle
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/processed/obx")
RESULTS_DIR = Path("results/obx")
GEN_DIR = RESULTS_DIR / "generated"
OUTPUT_DIR = Path("viz_data")

N_COND = 4  # θ, lat, lon, depth


def load_region_coords():
    """Load all region node coordinates from the ADCIRC mesh.

    The node-level prediction arrays (58,645) use region-local indices
    covering ALL nodes in the OBX bounding box. We need to reconstruct
    coordinates for all region nodes, not just the 39,494 valid ones.
    """
    import netCDF4 as nc

    # Use the same reference file as preprocessing
    cera_dir = Path("data/cera")
    ref_file = None
    for f in cera_dir.glob("*.nc"):
        ds = nc.Dataset(f)
        if ds.dimensions["node"].size == 1_813_443:
            ref_file = f
            ds.close()
            break
        ds.close()

    ds = nc.Dataset(ref_file)
    x = np.asarray(ds.variables["x"][:])  # longitude
    y = np.asarray(ds.variables["y"][:])  # latitude
    ds.close()

    # Extract OBX region
    region_mask = (
        (y >= 34.50) & (y <= 36.00) &
        (x >= -76.50) & (x <= -75.30)
    )
    region_indices = np.where(region_mask)[0]
    return y[region_indices], x[region_indices]


def export():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load spatial data ────────────────────────────────────────────────
    patch_centroids = np.load(DATA_DIR / "patch_centroids.npy")  # (1974, 3)
    with open(DATA_DIR / "patch_assignments.pkl", "rb") as f:
        patch_assignments = pickle.load(f)

    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    # ── Load region coordinates for node-level arrays ────────────────────
    print("Loading ADCIRC mesh coordinates...")
    region_lats, region_lons = load_region_coords()
    n_region = len(region_lats)
    print(f"  Region nodes: {n_region}")

    # ── Load validation data for ground truth at patch level ─────────────
    val_dorian = np.load(DATA_DIR / "val_dorian_data.npy")  # (17766, 24)
    val_arthur = np.load(DATA_DIR / "val_arthur_data.npy")  # (1974, 24)

    # ── Build scenarios ──────────────────────────────────────────────────
    scenarios = []

    # Dorian low-θ (success case) — θ ≈ 0.701
    dorian_low_theta = np.unique(val_dorian[:, 0])[0]
    dorian_low_mask = val_dorian[:, 0] == dorian_low_theta
    dorian_low_truth = val_dorian[dorian_low_mask][:, N_COND:]  # (1974, 20)

    pred_dorian_low_mean = np.load(GEN_DIR / "pred_dorian_theta_0.701_mean.npy")
    pred_dorian_low_std = np.load(GEN_DIR / "pred_dorian_theta_0.701_std.npy")

    scenarios.append({
        "id": "dorian_low",
        "name": "Hurricane Dorian",
        "label": "In-distribution (success)",
        "theta": round(float(dorian_low_theta), 3),
        "pred_mean": pred_dorian_low_mean,  # (1974, 20)
        "pred_std": pred_dorian_low_std,
        "truth": dorian_low_truth,
    })

    # Dorian high-θ (extrapolation failure) — use the highest θ
    dorian_high_theta_val = np.unique(val_dorian[:, 0])[-1]
    dorian_high_mask = val_dorian[:, 0] == dorian_high_theta_val
    dorian_high_truth = val_dorian[dorian_high_mask][:, N_COND:]

    # pred_dorian_mean/std are for the default scenario (high-θ Dorian)
    pred_dorian_mean = np.load(GEN_DIR / "pred_dorian_mean.npy")
    pred_dorian_std = np.load(GEN_DIR / "pred_dorian_std.npy")

    scenarios.append({
        "id": "dorian_high",
        "name": "Hurricane Dorian",
        "label": "Extrapolation (failure)",
        "theta": round(float(dorian_high_theta_val), 3),
        "pred_mean": pred_dorian_mean,
        "pred_std": pred_dorian_std,
        "truth": dorian_high_truth,
    })

    # Arthur (cross-storm failure)
    arthur_theta = val_arthur[0, 0]
    arthur_truth = val_arthur[:, N_COND:]

    pred_arthur_mean = np.load(GEN_DIR / "pred_arthur_mean.npy")
    pred_arthur_std = np.load(GEN_DIR / "pred_arthur_std.npy")

    scenarios.append({
        "id": "arthur",
        "name": "Hurricane Arthur",
        "label": "Cross-storm (failure)",
        "theta": round(float(arthur_theta), 3),
        "pred_mean": pred_arthur_mean,
        "pred_std": pred_arthur_std,
        "truth": arthur_truth,
    })

    # ── Export node coordinates ───────────────────────────────────────────
    # For efficiency, export only nodes that have valid predictions
    # (i.e., nodes assigned to patches)
    assigned_nodes = set()
    for node_list in patch_assignments.values():
        assigned_nodes.update(node_list)
    assigned_nodes = sorted(assigned_nodes)

    node_coords = {
        "lats": [round(float(region_lats[i]), 6) for i in assigned_nodes],
        "lons": [round(float(region_lons[i]), 6) for i in assigned_nodes],
        "indices": assigned_nodes,
    }
    # Create reverse mapping: region-local index -> position in our arrays
    idx_to_pos = {idx: pos for pos, idx in enumerate(assigned_nodes)}

    print(f"  Assigned nodes: {len(assigned_nodes)}")

    # ── Export patch centroids ───────────────────────────────────────────
    patches_data = {
        "n_patches": int(patch_centroids.shape[0]),
        "centroids": {
            "lats": [round(float(v), 6) for v in patch_centroids[:, 0]],
            "lons": [round(float(v), 6) for v in patch_centroids[:, 1]],
            "depths": [round(float(v), 2) for v in patch_centroids[:, 2]],
        },
        "assignments": {
            str(k): [idx_to_pos[n] for n in v]
            for k, v in patch_assignments.items()
        },
    }

    # ── Export scenarios ──────────────────────────────────────────────────
    metrics = json.load(open(GEN_DIR / "validation_metrics.json"))

    scenarios_export = []
    for sc in scenarios:
        pred_mean = sc["pred_mean"]  # (1974, 20)
        pred_std = sc["pred_std"]
        truth = sc["truth"]

        # Compute per-patch mean surge
        pred_patch_mean = pred_mean.mean(axis=1)  # (1974,)
        truth_patch_mean = truth.mean(axis=1)
        error_patch = pred_patch_mean - truth_patch_mean
        uncertainty_patch = pred_std.mean(axis=1)

        # Compute per-patch metrics
        ss_res = np.sum((pred_patch_mean - truth_patch_mean) ** 2)
        ss_tot = np.sum((truth_patch_mean - truth_patch_mean.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(error_patch ** 2)))
        bias = float(np.mean(error_patch))
        mae = float(np.mean(np.abs(error_patch)))

        # Stitch to node level for finer visualization
        node_pred = np.full(len(assigned_nodes), np.nan)
        node_truth = np.full(len(assigned_nodes), np.nan)
        node_error = np.full(len(assigned_nodes), np.nan)
        node_uncertainty = np.full(len(assigned_nodes), np.nan)

        for patch_id, node_list in patch_assignments.items():
            for j, region_idx in enumerate(node_list):
                pos = idx_to_pos[region_idx]
                node_pred[pos] = float(pred_mean[patch_id, j])
                node_truth[pos] = float(truth[patch_id, j])
                node_error[pos] = float(pred_mean[patch_id, j] - truth[patch_id, j])
                node_uncertainty[pos] = float(pred_std[patch_id, j])

        # Round for JSON size
        def r3(arr):
            return [round(float(v), 4) if not np.isnan(v) else None for v in arr]

        scenarios_export.append({
            "id": sc["id"],
            "name": sc["name"],
            "label": sc["label"],
            "theta": sc["theta"],
            "metrics": {
                "r2": round(r2, 4),
                "rmse": round(rmse, 4),
                "bias": round(bias, 4),
                "mae": round(mae, 4),
                "mean_uncertainty": round(float(uncertainty_patch.mean()), 4),
            },
            "node_data": {
                "prediction": r3(node_pred),
                "truth": r3(node_truth),
                "error": r3(node_error),
                "uncertainty": r3(node_uncertainty),
            },
            "patch_data": {
                "prediction": [round(float(v), 4) for v in pred_patch_mean],
                "truth": [round(float(v), 4) for v in truth_patch_mean],
                "error": [round(float(v), 4) for v in error_patch],
                "uncertainty": [round(float(v), 4) for v in uncertainty_patch],
            },
        })
        print(f"  Scenario '{sc['id']}': θ={sc['theta']}m, R²={r2:.4f}, RMSE={rmse:.4f}m")

    # ── Export θ distribution ────────────────────────────────────────────
    training_thetas = [s["theta"] for s in metadata["training_scenarios"]]
    theta_distribution = {
        "training": [round(t, 4) for t in sorted(training_thetas)],
        "validation": {
            name: [round(t, 4) for t in info["theta_values"]]
            for name, info in metadata["validation_storms"].items()
        },
        "stats": metadata["theta_stats"],
    }

    # ── Export all validation metrics ────────────────────────────────────
    all_metrics = []
    for storm_key, storm_metrics in metrics.items():
        for m in storm_metrics:
            all_metrics.append({
                "storm": storm_key.capitalize(),
                "theta": round(m["theta"], 3),
                "r2": round(m["r2"], 4),
                "rmse": round(m["rmse"], 4),
                "bias": round(m["bias"], 4),
                "mae": round(m["mae"], 4),
                "mean_std": round(m["mean_std"], 4),
            })

    # ── Write output files ───────────────────────────────────────────────
    # 1. Node coordinates (shared across scenarios)
    with open(OUTPUT_DIR / "node_coords.json", "w") as f:
        json.dump(node_coords, f)
    print(f"\n  Wrote node_coords.json ({len(assigned_nodes)} nodes)")

    # 2. Patch geometry
    with open(OUTPUT_DIR / "patches.json", "w") as f:
        json.dump(patches_data, f)
    print(f"  Wrote patches.json ({patches_data['n_patches']} patches)")

    # 3. Individual scenario files (keeps file sizes manageable)
    for sc in scenarios_export:
        fname = f"scenario_{sc['id']}.json"
        with open(OUTPUT_DIR / fname, "w") as f:
            json.dump(sc, f)
        fsize = (OUTPUT_DIR / fname).stat().st_size / 1024
        print(f"  Wrote {fname} ({fsize:.0f} KB)")

    # 4. Scenario index (lightweight — no node data)
    scenario_index = [{
        "id": sc["id"],
        "name": sc["name"],
        "label": sc["label"],
        "theta": sc["theta"],
        "metrics": sc["metrics"],
    } for sc in scenarios_export]

    with open(OUTPUT_DIR / "scenarios.json", "w") as f:
        json.dump(scenario_index, f, indent=2)
    print(f"  Wrote scenarios.json")

    # 5. θ distribution
    with open(OUTPUT_DIR / "theta_distribution.json", "w") as f:
        json.dump(theta_distribution, f, indent=2)
    print(f"  Wrote theta_distribution.json")

    # 6. All validation metrics
    with open(OUTPUT_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Wrote all_metrics.json ({len(all_metrics)} entries)")

    # ── Summary ──────────────────────────────────────────────────────────
    total_kb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.json")) / 1024
    print(f"\n  Total export size: {total_kb:.0f} KB ({total_kb/1024:.1f} MB)")
    print(f"  Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    export()
