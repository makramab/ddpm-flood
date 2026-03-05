"""
Preprocessing pipeline for DDPM flood prediction.

Reads raw ADCIRC maxele.63.nc files (Danso + CERA HSOFS), extracts regional
data for Outer Banks (OBX) and NYC, clusters nodes into spatial patches via
KMeans, and outputs .npy training matrices compatible with the DDPM code.

Usage:
    uv run python preprocess.py
"""

import json
import os
import pickle
from pathlib import Path

import netCDF4 as nc
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

ADCIRC_DIR = Path("data/adcirc")
CERA_DIR = Path("data/cera")
OUTPUT_DIR = Path("data/processed")

HSOFS_NODE_COUNT = 1_813_443
FILL_VALUE_THRESHOLD = 1e10
NODES_PER_PATCH = 20
VALID_NODE_THRESHOLD = 0.90  # node must have valid surge in 90% of scenarios

TYPE_A_STORMS = [
    "Delta", "Dorian", "Florence", "Ian", "Ida",
    "Irma", "Laura", "Matthew", "Michael",
]
TYPE_B_STORMS = [
    "Bonnie", "Charley", "Floyd", "Fran", "Frances",
    "Harvey", "Ike", "Isabel", "Jeanne", "Lili", "Opal",
]
SEA_LEVELS_A = ["reference", "slr_0.44m", "slr_0.74m"]
SEA_LEVELS_B = ["ref", "slr_0.44m", "slr_0.74m"]
FORCINGS = ["control", "historical", "Vmax_10"]

REGIONS = {
    "obx": {
        "name": "Outer Banks",
        "lat_min": 34.50, "lat_max": 36.00,
        "lon_min": -76.50, "lon_max": -75.30,
        "ref_lat_min": 35.00, "ref_lat_max": 35.40,
        "ref_lon_min": -76.10, "ref_lon_max": -75.50,
        "holdout_storms": {"Dorian", "Arthur"},
    },
    "nyc": {
        "name": "New York City",
        "lat_min": 40.45, "lat_max": 40.95,
        "lon_min": -74.30, "lon_max": -73.70,
        "ref_lat_min": 40.69, "ref_lat_max": 40.72,
        "ref_lon_min": -74.02, "ref_lon_max": -74.00,
        "holdout_storms": {"Sandy", "Irene"},
    },
}


# ── Step 1: Discover scenarios ───────────────────────────────────────────────

def discover_scenarios():
    """Find all ADCIRC maxele.63.nc files and return scenario metadata."""
    scenarios = []

    # Danso Type A
    for storm in TYPE_A_STORMS:
        for sl in SEA_LEVELS_A:
            for forcing in FORCINGS:
                fp = ADCIRC_DIR / storm / sl / forcing / "maxele.63.nc"
                if fp.exists():
                    scenarios.append({
                        "source": "danso",
                        "storm": storm,
                        "sea_level": sl,
                        "forcing": forcing,
                        "filepath": str(fp),
                    })

    # Danso Type B
    for storm in TYPE_B_STORMS:
        for sl in SEA_LEVELS_B:
            for forcing in FORCINGS:
                fp = ADCIRC_DIR / storm / "hot_start" / sl / forcing / "maxele.63.nc"
                if fp.exists():
                    scenarios.append({
                        "source": "danso",
                        "storm": storm,
                        "sea_level": sl,
                        "forcing": forcing,
                        "filepath": str(fp),
                    })

    # CERA HSOFS — identify by node count
    for f in sorted(CERA_DIR.glob("*.nc")):
        ds = nc.Dataset(str(f))
        n_nodes = ds.dimensions["node"].size
        ds.close()
        if n_nodes == HSOFS_NODE_COUNT:
            # Extract storm name from filename like 2012_18_SANDY_maxele.63.nc
            parts = f.stem.replace("_maxele.63", "").split("_")
            storm_name = parts[2] if len(parts) >= 3 else parts[-1]
            scenarios.append({
                "source": "cera",
                "storm": storm_name,
                "sea_level": "historical",
                "forcing": "historical",
                "filepath": str(f),
            })

    return scenarios


# ── Step 2: Extract region nodes ─────────────────────────────────────────────

def extract_region_nodes(ref_filepath):
    """Load HSOFS mesh coords and compute region/reference masks."""
    ds = nc.Dataset(ref_filepath)
    x = np.asarray(ds.variables["x"][:])  # longitude
    y = np.asarray(ds.variables["y"][:])  # latitude
    depth = np.asarray(ds.variables["depth"][:])  # bathymetric depth (static mesh property)
    ds.close()

    region_data = {}
    for region_key, cfg in REGIONS.items():
        region_mask = (
            (y >= cfg["lat_min"]) & (y <= cfg["lat_max"]) &
            (x >= cfg["lon_min"]) & (x <= cfg["lon_max"])
        )
        ref_mask = (
            (y >= cfg["ref_lat_min"]) & (y <= cfg["ref_lat_max"]) &
            (x >= cfg["ref_lon_min"]) & (x <= cfg["ref_lon_max"])
        )
        region_indices = np.where(region_mask)[0]
        ref_indices = np.where(ref_mask)[0]

        region_data[region_key] = {
            "region_indices": region_indices,
            "ref_indices": ref_indices,
            "lats": y[region_indices],
            "lons": x[region_indices],
            "depths": depth[region_indices],
        }
        print(f"  {cfg['name']}: {len(region_indices)} region nodes, "
              f"{len(ref_indices)} reference nodes")

    return region_data


# ── Step 3: Compute θ and surge per scenario ─────────────────────────────────

def read_zeta_max(filepath):
    """Read zeta_max from a maxele.63.nc file and filter fill values."""
    ds = nc.Dataset(filepath)
    zeta = np.asarray(ds.variables["zeta_max"][:], dtype=np.float64)
    ds.close()
    # ADCIRC uses multiple fill value conventions: -99999, > 1e10, < -1e10
    zeta = np.where(
        (zeta > FILL_VALUE_THRESHOLD) | (zeta < -FILL_VALUE_THRESHOLD) | (zeta <= -99999),
        np.nan, zeta,
    )
    return zeta


def compute_all_scenario_data(scenarios, region_data):
    """Compute θ and regional surge for every scenario."""
    results = []
    print(f"\nProcessing {len(scenarios)} scenarios...")

    for i, scenario in enumerate(tqdm(scenarios, desc="Reading ADCIRC files")):
        zeta = read_zeta_max(scenario["filepath"])
        entry = {
            "scenario_idx": i,
            "source": scenario["source"],
            "storm": scenario["storm"],
            "sea_level": scenario["sea_level"],
            "forcing": scenario["forcing"],
        }
        for region_key, rd in region_data.items():
            theta = np.nanmax(zeta[rd["ref_indices"]])
            surge = zeta[rd["region_indices"]].copy()
            entry[f"theta_{region_key}"] = float(theta)
            entry[f"surge_{region_key}"] = surge
        results.append(entry)

    return results


# ── Step 4: Filter valid nodes ───────────────────────────────────────────────

def filter_valid_nodes(scenario_results, region_data, region_key, training_indices):
    """Keep only nodes that have valid surge in >= threshold fraction of training scenarios."""
    n_region_nodes = len(region_data[region_key]["region_indices"])
    valid_count = np.zeros(n_region_nodes, dtype=int)

    for idx in training_indices:
        surge = scenario_results[idx][f"surge_{region_key}"]
        valid_count += ~np.isnan(surge)

    n_train = len(training_indices)
    valid_mask = valid_count >= (VALID_NODE_THRESHOLD * n_train)
    valid_local_indices = np.where(valid_mask)[0]

    print(f"  {region_key.upper()}: {valid_mask.sum()} / {n_region_nodes} nodes pass "
          f"{VALID_NODE_THRESHOLD*100:.0f}% validity threshold")
    return valid_local_indices


# ── Step 5: KMeans spatial clustering ────────────────────────────────────────

def cluster_patches(lats, lons, valid_local_indices):
    """Cluster valid nodes into patches of exactly NODES_PER_PATCH nodes."""
    valid_lats = lats[valid_local_indices]
    valid_lons = lons[valid_local_indices]
    n_valid = len(valid_local_indices)
    n_clusters = max(1, n_valid // NODES_PER_PATCH)

    print(f"  Clustering {n_valid} valid nodes into {n_clusters} patches...")
    coords = np.column_stack([valid_lats, valid_lons])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    # Build patch assignments with exactly NODES_PER_PATCH nodes each
    patches = {}
    for cluster_id in range(n_clusters):
        members = np.where(labels == cluster_id)[0]  # indices into valid_local_indices

        if len(members) == NODES_PER_PATCH:
            patches[cluster_id] = valid_local_indices[members].tolist()
        elif len(members) > NODES_PER_PATCH:
            # Keep the NODES_PER_PATCH closest to centroid
            centroid = coords[members].mean(axis=0)
            dists = np.linalg.norm(coords[members] - centroid, axis=1)
            closest = members[np.argsort(dists)[:NODES_PER_PATCH]]
            patches[cluster_id] = valid_local_indices[closest].tolist()
        elif len(members) > 0:
            # Pad by duplicating nearest neighbors
            centroid = coords[members].mean(axis=0)
            dists = np.linalg.norm(coords[members] - centroid, axis=1)
            order = members[np.argsort(dists)]
            padded = list(order)
            while len(padded) < NODES_PER_PATCH:
                padded.append(order[len(padded) % len(order)])
            patches[cluster_id] = valid_local_indices[np.array(padded)].tolist()

    print(f"  Created {len(patches)} patches of {NODES_PER_PATCH} nodes each")
    return patches


def compute_patch_centroids(patches, lats, lons, depths):
    """Compute centroid (lat, lon, mean_depth) for each patch."""
    n_patches = len(patches)
    centroids = {}
    centroids_array = np.zeros((n_patches, 3), dtype=np.float32)
    for i, pid in enumerate(sorted(patches.keys())):
        nodes = patches[pid]
        lat_c = float(np.mean(lats[nodes]))
        lon_c = float(np.mean(lons[nodes]))
        depth_c = float(np.mean(depths[nodes]))
        centroids[pid] = (lat_c, lon_c, depth_c)
        centroids_array[i] = [lat_c, lon_c, depth_c]
    print(f"  Patch centroids: lat [{centroids_array[:, 0].min():.3f}, {centroids_array[:, 0].max():.3f}], "
          f"lon [{centroids_array[:, 1].min():.3f}, {centroids_array[:, 1].max():.3f}], "
          f"depth [{centroids_array[:, 2].min():.1f}, {centroids_array[:, 2].max():.1f}]m")
    return centroids, centroids_array


# ── Step 6: Build training matrix ────────────────────────────────────────────

def build_matrix(scenario_results, patches, scenario_indices, region_key,
                 patch_centroids):
    """Build the (N, 24) training matrix: [θ, lat, lon, depth, surge_1, ..., surge_20]."""
    n_patches = len(patches)
    n_scenarios = len(scenario_indices)
    n_rows = n_scenarios * n_patches
    matrix = np.zeros((n_rows, NODES_PER_PATCH + 4), dtype=np.float32)

    row = 0
    for sc_idx in scenario_indices:
        entry = scenario_results[sc_idx]
        theta = entry[f"theta_{region_key}"]
        surge = entry[f"surge_{region_key}"]

        for patch_id in sorted(patches.keys()):
            node_indices = patches[patch_id]
            lat_c, lon_c, depth_c = patch_centroids[patch_id]
            matrix[row, 0] = theta
            matrix[row, 1] = lat_c
            matrix[row, 2] = lon_c
            matrix[row, 3] = depth_c
            for j, ni in enumerate(node_indices):
                val = surge[ni]
                matrix[row, j + 4] = 0.0 if np.isnan(val) else val
            row += 1

    assert row == n_rows, f"Expected {n_rows} rows, got {row}"
    return matrix


# ── Step 7: Train/validation split ───────────────────────────────────────────

def split_scenarios(scenarios, holdout_storms):
    """Split scenario indices into training and per-storm validation sets."""
    train_indices = []
    val_by_storm = {}

    for i, sc in enumerate(scenarios):
        if sc["storm"].upper() in {s.upper() for s in holdout_storms}:
            storm = sc["storm"].upper()
            val_by_storm.setdefault(storm, []).append(i)
        else:
            train_indices.append(i)

    return train_indices, val_by_storm


# ── Step 8: Save outputs ────────────────────────────────────────────────────

def save_region_outputs(region_key, train_matrix, val_matrices, patches,
                        region_data, valid_local_indices, train_indices,
                        val_by_storm, scenario_results, centroids_array):
    """Save all outputs for one region."""
    out_dir = OUTPUT_DIR / region_key
    out_dir.mkdir(parents=True, exist_ok=True)

    rd = region_data[region_key]

    # Training data
    np.save(out_dir / "train_data.npy", train_matrix)

    # Validation data
    for storm_name, matrix in val_matrices.items():
        np.save(out_dir / f"val_{storm_name.lower()}_data.npy", matrix)

    # Patch assignments
    with open(out_dir / "patch_assignments.pkl", "wb") as f:
        pickle.dump(patches, f)

    # Patch centroids — (n_patches, 3) = [lat, lon, depth] per patch
    np.save(out_dir / "patch_centroids.npy", centroids_array)

    # Node coordinates — valid nodes only
    coords = np.column_stack([
        rd["lats"][valid_local_indices],
        rd["lons"][valid_local_indices],
    ])
    np.save(out_dir / "node_coords.npy", coords)

    # Node depths — valid nodes only (for DEM visualization)
    node_depths = rd["depths"][valid_local_indices]
    np.save(out_dir / "node_depths.npy", node_depths)

    # HSOFS node indices — for mapping back to full mesh
    hsofs_indices = rd["region_indices"][valid_local_indices]
    np.save(out_dir / "node_indices.npy", hsofs_indices)

    # Metadata
    theta_col = train_matrix[:, 0]
    metadata = {
        "region": REGIONS[region_key]["name"],
        "bbox": {
            "lat_min": REGIONS[region_key]["lat_min"],
            "lat_max": REGIONS[region_key]["lat_max"],
            "lon_min": REGIONS[region_key]["lon_min"],
            "lon_max": REGIONS[region_key]["lon_max"],
        },
        "ref_bbox": {
            "lat_min": REGIONS[region_key]["ref_lat_min"],
            "lat_max": REGIONS[region_key]["ref_lat_max"],
            "lon_min": REGIONS[region_key]["ref_lon_min"],
            "lon_max": REGIONS[region_key]["ref_lon_max"],
        },
        "n_valid_nodes": int(len(valid_local_indices)),
        "n_patches": len(patches),
        "nodes_per_patch": NODES_PER_PATCH,
        "n_training_scenarios": len(train_indices),
        "n_training_rows": int(train_matrix.shape[0]),
        "train_matrix_shape": list(train_matrix.shape),
        "theta_stats": {
            "min": float(np.min(theta_col)),
            "max": float(np.max(theta_col)),
            "mean": float(np.mean(theta_col)),
            "std": float(np.std(theta_col)),
        },
        "column_min": train_matrix.min(axis=0).tolist(),
        "column_max": train_matrix.max(axis=0).tolist(),
        "training_scenarios": [
            {"storm": scenario_results[i]["storm"],
             "source": scenario_results[i]["source"],
             "sea_level": scenario_results[i]["sea_level"],
             "forcing": scenario_results[i]["forcing"],
             "theta": scenario_results[i][f"theta_{region_key}"]}
            for i in train_indices
        ],
        "validation_storms": {
            storm: {
                "n_scenarios": len(indices),
                "theta_values": [
                    scenario_results[i][f"theta_{region_key}"]
                    for i in indices
                ],
            }
            for storm, indices in val_by_storm.items()
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved {region_key.upper()} outputs to {out_dir}/")
    print(f"    train_data.npy: {train_matrix.shape}")
    for storm, matrix in val_matrices.items():
        print(f"    val_{storm.lower()}_data.npy: {matrix.shape}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DDPM Flood Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Discover scenarios
    print("\n[Step 1] Discovering scenarios...")
    scenarios = discover_scenarios()
    danso_count = sum(1 for s in scenarios if s["source"] == "danso")
    cera_count = sum(1 for s in scenarios if s["source"] == "cera")
    print(f"  Found {len(scenarios)} scenarios: {danso_count} Danso + {cera_count} CERA HSOFS")

    # Step 2: Extract region nodes
    print("\n[Step 2] Extracting region nodes from HSOFS mesh...")
    ref_file = scenarios[0]["filepath"]
    region_data = extract_region_nodes(ref_file)

    # Step 3: Compute θ and surge for all scenarios
    print("\n[Step 3] Computing θ and surge for all scenarios...")
    scenario_results = compute_all_scenario_data(scenarios, region_data)

    # Process each region
    for region_key, cfg in REGIONS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {cfg['name']} ({region_key.upper()})")
        print(f"{'=' * 60}")

        # Step 7 (split first so we filter nodes on training data only)
        print(f"\n[Step 7] Splitting train/validation for {region_key.upper()}...")
        holdout = cfg["holdout_storms"]
        train_indices, val_by_storm = split_scenarios(scenarios, holdout)
        print(f"  Training: {len(train_indices)} scenarios")
        for storm, indices in val_by_storm.items():
            thetas = [scenario_results[i][f"theta_{region_key}"] for i in indices]
            print(f"  Validation {storm}: {len(indices)} scenarios, "
                  f"θ range: {min(thetas):.2f}–{max(thetas):.2f}m")

        # Step 4: Filter valid nodes
        print(f"\n[Step 4] Filtering valid nodes for {region_key.upper()}...")
        rd = region_data[region_key]
        valid_local_indices = filter_valid_nodes(
            scenario_results, region_data, region_key, train_indices
        )

        # Step 5: KMeans clustering
        print(f"\n[Step 5] Clustering patches for {region_key.upper()}...")
        patches = cluster_patches(rd["lats"], rd["lons"], valid_local_indices)

        # Step 5b: Compute patch centroids (lat, lon, depth)
        patch_centroids, centroids_array = compute_patch_centroids(
            patches, rd["lats"], rd["lons"], rd["depths"]
        )

        # Step 6: Build matrices
        print(f"\n[Step 6] Building training matrix for {region_key.upper()}...")
        train_matrix = build_matrix(
            scenario_results, patches, train_indices, region_key, patch_centroids
        )
        print(f"  Training matrix: {train_matrix.shape}")
        print(f"  θ range: {train_matrix[:, 0].min():.2f} – {train_matrix[:, 0].max():.2f}m")

        val_matrices = {}
        for storm, indices in val_by_storm.items():
            val_matrix = build_matrix(
                scenario_results, patches, indices, region_key, patch_centroids
            )
            val_matrices[storm] = val_matrix
            print(f"  Validation {storm}: {val_matrix.shape}")

        # Step 8: Save
        print(f"\n[Step 8] Saving outputs for {region_key.upper()}...")
        save_region_outputs(
            region_key, train_matrix, val_matrices, patches,
            region_data, valid_local_indices, train_indices,
            val_by_storm, scenario_results, centroids_array,
        )

    # Summary report
    print(f"\n{'=' * 60}")
    print("Preprocessing Summary")
    print(f"{'=' * 60}")

    report = {"regions": {}}
    for region_key in REGIONS:
        meta_path = OUTPUT_DIR / region_key / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        report["regions"][region_key] = {
            "n_valid_nodes": meta["n_valid_nodes"],
            "n_patches": meta["n_patches"],
            "n_training_rows": meta["n_training_rows"],
            "theta_min": meta["theta_stats"]["min"],
            "theta_max": meta["theta_stats"]["max"],
        }
        print(f"\n  {meta['region']}:")
        print(f"    Valid nodes:    {meta['n_valid_nodes']}")
        print(f"    Patches:        {meta['n_patches']}")
        print(f"    Training rows:  {meta['n_training_rows']}")
        print(f"    θ range:        {meta['theta_stats']['min']:.2f} – "
              f"{meta['theta_stats']['max']:.2f}m")
        print(f"    Matrix shape:   {meta['train_matrix_shape']}")

    report_path = OUTPUT_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
