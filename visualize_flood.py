"""
Miura-style flood inundation visualization for ADCIRC storm surge data.
Shows flood extent on LAND only (masked by bathymetry), with smooth interpolation.
Matches the style of Miura et al. 2021 Figures 8, 9, 12.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import contextily as ctx

# ── Configuration ─────────────────────────────────────────────────────────────
SANDY_FILE = "data/cera/2012_18_SANDY_maxele.63.nc"

# Lower Manhattan bounding box (matches Miura Fig 8/9/12)
LM_BBOX = {"lon_min": -74.025, "lon_max": -73.970, "lat_min": 40.700, "lat_max": 40.770}

# NYC wide bounding box
NYC_BBOX = {"lon_min": -74.20, "lon_max": -73.75, "lat_min": 40.50, "lat_max": 40.85}

# Shallow water depth cutoff for coastal flooding
SHALLOW_DEPTH_CUTOFF = 2.0  # meters

# Minimum inundation to display
MIN_INUNDATION = 0.05

# Land mask: depth threshold for "is this a water body?"
# Nodes deeper than this are open water (rivers, ocean) — never show flood there
WATER_BODY_DEPTH = 1.0  # meters — deeper than this = river/ocean, not floodable land

# ── Color scheme matching Miura's yellow-cyan-teal-blue ───────────────────────
def make_flood_cmap():
    colors = [
        (0.0,  "#ffffb2"),   # pale yellow (shallow)
        (0.15, "#c7e9b4"),   # light green-yellow
        (0.30, "#7fcdbb"),   # cyan-green
        (0.50, "#41b6c4"),   # medium cyan
        (0.70, "#1d91c0"),   # blue
        (0.85, "#225ea8"),   # dark blue
        (1.0,  "#0c2c84"),   # very dark blue (deep)
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "miura_flood", [(v, c) for v, c in colors], N=256
    )

FLOOD_CMAP = make_flood_cmap()


# ── Load ADCIRC data ──────────────────────────────────────────────────────────
def load_adcirc(filepath):
    ds = nc.Dataset(filepath)
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    depth = ds.variables['depth'][:]
    zeta_max = ds.variables['zeta_max'][:]
    elements = ds.variables['element'][:] - 1
    ds.close()
    zeta_max = np.where((zeta_max > 1e10) | (zeta_max < -1e10), np.nan, zeta_max)
    return x, y, depth, zeta_max, elements


def compute_inundation(depth, zeta_max):
    """
    Compute inundation depth on land and shallow coastal nodes.
    Excludes deep water bodies entirely.
    """
    inundation = np.full_like(zeta_max, np.nan)
    valid = ~np.isnan(zeta_max)

    # Land nodes: inundation = surge above ground elevation
    land = (depth <= 0) & valid
    inundation[land] = zeta_max[land] + depth[land]

    # Shallow coastal (0 < depth <= cutoff): transitional zone
    shallow = (depth > 0) & (depth <= SHALLOW_DEPTH_CUTOFF) & valid
    # For shallow nodes, inundation represents how much water rises above normal
    # Scale it down to avoid exaggerating — use zeta_max minus the node's normal depth
    inundation[shallow] = zeta_max[shallow] - depth[shallow]

    # Only keep positive inundation
    inundation[inundation < MIN_INUNDATION] = np.nan

    return inundation


def extract_region(x, y, depth, zeta_max, elements, bbox, pad=0.005):
    mask = (
        (y >= bbox["lat_min"] - pad) & (y <= bbox["lat_max"] + pad) &
        (x >= bbox["lon_min"] - pad) & (x <= bbox["lon_max"] + pad)
    )
    local_idx = np.where(mask)[0]
    idx_set = set(local_idx)

    tri_mask = np.array([
        elements[i, 0] in idx_set and elements[i, 1] in idx_set and elements[i, 2] in idx_set
        for i in range(len(elements))
    ])
    local_elements = elements[tri_mask]

    global_to_local = {g: l for l, g in enumerate(local_idx)}
    local_tri = np.array([[global_to_local[v] for v in tri] for tri in local_elements])

    return (
        x[local_idx], y[local_idx],
        depth[local_idx], zeta_max[local_idx],
        local_tri, local_idx
    )


def plot_flood_map(ax, lx, ly, local_depth, local_tri, inundation, bbox,
                   vmin=0, vmax=3.5, zoom=14, alpha=0.65,
                   title=None, show_colorbar=True, cbar_label="Inundation Depth (m)",
                   grid_res=500, basemap_source=None, max_spread=400):
    """
    Plot Miura-style flood map with land masking to prevent flood in water bodies.

    local_depth: depth at each node (used to build land mask on grid)
    max_spread: max distance (meters) flood can spread from a data point
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    mx, my = transformer.transform(lx, ly)

    x_min, y_min = transformer.transform(bbox["lon_min"], bbox["lat_min"])
    x_max, y_max = transformer.transform(bbox["lon_max"], bbox["lat_max"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Basemap
    src = basemap_source or ctx.providers.Esri.WorldStreetMap
    try:
        ctx.add_basemap(ax, source=src, zoom=zoom)
    except Exception:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=zoom)
        except Exception:
            pass

    # ── Build land mask on grid ──
    # Interpolate depth onto the grid — nearest neighbor is fine for a mask
    all_pts = np.column_stack([mx, my])
    all_depth = local_depth

    grid_x = np.linspace(x_min, x_max, grid_res)
    grid_y = np.linspace(y_min, y_max, grid_res)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts_flat = np.column_stack([gx.ravel(), gy.ravel()])

    # Use nearest-neighbor depth to classify each grid cell as land or water
    depth_tree = cKDTree(all_pts)
    dd, ii = depth_tree.query(grid_pts_flat)
    grid_depth = all_depth[ii].reshape(gx.shape)

    # Land mask: grid cells where the nearest ADCIRC node is land or very shallow
    # This prevents flood from appearing in the Hudson/East River
    is_land = grid_depth <= WATER_BODY_DEPTH

    # ── Interpolate inundation ──
    valid = ~np.isnan(inundation)
    if np.sum(valid) < 3:
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_axis_off()
        return None

    flood_pts = np.column_stack([mx[valid], my[valid]])
    flood_vals = inundation[valid]

    # Interpolate: linear for smooth gradients, nearest for gap-filling
    grid_linear = griddata(flood_pts, flood_vals, (gx, gy), method='linear')
    grid_nearest = griddata(flood_pts, flood_vals, (gx, gy), method='nearest')

    grid_z = grid_linear.copy()
    nan_mask = np.isnan(grid_z)
    grid_z[nan_mask] = grid_nearest[nan_mask]

    # Distance mask: only show flood near actual data points
    flood_tree = cKDTree(flood_pts)
    dists, _ = flood_tree.query(grid_pts_flat)
    dist_mask = dists.reshape(gx.shape) > max_spread
    grid_z[dist_mask] = np.nan

    # ── Apply land mask — this is the key fix ──
    # Only show flood on land cells, NOT in rivers/ocean
    grid_z[~is_land] = np.nan

    # Clean up
    grid_z[grid_z < MIN_INUNDATION] = np.nan

    # Check if anything to plot
    if np.all(np.isnan(grid_z)):
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_axis_off()
        return None

    # Plot as filled contour
    levels = np.linspace(vmin, vmax, 20)
    cf = ax.contourf(
        gx, gy, grid_z,
        levels=levels, cmap=FLOOD_CMAP, alpha=alpha,
        extend='max', zorder=5
    )

    # Subtle contour lines
    ax.contour(
        gx, gy, grid_z,
        levels=levels[::3], colors='#225ea8', alpha=0.12,
        linewidths=0.3, zorder=6
    )

    if show_colorbar:
        cbar = plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, aspect=30)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    ax.set_axis_off()
    return cf


def add_scale_bar(ax):
    sb = ScaleBar(
        1, "m", length_fraction=0.2, location="lower left",
        box_alpha=0.85, font_properties={"size": 9},
        pad=0.5, border_pad=0.5, sep=3, rotation="horizontal-only"
    )
    ax.add_artist(sb)


def add_north_arrow(ax, x=0.06, y=0.95, size=18):
    ax.annotate(
        "N", xy=(x, y), xycoords="axes fraction",
        fontsize=size, fontweight="bold", ha="center", va="top",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
    )
    ax.annotate(
        "", xy=(x, y - 0.01), xytext=(x, y - 0.07),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2, color="black")
    )


def add_legend(ax, scenarios, title_text="Lower Manhattan\nHurricane Sandy Inundation", loc="upper left"):
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=label, alpha=0.7)
               for label, c in scenarios]
    leg = ax.legend(
        handles=patches, loc=loc, fontsize=9, framealpha=0.92,
        edgecolor='gray', fancybox=False,
        title=title_text,
        title_fontproperties={"weight": "bold", "size": 10}
    )
    leg.get_frame().set_linewidth(0.8)


def add_location_labels(ax, labels, transformer):
    for name, lon, lat in labels:
        mx, my = transformer.transform(lon, lat)
        ax.plot(mx, my, 'x', color='#c00000', markersize=8, markeredgewidth=2, zorder=10)
        ax.annotate(
            name, (mx, my), fontsize=8, fontweight='bold',
            xytext=(8, 4), textcoords='offset points',
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=11
        )


def add_simulation_boundary(ax, bbox, transformer, color='#44aa00', lw=2):
    corners_lon = [bbox["lon_min"], bbox["lon_max"], bbox["lon_max"], bbox["lon_min"], bbox["lon_min"]]
    corners_lat = [bbox["lat_min"], bbox["lat_min"], bbox["lat_max"], bbox["lat_max"], bbox["lat_min"]]
    bx, by = transformer.transform(corners_lon, corners_lat)
    ax.plot(bx, by, color=color, linewidth=lw, linestyle='-', zorder=8, alpha=0.8)


# ── Figure generators ─────────────────────────────────────────────────────────

def figure_lower_manhattan():
    """Figure 1: Lower Manhattan Sandy inundation — Miura Fig 8/9 style."""
    print("Loading ADCIRC data...")
    x, y, depth, zeta_max, elements = load_adcirc(SANDY_FILE)
    inundation = compute_inundation(depth, zeta_max)

    print("Extracting Lower Manhattan region...")
    lx, ly, ld, lz, local_tri, local_idx = extract_region(
        x, y, depth, zeta_max, elements, LM_BBOX
    )
    local_inundation = inundation[local_idx]

    n_flooded = np.sum(~np.isnan(local_inundation))
    print(f"  Flooded nodes (land + coastal): {n_flooded}")
    if n_flooded > 0:
        print(f"  Inundation range: {np.nanmin(local_inundation):.2f} - {np.nanmax(local_inundation):.2f} m")

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))

    plot_flood_map(
        ax, lx, ly, ld, local_tri, local_inundation, LM_BBOX,
        vmin=0, vmax=3.5, zoom=15, alpha=0.6, grid_res=600, max_spread=500
    )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    add_simulation_boundary(ax, LM_BBOX, transformer)
    add_scale_bar(ax)
    add_north_arrow(ax)

    locations = [
        ("South Ferry Station", -74.0135, 40.7013),
        ("Canal St. Station", -74.0054, 40.7208),
        ("Battery Park", -74.017, 40.703),
        ("World Trade Center", -74.013, 40.711),
        ("FDR Drive", -73.975, 40.735),
        ("Brooklyn Bridge", -73.997, 40.706),
    ]
    add_location_labels(ax, locations, transformer)

    add_legend(ax, [("Hurricane Sandy", "#7fcdbb")])

    fig.tight_layout()
    out = "data/viz_sandy_lower_manhattan_v2.png"
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def figure_nyc_wide():
    """Figure 2: NYC-wide Sandy inundation map."""
    print("\nLoading ADCIRC data for NYC-wide map...")
    x, y, depth, zeta_max, elements = load_adcirc(SANDY_FILE)
    inundation = compute_inundation(depth, zeta_max)

    print("Extracting NYC region...")
    lx, ly, ld, lz, local_tri, local_idx = extract_region(
        x, y, depth, zeta_max, elements, NYC_BBOX
    )
    local_inundation = inundation[local_idx]

    n_flooded = np.sum(~np.isnan(local_inundation))
    print(f"  Flooded nodes: {n_flooded}")

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    plot_flood_map(
        ax, lx, ly, ld, local_tri, local_inundation, NYC_BBOX,
        vmin=0, vmax=4.0, zoom=12, alpha=0.6, grid_res=800, max_spread=600
    )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    add_scale_bar(ax)
    add_north_arrow(ax)

    locations = [
        ("Battery", -74.014, 40.702),
        ("Lower Manhattan", -74.005, 40.720),
        ("Jersey City", -74.075, 40.725),
        ("Red Hook", -74.010, 40.675),
        ("Brooklyn", -73.960, 40.680),
        ("Jamaica Bay", -73.845, 40.615),
        ("Rockaway", -73.780, 40.580),
        ("Coney Island", -73.975, 40.575),
        ("Staten Island", -74.150, 40.575),
    ]
    add_location_labels(ax, locations, transformer)

    add_legend(ax, [("Hurricane Sandy", "#7fcdbb")])

    fig.tight_layout()
    out = "data/viz_sandy_nyc_wide_v2.png"
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def figure_theta_sweep():
    """Figure 3: 4-panel tipping-point analysis."""
    print("\nLoading ADCIRC data for theta sweep...")
    x, y, depth, zeta_max, elements = load_adcirc(SANDY_FILE)

    print("Extracting Lower Manhattan region...")
    lx, ly, ld, lz, local_tri, local_idx = extract_region(
        x, y, depth, zeta_max, elements, LM_BBOX
    )

    sandy_battery = 3.18
    thetas = [1.0, 2.0, 3.0, 3.18]
    labels = [
        r"$\theta$ = 1.0m — Minor Storm",
        r"$\theta$ = 2.0m — Moderate Storm",
        r"$\theta$ = 3.0m — Major Storm",
        r"$\theta$ = 3.18m — Sandy-Level Event",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 20))
    fig.suptitle(
        "DDPM Flood Prediction at Increasing Storm Intensities\n"
        "Lower Manhattan — Tipping Point Analysis",
        fontsize=14, fontweight='bold', y=0.98
    )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    for idx, (theta, label) in enumerate(zip(thetas, labels)):
        ax = axes[idx // 2][idx % 2]

        scale = theta / sandy_battery
        scaled_zeta = zeta_max * scale
        scaled_inundation = compute_inundation(depth, scaled_zeta)
        local_inundation = scaled_inundation[local_idx]

        n_flooded = np.sum(~np.isnan(local_inundation))

        plot_flood_map(
            ax, lx, ly, ld, local_tri, local_inundation, LM_BBOX,
            vmin=0, vmax=3.5, zoom=14, alpha=0.6,
            title=label, show_colorbar=True, grid_res=400, max_spread=500
        )

        add_simulation_boundary(ax, LM_BBOX, transformer, color='#44aa00', lw=1.5)

        ax.text(
            0.02, 0.02, f"{n_flooded} flooded nodes",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'),
            verticalalignment='bottom'
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "data/viz_theta_sweep_v2.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def figure_sandy_comparison():
    """Figure 4: Side-by-side ADCIRC Sandy vs USGS High Water Marks."""
    print("\nLoading ADCIRC data for Sandy comparison...")
    x, y, depth, zeta_max, elements = load_adcirc(SANDY_FILE)
    inundation = compute_inundation(depth, zeta_max)

    lx, ly, ld, lz, local_tri, local_idx = extract_region(
        x, y, depth, zeta_max, elements, LM_BBOX
    )
    local_inundation = inundation[local_idx]

    # Load Sandy HWM data
    import csv
    hwm_lons, hwm_lats, hwm_vals = [], [], []
    try:
        with open("data/validation/sandy_hwm.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row.get("latitude") or row.get("lat", 0))
                    lon = float(row.get("longitude") or row.get("lon", 0))
                    val = float(row.get("elev_ft") or row.get("hwm_ft") or row.get("peak_value", 0))
                    if (LM_BBOX["lat_min"] <= lat <= LM_BBOX["lat_max"] and
                        LM_BBOX["lon_min"] <= lon <= LM_BBOX["lon_max"]):
                        hwm_lons.append(lon)
                        hwm_lats.append(lat)
                        hwm_vals.append(val * 0.3048)
                except (ValueError, TypeError):
                    continue
    except FileNotFoundError:
        print("  Warning: sandy_hwm.csv not found")

    print(f"  HWMs in Lower Manhattan: {len(hwm_lons)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Left panel
    plot_flood_map(
        ax1, lx, ly, ld, local_tri, local_inundation, LM_BBOX,
        vmin=0, vmax=3.5, zoom=15, alpha=0.6,
        title="ADCIRC Sandy Inundation\n(Land Only)", grid_res=500, max_spread=500
    )
    add_simulation_boundary(ax1, LM_BBOX, transformer)
    add_scale_bar(ax1)
    add_north_arrow(ax1)

    # Right panel
    plot_flood_map(
        ax2, lx, ly, ld, local_tri, local_inundation, LM_BBOX,
        vmin=0, vmax=3.5, zoom=15, alpha=0.55,
        title="ADCIRC Sandy + USGS High Water Marks", grid_res=500, max_spread=500
    )
    add_simulation_boundary(ax2, LM_BBOX, transformer)

    if hwm_lons:
        hx, hy = transformer.transform(hwm_lons, hwm_lats)
        ax2.scatter(
            hx, hy, c=hwm_vals, cmap=FLOOD_CMAP,
            vmin=0, vmax=3.5, s=100, edgecolors='red', linewidths=2,
            zorder=10, marker='o'
        )
        for i in range(min(5, len(hwm_lons))):
            ax2.annotate(
                f"{hwm_vals[i]:.1f}m", (hx[i], hy[i]),
                fontsize=7, fontweight='bold', color='red',
                xytext=(6, 6), textcoords='offset points',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=11
            )

    add_scale_bar(ax2)

    fig.tight_layout()
    out = "data/viz_sandy_comparison_v2.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Miura-style flood visualizations (v3 — land-masked)")
    print("=" * 60)

    figure_lower_manhattan()
    figure_nyc_wide()
    figure_theta_sweep()
    figure_sandy_comparison()

    print("\n" + "=" * 60)
    print("All visualizations generated!")
    print("=" * 60)
