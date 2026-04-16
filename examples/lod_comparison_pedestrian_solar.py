"""Compare pedestrian-level cumulative solar irradiance: LoD2 vs LoD1.

The comparison is performed on **walk-network edges** — the same edges are
evaluated under both LoD2 and LoD1 voxelisations, enabling paired error
analysis.

Pipeline (per LoD):
  1. Load / build VoxCity at the given LoD.
  2. Compute cumulative ground-level global solar irradiance for August.
  3. Mask building-footprint cells to NaN.
  4. Aggregate irradiance onto the OSM walk network.

Then:
  5. Pair edges by index and compute error metrics (LoD1 vs LoD2 reference).
  6. Generate comparison figures:
       (a) LoD2 walk-network irradiance map on basemap
       (b) LoD1 walk-network irradiance map on basemap
       (c) Difference (LoD2 − LoD1) network map on basemap
       (d) Scatter plot (LoD1 vs LoD2 edges) with 1:1 line and regression
       (e) Histogram + KDE of edge-level differences
       (f) Summary error-metrics table
       + standalone: relative-difference network map, distribution overlay,
         empirical CDF, Q-Q plot
"""

import os
os.environ["TI_LOG_LEVEL"] = "warn"

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import contextily as ctx
from scipy import stats as sp_stats
from scipy.stats import gaussian_kde
from dclab.kde.methods import kde_gauss as dclab_kde_gauss
from shapely.geometry import Polygon

import sys
sys.path.insert(0, os.path.dirname(__file__))
from run_pedestrian_solar import (
    load_or_build_city,
    compute_august_solar,
    aggregate_on_walk_network,
)


# ======================================================================
# Configuration
# ======================================================================

CITY_LABEL = "Tokyo"
CITYGML_PATH = [
    "/path/to/citygml_dataset_1",  # Replace with your CityGML path(s)
    "/path/to/citygml_dataset_2",
]
CENTER_LON = 139.767125
CENTER_LAT = 35.681236
TARGET_SIZE = 2000
BUFFER_METERS = 200
MESHSIZE = 2.0
GEE_PROJECT = "your-gee-project-id"  # Replace with your GEE project

OUTPUT_DIR_LOD2 = "output/pedestrian_solar_lod2"
OUTPUT_DIR_LOD1 = "output/pedestrian_solar_lod1"
OUTPUT_DIR_COMPARE = "output/pedestrian_solar_lod_compare"

VALUE_COL = "solar_irradiance"

LOD_COLORS = {
    "LoD2": {"color": "#3e356b", "color2": "#2d2750"},
    "LoD1": {"color": "#c45e3e", "color2": "#a84e30"},
}


# ======================================================================
# Per-LoD pipeline
# ======================================================================

def run_single_lod(lod, output_dir, epw_file_path=None):
    """Run the pedestrian solar pipeline for a single LoD.

    Returns
    -------
    city : VoxCity
    solar_grid : np.ndarray  (R, C)
    edge_gdf_core : gpd.GeoDataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, "voxcity.pkl")
    grid_npy = os.path.join(output_dir, "solar_grid_august.npy")
    gpkg_path = os.path.join(output_dir, "pedestrian_solar_network.gpkg")

    # --- 1. Load / build city -------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  LoD{lod} — Loading / building VoxCity")
    print(f"{'=' * 60}")

    city = load_or_build_city(
        pkl_path=pkl_path,
        city_label=f"{CITY_LABEL}_lod{lod}",
        citygml_path=CITYGML_PATH,
        center_lon=CENTER_LON,
        center_lat=CENTER_LAT,
        target_size=TARGET_SIZE,
        buffer_meters=BUFFER_METERS,
        meshsize=MESHSIZE,
        output_dir=output_dir,
        building_lod=lod,
        gee_project=GEE_PROJECT,
    )

    # --- 2. Compute / load solar grid -----------------------------------
    if os.path.isfile(grid_npy):
        print(f"  Loading cached solar grid from {grid_npy}")
        solar_grid = np.load(grid_npy)
    else:
        epw = epw_file_path
        solar_extra = {}
        if epw is None:
            for d in [output_dir, OUTPUT_DIR_LOD2, OUTPUT_DIR_LOD1, "output"]:
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".epw"):
                            epw = os.path.join(d, f)
                            break
                if epw:
                    break
        if epw is None:
            solar_extra["download_nearest_epw"] = True
            solar_extra["output_dir"] = output_dir
        else:
            print(f"  Using EPW: {epw}")

        try:
            from voxcity.simulator_gpu.visibility.integration import (
                reset_visibility_taichi_cache,
            )
            reset_visibility_taichi_cache()
        except ImportError:
            pass

        solar_grid = compute_august_solar(
            city, epw_file_path=epw, output_dir=output_dir, **solar_extra,
        )
        np.save(grid_npy, solar_grid)
        print(f"  Saved solar grid to {grid_npy}")

    # --- 3. Network aggregation -----------------------------------------
    if os.path.isfile(gpkg_path):
        print(f"  Loading cached edge GeoPackage from {gpkg_path}")
        edge_gdf_core = gpd.read_file(gpkg_path)
    else:
        from voxcitygml.citygml.coordinates import create_rectangle
        core_rect = create_rectangle(CENTER_LON, CENTER_LAT, TARGET_SIZE)
        _, _, edge_gdf_core = aggregate_on_walk_network(
            solar_grid, city, core_rect=core_rect,
            output_dir=output_dir, value_name=VALUE_COL, network_type="walk",
        )

    return city, solar_grid, edge_gdf_core


# ======================================================================
# Edge-based error statistics (core analysis)
# ======================================================================

def compute_edge_error(edge_lod2, edge_lod1):
    """Compute paired edge-level error metrics (LoD2 as reference).

    Parameters
    ----------
    edge_lod2, edge_lod1 : gpd.GeoDataFrame
        Edge GeoDataFrames with ``VALUE_COL`` column.

    Returns
    -------
    stats : dict   – error metrics suitable for CSV / summary.
    paired_df : pd.DataFrame  – per-edge paired values & differences.
    """
    # --- Pair edges by index (same OSM network → identical topology) ---
    common_idx = edge_lod2.index.intersection(edge_lod1.index)
    v2 = edge_lod2.loc[common_idx, VALUE_COL].values.astype(np.float64)
    v1 = edge_lod1.loc[common_idx, VALUE_COL].values.astype(np.float64)

    valid = np.isfinite(v2) & np.isfinite(v1) & (v2 > 0) & (v1 > 0)
    v2 = v2[valid]
    v1 = v1[valid]
    valid_idx = common_idx[valid]

    d = v2 - v1          # positive → LoD2 receives more sun
    abs_d = np.abs(d)
    rel_d = d / v2 * 100  # relative to LoD2 (reference)

    # --- Compute edge lengths for length-weighted stats ----------------
    edge_proj = edge_lod2.loc[valid_idx].to_crs(epsg=3857)
    lengths = edge_proj.geometry.length.values
    w = lengths / lengths.sum()

    wmean_v2 = np.average(v2, weights=lengths)
    wmean_v1 = np.average(v1, weights=lengths)
    wmean_d = np.average(d, weights=lengths)
    wmae = np.average(abs_d, weights=lengths)
    wrmse = np.sqrt(np.average(d ** 2, weights=lengths))
    wnrmse = wrmse / wmean_v2 * 100
    wmean_abs_rel = np.average(np.abs(rel_d), weights=lengths)

    # Unweighted
    mae = abs_d.mean()
    rmse = np.sqrt((d ** 2).mean())
    nrmse = rmse / v2.mean() * 100

    r_pearson, p_pearson = sp_stats.pearsonr(v2, v1)
    r_spearman, p_spearman = sp_stats.spearmanr(v2, v1)
    slope, intercept, _, _, _ = sp_stats.linregress(v2, v1)
    ss_res = np.sum((v1 - v2) ** 2)
    ss_tot = np.sum((v2 - v2.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ks_stat, ks_p = sp_stats.ks_2samp(v2, v1)
    mw_stat, mw_p = sp_stats.mannwhitneyu(v2, v1, alternative="two-sided")

    # --- Print to console -----------------------------------------------
    print("\n" + "=" * 70)
    print("  Walk-Network Edge Irradiance — LoD Comparison")
    print("=" * 70)

    for tag, v in [("LoD2 (reference)", v2), ("LoD1", v1)]:
        print(f"\n  {tag} (Wh/m²) — {len(v):,} edges:")
        print(f"    Mean   = {v.mean():>12,.1f}")
        print(f"    Median = {np.median(v):>12,.1f}")
        print(f"    Std    = {v.std():>12,.1f}")
        print(f"    Min    = {v.min():>12,.1f}")
        print(f"    Max    = {v.max():>12,.1f}")

    print(f"\n  {'─' * 66}")
    print(f"  Paired difference (LoD2 − LoD1)  [{len(d):,} edges]")
    print(f"  {'─' * 66}")
    print(f"    Mean Diff          = {d.mean():>+12,.1f} Wh/m²")
    print(f"    Median Diff        = {np.median(d):>+12,.1f} Wh/m²")
    print(f"    Std Dev of Diff    = {d.std():>12,.1f} Wh/m²")

    print(f"\n  Error metrics (unweighted):")
    print(f"    MAE                = {mae:>12,.1f} Wh/m²")
    print(f"    RMSE               = {rmse:>12,.1f} Wh/m²")
    print(f"    NRMSE (% of mean)  = {nrmse:>11.2f} %")
    print(f"    Mean Rel Diff      = {rel_d.mean():>+11.2f} %")
    print(f"    Mean Abs Rel Diff  = {np.abs(rel_d).mean():>11.2f} %")

    print(f"\n  Error metrics (length-weighted):")
    print(f"    Wt. MAE            = {wmae:>12,.1f} Wh/m²")
    print(f"    Wt. RMSE           = {wrmse:>12,.1f} Wh/m²")
    print(f"    Wt. NRMSE (%)      = {wnrmse:>11.2f} %")
    print(f"    Wt. Mean Diff      = {wmean_d:>+12,.1f} Wh/m²")
    print(f"    Wt. Mean Abs Rel   = {wmean_abs_rel:>11.2f} %")

    print(f"\n    Min Diff           = {d.min():>+12,.1f} Wh/m²")
    print(f"    Max Diff           = {d.max():>+12,.1f} Wh/m²")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        pval = np.percentile(d, pct)
        print(f"    P{pct:<2d}               = {pval:>+12,.1f} Wh/m²")

    n_pos = np.count_nonzero(d > 0)
    n_neg = np.count_nonzero(d < 0)
    n_zero = np.count_nonzero(d == 0)
    n_total = len(d)
    print(f"\n    LoD2 > LoD1        = {n_pos:>8,} edges ({n_pos / n_total * 100:.1f}%)")
    print(f"    LoD2 < LoD1        = {n_neg:>8,} edges ({n_neg / n_total * 100:.1f}%)")
    print(f"    LoD2 = LoD1        = {n_zero:>8,} edges ({n_zero / n_total * 100:.1f}%)")

    print(f"\n    Pearson  r         = {r_pearson:.6f}  (p={p_pearson:.2e})")
    print(f"    R²                 = {r_squared:.6f}")
    print(f"    Spearman ρ         = {r_spearman:.6f}  (p={p_spearman:.2e})")
    print(f"    Lin. Regression    : LoD1 = {slope:.4f} × LoD2 + {intercept:,.1f}")
    print(f"    KS statistic       = {ks_stat:.6f}  (p={ks_p:.2e})")
    print(f"    Mann-Whitney U     = {mw_stat:,.0f}  (p={mw_p:.2e})")
    print("=" * 70)

    # --- Build paired DataFrame ----------------------------------------
    paired_df = pd.DataFrame({
        "lod2": v2,
        "lod1": v1,
        "diff": d,
        "abs_diff": abs_d,
        "rel_diff_pct": rel_d,
        "length_m": lengths,
    }, index=valid_idx)

    stats = {
        "n_edges": len(d),
        "total_length_km": lengths.sum() / 1000,
        "mean_lod2": v2.mean(),
        "mean_lod1": v1.mean(),
        "mean_diff": d.mean(),
        "median_diff": float(np.median(d)),
        "std_diff": d.std(),
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "mean_rel_diff_pct": rel_d.mean(),
        "mean_abs_rel_diff_pct": np.abs(rel_d).mean(),
        "wt_mae": wmae,
        "wt_rmse": wrmse,
        "wt_nrmse": wnrmse,
        "wt_mean_diff": wmean_d,
        "wt_mean_abs_rel_pct": wmean_abs_rel,
        "r_pearson": r_pearson,
        "r_squared": r_squared,
        "r_spearman": r_spearman,
        "slope": slope,
        "intercept": intercept,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "pct_lod2_gt_lod1": n_pos / n_total * 100,
    }
    return stats, paired_df


# ======================================================================
# Visualisation
# ======================================================================

def make_comparison_figures(edge_lod2, edge_lod1, paired_df, stats, output_dir):
    """Generate publication-quality comparison figures (edge-based).

    Main figure panels:
      (a) LoD2 walk-network irradiance map
      (b) LoD1 walk-network irradiance map
      (c) Difference (LoD2 − LoD1) network map
      (d) Scatter: LoD1 vs LoD2 edges, 1:1 line + regression
      (e) Histogram + KDE of edge-level differences
      (f) Summary error-metrics table
    """
    os.makedirs(output_dir, exist_ok=True)

    v2 = paired_df["lod2"].values
    v1 = paired_df["lod1"].values
    d = paired_df["diff"].values
    rel_d = paired_df["rel_diff_pct"].values
    lengths = paired_df["length_m"].values

    r_pearson = stats["r_pearson"]
    slope = stats["slope"]
    intercept = stats["intercept"]

    # ==================================================================
    # Figure 1: 6-panel main comparison (edge-based)
    # ==================================================================
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30)

    # Shared colour scale for (a) and (b)
    all_vals = np.concatenate([
        edge_lod2[VALUE_COL].dropna().values,
        edge_lod1[VALUE_COL].dropna().values,
    ])
    vmax_e = float(np.nanpercentile(all_vals, 95))

    # --- (a) LoD2 network map -------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    e2_web = edge_lod2.to_crs(epsg=3857)
    e2_web.plot(
        column=VALUE_COL, ax=ax_a, cmap="magma", legend=True,
        vmin=0.0, vmax=vmax_e, linewidth=1.0,
        legend_kwds={"label": "Wh/m²", "shrink": 0.6},
    )
    ctx.add_basemap(ax_a, source=ctx.providers.CartoDB.Positron)
    ax_a.set_axis_off()
    ax_a.set_title("(a) LoD2 — walk-network solar irradiance")

    # --- (b) LoD1 network map -------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    e1_web = edge_lod1.to_crs(epsg=3857)
    e1_web.plot(
        column=VALUE_COL, ax=ax_b, cmap="magma", legend=True,
        vmin=0.0, vmax=vmax_e, linewidth=1.0,
        legend_kwds={"label": "Wh/m²", "shrink": 0.6},
    )
    ctx.add_basemap(ax_b, source=ctx.providers.CartoDB.Positron)
    ax_b.set_axis_off()
    ax_b.set_title("(b) LoD1 — walk-network solar irradiance")

    # --- (c) Difference network map -------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    common_idx = paired_df.index
    edge_diff = edge_lod2.loc[common_idx].copy()
    edge_diff[VALUE_COL] = d
    ed_web = edge_diff.to_crs(epsg=3857)
    vabs_e = float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 95))
    ed_web.plot(
        column=VALUE_COL, ax=ax_c, cmap="RdBu_r", legend=True,
        vmin=-vabs_e, vmax=vabs_e, linewidth=1.0,
        legend_kwds={"label": "ΔWh/m²", "shrink": 0.6},
    )
    ctx.add_basemap(ax_c, source=ctx.providers.CartoDB.DarkMatter)
    ax_c.set_axis_off()
    ax_c.set_title("(c) Difference (LoD2 − LoD1) — walk network")

    # --- (d) Scatter: LoD1 vs LoD2 + regression -------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    n_pts = len(v2)
    if n_pts > 5_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pts, size=5_000, replace=False)
        v2_s, v1_s = v2[idx], v1[idx]
    else:
        v2_s, v1_s = v2, v1
    ax_d.scatter(v2_s, v1_s, s=4, alpha=0.3, rasterized=True, color="#3e356b")
    lim = [min(v2.min(), v1.min()), max(v2.max(), v1.max())]
    ax_d.plot(lim, lim, "k--", lw=1, label="1:1 line")
    ax_d.set_xlabel("LoD2 edge irradiance (Wh/m²)")
    ax_d.set_ylabel("LoD1 edge irradiance (Wh/m²)")
    ax_d.set_title("(d) LoD1 vs LoD2 — edge scatter")
    ax_d.legend(fontsize=9)
    ax_d.set_aspect("equal", adjustable="datalim")
    ax_d.grid(True, alpha=0.3)

    # --- (e) Histogram + KDE of edge differences -----------------------
    ax_e = fig.add_subplot(gs[2, 0])
    bins_d = np.linspace(d.min(), d.max(), 60)
    ax_e.hist(d, bins=bins_d, color="steelblue", edgecolor="none",
              alpha=0.7, density=True, label="Histogram")
    if len(d) > 3 and d.std() > 0:
        kde = gaussian_kde(d, bw_method=0.3)
        x_k = np.linspace(d.min(), d.max(), 300)
        ax_e.plot(x_k, kde(x_k), color="#1a3a5c", lw=2, label="KDE")
    ax_e.axvline(0, color="black", ls="--", lw=0.8)
    ax_e.axvline(d.mean(), color="red", ls="-", lw=1.5,
                 label=f"Mean = {d.mean():+,.0f}")
    ax_e.axvline(float(np.median(d)), color="blue", ls=":", lw=1.5,
                 label=f"Median = {np.median(d):+,.0f}")
    ax_e.set_xlabel("Edge difference LoD2 − LoD1 (Wh/m²)")
    ax_e.set_ylabel("Density")
    ax_e.set_title("(e) Distribution of edge-level differences")
    ax_e.legend(fontsize=8)

    # --- (f) Summary error-metrics table --------------------------------
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis("off")
    iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
    stats_text = (
        f"{'Metric':.<32s} {'Value':>14s}\n"
        f"{'─' * 48}\n"
        f"{'Paired edges':.<32s} {len(d):>14,}\n"
        f"{'Total length (km)':.<32s} {lengths.sum() / 1000:>14.1f}\n"
        f"{'─' * 48}\n"
        f"{'Mean LoD2 (Wh/m²)':.<32s} {v2.mean():>14,.1f}\n"
        f"{'Mean LoD1 (Wh/m²)':.<32s} {v1.mean():>14,.1f}\n"
        f"{'─' * 48}\n"
        f"{'Mean Diff (Wh/m²)':.<32s} {d.mean():>+14,.1f}\n"
        f"{'Median Diff':.<32s} {np.median(d):>+14,.1f}\n"
        f"{'Std Diff':.<32s} {d.std():>14,.1f}\n"
        f"{'IQR':.<32s} {iqr:>14,.1f}\n"
        f"{'─' * 48}\n"
        f"{'MAE (Wh/m²)':.<32s} {stats['mae']:>14,.1f}\n"
        f"{'RMSE (Wh/m²)':.<32s} {stats['rmse']:>14,.1f}\n"
        f"{'NRMSE (%)':.<32s} {stats['nrmse']:>14.2f}\n"
        f"{'Mean Abs Rel Diff (%)':.<32s} {stats['mean_abs_rel_diff_pct']:>14.2f}\n"
        f"{'─' * 48}\n"
        f"{'Wt. MAE (Wh/m²)':.<32s} {stats['wt_mae']:>14,.1f}\n"
        f"{'Wt. RMSE (Wh/m²)':.<32s} {stats['wt_rmse']:>14,.1f}\n"
        f"{'Wt. NRMSE (%)':.<32s} {stats['wt_nrmse']:>14.2f}\n"
        f"{'─' * 48}\n"
        f"{'Pearson r':.<32s} {r_pearson:>14.6f}\n"
        f"{'Regression slope':.<32s} {slope:>14.4f}\n"
        f"{'Regression intercept':.<32s} {intercept:>14,.1f}\n"
        f"{'─' * 48}\n"
        f"{'LoD2 > LoD1 (%)':.<32s} {stats['pct_lod2_gt_lod1']:>14.1f}\n"
        f"{'LoD2 < LoD1 (%)':.<32s} {100 - stats['pct_lod2_gt_lod1']:>14.1f}\n"
    )
    ax_f.text(
        0.05, 0.95, stats_text, transform=ax_f.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF5EB",
                  edgecolor="#E8963E", alpha=0.9),
    )
    ax_f.set_title("(f) Edge-level error metrics — LoD1 vs LoD2 (ref)")

    fig.suptitle(
        f"Pedestrian Solar Irradiance — LoD Comparison on Walk Network (August)\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m target, {BUFFER_METERS}m buffer, "
        f"{MESHSIZE}m voxel",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig_path = os.path.join(output_dir, "lod_comparison_main.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved main comparison figure to {fig_path}")
    plt.close(fig)

    # ==================================================================
    # Figure 2: Additional edge-level panels
    # ==================================================================
    e2_vals = edge_lod2[VALUE_COL].dropna().values
    e1_vals = edge_lod1[VALUE_COL].dropna().values

    fig2 = plt.figure(figsize=(22, 12))
    gs2 = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

    # --- (a) Relative difference network map ---------------------------
    ax2a = fig2.add_subplot(gs2[0, 0])
    edge_rel = edge_lod2.loc[common_idx].copy()
    edge_rel["rel_diff_pct"] = rel_d
    er_web = edge_rel.to_crs(epsg=3857)
    vabs_r = float(np.nanpercentile(np.abs(rel_d[np.isfinite(rel_d)]), 95))
    er_web.plot(
        column="rel_diff_pct", ax=ax2a, cmap="RdBu_r", legend=True,
        vmin=-vabs_r, vmax=vabs_r, linewidth=1.0,
        legend_kwds={"label": "Rel. diff (%)", "shrink": 0.6},
    )
    ctx.add_basemap(ax2a, source=ctx.providers.CartoDB.Positron)
    ax2a.set_axis_off()
    ax2a.set_title("(a) Relative difference (LoD2−LoD1)/LoD2 (%)")

    # --- (b) Edge irradiance distribution (KDE overlay) ----------------
    ax2b = fig2.add_subplot(gs2[0, 1])
    x_max = max(e2_vals.max(), e1_vals.max()) * 1.05
    x_kde = np.linspace(0, x_max, 300)
    bins_e = np.linspace(0, x_max, 40)
    for lod_tag, vals, c_main, c_line in [
        ("LoD2", e2_vals, LOD_COLORS["LoD2"]["color"], LOD_COLORS["LoD2"]["color2"]),
        ("LoD1", e1_vals, LOD_COLORS["LoD1"]["color"], LOD_COLORS["LoD1"]["color2"]),
    ]:
        ax2b.hist(vals, bins=bins_e, color=c_main, alpha=0.35,
                  density=True, label=f"{lod_tag} hist")
        if len(vals) > 3 and vals.std() > 0:
            kde_e = gaussian_kde(vals, bw_method=0.3)
            ax2b.plot(x_kde, kde_e(x_kde), color=c_line, lw=2,
                      label=f"{lod_tag} KDE")
    ax2b.set_xlabel("Edge irradiance (Wh/m²)")
    ax2b.set_ylabel("Density")
    ax2b.set_title("(b) Edge irradiance distribution")
    ax2b.legend(fontsize=9)

    # --- (c) Empirical CDF comparison -----------------------------------
    ax2c = fig2.add_subplot(gs2[0, 2])
    for lod_tag, vals, c_line in [
        ("LoD2", e2_vals, LOD_COLORS["LoD2"]["color2"]),
        ("LoD1", e1_vals, LOD_COLORS["LoD1"]["color2"]),
    ]:
        s_vals = np.sort(vals)
        cdf = np.arange(1, len(s_vals) + 1) / len(s_vals)
        ax2c.plot(s_vals, cdf, color=c_line, lw=2, label=lod_tag)
    ax2c.set_xlabel("Cumulative irradiance (Wh/m²)")
    ax2c.set_ylabel("Cumulative probability")
    ax2c.set_title("(c) Empirical CDF — edge irradiance")
    ax2c.legend(fontsize=10)
    ax2c.grid(True, alpha=0.3)

    # --- (d) Length-weighted histogram of differences -------------------
    ax2d = fig2.add_subplot(gs2[1, 0])
    bins_w = np.linspace(d.min(), d.max(), 50)
    ax2d.hist(d, bins=bins_w, weights=lengths / lengths.sum(),
              color="#E8963E", edgecolor="white", alpha=0.85,
              density=True, label="Length-weighted hist")
    if len(d) > 3 and d.std() > 0:
        kde_w = gaussian_kde(d, bw_method=0.3, weights=lengths)
        xw = np.linspace(d.min(), d.max(), 300)
        ax2d.plot(xw, kde_w(xw), color="#8B3E00", lw=2, label="Wt. KDE")
    ax2d.axvline(0, color="black", ls="--", lw=0.8)
    wmean_d = np.average(d, weights=lengths)
    ax2d.axvline(wmean_d, color="red", ls="-", lw=1.5,
                 label=f"Wt. Mean = {wmean_d:+,.0f}")
    ax2d.set_xlabel("Edge difference LoD2 − LoD1 (Wh/m²)")
    ax2d.set_ylabel("Density (length-weighted)")
    ax2d.set_title("(d) Length-weighted distribution of differences")
    ax2d.legend(fontsize=8)

    # --- (e) Box-and-violin plot (LoD2, LoD1) --------------------------
    ax2e = fig2.add_subplot(gs2[1, 1])
    data_box = [v2, v1]
    parts = ax2e.violinplot(data_box, positions=[0, 1],
                            showmeans=False, showmedians=False,
                            showextrema=False)
    lod_colors_list = [LOD_COLORS["LoD2"]["color"], LOD_COLORS["LoD1"]["color"]]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(lod_colors_list[i])
        pc.set_alpha(0.4)
    bp = ax2e.boxplot(data_box, positions=[0, 1], widths=0.25,
                      patch_artist=True, showfliers=False, zorder=3)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(lod_colors_list[i])
        box.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)
    ax2e.set_xticks([0, 1])
    ax2e.set_xticklabels(["LoD2", "LoD1"])
    ax2e.set_ylabel("Edge irradiance (Wh/m²)")
    ax2e.set_title("(e) Paired edge irradiance — violin + box")

    # --- (f) Q-Q plot (paired percentiles) -----------------------------
    ax2f = fig2.add_subplot(gs2[1, 2])
    quantiles = np.linspace(0, 100, 101)
    q2 = np.percentile(v2, quantiles)
    q1 = np.percentile(v1, quantiles)
    lo = min(q2.min(), q1.min()) * 0.95
    hi = max(q2.max(), q1.max()) * 1.05
    ax2f.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    sc = ax2f.scatter(q2, q1, c=quantiles, cmap="coolwarm", s=30, zorder=3,
                      edgecolors="grey", linewidths=0.5)
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2f, shrink=0.6)
    cbar.set_label("Percentile")
    ax2f.set_xlabel("LoD2 irradiance (Wh/m²)")
    ax2f.set_ylabel("LoD1 irradiance (Wh/m²)")
    ax2f.set_title("(f) Q-Q plot — edge percentiles")
    ax2f.set_aspect("equal", adjustable="datalim")
    ax2f.legend(fontsize=9, loc="upper left")
    ax2f.grid(True, alpha=0.3)

    fig2.suptitle(
        f"Walk-Network Irradiance — LoD Comparison Details (August)\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m, {MESHSIZE}m voxel",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig2_path = os.path.join(output_dir, "lod_comparison_details.png")
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    print(f"  Saved detail comparison figure to {fig2_path}")
    plt.close(fig2)

    # ==================================================================
    # Figure 3: Standalone — relative difference network map
    # ==================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    er_web3 = edge_rel.to_crs(epsg=3857)
    er_web3.plot(
        column="rel_diff_pct", ax=ax3, cmap="RdBu_r", legend=True,
        vmin=-vabs_r, vmax=vabs_r, linewidth=1.0,
        legend_kwds={"label": "Relative difference (%)", "shrink": 0.6},
    )
    ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron)
    ax3.set_axis_off()
    ax3.set_title(
        f"Relative difference (LoD2 − LoD1) / LoD2 (%) — Walk Network\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m, {MESHSIZE}m voxel",
        fontsize=12, fontweight="bold",
    )
    fig3_path = os.path.join(output_dir, "lod_relative_difference_network.png")
    fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
    print(f"  Saved relative difference map to {fig3_path}")
    plt.close(fig3)

    # ==================================================================
    # Figure 4: Standalone — signed difference network map (Wh/m²)
    # ==================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    common_idx = paired_df.index
    edge_diff4 = edge_lod2.loc[common_idx].copy()
    edge_diff4[VALUE_COL] = d
    ed4_web = edge_diff4.to_crs(epsg=3857)
    vabs4 = float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 95))
    ed4_web.plot(
        column=VALUE_COL, ax=ax4, cmap="RdBu_r", legend=True,
        vmin=-vabs4, vmax=vabs4, linewidth=1.0,
        legend_kwds={"label": "LoD2 − LoD1 (Wh/m²)", "shrink": 0.6},
    )
    ctx.add_basemap(ax4, source=ctx.providers.CartoDB.DarkMatter)
    ax4.set_axis_off()
    ax4.set_title(
        f"Difference (LoD2 − LoD1) — Walk Network\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m, {MESHSIZE}m voxel  |  "
        f"MAE = {stats['mae']:,.1f} Wh/m²",
        fontsize=12, fontweight="bold",
    )
    fig4_path = os.path.join(output_dir, "lod_difference_network.png")
    fig4.savefig(fig4_path, dpi=200, bbox_inches="tight")
    print(f"  Saved difference map to {fig4_path}")
    plt.close(fig4)

    # ==================================================================
    # Figure 5: Standalone — KDE scatter plot (LoD1 vs LoD2)
    # ==================================================================
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    v2_kw = v2 / 1000.0
    v1_kw = v1 / 1000.0
    density_5 = dclab_kde_gauss(v2_kw, v1_kw)
    order_5 = np.argsort(density_5)
    sc5 = ax5.scatter(
        v2_kw[order_5], v1_kw[order_5], c=density_5[order_5],
        s=6, cmap="mako", rasterized=True,
    )
    fig5.colorbar(sc5, ax=ax5, shrink=0.6, label="KDE density")
    ax5.plot([0, 175], [0, 175], "k--", lw=1, label="1:1 line")
    ax5.set_xlim(0, 175)
    ax5.set_ylim(0, 175)
    txt5 = (
        f"RMSE = {stats['rmse']/1000.0:,.1f} kWh/m²\n"
        f"MAE  = {stats['mae']/1000.0:,.1f} kWh/m²\n"
        f"R²   = {stats['r_squared']:.4f}"
    )
    ax5.text(
        0.05, 0.95, txt5, transform=ax5.transAxes, fontsize=12,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85),
    )
    ax5.set_xlabel("LoD2 edge irradiance (kWh/m²)")
    ax5.set_ylabel("LoD1 edge irradiance (kWh/m²)")
    ax5.legend(fontsize=9, loc="lower right")
    ax5.set_aspect("equal")
    ax5.grid(True, alpha=0.2)
    fig5_path = os.path.join(output_dir, "lod_kde_scatter.png")
    fig5.savefig(fig5_path, dpi=200, bbox_inches="tight")
    print(f"  Saved KDE scatter to {fig5_path}")
    plt.close(fig5)


# ======================================================================
# Comprehensive error evaluation by LoD1 shape simplification
# ======================================================================

def make_error_evaluation_figure(edge_lod2, paired_df, stats, output_dir):
    """Generate a comprehensive error-evaluation figure for LoD1 simplification.

    Panels:
      (a) Bland-Altman plot — mean of (LoD2,LoD1) vs difference
      (b) Absolute error vs LoD2 irradiance — binned box plot showing how
          LoD1 error varies with irradiance level (shaded → sunlit)
      (c) Absolute error network map — spatial distribution of |LoD2−LoD1|
      (d) Relative error histogram + KDE — distribution of (LoD2−LoD1)/LoD2 %
      (e) Error exceedance curve — fraction of edges with |err| > threshold
      (f) Cumulative error contribution — sorted |err| shows which edges
          account for the bulk of total absolute error
    """
    os.makedirs(output_dir, exist_ok=True)

    v2 = paired_df["lod2"].values
    v1 = paired_df["lod1"].values
    d = paired_df["diff"].values
    abs_d = paired_df["abs_diff"].values
    rel_d = paired_df["rel_diff_pct"].values
    lengths = paired_df["length_m"].values
    common_idx = paired_df.index

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.30)

    # ------------------------------------------------------------------
    # (a) Bland-Altman plot
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    mean_pair = (v2 + v1) / 2.0
    bias = d.mean()
    loa_upper = bias + 1.96 * d.std()
    loa_lower = bias - 1.96 * d.std()

    n_pts = len(v2)
    if n_pts > 5_000:
        rng = np.random.default_rng(42)
        idx_s = rng.choice(n_pts, size=5_000, replace=False)
        mp_s, d_s = mean_pair[idx_s], d[idx_s]
    else:
        mp_s, d_s = mean_pair, d

    ax_a.scatter(mp_s, d_s, s=4, alpha=0.25, rasterized=True, color="#3e356b")
    ax_a.axhline(bias, color="red", ls="-", lw=1.5,
                 label=f"Bias = {bias:+,.1f}")
    ax_a.axhline(loa_upper, color="grey", ls="--", lw=1,
                 label=f"+1.96 SD = {loa_upper:+,.1f}")
    ax_a.axhline(loa_lower, color="grey", ls="--", lw=1,
                 label=f"−1.96 SD = {loa_lower:+,.1f}")
    ax_a.axhline(0, color="black", ls=":", lw=0.6)
    ax_a.set_xlabel("Mean of LoD2 and LoD1 (Wh/m²)")
    ax_a.set_ylabel("Difference LoD2 − LoD1 (Wh/m²)")
    ax_a.set_title("(a) Bland-Altman — agreement analysis")
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # (b) Absolute error vs LoD2 irradiance (binned box plot)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    n_bins = 10
    bin_edges = np.percentile(v2, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_labels = []
    bin_data = []
    for i in range(len(bin_edges) - 1):
        lo_b, hi_b = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (v2 >= lo_b) & (v2 < hi_b)
        else:
            mask = (v2 >= lo_b) & (v2 <= hi_b)
        if mask.sum() > 0:
            bin_data.append(abs_d[mask])
            bin_labels.append(f"{lo_b:.0f}–\n{hi_b:.0f}")

    bp_b = ax_b.boxplot(bin_data, tick_labels=bin_labels, patch_artist=True,
                        showfliers=False, widths=0.6)
    for box in bp_b["boxes"]:
        box.set_facecolor("#E8963E")
        box.set_alpha(0.7)
    for med in bp_b["medians"]:
        med.set_color("black")
        med.set_linewidth(2)
    ax_b.set_xlabel("LoD2 irradiance bin (Wh/m²)")
    ax_b.set_ylabel("|LoD2 − LoD1| (Wh/m²)")
    ax_b.set_title("(b) Absolute error by irradiance level")
    ax_b.tick_params(axis="x", labelsize=7)
    ax_b.grid(True, alpha=0.2, axis="y")

    # ------------------------------------------------------------------
    # (c) Signed error network map (non-relative, Wh/m²)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    edge_err = edge_lod2.loc[common_idx].copy()
    edge_err["diff"] = d  # signed difference LoD2 − LoD1
    ee_web = edge_err.to_crs(epsg=3857)
    vabs = float(np.nanpercentile(abs_d, 95))
    ee_web.plot(
        column="diff", ax=ax_c, cmap="RdBu_r", legend=True,
        vmin=-vabs, vmax=vabs, linewidth=1.0,
        legend_kwds={"label": "LoD2 − LoD1 (Wh/m²)", "shrink": 0.6},
    )
    ctx.add_basemap(ax_c, source=ctx.providers.CartoDB.DarkMatter)
    ax_c.set_axis_off()
    mae_val = stats["mae"]
    ax_c.set_title(
        f"(c) LoD1 simplification error — walk network\n"
        f"MAE = {mae_val:,.1f} Wh/m²",
    )

    # ------------------------------------------------------------------
    # (d) Relative error histogram + KDE
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    finite_rel = rel_d[np.isfinite(rel_d)]
    clip_lo = np.percentile(finite_rel, 1)
    clip_hi = np.percentile(finite_rel, 99)
    rel_clipped = finite_rel[(finite_rel >= clip_lo) & (finite_rel <= clip_hi)]
    bins_r = np.linspace(clip_lo, clip_hi, 60)
    ax_d.hist(rel_clipped, bins=bins_r, color="#6b8e9e", edgecolor="none",
             alpha=0.7, density=True, label="Histogram (P1–P99)")
    if len(rel_clipped) > 3 and rel_clipped.std() > 0:
        kde_r = gaussian_kde(rel_clipped, bw_method=0.3)
        xr = np.linspace(clip_lo, clip_hi, 300)
        ax_d.plot(xr, kde_r(xr), color="#1a3a5c", lw=2, label="KDE")
    ax_d.axvline(0, color="black", ls="--", lw=0.8)
    ax_d.axvline(finite_rel.mean(), color="red", ls="-", lw=1.5,
                 label=f"Mean = {finite_rel.mean():+.2f}%")
    ax_d.axvline(float(np.median(finite_rel)), color="blue", ls=":", lw=1.5,
                 label=f"Median = {np.median(finite_rel):+.2f}%")
    ax_d.set_xlabel("Relative difference (LoD2−LoD1)/LoD2 (%)")
    ax_d.set_ylabel("Density")
    ax_d.set_title("(d) Distribution of relative error by LoD1 simplification")
    ax_d.legend(fontsize=8)

    # ------------------------------------------------------------------
    # (e) Error exceedance curve
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[2, 0])
    sorted_abs = np.sort(abs_d)
    exceedance = 1.0 - np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax_e.plot(sorted_abs, exceedance * 100, color="#c45e3e", lw=2)
    ax_e.fill_between(sorted_abs, exceedance * 100, alpha=0.15, color="#c45e3e")

    # Mark key thresholds
    for thr_pct in [50, 25, 10, 5]:
        thr_val = np.percentile(abs_d, 100 - thr_pct)
        ax_e.axhline(thr_pct, color="grey", ls=":", lw=0.6)
        ax_e.annotate(
            f"{thr_pct}% edges > {thr_val:.0f}",
            xy=(thr_val, thr_pct), fontsize=7,
            xytext=(thr_val + (sorted_abs.max() - sorted_abs.min()) * 0.03,
                    thr_pct + 2),
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.6),
            color="grey",
        )
    ax_e.set_xlabel("|LoD2 − LoD1| threshold (Wh/m²)")
    ax_e.set_ylabel("Edges exceeding threshold (%)")
    ax_e.set_title("(e) Error exceedance curve")
    ax_e.set_ylim(-2, 102)
    ax_e.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # (f) Cumulative error contribution (Lorenz-style)
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2, 1])
    order = np.argsort(abs_d)
    abs_sorted = abs_d[order]
    cum_err = np.cumsum(abs_sorted) / abs_sorted.sum() * 100
    edge_pct = np.arange(1, len(abs_sorted) + 1) / len(abs_sorted) * 100

    ax_f.plot(edge_pct, cum_err, color="#3e356b", lw=2, label="Cum. error")
    ax_f.plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.5, label="Uniform")
    ax_f.fill_between(edge_pct, cum_err, edge_pct, alpha=0.12, color="#3e356b")

    # Annotate: top X% of edges contribute Y% of total error
    for top_pct in [10, 20, 50]:
        cutoff_idx = int(np.ceil(len(abs_sorted) * (1 - top_pct / 100)))
        cum_at_cutoff = cum_err[cutoff_idx] if cutoff_idx < len(cum_err) else 100
        err_share = 100 - cum_at_cutoff
        ax_f.annotate(
            f"Top {top_pct}% edges → {err_share:.0f}% of total |err|",
            xy=(100 - top_pct, cum_at_cutoff), fontsize=7,
            xytext=(100 - top_pct - 20, cum_at_cutoff - 8),
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.6),
            color="#3e356b",
        )

    ax_f.set_xlabel("Edges sorted by |error|, cumulative (%)")
    ax_f.set_ylabel("Cumulative share of total |error| (%)")
    ax_f.set_title("(f) Error concentration — Lorenz curve")
    ax_f.legend(fontsize=9, loc="upper left")
    ax_f.set_xlim(0, 100)
    ax_f.set_ylim(0, 100)
    ax_f.set_aspect("equal")
    ax_f.grid(True, alpha=0.2)

    fig.suptitle(
        f"Comprehensive Error Evaluation — LoD1 Shape Simplification (August)\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m target, {BUFFER_METERS}m buffer, "
        f"{MESHSIZE}m voxel  |  MAE={stats['mae']:.1f} Wh/m², "
        f"RMSE={stats['rmse']:.1f} Wh/m², r={stats['r_pearson']:.4f}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig_path = os.path.join(output_dir, "lod_error_evaluation.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved error evaluation figure to {fig_path}")
    plt.close(fig)

    # ==================================================================
    # Figure 5: Error-vs-irradiance 2D density + marginals
    # ==================================================================
    fig5, ax5_main = plt.subplots(figsize=(10, 9))

    # 2d hexbin: LoD2 irradiance on x, error (diff) on y
    hb = ax5_main.hexbin(
        v2, d, gridsize=40, cmap="inferno", mincnt=1,
        reduce_C_function=np.mean,
    )
    ax5_main.axhline(0, color="white", ls="--", lw=0.8, alpha=0.7)
    # Moving-average trend
    n_trend = 20
    qvals = np.percentile(v2, np.linspace(5, 95, n_trend + 1))
    trend_x, trend_y, trend_y_abs = [], [], []
    for i in range(len(qvals) - 1):
        mask = (v2 >= qvals[i]) & (v2 < qvals[i + 1])
        if mask.sum() > 0:
            trend_x.append((qvals[i] + qvals[i + 1]) / 2)
            trend_y.append(d[mask].mean())
            trend_y_abs.append(abs_d[mask].mean())
    ax5_main.plot(trend_x, trend_y, "c-o", lw=2, ms=4,
                  label="Mean diff (binned)", zorder=5)
    ax5_main.plot(trend_x, trend_y_abs, color="lime", ls="-", marker="s",
                  lw=2, ms=4, label="Mean |diff| (binned)", zorder=5)

    cb = fig5.colorbar(hb, ax=ax5_main, shrink=0.7)
    cb.set_label("Count")
    ax5_main.set_xlabel("LoD2 edge irradiance (Wh/m²)")
    ax5_main.set_ylabel("LoD1 simplification error (Wh/m²)")
    ax5_main.set_title(
        f"LoD1 Simplification Error vs Irradiance Level\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m, {MESHSIZE}m voxel",
        fontsize=12, fontweight="bold",
    )
    ax5_main.legend(fontsize=9, loc="upper left")
    ax5_main.grid(True, alpha=0.15)

    fig5_path = os.path.join(output_dir, "lod_error_vs_irradiance.png")
    fig5.savefig(fig5_path, dpi=200, bbox_inches="tight")
    print(f"  Saved error-vs-irradiance figure to {fig5_path}")
    plt.close(fig5)

    # ==================================================================
    # Figure 6: Summary metrics — RMSE, MAE, R²
    # ==================================================================
    fig6 = plt.figure(figsize=(14, 5))
    gs6 = gridspec.GridSpec(1, 3, wspace=0.35)

    # --- (a) KDE scatter with RMSE / MAE / R² annotation ---------------
    ax6a = fig6.add_subplot(gs6[0, 0])
    density_6a = dclab_kde_gauss(v2, v1)
    order_6a = np.argsort(density_6a)
    sc_6a = ax6a.scatter(
        v2[order_6a], v1[order_6a], c=density_6a[order_6a],
        s=4, cmap="mako", rasterized=True,
    )
    fig6.colorbar(sc_6a, ax=ax6a, shrink=0.6, label="KDE density")
    ax6a.plot([0, 175000], [0, 175000], "k--", lw=0.8, label="1:1")
    ax6a.set_xlim(0, 175000)
    ax6a.set_ylim(0, 175000)
    ax6a.set_xlabel("LoD2 irradiance (Wh/m²)")
    ax6a.set_ylabel("LoD1 irradiance (Wh/m²)")
    ax6a.set_title("(a) LoD2 vs LoD1 — KDE scatter")
    txt = (
        f"RMSE = {stats['rmse']:,.1f} Wh/m²\n"
        f"MAE  = {stats['mae']:,.1f} Wh/m²\n"
        f"R²   = {stats['r_squared']:.4f}"
    )
    ax6a.text(
        0.05, 0.95, txt, transform=ax6a.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85),
    )
    ax6a.legend(fontsize=8, loc="lower right")
    ax6a.grid(True, alpha=0.2)

    # --- (b) Bar chart of error metrics --------------------------------
    ax6b = fig6.add_subplot(gs6[0, 1])
    metric_names = ["MAE", "RMSE", "Wt. MAE", "Wt. RMSE"]
    metric_vals = [
        stats["mae"], stats["rmse"], stats["wt_mae"], stats["wt_rmse"],
    ]
    colors_bar = ["#E8963E", "#c45e3e", "#E8963E", "#c45e3e"]
    hatches = ["", "", "//", "//"]
    bars = ax6b.bar(metric_names, metric_vals, color=colors_bar, edgecolor="black",
                    linewidth=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars, metric_vals):
        ax6b.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{val:,.0f}", ha="center", va="bottom", fontsize=10,
                  fontweight="bold")
    ax6b.set_ylabel("Wh/m²")
    ax6b.set_title("(b) Error metrics (unweighted vs length-weighted)")
    ax6b.grid(True, alpha=0.2, axis="y")

    # --- (c) Bar chart of R² / Pearson / Spearman ----------------------
    ax6c = fig6.add_subplot(gs6[0, 2])
    corr_names = ["R²", "Pearson r", "Spearman ρ"]
    corr_vals = [stats["r_squared"], stats["r_pearson"], stats["r_spearman"]]
    colors_corr = ["#3e356b", "#6b8e9e", "#2d8659"]
    bars_c = ax6c.bar(corr_names, corr_vals, color=colors_corr,
                      edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars_c, corr_vals):
        ax6c.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{val:.4f}", ha="center", va="bottom", fontsize=11,
                  fontweight="bold")
    ax6c.set_ylim(min(0.95, min(corr_vals) - 0.005), 1.0)
    ax6c.set_ylabel("Correlation / fit")
    ax6c.set_title("(c) Agreement metrics")
    ax6c.grid(True, alpha=0.2, axis="y")

    fig6.suptitle(
        f"LoD1 vs LoD2 — Summary Error Metrics (August)\n"
        f"{CITY_LABEL}, {TARGET_SIZE}m target, {BUFFER_METERS}m buffer, "
        f"{MESHSIZE}m voxel",
        fontsize=13, fontweight="bold",
    )
    fig6_path = os.path.join(output_dir, "lod_summary_metrics.png")
    fig6.savefig(fig6_path, dpi=200, bbox_inches="tight")
    print(f"  Saved summary metrics figure to {fig6_path}")
    plt.close(fig6)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    # --- Run pipeline for both LoDs ---
    city_lod2, solar_lod2, edge_lod2 = run_single_lod(2, OUTPUT_DIR_LOD2)
    city_lod1, solar_lod1, edge_lod1 = run_single_lod(1, OUTPUT_DIR_LOD1)

    os.makedirs(OUTPUT_DIR_COMPARE, exist_ok=True)

    # --- Edge-level paired comparison (primary analysis) ---------------
    edge_stats, paired_df = compute_edge_error(edge_lod2, edge_lod1)

    # --- Save paired edge data -----------------------------------------
    paired_csv = os.path.join(OUTPUT_DIR_COMPARE, "paired_edge_comparison.csv")
    paired_df.to_csv(paired_csv, index=True)
    print(f"  Saved paired edge CSV to {paired_csv}")

    # --- Visualisation --------------------------------------------------
    make_comparison_figures(
        edge_lod2, edge_lod1, paired_df, edge_stats, OUTPUT_DIR_COMPARE,
    )

    # --- Comprehensive error evaluation ---------------------------------
    make_error_evaluation_figure(
        edge_lod2, paired_df, edge_stats, OUTPUT_DIR_COMPARE,
    )

    # --- Save summary CSV -----------------------------------------------
    summary = {
        "city": CITY_LABEL,
        "target_size_m": TARGET_SIZE,
        "buffer_m": BUFFER_METERS,
        "meshsize_m": MESHSIZE,
        **edge_stats,
    }
    csv_path = os.path.join(OUTPUT_DIR_COMPARE, "lod_comparison_summary.csv")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f"  Saved summary CSV to {csv_path}")
    print("\nDone.")
