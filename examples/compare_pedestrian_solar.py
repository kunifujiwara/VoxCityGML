"""Statistical comparison of pedestrian solar irradiance: Tokyo vs Osaka.

Reads the per-edge GeoPackage and ground-level solar grids produced by
``run_pedestrian_solar.py`` / ``run_pedestrian_solar_osaka.py`` and
generates a multi-panel comparison figure.

All edge-level statistics are **length-weighted** so that longer road
segments contribute proportionally more than short stubs.

Panels:
  (a) Length-weighted KDE of edge-level irradiance
  (b) Box-and-violin plot (length-weighted resampled)
  (c) Length-weighted empirical CDF
  (d) Summary statistics table (weighted)
  (e) Ground irradiance histograms (pixel-level)
  (f) Paired weighted-percentile (Q-Q style) plot
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

# ======================================================================
# Configuration
# ======================================================================

CITIES = {
    "Tokyo": {
        "edge_gpkg": "output/pedestrian_solar_tokyo/pedestrian_solar_network.gpkg",
        "solar_grid": "output/pedestrian_solar_tokyo/solar_grid_august.npy",
        "color": "#3e356b",
        "color2": "#2d2750",
    },
    "Osaka": {
        "edge_gpkg": "output/pedestrian_solar_osaka/pedestrian_solar_network.gpkg",
        "solar_grid": "output/pedestrian_solar_osaka/solar_grid_august.npy",
        "color": "#3eb4ad",
        "color2": "#2e9a93",
    },
}

OUTPUT_DIR = "output"
VALUE_COL = "solar_irradiance"


# ======================================================================
# Helpers
# ======================================================================

def load_edge_data():
    """Return dict {city_name: dict with 'vals' and 'lengths' arrays}."""
    data = {}
    for name, cfg in CITIES.items():
        gdf = gpd.read_file(cfg["edge_gpkg"])
        mask = gdf[VALUE_COL].notna()
        gdf = gdf[mask].copy()
        # Compute edge lengths in metres (project to Web Mercator)
        gdf_proj = gdf.to_crs(epsg=3857)
        lengths = gdf_proj.geometry.length.values
        vals = gdf[VALUE_COL].values / 1000.0  # Convert to kWh/m2
        data[name] = {"vals": vals, "lengths": lengths}
        total_km = lengths.sum() / 1000
        print(f"  {name}: {len(vals)} edges, total length {total_km:.1f} km")
    return data


def load_grid_data():
    """Return dict {city_name: 1-D array of valid pixel irradiance values}."""
    data = {}
    for name, cfg in CITIES.items():
        grid = np.load(cfg["solar_grid"])
        vals = grid[~np.isnan(grid)] / 1000.0  # Convert to kWh/m2
        data[name] = vals
        print(f"  {name}: {len(vals)} valid ground pixels")
    return data


def weighted_percentile(vals, weights, percentiles):
    """Compute weighted percentiles.

    Parameters
    ----------
    vals : array-like
    weights : array-like (same length)
    percentiles : array-like in [0, 100]

    Returns
    -------
    np.ndarray of weighted percentile values.
    """
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(vals)
    vals = vals[order]
    weights = weights[order]
    cum_w = np.cumsum(weights)
    cum_w_norm = (cum_w - 0.5 * weights) / cum_w[-1]  # midpoint convention
    return np.interp(np.asarray(percentiles) / 100.0, cum_w_norm, vals)


def compute_stats(vals, weights):
    """Return an OrderedDict of length-weighted summary statistics."""
    from collections import OrderedDict
    w = weights / weights.sum()
    wmean = np.average(vals, weights=weights)
    wvar = np.average((vals - wmean) ** 2, weights=weights)
    wstd = np.sqrt(wvar)
    pcts = weighted_percentile(vals, weights, [5, 25, 50, 75, 95])
    total_length_km = weights.sum() / 1000
    return OrderedDict([
        ("N (edges)", len(vals)),
        ("Total length (km)", total_length_km),
        ("Wt. Mean", wmean),
        ("Wt. Std", wstd),
        ("Wt. Median", pcts[2]),
        ("Wt. IQR", float(pcts[3] - pcts[1])),
        ("Min", vals.min()),
        ("Max", vals.max()),
        ("Wt. P5", pcts[0]),
        ("Wt. P25", pcts[1]),
        ("Wt. P75", pcts[3]),
        ("Wt. P95", pcts[4]),
    ])


# ======================================================================
# Main figure
# ======================================================================

def make_comparison_figure(edge_data, grid_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    names = list(CITIES.keys())
    colors = [CITIES[n]["color"] for n in names]
    colors2 = [CITIES[n]["color2"] for n in names]

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30)

    # ---- (a) Length-weighted KDE of edge irradiance --------------------
    ax_a = fig.add_subplot(gs[0, 0])
    x_max = max(d["vals"].max() for d in edge_data.values()) * 1.05
    x_kde = np.linspace(0, x_max, 300)
    bins = np.linspace(0, x_max, 40)
    for name, c, c2 in zip(names, colors, colors2):
        vals = edge_data[name]["vals"]
        lengths = edge_data[name]["lengths"]
        # Length-weighted histogram (density=True so KDE overlay is visible)
        ax_a.hist(vals, bins=bins, weights=lengths / lengths.sum(),
                  color=c, alpha=0.35, density=True, label=f"{name} hist")
        # Length-weighted KDE
        if len(vals) > 3 and vals.std() > 0:
            kde = gaussian_kde(vals, bw_method=0.3, weights=lengths)
            ax_a.plot(x_kde, kde(x_kde), color=c2, lw=2, label=f"{name} KDE")
    ax_a.set_xlabel("Edge-level cumulative irradiance (kWh/m²)")
    ax_a.set_ylabel("Density (length-weighted)")
    ax_a.legend(fontsize=9)

    # ---- (b) Box-and-violin plot (length-weighted resample) -----------
    ax_b = fig.add_subplot(gs[0, 1])
    # Resample proportional to edge length for violin/box (10k samples)
    rng = np.random.default_rng(42)
    resampled = []
    for n in names:
        v, ln = edge_data[n]["vals"], edge_data[n]["lengths"]
        prob = ln / ln.sum()
        idx = rng.choice(len(v), size=min(10_000, len(v) * 5), p=prob)
        resampled.append(v[idx])
    parts = ax_b.violinplot(resampled, positions=range(len(names)),
                            showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
    bp = ax_b.boxplot(resampled, positions=range(len(names)), widths=0.25,
                      patch_artist=True, showfliers=False, zorder=3)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.8)
    for i, med in enumerate(bp["medians"]):
        med.set_color("black")
        med.set_linewidth(2)
    ax_b.set_xticks(range(len(names)))
    ax_b.set_xticklabels(names)
    ax_b.set_ylabel("Cumulative irradiance (kWh/m²)")

    # ---- (c) Length-weighted empirical CDF ------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    for name, c2 in zip(names, colors2):
        vals = edge_data[name]["vals"]
        lengths = edge_data[name]["lengths"]
        order = np.argsort(vals)
        vals_sorted = vals[order]
        len_sorted = lengths[order]
        cdf = np.cumsum(len_sorted) / len_sorted.sum()
        ax_c.plot(vals_sorted, cdf, color=c2, lw=2, label=name)
    ax_c.set_xlabel("Cumulative irradiance (kWh/m²)")
    ax_c.set_ylabel("Cumulative probability (length-weighted)")
    ax_c.legend(fontsize=10)
    ax_c.grid(True, alpha=0.3)

    # ---- (d) Summary statistics table ----------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    stats = {name: compute_stats(edge_data[name]["vals"], edge_data[name]["lengths"])
             for name in names}
    # Two-sample tests (unweighted — standard test)
    ks_stat, ks_p = ks_2samp(edge_data[names[0]]["vals"],
                              edge_data[names[1]]["vals"])
    mw_stat, mw_p = mannwhitneyu(edge_data[names[0]]["vals"],
                                  edge_data[names[1]]["vals"],
                                  alternative="two-sided")

    header = f"{'Statistic':.<22s} {'Tokyo':>12s} {'Osaka':>12s}\n"
    header += "─" * 48 + "\n"
    rows = ""
    for key in stats[names[0]]:
        v0 = stats[names[0]][key]
        v1 = stats[names[1]][key]
        if key == "N (edges)":
            rows += f"{key:.<22s} {v0:>12d} {v1:>12d}\n"
        else:
            rows += f"{key:.<22s} {v0:>12.1f} {v1:>12.1f}\n"
    rows += "─" * 48 + "\n"
    rows += f"{'KS statistic':.<22s} {ks_stat:>12.4f}\n"
    rows += f"{'KS p-value':.<22s} {ks_p:>12.2e}\n"
    rows += f"{'Mann-Whitney U':.<22s} {mw_stat:>12.0f}\n"
    rows += f"{'MW p-value':.<22s} {mw_p:>12.2e}\n"

    ax_d.text(
        0.05, 0.95, header + rows, transform=ax_d.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                  edgecolor="#888888", alpha=0.9),
    )

    # ---- (e) Ground-pixel irradiance histograms ------------------------
    ax_e = fig.add_subplot(gs[2, 0])
    x_max_g = max(v.max() for v in grid_data.values()) * 1.02
    bins_g = np.linspace(0, x_max_g, 50)
    for name, c, c2 in zip(names, colors, colors2):
        vals = grid_data[name]
        ax_e.hist(vals, bins=bins_g, color=c, alpha=0.40, density=True,
                  label=f"{name} (n={len(vals):,})")
        if len(vals) > 3 and vals.std() > 0:
            kde = gaussian_kde(vals, bw_method=0.25)
            x_kde_g = np.linspace(0, x_max_g, 300)
            ax_e.plot(x_kde_g, kde(x_kde_g), color=c2, lw=2)
    ax_e.set_xlabel("Pixel-level cumulative irradiance (kWh/m²)")
    ax_e.set_ylabel("Density")
    ax_e.legend(fontsize=9)

    # ---- (f) Paired weighted-percentile (Q-Q style) plot ---------------
    ax_f = fig.add_subplot(gs[2, 1])
    quantiles = np.linspace(0, 100, 101)
    q0 = weighted_percentile(edge_data[names[0]]["vals"],
                              edge_data[names[0]]["lengths"], quantiles)
    q1 = weighted_percentile(edge_data[names[1]]["vals"],
                              edge_data[names[1]]["lengths"], quantiles)
    lo = min(q0.min(), q1.min()) * 0.95
    hi = max(q0.max(), q1.max()) * 1.05
    ax_f.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax_f.scatter(q0, q1, c=quantiles, cmap="coolwarm", s=30, zorder=3,
                 edgecolors="grey", linewidths=0.5)
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_f, shrink=0.6)
    cbar.set_label("Percentile")
    ax_f.set_xlabel(f"{names[0]} irradiance (kWh/m²)")
    ax_f.set_ylabel(f"{names[1]} irradiance (kWh/m²)")
    ax_f.set_aspect("equal", adjustable="datalim")
    ax_f.legend(fontsize=9, loc="upper left")
    ax_f.grid(True, alpha=0.3)

    # ---- Save -----------------------------------------------
    fig_path = os.path.join(output_dir, "pedestrian_solar_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\n  Saved comparison figure to {fig_path}")
    plt.close(fig)

    # ---- Individual image: edge irradiance histogram -------------------
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    x_max = max(d["vals"].max() for d in edge_data.values()) * 1.05
    x_kde = np.linspace(0, x_max, 300)
    bins = np.linspace(0, x_max, 40)
    for name, c, c2 in zip(names, colors, colors2):
        vals = edge_data[name]["vals"]
        lengths = edge_data[name]["lengths"]
        ax_hist.hist(vals, bins=bins, weights=lengths / lengths.sum(),
                     color=c, alpha=0.35, density=True, label=f"{name} hist")
        if len(vals) > 3 and vals.std() > 0:
            kde = gaussian_kde(vals, bw_method=0.3, weights=lengths)
            ax_hist.plot(x_kde, kde(x_kde), color=c2, lw=2, label=f"{name} KDE")
    ax_hist.set_xlabel("Edge-level cumulative irradiance (kWh/m²)")
    ax_hist.set_ylabel("Density (length-weighted)")
    ax_hist.legend(fontsize=10)
    hist_path = os.path.join(output_dir, "edge_irradiance_histogram.png")
    fig_hist.savefig(hist_path, dpi=200, bbox_inches="tight")
    print(f"  Saved edge irradiance histogram to {hist_path}")
    plt.close(fig_hist)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("Loading edge-level data …")
    edge_data = load_edge_data()

    print("Loading ground-pixel data …")
    grid_data = load_grid_data()

    print("Generating comparison figure …")
    make_comparison_figure(edge_data, grid_data, OUTPUT_DIR)
    print("Done.")
