"""Comprehensive comparison of building-surface GVI between Tokyo and Osaka.

Loads per-building GVI CSV files produced by ``run_building_gvi.py`` and
``run_building_gvi_osaka.py``, then generates a multi-panel statistical
comparison figure.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
import seaborn as sns

# ======================================================================
# Configuration
# ======================================================================
_mako = sns.color_palette("mako", as_cmap=True)
CITIES = {
    "Tokyo": {
        "csv": "output/building_gvi_tokyo/building_gvi.csv",
        "color": matplotlib.colors.to_hex(_mako(0.25)),   # dark teal
        "color_light": matplotlib.colors.to_hex(_mako(0.15)),
    },
    "Osaka": {
        "csv": "output/building_gvi_osaka/building_gvi.csv",
        "color": matplotlib.colors.to_hex(_mako(0.55)),   # mid green
        "color_light": matplotlib.colors.to_hex(_mako(0.45)),
    },
}
OUT_DIR = "output/building_gvi_compare"

# ======================================================================
# Load data
# ======================================================================
data = {}
for city, info in CITIES.items():
    path = info["csv"]
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run the {city} GVI script first.")
        sys.exit(1)
    df = pd.read_csv(path)
    data[city] = df
    print(f"Loaded {city}: {len(df)} buildings, "
          f"mean GVI = {df['mean_gvi'].mean():.4f}")

os.makedirs(OUT_DIR, exist_ok=True)

# Convenience arrays
cities = list(data.keys())
gvi = {c: data[c]["mean_gvi"].values for c in cities}
fc = {c: data[c]["face_count"].values for c in cities}
colors = {c: CITIES[c]["color"] for c in cities}
colors_light = {c: CITIES[c]["color_light"] for c in cities}

# ======================================================================
# Create comparison figure (4 × 3 = 12 panels)
# ======================================================================
fig = plt.figure(figsize=(24, 22))
gs = gridspec.GridSpec(4, 3, hspace=0.38, wspace=0.32)

# ------------------------------------------------------------------
# (a) Overlaid histogram + KDE
# ------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
x_max = 0.1
bins = np.linspace(0, x_max, 80)
from scipy.stats import gaussian_kde

for c in cities:
    ax1.hist(gvi[c], bins=bins, alpha=0.35, color=colors[c],
             edgecolor="white", density=True, label=f"{c} (n={len(gvi[c])})")
    if len(gvi[c]) > 3 and gvi[c].std() > 0:
        kde = gaussian_kde(gvi[c], bw_method=0.3)
        x_kde = np.linspace(0, x_max, 300)
        ax1.plot(x_kde, kde(x_kde), color=colors[c], lw=2)
ax1.set_xlabel("Mean GVI")
ax1.set_ylabel("Density")
ax1.set_title("(a) Distribution comparison")
ax1.legend(fontsize=9)

# ------------------------------------------------------------------
# (b) Overlaid ECDF
# ------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
for c in cities:
    sorted_g = np.sort(gvi[c])
    ecdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
    ax2.step(sorted_g, ecdf, color=colors[c], lw=2, label=c)
ax2.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax2.set_xlabel("Mean GVI")
ax2.set_ylabel("Cumulative proportion")
ax2.set_title("(b) Empirical CDF comparison")
ax2.set_xlim(left=0)
ax2.legend(fontsize=9)

# ------------------------------------------------------------------
# (c) Side-by-side box + violin
# ------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
positions = [1, 2]
for i, c in enumerate(cities):
    parts = ax3.violinplot(gvi[c], positions=[positions[i]],
                           showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(colors_light[c])
        pc.set_alpha(0.6)
    ax3.boxplot(gvi[c], positions=[positions[i]], widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor=colors[c], alpha=0.7),
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color="#333"),
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor="#888", alpha=0.5))
ax3.set_xticks(positions)
ax3.set_xticklabels(cities)
ax3.set_ylabel("Mean GVI")
ax3.set_title("(c) Box & violin comparison")

# ------------------------------------------------------------------
# (d) Percentile comparison (paired horizontal bars)
# ------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])
pcts = [5, 10, 25, 50, 75, 90, 95]
bar_h = 0.35
y_pos = np.arange(len(pcts))
for i, c in enumerate(cities):
    pct_vals = np.percentile(gvi[c], pcts)
    offset = -bar_h / 2 if i == 0 else bar_h / 2
    bars = ax4.barh(y_pos + offset, pct_vals, height=bar_h,
                    color=colors[c], edgecolor="white", label=c)
    for bar, v in zip(bars, pct_vals):
        ax4.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=7)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"P{p}" for p in pcts])
ax4.set_xlabel("Mean GVI")
ax4.set_title("(d) Percentile comparison")
ax4.legend(fontsize=8)

# ------------------------------------------------------------------
# (e) GVI category proportions (stacked bars)
# ------------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 1])
categories = [
    ("GVI = 0", lambda g: g == 0),
    ("0 < GVI ≤ 0.02", lambda g: (g > 0) & (g <= 0.02)),
    ("0.02 < GVI ≤ 0.05", lambda g: (g > 0.02) & (g <= 0.05)),
    ("0.05 < GVI ≤ 0.10", lambda g: (g > 0.05) & (g <= 0.10)),
    ("0.10 < GVI ≤ 0.20", lambda g: (g > 0.10) & (g <= 0.20)),
    ("GVI > 0.20", lambda g: g > 0.20),
]
cat_colors = ["#D5D8DC", "#FAD7A0", "#ABEBC6", "#82E0AA", "#27AE60", "#1E8449"]
x = np.arange(len(cities))
bar_width = 0.5
bottoms = {c: 0.0 for c in cities}
for (cat_label, cat_fn), cat_col in zip(categories, cat_colors):
    fracs = []
    for c in cities:
        fracs.append(cat_fn(gvi[c]).sum() / len(gvi[c]) * 100)
    rects = ax5.bar(x, fracs, bar_width, bottom=[bottoms[c] for c in cities],
                    color=cat_col, edgecolor="white", label=cat_label)
    for ci, c in enumerate(cities):
        bottoms[c] += fracs[ci]
ax5.set_xticks(x)
ax5.set_xticklabels(cities)
ax5.set_ylabel("Percentage of buildings (%)")
ax5.set_title("(e) GVI category breakdown")
ax5.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0))

# ------------------------------------------------------------------
# (f) Face count vs GVI scatter (both cities)
# ------------------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
for c in cities:
    ax6.scatter(fc[c], gvi[c], s=10, alpha=0.35, color=colors[c],
                edgecolors="none", label=c)
ax6.set_xlabel("Vertical face count (building size proxy)")
ax6.set_ylabel("Mean GVI")
ax6.set_title("(f) GVI vs. building size")
ax6.set_xscale("log")
ax6.legend(fontsize=9)

# ------------------------------------------------------------------
# (g) Mean / median comparison bar chart
# ------------------------------------------------------------------
ax7 = fig.add_subplot(gs[2, 0])
metrics = ["Mean", "Weighted\nmean", "Median", "Std dev"]
vals = {}
for c in cities:
    w_mean = (gvi[c] * fc[c]).sum() / fc[c].sum() if fc[c].sum() else 0
    vals[c] = [gvi[c].mean(), w_mean, np.median(gvi[c]), gvi[c].std()]
x = np.arange(len(metrics))
bw = 0.35
for i, c in enumerate(cities):
    bars = ax7.bar(x + (i - 0.5) * bw, vals[c], bw, color=colors[c],
                   edgecolor="white", label=c)
    for bar, v in zip(bars, vals[c]):
        ax7.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                 f"{v:.4f}", ha="center", fontsize=7, rotation=0)
ax7.set_xticks(x)
ax7.set_xticklabels(metrics)
ax7.set_ylabel("GVI value")
ax7.set_title("(g) Key statistics comparison")
ax7.legend(fontsize=9)
ax7.set_ylim(0, max(max(v) for v in vals.values()) * 1.45)

# ------------------------------------------------------------------
# (h) Difference in ECDF (Osaka − Tokyo)
# ------------------------------------------------------------------
ax8 = fig.add_subplot(gs[2, 1])
# Create a common x-grid
x_grid = np.linspace(0, x_max, 500)
ecdfs = {}
for c in cities:
    sorted_g = np.sort(gvi[c])
    ecdf_vals = np.searchsorted(sorted_g, x_grid, side="right") / len(sorted_g)
    ecdfs[c] = ecdf_vals
diff = ecdfs[cities[1]] - ecdfs[cities[0]]  # Osaka − Tokyo
ax8.fill_between(x_grid, diff, 0, where=diff >= 0,
                 color=CITIES[cities[1]]["color_light"], alpha=0.6,
                 label=f"More in {cities[1]}")
ax8.fill_between(x_grid, diff, 0, where=diff < 0,
                 color=CITIES[cities[0]]["color_light"], alpha=0.6,
                 label=f"More in {cities[0]}")
ax8.plot(x_grid, diff, color="#333", lw=1)
ax8.axhline(0, color="gray", ls="--", lw=0.8)
ax8.set_xlabel("Mean GVI")
ax8.set_ylabel(f"ECDF({cities[1]}) − ECDF({cities[0]})")
ax8.set_title("(h) ECDF difference")
ax8.legend(fontsize=8)

# ------------------------------------------------------------------
# (i) Statistical tests
# ------------------------------------------------------------------
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")

# Two-sample tests
t_stat, t_pval = sp_stats.ttest_ind(gvi[cities[0]], gvi[cities[1]],
                                     equal_var=False)
u_stat, u_pval = sp_stats.mannwhitneyu(gvi[cities[0]], gvi[cities[1]],
                                        alternative="two-sided")
ks_stat, ks_pval = sp_stats.ks_2samp(gvi[cities[0]], gvi[cities[1]])

# Effect size (Cohen's d)
pooled_std = np.sqrt(
    (gvi[cities[0]].std()**2 + gvi[cities[1]].std()**2) / 2
)
cohens_d = (gvi[cities[0]].mean() - gvi[cities[1]].mean()) / pooled_std if pooled_std > 0 else 0

def fmt_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

test_text = (
    "Statistical Tests\n"
    f"{'═' * 42}\n"
    f"\n"
    f"Welch's t-test\n"
    f"  t = {t_stat:+.4f}\n"
    f"  p = {fmt_p(t_pval)}  {sig_stars(t_pval)}\n"
    f"\n"
    f"Mann-Whitney U test\n"
    f"  U = {u_stat:.0f}\n"
    f"  p = {fmt_p(u_pval)}  {sig_stars(u_pval)}\n"
    f"\n"
    f"Kolmogorov-Smirnov test\n"
    f"  D = {ks_stat:.4f}\n"
    f"  p = {fmt_p(ks_pval)}  {sig_stars(ks_pval)}\n"
    f"\n"
    f"Effect size (Cohen's d)\n"
    f"  d = {cohens_d:+.4f}\n"
    f"  {'(negligible)' if abs(cohens_d) < 0.2 else '(small)' if abs(cohens_d) < 0.5 else '(medium)' if abs(cohens_d) < 0.8 else '(large)'}\n"
)
ax9.text(0.05, 0.95, test_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDEBD0",
                   edgecolor="#E67E22", alpha=0.9))
ax9.set_title("(i) Statistical hypothesis tests")

# ------------------------------------------------------------------
# (j) Top-20 GVI buildings from each city
# ------------------------------------------------------------------
ax10 = fig.add_subplot(gs[3, 0])
n_show = min(15, min(len(data[c]) for c in cities))
for i, c in enumerate(cities):
    top = data[c].nlargest(n_show, "mean_gvi")
    y_offset = np.arange(n_show) + i * 0.4
    ax10.barh(y_offset, top["mean_gvi"].values, height=0.35,
              color=colors[c], edgecolor="white", label=c)
ax10.set_yticks(np.arange(n_show) + 0.2)
ax10.set_yticklabels([f"Rank {i+1}" for i in range(n_show)], fontsize=7)
ax10.set_xlabel("Mean GVI")
ax10.set_title(f"(j) Top-{n_show} greenest buildings")
ax10.legend(fontsize=8)
ax10.invert_yaxis()

# ------------------------------------------------------------------
# (k) Quartile distribution comparison (100% stacked)
# ------------------------------------------------------------------
ax11 = fig.add_subplot(gs[3, 1])
quartile_colors = ["#E74C3C", "#F39C12", "#3498DB", "#27AE60"]
quartile_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
# Use global quartile boundaries (combined distribution)
combined = np.concatenate([gvi[c] for c in cities])
quartile_bounds = np.percentile(combined, [25, 50, 75])

x = np.arange(len(cities))
for ci, c in enumerate(cities):
    g = gvi[c]
    q_counts = [
        (g <= quartile_bounds[0]).sum(),
        ((g > quartile_bounds[0]) & (g <= quartile_bounds[1])).sum(),
        ((g > quartile_bounds[1]) & (g <= quartile_bounds[2])).sum(),
        (g > quartile_bounds[2]).sum(),
    ]
    q_pcts = [cnt / len(g) * 100 for cnt in q_counts]
    bottom = 0
    for qi, (qp, qc, ql) in enumerate(zip(q_pcts, quartile_colors,
                                            quartile_labels)):
        label = ql if ci == 0 else None
        ax11.bar(ci, qp, bottom=bottom, color=qc, edgecolor="white",
                 width=0.5, label=label)
        if qp > 5:
            ax11.text(ci, bottom + qp / 2, f"{qp:.1f}%",
                      ha="center", va="center", fontsize=8, fontweight="bold",
                      color="white")
        bottom += qp

ax11.set_xticks(range(len(cities)))
ax11.set_xticklabels(cities)
ax11.set_ylabel("Percentage (%)")
ax11.set_title("(k) Global-quartile distribution")
ax11.legend(fontsize=7, loc="upper right")

# ------------------------------------------------------------------
# (l) Comprehensive comparison table
# ------------------------------------------------------------------
ax12 = fig.add_subplot(gs[3, 2])
ax12.axis("off")

rows = []
for c in cities:
    g, f = gvi[c], fc[c]
    w_mean = (g * f).sum() / f.sum() if f.sum() else 0
    iqr = np.percentile(g, 75) - np.percentile(g, 25)
    rows.append({
        "Metric": "",
        c: "",
    })

# Build table text
hdr = f"{'Metric':.<24s}"
for c in cities:
    hdr += f" {c:>10s}"
lines = [hdr, "─" * (24 + 11 * len(cities))]

def add_row(label, vals, fmt=".4f"):
    line = f"{label:.<24s}"
    for v in vals:
        if isinstance(v, int):
            line += f" {v:>10d}"
        else:
            line += f" {v:>10{fmt}}"
    return line

for metric_name, metric_fn in [
    ("Buildings",        lambda c: len(data[c])),
    ("Vertical faces",   lambda c: int(fc[c].sum())),
    ("Mean",             lambda c: gvi[c].mean()),
    ("Weighted mean",    lambda c: (gvi[c]*fc[c]).sum()/fc[c].sum()),
    ("Median",           lambda c: float(np.median(gvi[c]))),
    ("Std dev",          lambda c: gvi[c].std()),
    ("IQR",              lambda c: float(np.percentile(gvi[c],75)-np.percentile(gvi[c],25))),
    ("Min",              lambda c: gvi[c].min()),
    ("Max",              lambda c: gvi[c].max()),
    ("P5",               lambda c: float(np.percentile(gvi[c], 5))),
    ("P25",              lambda c: float(np.percentile(gvi[c], 25))),
    ("P75",              lambda c: float(np.percentile(gvi[c], 75))),
    ("P95",              lambda c: float(np.percentile(gvi[c], 95))),
    ("Skewness",         lambda c: float(pd.Series(gvi[c]).skew())),
    ("Kurtosis",         lambda c: float(pd.Series(gvi[c]).kurtosis())),
    ("GVI = 0 (%)",      lambda c: (gvi[c]==0).sum()/len(gvi[c])*100),
    ("GVI > 0.10 (%)",   lambda c: (gvi[c]>0.10).sum()/len(gvi[c])*100),
    ("GVI > 0.20 (%)",   lambda c: (gvi[c]>0.20).sum()/len(gvi[c])*100),
]:
    vals = [metric_fn(c) for c in cities]
    if metric_name in ("Buildings", "Vertical faces"):
        lines.append(add_row(metric_name, vals))
    else:
        lines.append(add_row(metric_name, vals))

table_text = "\n".join(lines)
ax12.text(0.02, 0.95, table_text, transform=ax12.transAxes,
          fontsize=9, verticalalignment="top", fontfamily="monospace",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB",
                    edgecolor="#2E86C1", alpha=0.9))
ax12.set_title("(l) Summary comparison table")

# ------------------------------------------------------------------
# Suptitle
# ------------------------------------------------------------------
fig.suptitle(
    "Green View Index \u2014 Tokyo vs Osaka Comprehensive Comparison\n"
    "(2 km target area, 1 km buffer, 5 m voxel, vertical wall faces only)",
    fontsize=16, fontweight="bold", y=0.995,
)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
out_path = f"{OUT_DIR}/gvi_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved comparison figure to {out_path}")
plt.close(fig)

# ------------------------------------------------------------------
# Save Individual Histograms
# ------------------------------------------------------------------
# Use a static fixed max on the X axis to zoom in and see detail better
x_max_plot = 0.1

# 1. Histogram of buildings
fig_b = plt.figure(figsize=(6, 3))
ax_b = fig_b.add_subplot(111)
detailed_bins = np.linspace(0, x_max_plot, 25)
x_kde = np.linspace(0, x_max_plot, 500)
for c in cities:
    ax_b.hist(gvi[c], bins=detailed_bins, alpha=0.5, color=colors[c],
             edgecolor="none", density=True, label=f"{c} (n={len(gvi[c])})")
    if len(gvi[c]) > 3 and gvi[c].std() > 0:
        # Re-soften the KDE a bit since bins are wider
        kde = gaussian_kde(gvi[c], bw_method=0.2)
        ax_b.plot(x_kde, kde(x_kde), color=colors[c], lw=1.5)
ax_b.set_xlim(0, x_max_plot)
ax_b.set_xlabel("Mean GVI")
ax_b.set_ylabel("Density")
ax_b.set_title("Distribution of Building Mean GVI (Detailed)")
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.2, linestyle='--')
fig_b.savefig(f"{OUT_DIR}/histogram_buildings.png", dpi=300, bbox_inches="tight")
print(f"Saved building histogram to {OUT_DIR}/histogram_buildings.png")
plt.close(fig_b)

# 2. Histogram of vertical surface meshes (weighted by face count)
fig_f = plt.figure(figsize=(6, 3))
ax_f = fig_f.add_subplot(111)
for c in cities:
    weights = fc[c]
    ax_f.hist(gvi[c], bins=detailed_bins, weights=weights, alpha=0.5, color=colors[c],
             edgecolor="none", density=True, label=f"{c} (faces={int(weights.sum())})")
    if len(gvi[c]) > 3 and gvi[c].std() > 0:
        try:
            kde = gaussian_kde(gvi[c], bw_method=0.2, weights=weights)
            ax_f.plot(x_kde, kde(x_kde), color=colors[c], lw=1.5)
        except TypeError:
            pass # fallback if gaussian_kde version doesn't support weights
ax_f.set_xlim(0, x_max_plot)
ax_f.set_xlabel("Mean GVI")
ax_f.set_ylabel("Density")
ax_f.set_title("Distribution of Vertical Surface Mesh GVI (Detailed)")
ax_f.legend(fontsize=9)
ax_f.grid(True, alpha=0.2, linestyle='--')
fig_f.savefig(f"{OUT_DIR}/histogram_faces.png", dpi=300, bbox_inches="tight")
print(f"Saved face histogram to {OUT_DIR}/histogram_faces.png")
plt.close(fig_f)


# Also save a combined CSV
combined_df = pd.concat(
    [data[c].assign(city=c) for c in cities], ignore_index=True
)
csv_path = f"{OUT_DIR}/gvi_combined.csv"
combined_df.to_csv(csv_path, index=False)
print(f"Saved combined CSV to {csv_path}")

# Print key results
print("\n" + "=" * 60)
print("KEY COMPARISON RESULTS")
print("=" * 60)
for c in cities:
    g = gvi[c]
    w = (g * fc[c]).sum() / fc[c].sum()
    print(f"\n{c}:")
    print(f"  Buildings:     {len(g)}")
    print(f"  Mean GVI:      {g.mean():.4f}")
    print(f"  Weighted mean: {w:.4f}")
    print(f"  Median:        {np.median(g):.4f}")
    print(f"  Std dev:       {g.std():.4f}")

print(f"\nWelch's t-test:  t={t_stat:+.4f}, p={fmt_p(t_pval)} {sig_stars(t_pval)}")
print(f"Mann-Whitney U:  U={u_stat:.0f}, p={fmt_p(u_pval)} {sig_stars(u_pval)}")
print(f"KS test:         D={ks_stat:.4f}, p={fmt_p(ks_pval)} {sig_stars(ks_pval)}")
print(f"Cohen's d:       {cohens_d:+.4f}")

# ------------------------------------------------------------------
# 3. Scatter plot of mesh GVI and mesh elevation
# ------------------------------------------------------------------
import dclab.kde_methods

fig_s, axes_s = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, c in enumerate(cities):
    mesh_csv = f"output/building_gvi_{c.lower()}/mesh_gvi_vertical.csv"
    if os.path.exists(mesh_csv):
        print(f"Loading {c} mesh faces from {mesh_csv}...")
        df_mesh = pd.read_csv(mesh_csv)
        ax = axes_s[i]
        
        x_vals = df_mesh['gvi'].values
        y_vals = df_mesh['z'].values
        
        print(f"Computing KDE density for {c} ({len(x_vals)} points)...")
        # Use dclab's fast histogram-based KDE estimation
        density = dclab.kde_methods.kde_histogram(x_vals, y_vals)
        
        # Sort points by density so highest density points are drawn on top
        idx = density.argsort()
        x_vals, y_vals, density = x_vals[idx], y_vals[idx], density[idx]

        sc = ax.scatter(x_vals, y_vals, s=0.1, c=density, cmap='viridis', 
                        alpha=0.6, rasterized=True)
        
        ax.set_title(f"{c} Mesh Faces (n={len(df_mesh):,})")
        ax.set_xlabel("Mesh Face GVI")
        if i == 0:
            ax.set_ylabel("Mesh Face Elevation (m)")
        ax.set_xlim(0, x_max_plot)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Add colorbar for each subplot
        cbar = fig_s.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("KDE Density")
    else:
        print(f"Skipping mesh scatter for {c} - {mesh_csv} not found.")

fig_s.suptitle("KDE Scatter Plot: Mesh GVI vs. Mesh Elevation (Vertical Faces)", fontsize=14, y=1.02)
out_scatter = f"{OUT_DIR}/scatter_elevation_gvi.png"
fig_s.savefig(out_scatter, dpi=300, bbox_inches="tight")
print(f"\nSaved scatter plot to {out_scatter}")
plt.close(fig_s)
