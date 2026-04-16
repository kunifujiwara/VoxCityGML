"""Calculate ground-level cumulative solar irradiance in August and aggregate
on the walk road network.

Pipeline:
  1. Load the VoxCity pkl saved by ``run_building_gvi.py`` (or voxelize from
     scratch when no pkl is available).
  2. Compute cumulative ground-level global solar irradiance for August using
     an EPW weather file and the GPU-accelerated solar simulator.
  3. Aggregate the 2-D irradiance grid onto the OSM walk network using
     ``voxcity.geoprocessor.network.get_network_values``.
  4. Visualize the per-edge irradiance on a basemap and export statistics.

Note:
  The GPU solar ray tracer requires the full 3-D voxel grid to fit in GPU
  memory.  A 2 m meshsize over a 4 km area produces a 2000×2000×Z grid
  that will OOM on most consumer GPUs.  This script therefore defaults to
  **5 m** meshsize (800×800 grid) and builds / caches its own ``voxcity.pkl``
  separately from the finer GVI pkl.
"""

import os
os.environ["TI_LOG_LEVEL"] = "warn"  # suppress Taichi startup spam

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import contextily as ctx
from shapely.geometry import Polygon


# ======================================================================
# Loading helpers
# ======================================================================

def load_or_build_city(
    pkl_path,
    city_label,
    citygml_path,
    center_lon,
    center_lat,
    target_size,
    buffer_meters,
    meshsize,
    output_dir,
    building_lod=2,
    gee_project=None,
):
    """Load VoxCity from *pkl_path* if it exists; otherwise run the pipeline."""
    if os.path.isfile(pkl_path):
        print(f"Loading VoxCity model from {pkl_path} …")
        from voxcity.generator.io import load_voxcity
        city = load_voxcity(pkl_path)
        print(f"  Loaded. Voxel grid shape: {city.voxels.classes.shape}")
        return city

    print(f"No pkl found at {pkl_path} — running VoxCityGML pipeline …")
    from voxcitygml.models import VoxelizerConfig
    from voxcitygml.pipeline import VoxCityGML

    cfg = VoxelizerConfig(
        citygml_path=citygml_path,
        center_lon=center_lon,
        center_lat=center_lat,
        size_meters=target_size + 2 * buffer_meters,
        meshsize=meshsize,
        gee_project=gee_project,
        canopy_height_source="Static",
        save_output=True,
        output_dir=output_dir,
        n_workers=4,
        max_voxel_ram_mb=4000,
        building_lod=building_lod,
        # occupancy_threshold=0.5,
        # occupancy_subdivisions=4,
    )
    city = VoxCityGML(cfg).run()
    return city


# ======================================================================
# Core computation
# ======================================================================

def compute_august_solar(city, epw_file_path, output_dir, **kwargs):
    """Compute ground-level cumulative solar irradiance for August.

    Parameters
    ----------
    city : VoxCity
        The assembled voxel city model.
    epw_file_path : str
        Path to an EPW weather file.
    output_dir : str
        Directory for outputs (plots, CSV, etc.).
    **kwargs
        Forwarded to ``get_global_solar_irradiance_using_epw``.

    Returns
    -------
    solar_grid : np.ndarray
        2-D array of cumulative irradiance (Wh/m²) for August.
    """
    from voxcity.simulator_gpu.solar import get_global_solar_irradiance_using_epw

    solar_kwargs = dict(
        epw_file_path=epw_file_path,
        start_time="08-01 05:00:00",
        end_time="08-31 20:00:00",
        view_point_height=1.5,
        output_directory=output_dir,
        show_plot=False,
        with_reflections=False,
        use_sky_patches=True,
        sky_discretization="tregenza",
        progress_report=True,
    )
    solar_kwargs.update(kwargs)

    print("\n" + "=" * 60)
    print("Computing cumulative ground-level solar irradiance (August) …")
    print("=" * 60)

    solar_grid = get_global_solar_irradiance_using_epw(
        city,
        temporal_mode="cumulative",
        spatial_mode="horizontal",
        **solar_kwargs,
    )

    # ------------------------------------------------------------------
    # Post-process: mask out columns where a building occupies the
    # pedestrian-height voxel.  The library's ground-level path
    # (compute_valid_ground_vectorized) correctly excludes columns whose
    # *first* solid→air transition is a building, but it does NOT
    # exclude columns that have valid terrain below with a building
    # starting just above (ground_k+1).  A pedestrian cannot stand
    # inside a building, so those cells must be NaN.
    # ------------------------------------------------------------------
    from voxcity.simulator_gpu.solar.integration.utils import (
        compute_valid_ground_vectorized,
    )

    voxel_data = city.voxels.classes
    meshsize = city.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    BUILDING_CODE = -3

    view_point_height = solar_kwargs.get("view_point_height", 1.5)
    height_offset = max(1, int(round(view_point_height / meshsize)))

    valid_ground, ground_k = compute_valid_ground_vectorized(voxel_data)
    idx_i, idx_j = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")

    # Check several levels around pedestrian height for robustness
    building_mask = np.zeros((ni, nj), dtype=bool)
    for delta_k in range(0, height_offset + 3):
        check_k = np.clip(ground_k + delta_k, 0, nk - 1)
        vals = voxel_data[idx_i, idx_j, check_k]
        building_mask |= (vals == BUILDING_CODE) & valid_ground

    # Flip to match the solar_grid coordinate system (flipud applied inside lib)
    building_mask_flipped = np.flipud(building_mask)

    n_masked = int(building_mask_flipped.sum())
    if n_masked > 0:
        solar_grid = np.where(building_mask_flipped, np.nan, solar_grid)
        print(f"  Masked {n_masked} additional building-footprint cells to NaN")

    return solar_grid


# ======================================================================
# Network aggregation
# ======================================================================

def aggregate_on_walk_network(
    solar_grid,
    city,
    core_rect,
    output_dir,
    value_name="solar_irradiance",
    network_type="walk",
):
    """Overlay the irradiance grid on the OSM walk network and aggregate.

    Parameters
    ----------
    solar_grid : np.ndarray  (R, C)
        Cumulative solar irradiance grid (Wh/m²).
    city : VoxCity
    core_rect : list[(lon, lat), …]
        Target rectangle [SW, NW, NE, SE].
    output_dir : str
    value_name : str
    network_type : str

    Returns
    -------
    G : networkx.MultiDiGraph  –  network with per-edge irradiance.
    edge_gdf : gpd.GeoDataFrame  –  edge geometries + irradiance values.
    """
    from voxcity.geoprocessor.network import get_network_values

    # Derive rectangle_vertices and meshsize from the VoxCity object
    rect = None
    meshsize = None

    extras = getattr(city, "extras", None)
    if isinstance(extras, dict):
        rect = extras.get("rectangle_vertices")

    voxels = getattr(city, "voxels", None)
    meta = getattr(voxels, "meta", None) if voxels is not None else None
    if meta is not None:
        meshsize = getattr(meta, "meshsize", None)
        if rect is None:
            bounds = getattr(meta, "bounds", None)
            if bounds is not None:
                west, south, east, north = bounds
                rect = [(west, south), (west, north),
                        (east, north), (east, south)]

    if rect is None or meshsize is None:
        raise RuntimeError(
            "Cannot determine rectangle_vertices / meshsize from the VoxCity object."
        )

    print("\n" + "=" * 60)
    print("Aggregating solar irradiance on walk road network …")
    print("=" * 60)

    # Use core_rect as the network download extent (target area only)
    G, edge_gdf = get_network_values(
        solar_grid,
        rectangle_vertices=rect,
        meshsize=meshsize,
        value_name=value_name,
        network_type=network_type,
        vis_graph=False,  # we'll do our own visualisation
    )

    # Filter edges to the core target area
    target_poly = Polygon(core_rect)
    edge_gdf_core = edge_gdf[
        edge_gdf.geometry.intersects(target_poly)
    ].copy()
    print(f"  Edges in core area: {len(edge_gdf_core)} / {len(edge_gdf)}")

    # Save GeoPackage
    gpkg_path = os.path.join(output_dir, "pedestrian_solar_network.gpkg")
    edge_gdf_core.to_file(gpkg_path, driver="GPKG")
    print(f"  Saved edge GeoDataFrame to {gpkg_path}")

    return G, edge_gdf, edge_gdf_core


# ======================================================================
# Visualisation
# ======================================================================

def visualize_results(
    solar_grid,
    city,
    edge_gdf_core,
    core_rect,
    city_label,
    target_size,
    buffer_meters,
    meshsize,
    output_dir,
    value_name="solar_irradiance",
):
    """Generate publication-quality figures.

    Produces:
    * (a) Solar irradiance raster map on basemap
    * (b) Walk-network irradiance map on basemap
    * (c) Histogram + KDE of edge irradiance
    * (d) Summary statistics text box
    """
    os.makedirs(output_dir, exist_ok=True)
    vals = edge_gdf_core[value_name].dropna().values

    if len(vals) == 0:
        print("No valid edge values — skipping visualisation.")
        return

    # --- Derive rectangle_vertices for raster overlay ---
    rect = None
    extras = getattr(city, "extras", None)
    if isinstance(extras, dict):
        rect = extras.get("rectangle_vertices")
    if rect is None:
        meta = getattr(getattr(city, "voxels", None), "meta", None)
        if meta is not None:
            bounds = getattr(meta, "bounds", None)
            if bounds is not None:
                w, s, e, n = bounds
                rect = [(w, s), (w, n), (e, n), (e, s)]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.30)

    # ---- (a) Raster irradiance map on basemap --------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if rect is not None:
        from voxcity.geoprocessor.raster import grid_to_geodataframe
        meshsize_val = city.voxels.meta.meshsize
        raster_gdf = grid_to_geodataframe(solar_grid, rect, meshsize_val)
        raster_gdf = raster_gdf[raster_gdf["value"].notna()].copy()

        # Clip to core
        target_poly = Polygon(core_rect)
        raster_gdf = raster_gdf[raster_gdf.geometry.intersects(target_poly)].copy()

        raster_web = raster_gdf.to_crs(epsg=3857)
        vmax_r = float(np.nanpercentile(raster_gdf["value"].values, 95))
        raster_web.plot(
            column="value", ax=ax1, alpha=0.7, cmap="magma",
            vmin=0.0, vmax=vmax_r, legend=True,
            legend_kwds={"label": "Cumulative irradiance (Wh/m²)", "shrink": 0.6},
            edgecolor="none",
        )
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
    ax1.set_axis_off()
    ax1.set_title("(a) Ground-level cumulative solar irradiance — August")

    # ---- (b) Network map on basemap ------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    edge_web = edge_gdf_core.to_crs(epsg=3857)
    vmax_e = 150000.0
    edge_web.plot(
        column=value_name, ax=ax2, cmap="magma", legend=True,
        vmin=0.0, vmax=vmax_e, linewidth=1.0,
        legend_kwds={"label": "Irradiance (Wh/m²)", "shrink": 0.6},
    )
    ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron)
    ax2.set_axis_off()
    ax2.set_title("(b) Walk-network aggregated solar irradiance — August")

    # ---- (c) Histogram + KDE -------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, max(vals.max() * 1.05, 1), 30)
    ax3.hist(vals, bins=bins, color="#E8963E", edgecolor="white",
             alpha=0.85, density=True, label="Histogram")
    from scipy.stats import gaussian_kde
    if len(vals) > 3 and vals.std() > 0:
        kde = gaussian_kde(vals, bw_method=0.3)
        x_kde = np.linspace(0, vals.max() * 1.1, 200)
        ax3.plot(x_kde, kde(x_kde), color="#8B3E00", lw=2, label="KDE")
    ax3.axvline(vals.mean(), color="red", ls="--", lw=1.5,
                label=f"Mean = {vals.mean():.0f}")
    ax3.axvline(np.median(vals), color="blue", ls=":", lw=1.5,
                label=f"Median = {np.median(vals):.0f}")
    ax3.set_xlabel("Cumulative solar irradiance (Wh/m²)")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) Distribution of edge-level solar irradiance")
    ax3.legend(fontsize=8)

    # ---- (d) Summary statistics ----------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    stats_text = (
        f"{'Edges (with data)':.<30s} {len(vals):>10d}\n"
        f"{'─' * 42}\n"
        f"{'Mean (Wh/m²)':.<30s} {vals.mean():>10.1f}\n"
        f"{'Std dev':.<30s} {vals.std():>10.1f}\n"
        f"{'Median':.<30s} {np.median(vals):>10.1f}\n"
        f"{'IQR (Q3−Q1)':.<30s} {iqr:>10.1f}\n"
        f"{'Min':.<30s} {vals.min():>10.1f}\n"
        f"{'Max':.<30s} {vals.max():>10.1f}\n"
        f"{'P5':.<30s} {np.percentile(vals, 5):>10.1f}\n"
        f"{'P25':.<30s} {np.percentile(vals, 25):>10.1f}\n"
        f"{'P75':.<30s} {np.percentile(vals, 75):>10.1f}\n"
        f"{'P95':.<30s} {np.percentile(vals, 95):>10.1f}\n"
        f"{'Skewness':.<30s} {pd.Series(vals).skew():>10.4f}\n"
        f"{'Kurtosis':.<30s} {pd.Series(vals).kurtosis():>10.4f}\n"
    )
    ax4.text(
        0.05, 0.95, stats_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF5EB",
                  edgecolor="#E8963E", alpha=0.9),
    )
    ax4.set_title("(d) Summary statistics")

    fig.suptitle(
        f"Pedestrian Solar Irradiance — August  "
        f"({city_label}, {target_size}m target, {buffer_meters}m buffer, "
        f"{meshsize}m voxel)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    fig_path = os.path.join(output_dir, "pedestrian_solar_stats.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"  Saved figure to {fig_path}")
    plt.close(fig)

    # ---- Individual image: ground irradiance map -----------------------
    if rect is not None:
        fig_ground, ax_g = plt.subplots(figsize=(10, 10))
        from voxcity.geoprocessor.raster import grid_to_geodataframe
        meshsize_val = city.voxels.meta.meshsize
        raster_gdf = grid_to_geodataframe(solar_grid, rect, meshsize_val)
        raster_gdf = raster_gdf[raster_gdf["value"].notna()].copy()
        target_poly = Polygon(core_rect)
        raster_gdf = raster_gdf[raster_gdf.geometry.intersects(target_poly)].copy()
        raster_web = raster_gdf.to_crs(epsg=3857)
        vmax_r = float(np.nanpercentile(raster_gdf["value"].values, 95))
        raster_web.plot(
            column="value", ax=ax_g, alpha=0.7, cmap="magma",
            vmin=0.0, vmax=vmax_r, legend=True,
            legend_kwds={"label": "Cumulative irradiance (Wh/m²)", "shrink": 0.6},
            edgecolor="none",
        )
        ctx.add_basemap(ax_g, source=ctx.providers.CartoDB.Positron)
        ax_g.set_axis_off()
        ax_g.set_title(
            f"Ground-level cumulative solar irradiance — August\n"
            f"({city_label}, {target_size}m, {meshsize}m voxel)",
            fontsize=12, fontweight="bold",
        )
        ground_path = os.path.join(output_dir, "ground_irradiance_map.png")
        fig_ground.savefig(ground_path, dpi=200, bbox_inches="tight")
        print(f"  Saved ground irradiance map to {ground_path}")
        plt.close(fig_ground)

    # ---- Individual image: edge irradiance map -------------------------
    fig_edge, ax_e = plt.subplots(figsize=(10, 10))
    edge_web = edge_gdf_core.to_crs(epsg=3857)
    vmax_e = 150000.0
    edge_web.plot(
        column=value_name, ax=ax_e, cmap="magma", legend=True,
        vmin=0.0, vmax=vmax_e, linewidth=1.0,
        legend_kwds={"label": "Irradiance (Wh/m²)", "shrink": 0.6},
    )
    ctx.add_basemap(ax_e, source=ctx.providers.CartoDB.Positron)
    ax_e.set_axis_off()
    ax_e.set_title(
        f"Walk-network aggregated solar irradiance — August\n"
        f"({city_label}, {target_size}m, {meshsize}m voxel)",
        fontsize=12, fontweight="bold",
    )
    edge_path = os.path.join(output_dir, "edge_irradiance_map.png")
    fig_edge.savefig(edge_path, dpi=200, bbox_inches="tight")
    print(f"  Saved edge irradiance map to {edge_path}")
    plt.close(fig_edge)


# ======================================================================
# Main pipeline
# ======================================================================

def run_pedestrian_solar(
    city_label,
    citygml_path,
    center_lon,
    center_lat,
    target_size=2000,
    buffer_meters=1000,
    meshsize=5.0,
    output_dir=None,
    gvi_output_dir=None,
    building_lod=2,
    gee_project=None,
    epw_file_path=None,
):
    """Run the full pedestrian solar irradiance pipeline.

    Parameters
    ----------
    city_label : str
        Human-readable label (used in titles and filenames).
    citygml_path : str | list[str]
        Path(s) to CityGML dataset directories.
    center_lon, center_lat : float
        Centre of the target area (WGS 84).
    target_size : int
        Side length of the core target area in metres.
    buffer_meters : int
        Buffer around the target area (same as ``gvi_buffer`` in
        ``run_building_gvi.py``).
    meshsize : float
        Voxel resolution in metres.
    output_dir : str | None
        Directory for this script's outputs (defaults to
        ``output/pedestrian_solar_<label>``).
    gvi_output_dir : str | None
        Directory where ``run_building_gvi.py`` saved its outputs.  Used
        only to look for an existing EPW file (the GVI pkl at 2 m is too
        large for the GPU solar simulator).  Defaults to
        ``output/building_gvi_<label>``.
    building_lod : int
    gee_project : str
    epw_file_path : str | None
        Path to an EPW weather file.  If ``None``, the nearest EPW is
        downloaded automatically.

    Returns
    -------
    solar_grid : np.ndarray  –  2-D cumulative irradiance (Wh/m²).
    G : networkx.MultiDiGraph  –  walk network with per-edge irradiance.
    edge_gdf_core : gpd.GeoDataFrame  –  filtered edge geometries.
    """
    label_slug = city_label.lower().replace(" ", "_")
    if output_dir is None:
        output_dir = f"output/pedestrian_solar_{label_slug}"
    if gvi_output_dir is None:
        gvi_output_dir = f"output/building_gvi_{label_slug}"
    os.makedirs(output_dir, exist_ok=True)

    # Use *this script's* output_dir for the pkl — not the GVI pkl, which
    # may be at 2 m resolution and too large for GPU solar simulation.
    pkl_path = os.path.join(output_dir, "voxcity.pkl")

    phase_times = {}  # collect per-phase wall-clock times

    # --- 1. Load / build VoxCity -----------------------------------------
    t_phase_start = time.perf_counter()
    city = load_or_build_city(
        pkl_path=pkl_path,
        city_label=city_label,
        citygml_path=citygml_path,
        center_lon=center_lon,
        center_lat=center_lat,
        target_size=target_size,
        buffer_meters=buffer_meters,
        meshsize=meshsize,
        output_dir=output_dir,
        building_lod=building_lod,
        gee_project=gee_project,
    )
    phase_times["1_voxelization"] = time.perf_counter() - t_phase_start
    print(f"\n  Phase 1 (CityGML extraction & voxelization): "
          f"{phase_times['1_voxelization']:.1f} s")

    # --- 2. Core target rectangle ----------------------------------------
    from voxcitygml.citygml.coordinates import create_rectangle
    core_rect = create_rectangle(center_lon, center_lat, target_size)

    # --- 3. EPW file path ------------------------------------------------
    solar_kwargs = {}
    if epw_file_path is None:
        # Try to reuse an EPW already present in the output directories
        for candidate_dir in [output_dir, gvi_output_dir, "output"]:
            for f in sorted(os.listdir(candidate_dir)) if os.path.isdir(candidate_dir) else []:
                if f.endswith(".epw"):
                    epw_file_path = os.path.join(candidate_dir, f)
                    break
            if epw_file_path:
                break
    if epw_file_path is None:
        # Let the solar function download the nearest EPW
        print("No EPW file found — will download the nearest one.")
        solar_kwargs["download_nearest_epw"] = True
        solar_kwargs["output_dir"] = output_dir
    else:
        print(f"Using EPW file: {epw_file_path}")

    # --- 4. Cumulative solar irradiance (August) -------------------------
    t_phase_start = time.perf_counter()
    solar_grid = compute_august_solar(
        city,
        epw_file_path=epw_file_path,
        output_dir=output_dir,
        **solar_kwargs,
    )

    # Save the raw grid as numpy file
    grid_npy_path = os.path.join(output_dir, "solar_grid_august.npy")
    np.save(grid_npy_path, solar_grid)
    print(f"  Saved solar grid to {grid_npy_path}")

    # Quick grid statistics
    valid = solar_grid[~np.isnan(solar_grid)]
    print(f"\n  Grid statistics (valid cells: {len(valid)} / {solar_grid.size}):")
    print(f"    Mean : {valid.mean():.1f} Wh/m²")
    print(f"    Min  : {valid.min():.1f} Wh/m²")
    print(f"    Max  : {valid.max():.1f} Wh/m²")
    print(f"    P50  : {np.median(valid):.1f} Wh/m²")
    phase_times["2_solar_computation"] = time.perf_counter() - t_phase_start
    print(f"\n  Phase 2 (Ground-level solar irradiance computation): "
          f"{phase_times['2_solar_computation']:.1f} s")

    # --- 5. Aggregate on walk road network --------------------------------
    t_phase_start = time.perf_counter()
    G, edge_gdf, edge_gdf_core = aggregate_on_walk_network(
        solar_grid,
        city,
        core_rect=core_rect,
        output_dir=output_dir,
        value_name="solar_irradiance",
        network_type="walk",
    )

    # Edge-level CSV
    csv_path = os.path.join(output_dir, "pedestrian_solar_edges.csv")
    edge_gdf_core.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"  Saved edge CSV to {csv_path}")
    phase_times["3_network_aggregation"] = time.perf_counter() - t_phase_start
    print(f"\n  Phase 3 (Aggregation by road network edges): "
          f"{phase_times['3_network_aggregation']:.1f} s")

    # --- 6. Visualise & export statistics ---------------------------------
    visualize_results(
        solar_grid=solar_grid,
        city=city,
        edge_gdf_core=edge_gdf_core,
        core_rect=core_rect,
        city_label=city_label,
        target_size=target_size,
        buffer_meters=buffer_meters,
        meshsize=meshsize,
        output_dir=output_dir,
        value_name="solar_irradiance",
    )

    # --- 7. Print summary -------------------------------------------------
    vals = edge_gdf_core["solar_irradiance"].dropna().values
    print("\n" + "=" * 60)
    print(f"Pedestrian Solar Irradiance — Summary ({city_label})")
    print("=" * 60)
    print(f"  Period           : August (01–31)")
    print(f"  Target area      : {target_size} m  (buffer {buffer_meters} m)")
    print(f"  Voxel resolution : {meshsize} m")
    print(f"  Network edges    : {len(edge_gdf_core)} ({len(vals)} with data)")
    if len(vals) > 0:
        print(f"  Mean irradiance  : {vals.mean():.1f} Wh/m²")
        print(f"  Median           : {np.median(vals):.1f} Wh/m²")
        print(f"  Min              : {vals.min():.1f} Wh/m²")
        print(f"  Max              : {vals.max():.1f} Wh/m²")

    # --- Timing summary --------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase timing summary")
    print("=" * 60)
    total = 0.0
    for label, secs in phase_times.items():
        mins, s = divmod(secs, 60)
        print(f"  {label:.<45s} {int(mins):3d}m {s:05.1f}s  ({secs:.1f}s)")
        total += secs
    mins_t, s_t = divmod(total, 60)
    print(f"  {'TOTAL':.<45s} {int(mins_t):3d}m {s_t:05.1f}s  ({total:.1f}s)")

    return solar_grid, G, edge_gdf_core


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    run_pedestrian_solar(
        city_label="Tokyo",
        citygml_path=[
            "/path/to/citygml_dataset_1",  # Replace with your CityGML path(s)
            "/path/to/citygml_dataset_2",
        ],
        center_lon=139.767125,
        center_lat=35.681236,
        # Same target & buffer as run_building_gvi.py defaults:
        target_size=2000,
        buffer_meters=200,
        # 5 m meshsize keeps the 3-D grid small enough for GPU solar.
        # (2 m → 2000×2000×141 → CUDA OOM on most consumer GPUs)
        meshsize=2.0,
    )
