"""Calculate average building-surface Green View Index (GVI) for each building.

Pipeline:
  1. Voxelize a (target + buffer) area using VoxCityGML.  The buffer ensures
     buildings at the edge of the target area have complete visibility.
  2. Use voxcity's GPU-accelerated ``get_surface_view_factor`` to cast rays
     from every exposed building face and measure the fraction of green
     targets (tree canopy, rangeland, shrub, tree, moss/lichen, wetland,
     mangrove).
  3. Aggregate per-face GVI into a per-building average **using only
     vertical (wall) faces** — horizontal roof/floor faces are excluded.
  4. Filter results to only buildings whose centroid falls within the
     core target rectangle (discard buffer-only buildings).
  5. Print / export a summary table and an OBJ coloured by GVI.
  6. Visualize per-building GVI on a basemap using LOD1 footprints.
"""

import os
os.environ["TI_LOG_LEVEL"] = "warn"  # suppress Taichi startup spam

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline import VoxCityGML


# ======================================================================
# Building-ID grid aligned to the 3-D voxel grid
# ======================================================================

def _fill_building_id_gaps(bid_grid, voxel_grid, building_code=-3):
    """Fill gaps in the 2-D building-ID grid caused by the 2-D/3-D mismatch.

    The pipeline's ``meshes_to_building_grids`` rasterises CityGML
    triangles to a 2-D grid (tallest building per cell).  The 3-D
    voxelizer, however, fills the *interior volume* of each building
    using a signed-distance / winding-number method that can mark cells
    as BUILDING even where no surface triangle projects.  This creates
    two classes of mismatched cells:

    *  ``bid_grid[r, c] == 0`` but ``voxel_grid[r, c, :]`` has building
       voxels  →  13 % of the mesh faces get ``building_id = 0``, and
       those GVI values are lost.
    *  ``bid_grid[r, c]`` belongs to a neighbouring building, not the one
       whose voxels actually occupy that column  →  misattributed GVI.

    This helper propagates the nearest non-zero building-ID into every
    cell that contains building voxels but has ``bid == 0``, using
    ``scipy.ndimage.distance_transform_edt``.

    Parameters
    ----------
    bid_grid : (R, C) int array
        Original building-ID grid (north-up, tallest-wins).
    voxel_grid : (R, C, Z) int16 array
        3-D voxel grid (same axes 0-1 as *bid_grid*).
    building_code : int
        Class code for buildings / bridges in *voxel_grid*.

    Returns
    -------
    bid_filled : (R, C) int32  –  copy of *bid_grid* with gaps filled.
    """
    from scipy.ndimage import distance_transform_edt

    has_building_voxels = np.any(voxel_grid == building_code, axis=2)
    has_id = bid_grid > 0
    needs_id = has_building_voxels & ~has_id

    n_gaps = int(needs_id.sum())
    if n_gaps == 0:
        return bid_grid.copy()

    # EDT: for each cell without a building-ID, find the nearest cell
    # that *does* have one.  `indices` contains the (row, col) of that
    # nearest source cell.
    _, indices = distance_transform_edt(~has_id, return_indices=True)

    bid_filled = bid_grid.copy()
    bid_filled[needs_id] = bid_grid[
        indices[0][needs_id], indices[1][needs_id]
    ]
    print(f"  Filled {n_gaps} building-ID gaps "
          f"({n_gaps / has_building_voxels.sum() * 100:.1f}% of building columns)")
    return bid_filled


def compute_building_gvi(city, **kwargs):
    """Return a DataFrame with per-building average surface GVI.

    Parameters
    ----------
    city : voxcity VoxCity object
        The assembled voxel city model.
    **kwargs
        Forwarded to ``get_surface_view_factor``. Useful overrides:
        - N_azimuth (int): azimuth ray count (default 120)
        - N_elevation (int): elevation ray count (default 20)
        - progress_report (bool): print progress (default True)

    Returns
    -------
    df : pd.DataFrame
        Columns: building_id, mean_gvi, face_count
    mesh : trimesh.Trimesh
        Building mesh with per-face GVI in metadata['gvi_values'].
    """
    from voxcity.simulator_gpu import get_surface_view_factor

    # Green targets: tree canopy (-2), rangeland (2), tree (5),
    # moss/lichen (6), wetland (7), mangrove (8)
    green_targets = (-2, 2, 5, 6, 7, 8)

    defaults = dict(
        target_values=green_targets,
        inclusion_mode=True,
        value_name="gvi_values",
        progress_report=True,
        building_class_id=-3,
    )
    defaults.update(kwargs)

    # ---- Fill gaps in the building-ID grid --------------------------
    filled_ids = _fill_building_id_gaps(
        city.buildings.ids,
        city.voxels.classes,
        building_code=defaults.get("building_class_id", -3),
    )

    # WORKAROUND for voxcity.geoprocessor.mesh:
    # create_voxel_mesh internally uses ensure_orientation on building_id_grid
    # which flips the grid vertically, mismatching the unflipped 3D array index.
    original_ids = np.copy(city.buildings.ids)
    city.buildings.ids = np.flipud(filled_ids)
    try:
        mesh = get_surface_view_factor(city, mode='green', **defaults)
    finally:
        city.buildings.ids = original_ids

    if mesh is None:
        raise RuntimeError("No building surface faces were generated.")

    gvi_values = mesh.metadata["gvi_values"]
    building_ids = mesh.metadata.get("building_id")
    if building_ids is None:
        raise RuntimeError(
            "Building-ID metadata is missing. "
            "Ensure the VoxCity object contains a building_id grid."
        )

    # Face normals: vertical (wall) faces have |normal_z| ≈ 0
    face_normals = np.array(mesh.face_normals)
    is_vertical = np.abs(face_normals[:, 2]) < 0.1

    # Build per-building summary.
    # Prefer vertical (wall) faces, but fall back to ALL faces for
    # buildings that have no vertical faces (e.g. low-rise buildings
    # that are only 1 voxel tall at the given meshsize).
    valid_vert = ~np.isnan(gvi_values) & is_vertical
    valid_all  = ~np.isnan(gvi_values)

    # Gather all building IDs that have at least one valid face
    all_bids_vert = set(np.unique(building_ids[valid_vert])) - {0}
    all_bids_all  = set(np.unique(building_ids[valid_all]))  - {0}
    fallback_bids = all_bids_all - all_bids_vert   # no vertical faces

    if fallback_bids:
        print(f"  {len(fallback_bids)} buildings have no vertical faces "
              f"→ using all faces for those buildings")

    ids_vert  = building_ids[valid_vert]
    gvi_vert  = gvi_values[valid_vert]
    ids_all   = building_ids[valid_all]
    gvi_all   = gvi_values[valid_all]

    rows = []
    for bid in sorted(all_bids_all):
        if bid in fallback_bids:
            mask = ids_all == bid
            rows.append({
                "building_id": int(bid),
                "mean_gvi": float(np.mean(gvi_all[mask])),
                "face_count": int(mask.sum()),
            })
        else:
            mask = ids_vert == bid
            rows.append({
                "building_id": int(bid),
                "mean_gvi": float(np.mean(gvi_vert[mask])),
                "face_count": int(mask.sum()),
            })

    df = pd.DataFrame(rows).sort_values("mean_gvi", ascending=False).reset_index(drop=True)
    return df, mesh


def filter_buildings_to_core_rect(
    df, city, core_rect, collection_lod2=None,
):
    """Keep only buildings whose centroid lies inside *core_rect*.

    Two-pass strategy:
    1.  Use the CityGML mesh vertex centroid for each building parsed in
        *collection_lod2* to decide if it belongs to the core rect.  This
        catches **all** buildings regardless of whether the voxelisation
        produced exposed faces for them.
    2.  If *collection_lod2* is not supplied, fall back to the voxel-grid
        centroid (legacy behaviour — may lose short/overlapped buildings).

    Buildings present in the CityGML parse but absent from *df* (i.e.
    buildings that produced zero SVF faces) are appended with
    ``mean_gvi=0.0, face_count=0``.

    Parameters
    ----------
    df : pd.DataFrame
        Per-building GVI table (must contain ``building_id``).
    city : VoxCity
        The assembled voxel city model.
    core_rect : list[(lon, lat), …]
        The core target rectangle [SW, NW, NE, SE].
    collection_lod2 : CityGmlCollection | None
        The parsed CityGML collection (same order used for bid assignment).

    Returns
    -------
    df_filtered : pd.DataFrame
    """
    from voxcitygml.citygml.coordinates import swap_coordinates_3d

    core_lons = [v[0] for v in core_rect]
    core_lats = [v[1] for v in core_rect]
    lon_min, lon_max = min(core_lons), max(core_lons)
    lat_min, lat_max = min(core_lats), max(core_lats)

    def _in_core(lon, lat):
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    # ---- Strategy A: CityGML mesh centroids ----------------------------
    if collection_lod2 is not None:
        all_meshes = list(collection_lod2.buildings) + list(collection_lod2.bridges)
        core_bids = set()           # bid values inside core rect
        for idx, m in enumerate(all_meshes):
            bid = idx + 1
            if len(m.vertices) == 0:
                continue
            pts_ll = swap_coordinates_3d(m.vertices)       # (lat,lon,z) → (lon,lat,z)
            clon, clat = float(pts_ll[:, 0].mean()), float(pts_ll[:, 1].mean())
            if _in_core(clon, clat):
                core_bids.add(bid)

        # Filter SVF rows to core
        df_core = df[df["building_id"].isin(core_bids)].copy()

        # Backfill: buildings in core that have no SVF row → GVI 0
        svf_bids = set(df_core["building_id"].unique())
        missing_bids = core_bids - svf_bids
        if missing_bids:
            backfill = pd.DataFrame({
                "building_id": sorted(missing_bids),
                "mean_gvi": 0.0,
                "face_count": 0,
            })
            df_core = pd.concat([df_core, backfill], ignore_index=True)
            print(f"  Backfilled {len(missing_bids)} buildings with GVI=0 "
                  f"(no SVF faces produced)")

        df_core = df_core.sort_values("mean_gvi", ascending=False).reset_index(drop=True)
        print(f"  Core-area filter: {len(df)} SVF → {len(df_core)} buildings "
              f"({len(core_bids)} in CityGML)")
        return df_core

    # ---- Strategy B: voxel-grid centroid (fallback) --------------------
    from voxcitygml.grid_utils import compute_grid_params

    rect = city.extras["rectangle_vertices"]
    meshsize = city.voxels.meta.meshsize
    gp = compute_grid_params(rect, meshsize)
    bid_grid = city.buildings.ids

    keep_ids = set()
    for bid in df["building_id"].unique():
        rows_arr, cols_arr = np.where(bid_grid == bid)
        if len(rows_arr) == 0:
            continue
        lon, lat = gp.cell_centre(int(rows_arr.mean()), int(cols_arr.mean()))
        if _in_core(lon, lat):
            keep_ids.add(int(bid))

    df_filtered = df[df["building_id"].isin(keep_ids)].reset_index(drop=True)
    print(f"  Core-area filter: {len(df)} → {len(df_filtered)} buildings")
    return df_filtered


def build_footprint_gdf(cfg, collection_buildings, collection_bridges, df):
    """Build a GeoDataFrame of LOD1 building footprints coloured by GVI.

    Parameters
    ----------
    cfg : VoxelizerConfig
    collection_buildings : list[Mesh3D]
        Building meshes from the pipeline's CityGML parse (LOD2), used for
        the feature_id → building_id mapping.
    collection_bridges : list[Mesh3D]
        Bridge meshes from the same parse.
    df : pd.DataFrame
        Per-building GVI table with columns ``building_id``, ``mean_gvi``.

    Returns
    -------
    gdf : gpd.GeoDataFrame  (CRS EPSG:4326)
        One row per building with columns: building_id, gml_id, mean_gvi,
        geometry (footprint polygon).
    """
    from voxcitygml.citygml.parser import parse_citygml_directory
    from voxcitygml.citygml.coordinates import (
        create_rectangle, swap_coordinates_3d,
    )

    # --- bid → gml_id mapping from the original (LOD2) parse order ------
    all_meshes_lod2 = list(collection_buildings) + list(collection_bridges)
    bid_to_gmlid = {}
    for idx, m in enumerate(all_meshes_lod2):
        bid_to_gmlid[idx + 1] = m.feature_id

    # --- Parse CityGML again with LOD1 (buildings only) -----------------
    buffered_rect = create_rectangle(
        cfg.center_lon, cfg.center_lat,
        cfg.size_meters + 2 * cfg.buffer_meters,
    )
    print("\nParsing CityGML (LOD1) for building footprints...")
    citygml_paths = cfg.citygml_path if isinstance(cfg.citygml_path, list) else [cfg.citygml_path]
    from voxcitygml.models import CityGMLMeshCollection
    lod1_collection = parse_citygml_directory(
        citygml_paths[0],
        rectangle_vertices=buffered_rect,
        n_workers=cfg.n_workers,
        feature_types=['building'],
        building_lod=1,
    )
    for extra_path in citygml_paths[1:]:
        extra = parse_citygml_directory(
            extra_path,
            rectangle_vertices=buffered_rect,
            n_workers=cfg.n_workers,
            feature_types=['building'],
            building_lod=1,
        )
        lod1_collection.merge(extra)

    # gml_id → footprint polygon from LOD1 triangulated faces
    gmlid_to_footprint = {}
    for m in lod1_collection.buildings:
        if len(m.vertices) < 3 or len(m.faces) == 0:
            continue
        pts_ll = swap_coordinates_3d(m.vertices)  # → (lon, lat, z)
        # Project each triangle to 2D and union them.
        # Wall triangles project to degenerate lines (area ≈ 0) and are
        # automatically excluded; roof/ground triangles give the true shape.
        triangles = []
        for face in m.faces:
            tri_2d = pts_ll[face, :2]  # (3, 2) — lon, lat
            poly = Polygon(tri_2d)
            if poly.is_valid and poly.area > 0:
                triangles.append(poly)
        if not triangles:
            continue
        footprint = unary_union(triangles)
        if footprint.is_empty or footprint.geom_type == 'Point':
            continue
        gmlid_to_footprint[m.feature_id] = footprint

    # --- Join: bid → gml_id → footprint, then merge GVI ----------------
    records = []
    for _, row in df.iterrows():
        bid = int(row["building_id"])
        gml_id = bid_to_gmlid.get(bid)
        if gml_id is None:
            continue
        footprint = gmlid_to_footprint.get(gml_id)
        if footprint is None:
            continue
        records.append({
            "building_id": bid,
            "gml_id": gml_id,
            "mean_gvi": row["mean_gvi"],
            "geometry": footprint,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Deduplicate: when overlapping CityGML directories produce two bids
    # for the same gml_id, keep only the entry with the most faces
    # (most reliable GVI estimate).  Ties break toward higher GVI.
    if "gml_id" in gdf.columns and gdf["gml_id"].duplicated().any():
        n_before = len(gdf)
        gdf = (
            gdf.sort_values(["face_count", "mean_gvi"], ascending=False)
               .drop_duplicates(subset="gml_id", keep="first")
               .reset_index(drop=True)
        )
        print(f"  Deduplicated {n_before - len(gdf)} duplicate gml_id "
              f"footprints ({n_before} → {len(gdf)})")

    print(f"  Built footprint GeoDataFrame: {len(gdf)} buildings")
    return gdf


# ======================================================================
# City GVI pipeline
# ======================================================================
def run_city_gvi(
    city_label,
    citygml_path,
    center_lon,
    center_lat,
    target_size=2000,
    gvi_buffer=1000,
    meshsize=2.0,
    output_dir=None,
    building_lod=2,
    gee_project=None,
):
    """Run the full per-building GVI pipeline for one city.

    Returns
    -------
    df : pd.DataFrame
        Per-building GVI with columns building_id, mean_gvi, face_count.
    """
    if output_dir is None:
        output_dir = f"output/building_gvi_{city_label.lower().replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    phase_times = {}  # collect per-phase wall-clock times

    # --- 1. Voxelization ------------------------------------------------
    t_phase_start = time.perf_counter()
    cfg = VoxelizerConfig(
        citygml_path=citygml_path,
        center_lon=center_lon,
        center_lat=center_lat,
        size_meters=target_size + 2 * gvi_buffer,
        meshsize=meshsize,
        gee_project=gee_project,
        canopy_height_source="Static",
        save_output=True,
        output_dir=output_dir,
        n_workers=4,
        max_voxel_ram_mb=4000,
        building_lod=building_lod,
        occupancy_threshold=0.5,
        occupancy_subdivisions=4,
    )

    city = VoxCityGML(cfg).run()
    phase_times["1_voxelization"] = time.perf_counter() - t_phase_start
    print(f"\n  Phase 1 (CityGML extraction & voxelization): "
          f"{phase_times['1_voxelization']:.1f} s")

    # Core target rectangle (for filtering results)
    from voxcitygml.citygml.coordinates import create_rectangle
    core_rect = create_rectangle(cfg.center_lon, cfg.center_lat, target_size)

    # Retrieve the parsed CityGML collection from the pipeline
    # (stored in city.extras to avoid redundant re-parsing)
    collection_lod2 = city.extras.get("citygml_collection")
    if collection_lod2 is None:
        raise RuntimeError(
            "CityGML collection not found in city.extras. "
            "Ensure you are using an up-to-date voxcitygml pipeline."
        )

    # --- 2. Surface GVI computation -------------------------------------
    t_phase_start = time.perf_counter()
    print("\n" + "=" * 60)
    print("Computing building-surface Green View Index …")
    print("=" * 60)

    df, mesh = compute_building_gvi(
        city,
        N_azimuth=120,
        N_elevation=20,
        obj_export=True,
        output_directory=output_dir,
        output_file_name="building_surface_gvi",
        colormap="viridis",
        vmin=0.0,
        vmax=0.2,
    )
    phase_times["2_gvi_computation"] = time.perf_counter() - t_phase_start
    print(f"\n  Phase 2 (Building GVI computation): "
          f"{phase_times['2_gvi_computation']:.1f} s")

    # --- 3. Filter to core target area -----------------------------------
    t_phase_start = time.perf_counter()
    df = filter_buildings_to_core_rect(df, city, core_rect,
                                       collection_lod2=collection_lod2)

    # --- 4. Print results ------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Per-building average surface GVI  ({len(df)} buildings)")
    print("=" * 60)
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))

    # --- 5. Save CSV -----------------------------------------------------
    csv_path = f"{output_dir}/building_gvi.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-building GVI table to {csv_path}")

    # --- 6. Quick statistics ---------------------------------------------
    print(f"\nOverall statistics (weighted by face count):")
    total_faces = df["face_count"].sum()
    weighted_mean = (df["mean_gvi"] * df["face_count"]).sum() / total_faces if total_faces else 0
    print(f"  Buildings        : {len(df)}")
    print(f"  Total faces      : {total_faces}")
    print(f"  Weighted mean GVI: {weighted_mean:.4f}")
    print(f"  Min building GVI : {df['mean_gvi'].min():.4f}")
    print(f"  Max building GVI : {df['mean_gvi'].max():.4f}")
    print(f"  Median           : {df['mean_gvi'].median():.4f}")

    phase_times["3_aggregation"] = time.perf_counter() - t_phase_start
    print(f"\n  Phase 3 (Aggregation by individual buildings): "
          f"{phase_times['3_aggregation']:.1f} s")

    # --- 7. Build footprint GeoDataFrame & visualize on basemap ----------
    gdf = build_footprint_gdf(
        cfg,
        collection_lod2.buildings,
        collection_lod2.bridges,
        df,
    )

    if len(gdf) > 0:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import contextily as ctx
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        out_dir = output_dir

        # Filter to footprints completely inside the target area
        target_poly = Polygon(core_rect)
        gdf = gdf[gdf.geometry.within(target_poly)].copy()
        print(f"  Footprints fully within target area: {len(gdf)}")

        # ---------------------------------------------------------------
        # 7a. Basemap with building footprints coloured by GVI
        # ---------------------------------------------------------------
        print("\nVisualizing per-building GVI on basemap...")
        gdf_web = gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf_web.plot(
            column="mean_gvi", ax=ax, alpha=0.8, cmap="viridis",
            vmin=0.0, vmax=0.2, legend=True,
            legend_kwds={"label": "Mean Green View Index"},
            edgecolor="gray", linewidth=0.3,
        )
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        ax.set_axis_off()
        ax.set_title(f"Per-building average Green View Index \u2014 {city_label}")
        plt.tight_layout()
        map_path = f"{out_dir}/building_gvi_map.png"
        fig.savefig(map_path, dpi=200, bbox_inches="tight")
        print(f"  Saved basemap figure to {map_path}")
        plt.close(fig)

        # ---------------------------------------------------------------
        # 7b. Comprehensive statistical analysis figure
        # ---------------------------------------------------------------
        print("\nGenerating comprehensive statistical analysis figure...")
        gvi = df["mean_gvi"].values
        fc = df["face_count"].values

        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

        # --- (a) Histogram with KDE ---
        ax1 = fig.add_subplot(gs[0, 0])
        bins = np.linspace(0, max(0.5, gvi.max() * 1.05), 30)
        ax1.hist(gvi, bins=bins, color="#6EB257", edgecolor="white",
                 alpha=0.85, density=True, label="Histogram")
        # KDE overlay
        from scipy.stats import gaussian_kde
        if len(gvi) > 3 and gvi.std() > 0:
            kde = gaussian_kde(gvi, bw_method=0.3)
            x_kde = np.linspace(0, gvi.max() * 1.1, 200)
            ax1.plot(x_kde, kde(x_kde), color="#2D6A1E", lw=2, label="KDE")
        ax1.axvline(gvi.mean(), color="red", ls="--", lw=1.5,
                    label=f"Mean = {gvi.mean():.4f}")
        ax1.axvline(np.median(gvi), color="blue", ls=":", lw=1.5,
                    label=f"Median = {np.median(gvi):.4f}")
        ax1.set_xlabel("Mean GVI")
        ax1.set_ylabel("Density")
        ax1.set_title("(a) Distribution of building mean GVI")
        ax1.legend(fontsize=8)

        # --- (b) Cumulative distribution (ECDF) ---
        ax2 = fig.add_subplot(gs[0, 1])
        sorted_gvi = np.sort(gvi)
        ecdf = np.arange(1, len(sorted_gvi) + 1) / len(sorted_gvi)
        ax2.step(sorted_gvi, ecdf, color="#2D6A1E", lw=2)
        ax2.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax2.axvline(np.median(gvi), color="blue", ls=":", lw=1.2, alpha=0.6)
        ax2.set_xlabel("Mean GVI")
        ax2.set_ylabel("Cumulative proportion")
        ax2.set_title("(b) Empirical CDF")
        ax2.set_xlim(left=0)

        # --- (c) Box plot + violin ---
        ax3 = fig.add_subplot(gs[0, 2])
        parts = ax3.violinplot(gvi, positions=[1], showmedians=False,
                               showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#A8D99C")
            pc.set_alpha(0.6)
        bp = ax3.boxplot(gvi, positions=[1], widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor="#6EB257", alpha=0.7),
                         medianprops=dict(color="red", lw=2),
                         whiskerprops=dict(color="#333"),
                         flierprops=dict(marker="o", markersize=3,
                                         markerfacecolor="#888", alpha=0.5))
        ax3.set_ylabel("Mean GVI")
        ax3.set_title("(c) Box & violin plot")
        ax3.set_xticks([1])
        ax3.set_xticklabels(["All buildings"])

        # --- (d) GVI vs. face count (building size proxy) ---
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(fc, gvi, s=12, alpha=0.5, c=gvi, cmap="viridis",
                    edgecolors="gray", linewidths=0.3)
        ax4.set_xlabel("Vertical face count (building size proxy)")
        ax4.set_ylabel("Mean GVI")
        ax4.set_title("(d) GVI vs. building size")
        ax4.set_xscale("log")

        # --- (e) Percentile bar chart ---
        ax5 = fig.add_subplot(gs[1, 1])
        pcts = [5, 10, 25, 50, 75, 90, 95]
        pct_vals = np.percentile(gvi, pcts)
        bars = ax5.barh([f"P{p}" for p in pcts], pct_vals,
                        color="#6EB257", edgecolor="white", height=0.6)
        for bar, v in zip(bars, pct_vals):
            ax5.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}", va="center", fontsize=8)
        ax5.set_xlabel("Mean GVI")
        ax5.set_title("(e) Percentile values")
        ax5.set_xlim(right=max(pct_vals) * 1.3)

        # --- (f) Top-20 / bottom-20 buildings ---
        ax6 = fig.add_subplot(gs[1, 2])
        top20 = df.nlargest(20, "mean_gvi")
        bot20 = df.nsmallest(20, "mean_gvi")
        combined = pd.concat([top20, bot20])
        y_labels = [f"B{bid}" for bid in combined["building_id"]]
        colors = ["#2D6A1E" if v >= np.median(gvi) else "#C44E52"
                  for v in combined["mean_gvi"]]
        ax6.barh(range(len(combined)), combined["mean_gvi"].values,
                 color=colors, edgecolor="white", height=0.7)
        ax6.set_yticks(range(len(combined)))
        ax6.set_yticklabels(y_labels, fontsize=6)
        ax6.set_xlabel("Mean GVI")
        ax6.set_title("(f) Top-20 & bottom-20 buildings")
        ax6.invert_yaxis()

        # --- (g) Weighted vs. unweighted comparison ---
        ax7 = fig.add_subplot(gs[2, 0])
        weighted_mean = (gvi * fc).sum() / fc.sum() if fc.sum() else 0
        unweighted_mean = gvi.mean()
        vals = [unweighted_mean, weighted_mean]
        bar_labels = ["Unweighted\nmean", "Face-count\nweighted mean"]
        ax7.bar(bar_labels, vals, color=["#6EB257", "#2D6A1E"],
                edgecolor="white", width=0.5)
        for i, v in enumerate(vals):
            ax7.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
        ax7.set_ylabel("Mean GVI")
        ax7.set_title("(g) Unweighted vs. weighted mean")
        ax7.set_ylim(0, max(vals) * 1.4)

        # --- (h) Binned spatial map (GVI quintiles on basemap) ---
        ax8 = fig.add_subplot(gs[2, 1])
        gdf_web_q = gdf_web.copy()
        # Use qcut without labels first to determine the actual bin count,
        # then assign matching labels (handles duplicate bin edges).
        _qcut_raw = pd.qcut(gdf_web_q["mean_gvi"], q=5, duplicates="drop")
        n_bins = _qcut_raw.cat.categories.size
        _labels = ["Q1 (low)"] + [f"Q{i+1}" for i in range(1, n_bins - 1)] + ["Q{} (high)".format(n_bins)] if n_bins > 1 else ["All"]
        gdf_web_q["quintile"] = pd.qcut(gdf_web_q["mean_gvi"], q=5,
                                         labels=_labels,
                                         duplicates="drop")
        gdf_web_q.plot(column="quintile", ax=ax8, alpha=0.8,
                       categorical=True, legend=True, cmap="RdYlGn",
                       edgecolor="gray", linewidth=0.3,
                       legend_kwds={"fontsize": 7, "title": "Quintile",
                                    "title_fontsize": 8})
        ctx.add_basemap(ax8, source=ctx.providers.CartoDB.Positron)
        ax8.set_axis_off()
        ax8.set_title("(h) GVI quintile map")

        # --- (i) Summary statistics text box ---
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")
        iqr = np.percentile(gvi, 75) - np.percentile(gvi, 25)
        stats_text = (
            f"{'Buildings':.<28s} {len(df):>8d}\n"
            f"{'Vertical faces':.<28s} {int(fc.sum()):>8d}\n"
            f"{'─' * 38}\n"
            f"{'Mean':.<28s} {gvi.mean():>8.4f}\n"
            f"{'Std dev':.<28s} {gvi.std():>8.4f}\n"
            f"{'Median':.<28s} {np.median(gvi):>8.4f}\n"
            f"{'IQR (Q3−Q1)':.<28s} {iqr:>8.4f}\n"
            f"{'Min':.<28s} {gvi.min():>8.4f}\n"
            f"{'Max':.<28s} {gvi.max():>8.4f}\n"
            f"{'Skewness':.<28s} {pd.Series(gvi).skew():>8.4f}\n"
            f"{'Kurtosis':.<28s} {pd.Series(gvi).kurtosis():>8.4f}\n"
            f"{'─' * 38}\n"
            f"{'Weighted mean (by faces)':.<28s} {weighted_mean:>8.4f}\n"
            f"{'Buildings with GVI=0':.<28s} {int((gvi == 0).sum()):>8d}\n"
            f"{'Buildings with GVI>0.1':.<28s} {int((gvi > 0.1).sum()):>8d}\n"
            f"{'Buildings with GVI>0.2':.<28s} {int((gvi > 0.2).sum()):>8d}\n"
        )
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
                 fontsize=10, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F7EC",
                           edgecolor="#6EB257", alpha=0.9))
        ax9.set_title("(i) Summary statistics")

        fig.suptitle(
            f"Green View Index — Statistical Analysis  "
            f"({city_label}, {target_size}m target, {gvi_buffer}m buffer, "
            f"{meshsize}m voxel)",
            fontsize=14, fontweight="bold", y=0.98,
        )

        stats_path = f"{out_dir}/building_gvi_stats.png"
        fig.savefig(stats_path, dpi=200, bbox_inches="tight")
        print(f"  Saved statistical analysis figure to {stats_path}")
        plt.close(fig)
    else:
        print("No footprints matched — skipping basemap visualisation.")

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

    return df


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    run_city_gvi(
        city_label="Tokyo",
        citygml_path=[
            "/path/to/citygml_dataset_1",  # Replace with your CityGML path(s)
            "/path/to/citygml_dataset_2",
        ],
        center_lon=139.767125,
        center_lat=35.681236,
    )
