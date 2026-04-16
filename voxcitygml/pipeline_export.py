"""
Pipeline variant that runs the full VoxCityGML pipeline and exports
both mesh and voxel OBJ files in a shared coordinate system.

This reuses the existing pipeline logic and hooks into it after the
CityGML parse (for mesh export) and after voxelization (for voxel export).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from .models import VoxelizerConfig, CityGMLMeshCollection, resolve_citygml_paths
from .citygml.coordinates import create_rectangle
from .citygml.parser import parse_citygml_directory, merge_terrain_meshes
from .terrain.processor import terrain_meshes_to_dem_grid
from .landcover.processor import get_land_cover_grid
from .buildings.processor import meshes_to_building_grids
from .canopy.processor import get_canopy_grids
from .voxelizer3d import voxelize_citygml_meshes
from .export_obj import export_meshes_obj, export_voxels_obj, export_per_category_voxels_obj, export_landcover_obj


def run_and_export(
    cfg: VoxelizerConfig,
    mesh_basename: str = "meshes",
    voxel_basename: str = "voxels",
    per_category_basename: str = "mesh_voxels",
    landcover_basename: str = "landcover",
    watertight_meshes: bool = True,
):
    """Run the full pipeline and export OBJ files.

    Parameters
    ----------
    cfg : VoxelizerConfig
        Pipeline configuration.
    mesh_basename : str
        Base filename for the mesh OBJ/MTL (default ``meshes``).
    voxel_basename : str
        Base filename for the voxel OBJ/MTL (default ``voxels``).
    per_category_basename : str | None
        Base filename for the per-category voxelized mesh OBJ/MTL
        (default ``mesh_voxels``).  Set to ``None`` to skip.
    landcover_basename : str | None
        Base filename for the 2-D land-cover mesh OBJ/MTL
        (default ``landcover``).  Set to ``None`` to skip.
    watertight_meshes : bool
        Apply the watertight cascade to buildings before mesh export.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    rectangle = create_rectangle(cfg.center_lon, cfg.center_lat, cfg.size_meters)
    buffered_rect = create_rectangle(
        cfg.center_lon, cfg.center_lat,
        cfg.size_meters + 2 * cfg.buffer_meters,
    )

    # GEE init
    if cfg.gee_project:
        from voxcity.downloader.gee import initialize_earth_engine
        initialize_earth_engine(project=cfg.gee_project)

    # Auto-select data sources
    if cfg.land_cover_source is None or cfg.canopy_height_source is None:
        from voxcity.generator.api import auto_select_data_sources
        auto = auto_select_data_sources(rectangle)
        if cfg.land_cover_source is None:
            cfg.land_cover_source = auto['land_cover_source']
        if cfg.canopy_height_source is None:
            cfg.canopy_height_source = auto['canopy_height_source']

    # Resolve citygml_path (auto-discover datasets in parent folders)
    citygml_paths = resolve_citygml_paths(cfg.citygml_path)

    print("=" * 60)
    print("VoxCityGML Pipeline  +  OBJ Export")
    print("=" * 60)
    print(f"  CityGML path(s):    {', '.join(citygml_paths)}")
    print(f"  Centre:             ({cfg.center_lon:.6f}, {cfg.center_lat:.6f})")
    print(f"  Area:               {cfg.size_meters} m")
    print(f"  Voxel size:         {cfg.meshsize} m")
    print(f"  Workers:            {cfg.n_workers}")
    if cfg.building_lod is not None:
        print(f"  Building LOD:       {cfg.building_lod}")
    print("=" * 60)

    # ── Step 1 – Parse CityGML ───────────────────────────────────────
    print("\n[1/6] Parsing CityGML data...")
    collection = parse_citygml_directory(
        citygml_paths[0],
        rectangle_vertices=buffered_rect,
        n_workers=cfg.n_workers,
        feature_types=['terrain', 'building', 'bridge', 'vegetation'],
        building_lod=cfg.building_lod,
        dem_path=cfg.dem_path,
    )
    for extra_path in citygml_paths[1:]:
        print(f"  Parsing additional CityGML directory: {extra_path}")
        extra_collection = parse_citygml_directory(
            extra_path,
            rectangle_vertices=buffered_rect,
            n_workers=cfg.n_workers,
            feature_types=['terrain', 'building', 'bridge', 'vegetation'],
            building_lod=cfg.building_lod,
        )
        collection.merge(extra_collection)
    if collection.terrain:
        collection.terrain = merge_terrain_meshes(collection.terrain)

    # ── Step 2 – DEM ─────────────────────────────────────────────────
    print("\n[2/6] Creating DEM grid from terrain TIN...")
    dem_grid = terrain_meshes_to_dem_grid(
        collection.terrain, rectangle, cfg.meshsize,
    )
    print(f"  DEM grid shape: {dem_grid.shape}, "
          f"elevation range: [{dem_grid.min():.1f}, {dem_grid.max():.1f}] m")

    # ── Step 3 – Land cover ──────────────────────────────────────────
    print("\n[3/6] Acquiring land cover grid...")
    land_cover_grid = get_land_cover_grid(
        rectangle, cfg.meshsize, cfg.land_cover_source, cfg.output_dir,
        citygml_path=citygml_paths[0],
    )
    if land_cover_grid.shape != dem_grid.shape:
        from scipy.ndimage import zoom
        factor = (dem_grid.shape[0] / land_cover_grid.shape[0],
                  dem_grid.shape[1] / land_cover_grid.shape[1])
        land_cover_grid = zoom(land_cover_grid, factor, order=0).astype(land_cover_grid.dtype)

    # ── Step 4 – Buildings & bridges (2-D grids for metadata) ────────
    print("\n[4/6] Rasterising buildings and bridges...")
    building_height_grid, building_min_height_grid, building_id_grid = \
        meshes_to_building_grids(
            collection.buildings, collection.bridges,
            rectangle, cfg.meshsize, dem_grid,
        )

    # ── Step 5 – Canopy ──────────────────────────────────────────────
    print("\n[5/6] Acquiring canopy height data...")
    canopy_top, canopy_bottom = get_canopy_grids(
        rectangle, cfg.meshsize,
        cfg.canopy_height_source, cfg.land_cover_source,
        land_cover_grid, dem_grid, cfg.output_dir,
        vegetation_meshes=collection.vegetation,
        trunk_height_ratio=cfg.trunk_height_ratio,
        static_tree_height=cfg.static_tree_height,
    )
    if canopy_top.shape != dem_grid.shape:
        from scipy.ndimage import zoom
        factor = (dem_grid.shape[0] / canopy_top.shape[0],
                  dem_grid.shape[1] / canopy_top.shape[1])
        canopy_top = zoom(canopy_top, factor, order=1)
        canopy_bottom = zoom(canopy_bottom, factor, order=1)

    # ── Step 6 – 3-D Voxelization ────────────────────────────────────
    print("\n[6/6] Voxelising all components...")
    voxel_grid = voxelize_citygml_meshes(
        collection,
        rectangle,
        cfg.center_lon,
        cfg.center_lat,
        cfg.meshsize,
        dem_grid=dem_grid,
        land_cover_grid=land_cover_grid,
        canopy_top=canopy_top,
        canopy_bottom=canopy_bottom,
        land_cover_source=cfg.land_cover_source,
        trunk_height_ratio=cfg.trunk_height_ratio,
        max_voxel_ram_mb=cfg.max_voxel_ram_mb,
        occupancy_threshold=cfg.occupancy_threshold,
        occupancy_subdivisions=cfg.occupancy_subdivisions,
    )

    # ── Export 1: Voxel OBJ (greedy meshed) ──────────────────────────
    # Export voxels first to obtain Grid3DParams for the mesh transform
    print("\n" + "=" * 60)
    print("Exporting voxel OBJ (greedy meshed)...")
    print("=" * 60)
    voxel_obj, gp = export_voxels_obj(
        voxel_grid,
        collection,
        rectangle,
        center_lon=cfg.center_lon,
        center_lat=cfg.center_lat,
        meshsize=cfg.meshsize,
        output_dir=cfg.output_dir,
        basename=voxel_basename,
    )

    # ── Export 2: Mesh OBJ (same coordinate system as voxel OBJ) ─────
    print("\n" + "=" * 60)
    print("Exporting mesh OBJ (after watertight, before voxelization)...")
    print("=" * 60)
    mesh_obj, mesh_groups = export_meshes_obj(
        collection,
        center_lon=cfg.center_lon,
        center_lat=cfg.center_lat,
        output_dir=cfg.output_dir,
        gp=gp,
        basename=mesh_basename,
        watertight=watertight_meshes,
        voxel_size=cfg.meshsize,
    )

    # ── Export 3: Per-category voxelized mesh OBJ ────────────────────
    per_cat_obj = None
    if per_category_basename:
        print("\n" + "=" * 60)
        print("Exporting per-category voxelized mesh OBJ...")
        print("=" * 60)
        per_cat_obj, _ = export_per_category_voxels_obj(
            collection,
            rectangle,
            center_lon=cfg.center_lon,
            center_lat=cfg.center_lat,
            meshsize=cfg.meshsize,
            output_dir=cfg.output_dir,
            basename=per_category_basename,
            max_voxel_ram_mb=cfg.max_voxel_ram_mb,
            occupancy_threshold=cfg.occupancy_threshold,
            occupancy_subdivisions=cfg.occupancy_subdivisions,
            mesh_groups=mesh_groups,
        )

    # ── Export 4: Land-cover 2-D flat mesh OBJ ───────────────────────
    lc_obj = None
    if landcover_basename:
        print("\n" + "=" * 60)
        print("Exporting land-cover 2-D mesh OBJ...")
        print("=" * 60)
        lc_obj = export_landcover_obj(
            land_cover_grid,
            cfg.land_cover_source,
            dem_grid,
            gp,
            output_dir=cfg.output_dir,
            basename=landcover_basename,
            citygml_path=citygml_paths[0],
            rectangle_vertices=rectangle,
            center_lon=cfg.center_lon,
            center_lat=cfg.center_lat,
        )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"  Mesh OBJ:        {mesh_obj}")
    print(f"  Voxel OBJ:       {voxel_obj}")
    if per_cat_obj:
        print(f"  Per-cat Voxel:   {per_cat_obj}")
    if lc_obj:
        print(f"  Land cover:      {lc_obj}")
    print(f"\nAll files share the same coordinate system.")
    print(f"  OBJ X = row direction (south), Y = col (east), Z = elevation")
    print(f"  Origin at grid corner, units in metres.")
    print(f"Load all in Rhino, Blender, MeshLab, etc. for overlay.")
    print("=" * 60)

    return mesh_obj, voxel_obj, per_cat_obj, lc_obj
