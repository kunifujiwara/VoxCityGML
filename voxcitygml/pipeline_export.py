"""
Pipeline variant that runs the full VoxCityGML pipeline and exports
both mesh and voxel OBJ files in a shared coordinate system.

This reuses the existing pipeline logic and hooks into it after the
CityGML parse (for mesh export) and after voxelization (for voxel export).
"""

import matplotlib
matplotlib.use("Agg")

from .models import VoxelizerConfig
from .pipeline import run_core
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
    art = run_core(cfg)
    collection = art.collection
    rectangle = art.rectangle
    land_cover_grid = art.land_cover_grid
    dem_grid = art.dem_grid
    voxel_grid = art.voxel_grid
    citygml_paths = art.citygml_paths
    center_lon, center_lat = art.center_lon, art.center_lat

    # ── Export 1: Voxel OBJ (greedy meshed) ──────────────────────────
    # Export voxels first to obtain Grid3DParams for the mesh transform
    print("\n" + "=" * 60)
    print("Exporting voxel OBJ (greedy meshed)...")
    print("=" * 60)
    voxel_obj, gp = export_voxels_obj(
        voxel_grid,
        collection,
        rectangle,
        center_lon=center_lon,
        center_lat=center_lat,
        meshsize=cfg.meshsize,
        output_dir=cfg.output_dir,
        basename=voxel_basename,
        underground_depth=cfg.terrain_underground_depth,
    )

    # ── Export 2: Mesh OBJ (same coordinate system as voxel OBJ) ─────
    print("\n" + "=" * 60)
    print("Exporting mesh OBJ (after watertight, before voxelization)...")
    print("=" * 60)
    mesh_obj, mesh_groups = export_meshes_obj(
        collection,
        center_lon=center_lon,
        center_lat=center_lat,
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
            center_lon=center_lon,
            center_lat=center_lat,
            meshsize=cfg.meshsize,
            output_dir=cfg.output_dir,
            basename=per_category_basename,
            max_voxel_ram_mb=cfg.max_voxel_ram_mb,
            occupancy_threshold=cfg.occupancy_threshold,
            occupancy_subdivisions=cfg.occupancy_subdivisions,
            mesh_groups=mesh_groups,
            underground_depth=cfg.terrain_underground_depth,
        )

    # ── Export 4: Land-cover 2-D flat mesh OBJ ───────────────────────
    lc_obj = None
    if landcover_basename:
        print("\n" + "=" * 60)
        print("Exporting land-cover 2-D mesh OBJ...")
        print("=" * 60)
        lc_obj = export_landcover_obj(
            land_cover_grid,
            art.land_cover_source,
            dem_grid,
            gp,
            output_dir=cfg.output_dir,
            basename=landcover_basename,
            citygml_path=citygml_paths,
            rectangle_vertices=rectangle,
            center_lon=center_lon,
            center_lat=center_lat,
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
