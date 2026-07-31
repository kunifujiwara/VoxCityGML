"""
VoxCityGML integration pipeline.

Orchestrates the full conversion:
    CityGML dataset  →  VoxCity semantic voxel model

Steps
-----
1. Parse CityGML directory to extract terrain, building, bridge, and
   vegetation triangle meshes.
2. Convert terrain TIN to a DEM elevation grid and voxelise the
   subsurface.
3. Download a land-cover classification grid; set semantic labels on
   the topmost terrain voxels.
4. Rasterise building and bridge meshes into height grids (for metadata).
5. Download canopy-height data (merged with CityGML vegetation).
6. Voxelize all CityGML meshes in a shared 3-D grid, then overlay land
    cover and canopy voxels.
7. Combine all layers into a single 3-D voxel grid and wrap in a
    ``VoxCity`` data class.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; avoids Tk crash in voxcity's plt.show()

from .models import (
    VoxelizerConfig, CityGMLMeshCollection, resolve_citygml_paths,
    resolve_rectangles,
)

# CityGML parsing
from .citygml.parser import parse_citygml_directory, merge_terrain_meshes

# Component processors
from .terrain.processor import terrain_meshes_to_dem_grid
from .landcover.processor import get_land_cover_grid
from .buildings.processor import meshes_to_building_grids
from .canopy.processor import get_canopy_grids
from .voxelizer3d import voxelize_citygml_meshes


class VoxCityGML:
    """End-to-end CityGML → VoxCity pipeline.

    Example
    -------
    >>> from voxcitygml import VoxCityGML, VoxelizerConfig
    >>> cfg = VoxelizerConfig(
    ...     citygml_path="/data/plateau/13101_chiyoda",
    ...     center_lon=139.7671, center_lat=35.6812,
    ...     size_meters=500, meshsize=1.0,
    ... )
    >>> city = VoxCityGML(cfg).run()
    """

    def __init__(self, config: VoxelizerConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full pipeline and return a ``VoxCity`` object."""
        cfg = self.cfg
        art = run_core(cfg)

        print("\nAssembling VoxCity model...")
        from voxcity.generator.pipeline import VoxCityPipeline
        pipeline = VoxCityPipeline(
            meshsize=cfg.meshsize,
            rectangle_vertices=art.rectangle,
        )
        city = pipeline.assemble_voxcity(
            voxcity_grid=art.voxel_grid,
            building_height_grid=art.building_height_grid,
            building_min_height_grid=art.building_min_height_grid,
            building_id_grid=art.building_id_grid,
            land_cover_grid=art.land_cover_grid,
            dem_grid=art.dem_grid,
            canopy_height_top=art.canopy_top,
            canopy_height_bottom=art.canopy_bottom,
            extras={
                "citygml_path": cfg.citygml_path,
                "citygml_paths": art.citygml_paths,
                "land_cover_source": art.land_cover_source,
                "canopy_height_source": art.canopy_height_source,
                "citygml_collection": art.collection,
            },
        )

        if cfg.save_output:
            from voxcity.generator.io import save_voxcity
            save_path = os.path.join(cfg.output_dir, "voxcity.pkl")
            save_voxcity(save_path, city)
            print(f"Saved VoxCity model to {save_path}")

        print("\n" + "=" * 60)
        print("VoxCityGML Pipeline Complete!")
        print("=" * 60)
        from voxcity.utils.classes import summarize_voxel_grid
        summarize_voxel_grid(art.voxel_grid)
        return city


def generate_voxcity(config: VoxelizerConfig):
    """Run the full CityGML → VoxCity pipeline and return a ``VoxCity``.

    Thin public wrapper around :class:`VoxCityGML` for application use::

        from voxcitygml import generate_voxcity, VoxelizerConfig
        city = generate_voxcity(VoxelizerConfig(...))
    """
    return VoxCityGML(config).run()


# ------------------------------------------------------------------
# Shared pipeline core
# ------------------------------------------------------------------

@dataclass
class PipelineArtifacts:
    """Everything the pipeline produces before VoxCity assembly/export."""
    collection: CityGMLMeshCollection
    rectangle: List[Tuple[float, float]]      # [(lon, lat), ...] target rect
    buffered_rectangle: List[Tuple[float, float]]
    center_lon: float
    center_lat: float
    citygml_paths: List[str]
    land_cover_source: str
    canopy_height_source: str
    dem_grid: np.ndarray
    land_cover_grid: np.ndarray
    building_height_grid: np.ndarray
    building_min_height_grid: np.ndarray
    building_id_grid: np.ndarray
    canopy_top: np.ndarray
    canopy_bottom: np.ndarray
    voxel_grid: np.ndarray


def run_core(cfg: VoxelizerConfig) -> PipelineArtifacts:
    """Shared pipeline core: parse → grids → 3-D voxel grid.

    Used by both ``VoxCityGML.run()`` and ``run_and_export()``.

    Does not mutate ``cfg``: auto-selected land-cover / canopy-height
    sources are resolved to local variables and returned on
    ``PipelineArtifacts`` instead of being written back onto the config.

    Raises ValueError when no buildings intersect the target area —
    deliberate behavior change from the pre-refactor pipeline, required
    by the web-app integration.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    rectangle, buffered_rect, center_lon, center_lat = resolve_rectangles(cfg)

    if cfg.gee_project:
        from voxcity.downloader.gee import initialize_earth_engine
        initialize_earth_engine(project=cfg.gee_project)

    land_cover_source = cfg.land_cover_source
    canopy_height_source = cfg.canopy_height_source
    if land_cover_source is None or canopy_height_source is None:
        from voxcity.generator.api import auto_select_data_sources
        auto = auto_select_data_sources(rectangle)
        if land_cover_source is None:
            land_cover_source = auto['land_cover_source']
        if canopy_height_source is None:
            canopy_height_source = auto['canopy_height_source']

    citygml_paths = resolve_citygml_paths(cfg.citygml_path)
    print("=" * 60)
    print("VoxCityGML Pipeline")
    print("=" * 60)
    print(f"  CityGML path(s):    {', '.join(citygml_paths)}")
    print(f"  Centre:             ({center_lon:.6f}, {center_lat:.6f})")
    print(f"  Voxel size:         {cfg.meshsize} m")
    print(f"  Land cover source:  {land_cover_source}")
    print(f"  Canopy source:      {canopy_height_source}")
    if cfg.building_lod is not None:
        print(f"  Building LOD:       {cfg.building_lod}")
    print("=" * 60)

    # -- Step 1: Parse ------------------------------------------------
    print("\n[1/5] Parsing CityGML data...")
    collection = parse_citygml_directory(
        citygml_paths[0],
        rectangle_vertices=buffered_rect,
        n_workers=cfg.n_workers,
        feature_types=['terrain', 'building', 'bridge', 'vegetation'],
        building_lod=cfg.building_lod,
        dem_path=cfg.dem_path,
        tree_citygml_path=cfg.tree_citygml_path,
        use_parse_cache=cfg.use_parse_cache,
    )
    for extra_path in citygml_paths[1:]:
        print(f"  Parsing additional CityGML directory: {extra_path}")
        extra_collection = parse_citygml_directory(
            extra_path,
            rectangle_vertices=buffered_rect,
            n_workers=cfg.n_workers,
            feature_types=['terrain', 'building', 'bridge', 'vegetation'],
            building_lod=cfg.building_lod,
            use_parse_cache=cfg.use_parse_cache,
        )
        collection.merge(extra_collection)

    if not collection.buildings:
        raise ValueError(
            "No CityGML buildings found in the selected area. "
            "Check that the dataset covers the target rectangle.")

    if collection.terrain:
        collection.terrain = merge_terrain_meshes(collection.terrain)

    if not cfg.include_bridges and collection.bridges:
        print(f"  Excluding {len(collection.bridges)} bridge(s) "
              f"from voxelization")
        collection.bridges = []

    # -- Step 2: DEM --------------------------------------------------
    print("\n[2/5] Creating DEM grid from terrain TIN...")
    dem_grid = terrain_meshes_to_dem_grid(
        collection.terrain, rectangle, cfg.meshsize,
    )
    print(f"  DEM grid shape: {dem_grid.shape}, "
          f"elevation range: [{dem_grid.min():.1f}, {dem_grid.max():.1f}] m")
    if cfg.gridvis:
        _visualise_grid(dem_grid, cfg.meshsize, "Digital Elevation Model",
                        cmap='terrain', label='Elevation (m)')

    # -- Step 3: Land cover -------------------------------------------
    print("\n[3/5] Acquiring land cover grid...")
    land_cover_grid = get_land_cover_grid(
        rectangle, cfg.meshsize, land_cover_source, cfg.output_dir,
        citygml_path=citygml_paths,
    )
    if land_cover_grid.shape != dem_grid.shape:
        land_cover_grid = _resize_int_grid(land_cover_grid, *dem_grid.shape)
        print(f"  Resized land cover grid to {land_cover_grid.shape}")

    # -- Step 4: Building grids ---------------------------------------
    print("\n[4/5] Rasterising buildings and bridges...")
    building_height_grid, building_min_height_grid, building_id_grid = \
        meshes_to_building_grids(
            collection.buildings, collection.bridges,
            rectangle, cfg.meshsize, dem_grid,
        )
    if cfg.gridvis:
        bh_vis = building_height_grid.copy()
        bh_vis[bh_vis == 0] = np.nan
        _visualise_grid(bh_vis, cfg.meshsize, "Building height (m)",
                        cmap='viridis', label='Height (m)')

    # -- Step 5: Canopy -----------------------------------------------
    print("\n[5/5] Acquiring canopy height data...")
    canopy_top, canopy_bottom = get_canopy_grids(
        rectangle, cfg.meshsize,
        canopy_height_source, land_cover_source,
        land_cover_grid, dem_grid, cfg.output_dir,
        vegetation_meshes=collection.vegetation,
        trunk_height_ratio=cfg.trunk_height_ratio,
        static_tree_height=cfg.static_tree_height,
    )
    if canopy_top.shape != dem_grid.shape:
        canopy_top = _resize_float_grid(canopy_top, *dem_grid.shape)
        canopy_bottom = _resize_float_grid(canopy_bottom, *dem_grid.shape)

    # -- Voxelize -----------------------------------------------------
    print("\nVoxelising all components...")
    if cfg.use_3d_voxelizer:
        voxel_grid = voxelize_citygml_meshes(
            collection, rectangle, center_lon, center_lat, cfg.meshsize,
            dem_grid=dem_grid,
            land_cover_grid=land_cover_grid,
            canopy_top=canopy_top,
            canopy_bottom=canopy_bottom,
            land_cover_source=land_cover_source,
            trunk_height_ratio=cfg.trunk_height_ratio,
            max_voxel_ram_mb=cfg.max_voxel_ram_mb,
            occupancy_threshold=cfg.occupancy_threshold,
            occupancy_subdivisions=cfg.occupancy_subdivisions,
            underground_depth=cfg.terrain_underground_depth,
        )
    else:
        from voxcity.generator.voxelizer import Voxelizer
        voxelizer = Voxelizer(
            voxel_size=cfg.meshsize,
            land_cover_source=land_cover_source,
            trunk_height_ratio=cfg.trunk_height_ratio,
        )
        voxel_grid = voxelizer.generate_combined(
            building_height_grid_ori=building_height_grid,
            building_min_height_grid_ori=building_min_height_grid,
            building_id_grid_ori=building_id_grid,
            land_cover_grid_ori=land_cover_grid,
            dem_grid_ori=dem_grid,
            tree_grid_ori=canopy_top,
            canopy_bottom_height_grid_ori=canopy_bottom,
        )

    return PipelineArtifacts(
        collection=collection,
        rectangle=rectangle,
        buffered_rectangle=buffered_rect,
        center_lon=center_lon,
        center_lat=center_lat,
        citygml_paths=citygml_paths,
        land_cover_source=land_cover_source,
        canopy_height_source=canopy_height_source,
        dem_grid=dem_grid,
        land_cover_grid=land_cover_grid,
        building_height_grid=building_height_grid,
        building_min_height_grid=building_min_height_grid,
        building_id_grid=building_id_grid,
        canopy_top=canopy_top,
        canopy_bottom=canopy_bottom,
        voxel_grid=voxel_grid,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resize_int_grid(grid: np.ndarray, target_rows: int, target_cols: int):
    from scipy.ndimage import zoom
    factor = (target_rows / grid.shape[0], target_cols / grid.shape[1])
    return zoom(grid, factor, order=0).astype(grid.dtype)


def _resize_float_grid(grid: np.ndarray, target_rows: int, target_cols: int):
    from scipy.ndimage import zoom
    factor = (target_rows / grid.shape[0], target_cols / grid.shape[1])
    return zoom(grid, factor, order=1)


def _visualise_grid(grid, meshsize, title, cmap='viridis', label='Value'):
    """Thin wrapper around voxcity's grid visualiser."""
    try:
        from voxcity.visualizer.grids import visualize_numerical_grid
        visualize_numerical_grid(np.flipud(grid), meshsize, title=title,
                                 cmap=cmap, label=label)
    except Exception:
        pass
