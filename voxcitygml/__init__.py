"""
VoxCityGML - Voxelize CityGML data to create semantic 3D city models.

This package converts CityGML data (terrain, buildings, bridges, vegetation)
into voxel-based semantic city models compatible with the VoxCity framework.

Pipeline:
    1. Extract terrain/buildings/bridges/vegetation meshes from CityGML
    2. Create DEM and auxiliary grids for land cover / canopy data
    3. Voxelize all CityGML meshes into a shared 3-D grid
    4. Overlay land cover and canopy voxels
    5. Integrate all components → VoxCity model

Usage:
    from voxcitygml import generate_voxcity, VoxelizerConfig

    config = VoxelizerConfig(
        citygml_path="path/to/plateau_dataset",
        center_lon=139.7671,
        center_lat=35.6812,
        size_meters=500,
        meshsize=1.0,
    )
    city = generate_voxcity(config)

    # Overlay a revised canopy onto the finished grid, in place, without
    # rebuilding it -- so mesh-voxelized LOD2 geometry survives.  The canopy
    # must be north-up like ``city.voxels.classes``; see ``reapply_canopy``.
    from voxcitygml import reapply_canopy
    reapply_canopy(city, refined_canopy_top, refined_canopy_bottom)

Model extras:
    Models built with ``use_3d_voxelizer=True`` carry two keys in
    ``city.extras`` that ``reapply_canopy`` needs and that cannot be recovered
    from a finished grid:

    ``voxel_min_z``
        float -- elevation (m) of the bottom face of the z=0 voxel layer, the
        grid's vertical datum.  ``None`` on the legacy voxelizer path.
    ``mesh_vegetation_mask``
        (n_rows, n_cols) bool, north-up -- columns whose tree voxels came from
        CityGML vegetation meshes rather than from the canopy overlay.
"""

from .models import VoxelizerConfig
from .pipeline import VoxCityGML, generate_voxcity
from .reapply import reapply_canopy

__version__ = "0.3.0"
__author__ = "Kunihiko Fujiwara"

__all__ = [
    "VoxCityGML",
    "VoxelizerConfig",
    "generate_voxcity",
    "reapply_canopy",
]
