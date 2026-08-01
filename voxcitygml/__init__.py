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
"""

from .models import VoxelizerConfig
from .pipeline import VoxCityGML, generate_voxcity

__version__ = "0.3.0"
__author__ = "Kunihiko Fujiwara"

__all__ = [
    "VoxCityGML",
    "VoxelizerConfig",
    "generate_voxcity",
]
