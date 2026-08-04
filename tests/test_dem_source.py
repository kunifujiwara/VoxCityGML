"""Tests for DEM source selection (dem_source) in the LOD2 pipeline."""
import numpy as np
import pytest

from voxcitygml.models import VoxelizerConfig, Mesh3D, CityGMLMeshCollection


def test_voxelizer_config_has_dem_source_defaulting_to_none():
    cfg = VoxelizerConfig()
    assert cfg.dem_source is None
    assert cfg.dem_interpolation is None
    # The app probes __dataclass_fields__ to detect this capability.
    assert "dem_source" in VoxelizerConfig.__dataclass_fields__
    assert "dem_interpolation" in VoxelizerConfig.__dataclass_fields__
