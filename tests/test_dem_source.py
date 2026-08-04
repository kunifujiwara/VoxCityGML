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


# Axis-aligned test rectangle near Tokyo, [SW, NW, NE, SE] (lon, lat)
RECT = [(139.770, 35.646), (139.770, 35.650),
        (139.775, 35.650), (139.775, 35.646)]


def test_named_source_grid_is_fetched_via_voxcity_and_flipped_north_up(
        monkeypatch, tmp_path):
    """get_dem_grid returns voxcity-convention (south-up) grids; the
    pipeline is north-up internally, so the helper must flipud."""
    from voxcitygml.terrain import processor

    captured = {}

    def fake_get_dem_grid(rectangle_vertices, meshsize, source, output_dir,
                          **kwargs):
        captured["source"] = source
        captured["dem_interpolation"] = kwargs.get("dem_interpolation")
        captured["gridvis"] = kwargs.get("gridvis")
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    import voxcity.generator.grids as vgrids
    monkeypatch.setattr(vgrids, "get_dem_grid", fake_get_dem_grid)

    grid = processor.dem_grid_from_named_source(
        RECT, 5.0, "FABDEM", str(tmp_path), dem_interpolation=True)

    assert captured["source"] == "FABDEM"
    assert captured["dem_interpolation"] is True
    assert captured["gridvis"] is False  # never pop matplotlib windows
    np.testing.assert_array_equal(grid, [[3.0, 4.0], [1.0, 2.0]])
    assert grid.dtype == np.float64
    assert grid.flags["C_CONTIGUOUS"]
