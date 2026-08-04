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


def _box_mesh(lat, lon, z_base, z_top):
    """Minimal Mesh3D: a vertical quad at one point, vertices (lat, lon, z)."""
    verts = np.array([
        [lat, lon, z_base], [lat, lon, z_top],
        [lat, lon + 1e-6, z_base], [lat, lon + 1e-6, z_top],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return Mesh3D(vertices=verts, faces=faces, feature_type="building")


def test_anchor_shifts_buildings_by_dem_delta_at_centroid():
    """With CityGML terrain as reference: dz = dem_new - dem_citygml."""
    from voxcitygml.terrain.processor import anchor_meshes_to_dem
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)
    # Building near the rectangle centre, base at 30 m (CityGML elevation).
    b = _box_mesh(35.648, 139.7725, 30.0, 40.0)
    coll = CityGMLMeshCollection(buildings=[b])

    dem_citygml = np.full(gp.shape, 30.0)
    dem_new = np.full(gp.shape, 4.0)

    anchor_meshes_to_dem(coll, dem_citygml, dem_new, RECT, 5.0)

    # 30 -> 4: shifted down by 26.
    np.testing.assert_allclose(b.vertices[:, 2], [4.0, 14.0, 4.0, 14.0])


def test_anchor_without_citygml_terrain_seats_mesh_base_on_new_dem():
    """No terrain reference (dem_citygml is None): base lands ON the DEM."""
    from voxcitygml.terrain.processor import anchor_meshes_to_dem
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)
    b = _box_mesh(35.648, 139.7725, 30.0, 40.0)
    coll = CityGMLMeshCollection(buildings=[b])

    dem_new = np.zeros(gp.shape)  # Flat
    anchor_meshes_to_dem(coll, None, dem_new, RECT, 5.0)

    np.testing.assert_allclose(b.vertices[:, 2], [0.0, 10.0, 0.0, 10.0])


def test_anchor_shifts_bridges_and_vegetation_but_not_terrain():
    from voxcitygml.terrain.processor import anchor_meshes_to_dem
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)
    br = _box_mesh(35.648, 139.7725, 32.0, 35.0)
    veg = _box_mesh(35.647, 139.7720, 31.0, 39.0)
    terr = _box_mesh(35.648, 139.7725, 29.0, 30.0)
    coll = CityGMLMeshCollection(bridges=[br], vegetation=[veg],
                                 terrain=[terr])

    dem_citygml = np.full(gp.shape, 30.0)
    dem_new = np.full(gp.shape, 0.0)
    anchor_meshes_to_dem(coll, dem_citygml, dem_new, RECT, 5.0)

    np.testing.assert_allclose(br.vertices[:, 2], [2.0, 5.0, 2.0, 5.0])
    np.testing.assert_allclose(veg.vertices[:, 2], [1.0, 9.0, 1.0, 9.0])
    # Terrain meshes are the thing being *replaced* — never shifted.
    np.testing.assert_allclose(terr.vertices[:, 2], [29.0, 30.0, 29.0, 30.0])


def test_anchor_handles_empty_meshes():
    from voxcitygml.terrain.processor import anchor_meshes_to_dem
    from voxcitygml.grid_utils import compute_grid_params
    gp = compute_grid_params(RECT, 5.0)
    empty = Mesh3D(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), np.int32))
    coll = CityGMLMeshCollection(buildings=[empty])
    anchor_meshes_to_dem(coll, None, np.zeros(gp.shape), RECT, 5.0)  # no raise
