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


def test_anchor_rejects_mismatched_dem_new_shape():
    from voxcitygml.terrain.processor import anchor_meshes_to_dem
    from voxcitygml.grid_utils import compute_grid_params
    gp = compute_grid_params(RECT, 5.0)
    coll = CityGMLMeshCollection(buildings=[_box_mesh(35.648, 139.7725, 30.0, 40.0)])
    wrong_shape_dem = np.zeros((gp.n_rows + 1, gp.n_cols))
    with pytest.raises(ValueError):
        anchor_meshes_to_dem(coll, None, wrong_shape_dem, RECT, 5.0)


def test_anchor_rejects_mismatched_dem_citygml_shape():
    from voxcitygml.terrain.processor import anchor_meshes_to_dem
    from voxcitygml.grid_utils import compute_grid_params
    gp = compute_grid_params(RECT, 5.0)
    coll = CityGMLMeshCollection(buildings=[_box_mesh(35.648, 139.7725, 30.0, 40.0)])
    wrong_shape_dem_citygml = np.zeros((gp.n_rows, gp.n_cols + 1))
    with pytest.raises(ValueError):
        anchor_meshes_to_dem(coll, wrong_shape_dem_citygml, np.zeros(gp.shape),
                             RECT, 5.0)


def _terrain_mesh(z):
    """Flat terrain TIN at elevation ``z`` spanning the whole RECT (with a
    margin), so terrain_meshes_to_dem_grid gets a well-posed triangulation.
    Vertices are (lat, lon, z)."""
    lons = [139.769, 139.776]
    lats = [35.645, 35.651]
    verts = np.array([
        [lats[0], lons[0], z], [lats[0], lons[1], z],
        [lats[1], lons[0], z], [lats[1], lons[1], z],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return Mesh3D(vertices=verts, faces=faces, feature_type="terrain")


def test_resolve_dem_step_citygml_default_uses_terrain_tin():
    from voxcitygml.pipeline import _resolve_dem_step
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)
    coll = CityGMLMeshCollection(terrain=[_terrain_mesh(12.0)])

    cfg = VoxelizerConfig(rectangle_vertices=list(RECT), meshsize=5.0)
    dem, effective = _resolve_dem_step(cfg, coll, list(RECT))

    assert effective == "CityGML Terrain"
    assert dem.shape == gp.shape
    np.testing.assert_allclose(dem, 12.0)
    assert coll.terrain  # untouched


def test_resolve_dem_step_flat_zeroes_dem_anchors_and_drops_terrain():
    from voxcitygml.pipeline import _resolve_dem_step
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)
    bldg = _box_mesh(35.648, 139.7725, 30.0, 40.0)
    coll = CityGMLMeshCollection(terrain=[_terrain_mesh(30.0)],
                                 buildings=[bldg])

    cfg = VoxelizerConfig(rectangle_vertices=list(RECT), meshsize=5.0,
                          dem_source="Flat")
    dem, effective = _resolve_dem_step(cfg, coll, list(RECT))

    assert effective == "Flat"
    assert dem.shape == gp.shape
    assert (dem == 0).all()
    assert coll.terrain == []          # terrain solid must not be voxelized
    # Building re-seated: was on terrain at 30 m, now on the flat plane.
    assert float(np.min(bldg.vertices[:, 2])) == pytest.approx(0.0)


def test_resolve_dem_step_named_source_fetches_and_anchors(monkeypatch):
    from voxcitygml import pipeline as pl
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)

    def fake_named(rectangle, meshsize, source, output_dir,
                   dem_interpolation=None):
        assert source == "FABDEM"
        assert dem_interpolation is True
        return np.full(gp.shape, 4.0)

    monkeypatch.setattr(pl, "dem_grid_from_named_source", fake_named)

    bldg = _box_mesh(35.648, 139.7725, 30.0, 40.0)
    coll = CityGMLMeshCollection(terrain=[_terrain_mesh(30.0)],
                                 buildings=[bldg])

    cfg = VoxelizerConfig(rectangle_vertices=list(RECT), meshsize=5.0,
                          dem_source="FABDEM", dem_interpolation=True)
    dem, effective = pl._resolve_dem_step(cfg, coll, list(RECT))

    assert effective == "FABDEM"
    assert (dem == 4.0).all()
    assert coll.terrain == []
    assert float(np.min(bldg.vertices[:, 2])) == pytest.approx(4.0)


def test_resolve_dem_step_named_source_resizes_mismatched_grid_even_without_citygml_terrain(
        monkeypatch):
    """Regression: when the CityGML dataset ships no terrain (dem_citygml is
    None), a fetched named-source grid whose shape doesn't match the target
    grid must still be resized -- not silently passed through mismatched."""
    from voxcitygml import pipeline as pl
    from voxcitygml.grid_utils import compute_grid_params

    gp = compute_grid_params(RECT, 5.0)

    def fake_named(rectangle, meshsize, source, output_dir,
                   dem_interpolation=None):
        # Deliberately wrong shape.
        return np.full((gp.n_rows + 3, gp.n_cols + 3), 7.0)

    monkeypatch.setattr(pl, "dem_grid_from_named_source", fake_named)

    bldg = _box_mesh(35.648, 139.7725, 30.0, 40.0)
    coll = CityGMLMeshCollection(buildings=[bldg])  # no terrain

    cfg = VoxelizerConfig(rectangle_vertices=list(RECT), meshsize=5.0,
                          dem_source="FABDEM")
    dem, effective = pl._resolve_dem_step(cfg, coll, list(RECT))

    assert effective == "FABDEM"
    assert dem.shape == gp.shape
    assert coll.terrain == []
