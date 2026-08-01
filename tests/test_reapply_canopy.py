"""nDSM re-apply support: extras carry the z datum and the mesh-vegetation mask."""
import numpy as np
import pytest

from voxcitygml.models import CityGMLMeshCollection, Mesh3D
# The stub config is identical to the one the pipeline-core tests use.
from tests.test_pipeline_core import _config as _stub_cfg


def _stub_pipeline(monkeypatch, tmp_path, *, use_3d=True):
    """Monkeypatch every heavy pipeline stage, mirroring test_pipeline_core.

    The one deliberate difference: ``fake_voxelize`` honours the ``info_out``
    contract, so the assertions exercise the threading of those values from
    the voxelizer through ``run_core`` into ``extras``.
    """
    import voxcitygml.pipeline as pl

    def fake_parse(path, **kwargs):
        col = CityGMLMeshCollection()
        col.buildings = [Mesh3D(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
            feature_type='building', feature_id='b1')]
        return col

    monkeypatch.setattr(pl, 'parse_citygml_directory', fake_parse)
    monkeypatch.setattr(pl, 'merge_terrain_meshes', lambda t: t)
    monkeypatch.setattr(pl, 'terrain_meshes_to_dem_grid',
                        lambda *a, **k: np.zeros((10, 10)))
    monkeypatch.setattr(pl, 'get_land_cover_grid',
                        lambda *a, **k: np.zeros((10, 10), dtype=np.int32))
    monkeypatch.setattr(pl, 'meshes_to_building_grids',
                        lambda *a, **k: (np.zeros((10, 10)),
                                         np.empty((10, 10), dtype=object),
                                         np.zeros((10, 10))))
    monkeypatch.setattr(pl, 'get_canopy_grids',
                        lambda *a, **k: (np.zeros((10, 10)),
                                         np.zeros((10, 10))))

    def fake_voxelize(collection, rectangle, center_lon, center_lat, meshsize,
                      *, info_out=None, **kwargs):
        if info_out is not None:
            info_out["voxel_min_z"] = -3.5
            mask = np.zeros((10, 10), dtype=bool)
            mask[2, 3] = True
            info_out["mesh_vegetation_mask"] = mask
        return np.zeros((10, 10, 5), dtype=np.int16)

    monkeypatch.setattr(pl, 'voxelize_citygml_meshes', fake_voxelize)
    monkeypatch.setattr(pl, 'resolve_citygml_paths', lambda p: [str(tmp_path)])


# ---------------------------------------------------------------------
# The real producer: voxelize_citygml_meshes' info_out contract.
# These run the actual voxelizer on a tiny synthetic collection (30x30x6,
# ~1.5 s, no network, no dataset) rather than a stub, so a wrong value
# written into info_out is caught rather than threaded through unexamined.
# ---------------------------------------------------------------------

_LON, _LAT, _SIZE, _MS = 139.7671, 35.6812, 60.0, 2.0


def _vegetation_box():
    """An ~8 m cube of vegetation near the target centre, in (lat, lon, z)."""
    dlat = 4.0 / 111320.0
    dlon = 4.0 / (111320.0 * np.cos(np.radians(_LAT)))
    lat0, lat1 = _LAT - dlat, _LAT + dlat
    lon0, lon1 = _LON - dlon, _LON + dlon
    verts = []
    for z in (0.0, 8.0):
        verts += [[lat0, lon0, z], [lat0, lon1, z],
                  [lat1, lon1, z], [lat1, lon0, z]]
    faces = np.array([
        [0, 1, 2], [0, 2, 3],          # bottom
        [4, 6, 5], [4, 7, 6],          # top
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)
    return Mesh3D(vertices=np.array(verts, dtype=np.float64), faces=faces,
                  feature_type='vegetation', feature_id='v1')


@pytest.fixture(scope="module")
def veg_scene():
    """A tiny real voxelization with one vegetation mesh under full canopy."""
    from voxcitygml.citygml.coordinates import create_rectangle
    from voxcitygml.voxelizer3d import _compute_grid_params_3d

    rect = create_rectangle(_LON, _LAT, _SIZE)
    collection = CityGMLMeshCollection(vegetation=[_vegetation_box()])
    gp, _ = _compute_grid_params_3d(rect, _LON, _LAT, _MS, collection)
    return rect, collection, gp


def test_info_out_voxel_min_z_is_the_grid_datum(veg_scene):
    """voxel_min_z must be the grid's own min_z, not some other z bound."""
    from voxcitygml.voxelizer3d import voxelize_citygml_meshes, _compute_grid_params_3d

    rect, collection, gp = veg_scene
    dem = np.zeros((gp.n_rows, gp.n_cols))
    info = {}
    voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem,
        canopy_top=np.full((gp.n_rows, gp.n_cols), 10.0), info_out=info)

    # Cross-checked against the grid params the voxelizer derives internally,
    # not a hardcoded number, so this cannot drift as the grid sizing changes.
    assert isinstance(info["voxel_min_z"], float)
    assert info["voxel_min_z"] == gp.min_z
    # Pin the distinction from the other z bounds a typo could reach for.
    assert gp.min_z != gp.max_z
    assert info["voxel_min_z"] != gp.max_z


def test_info_out_mask_marks_only_mesh_vegetation_columns(veg_scene):
    """The mask must be the mesh-vegetation columns, not the canopy ones."""
    from voxcitygml.voxelizer3d import voxelize_citygml_meshes, TREE_CODE

    rect, collection, gp = veg_scene
    dem = np.zeros((gp.n_rows, gp.n_cols))
    info = {}
    grid = voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem,
        canopy_top=np.full((gp.n_rows, gp.n_cols), 10.0), info_out=info)

    mask = info["mesh_vegetation_mask"]
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == grid.shape[:2]

    after = np.any(grid == TREE_CODE, axis=2)
    # The canopy overlay covered the whole grid, so *many* more columns hold
    # TREE_CODE afterwards than the vegetation mesh alone produced.  A mask
    # computed after the canopy write would equal `after`; a strict subset is
    # the proof it was captured before.
    assert mask.any(), "the vegetation mesh produced no tree columns"
    assert np.all(after[mask]), "mesh columns must still be tree columns"
    assert mask.sum() < after.sum(), \
        "mask must exclude canopy-only columns, not equal the post-canopy set"
    assert not np.array_equal(mask, after)
    assert mask is not after


def test_info_out_mask_without_canopy_matches_the_captured_one(veg_scene):
    """The no-canopy fallback scan must agree with the captured mask."""
    from voxcitygml.voxelizer3d import voxelize_citygml_meshes

    rect, collection, gp = veg_scene
    dem = np.zeros((gp.n_rows, gp.n_cols))

    with_canopy = {}
    voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem,
        canopy_top=np.full((gp.n_rows, gp.n_cols), 10.0), info_out=with_canopy)

    # No canopy at all -> _apply_canopy never runs -> the fallback scan path.
    no_canopy = {}
    voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem,
        canopy_top=None, info_out=no_canopy)

    # All-zero canopy -> _apply_canopy runs but takes its early return.
    zero_canopy = {}
    voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem,
        canopy_top=np.zeros((gp.n_rows, gp.n_cols)), info_out=zero_canopy)

    ref = with_canopy["mesh_vegetation_mask"]
    assert np.array_equal(no_canopy["mesh_vegetation_mask"], ref)
    assert np.array_equal(zero_canopy["mesh_vegetation_mask"], ref)
    assert no_canopy["voxel_min_z"] == with_canopy["voxel_min_z"]


def test_voxelize_without_info_out_is_unchanged(veg_scene):
    """info_out is additive: omitting it must not alter the returned grid."""
    from voxcitygml.voxelizer3d import voxelize_citygml_meshes

    rect, collection, gp = veg_scene
    dem = np.zeros((gp.n_rows, gp.n_cols))
    canopy = np.full((gp.n_rows, gp.n_cols), 10.0)

    info = {}
    with_info = voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem,
        canopy_top=canopy, info_out=info)
    without_info = voxelize_citygml_meshes(
        collection, rect, _LON, _LAT, _MS, dem_grid=dem, canopy_top=canopy)

    assert isinstance(without_info, np.ndarray)
    assert np.array_equal(with_info, without_info)


def test_apply_canopy_reports_mesh_tree_mask():
    """_apply_canopy must surface the already_has_tree array it computes."""
    from voxcitygml.voxelizer3d import _apply_canopy, Grid3DParams, TREE_CODE

    gp = Grid3DParams(n_rows=4, n_cols=4, n_z=10,
                      min_x=0.0, max_x=8.0, min_y=0.0, max_y=8.0,
                      min_z=0.0, max_z=20.0, voxel_size=2.0)
    grid = np.zeros((4, 4, 10), dtype=np.int16)
    grid[1, 1, 0:3] = TREE_CODE          # a mesh-derived tree column
    dem = np.zeros((4, 4))
    canopy_top = np.full((4, 4), 6.0)
    out = {}
    _apply_canopy(grid, gp, dem, canopy_top, None, None, mesh_tree_mask_out=out)

    mask = out["mesh_tree_mask"]
    assert mask.shape == (4, 4)
    assert mask[1, 1]
    assert mask.sum() == 1, "only the pre-seeded mesh column may be marked"
    # Guard the capture *point*, not just the value: this run wrote canopy
    # voxels into the other 15 columns, so a mask captured after the write
    # would have sum() == 16.  Assert the write actually happened, otherwise
    # the sum()==1 check above proves nothing.
    assert np.any(grid[0, 0] == TREE_CODE), "canopy fill did not run"


def test_apply_canopy_reports_mask_even_when_no_canopy():
    """The early return for 'no canopy anywhere' must still yield a mask."""
    from voxcitygml.voxelizer3d import _apply_canopy, Grid3DParams, TREE_CODE

    gp = Grid3DParams(n_rows=4, n_cols=4, n_z=10,
                      min_x=0.0, max_x=8.0, min_y=0.0, max_y=8.0,
                      min_z=0.0, max_z=20.0, voxel_size=2.0)
    grid = np.zeros((4, 4, 10), dtype=np.int16)
    grid[3, 2, 0:2] = TREE_CODE
    out = {}
    _apply_canopy(grid, gp, np.zeros((4, 4)), np.zeros((4, 4)), None, None,
                  mesh_tree_mask_out=out)

    mask = out["mesh_tree_mask"]
    assert mask.shape == (4, 4)
    assert mask[3, 2]
    assert mask.sum() == 1


def test_extras_carry_voxel_min_z_and_mask(monkeypatch, tmp_path):
    """run() must expose the z datum and mask so canopy can be re-applied."""
    import voxcitygml.pipeline as pl

    _stub_pipeline(monkeypatch, tmp_path)
    city = pl.VoxCityGML(_stub_cfg(tmp_path)).run()
    assert "voxel_min_z" in city.extras
    assert isinstance(city.extras["voxel_min_z"], float)
    assert city.extras["voxel_min_z"] == -3.5
    mask = city.extras["mesh_vegetation_mask"]
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == city.voxels.classes.shape[:2]
    assert mask[2, 3]


def test_extras_sane_without_3d_voxelizer(monkeypatch, tmp_path):
    """The legacy voxcity Voxelizer path has no z datum and no mesh trees."""
    from dataclasses import replace
    import voxcitygml.pipeline as pl

    _stub_pipeline(monkeypatch, tmp_path)
    cfg = replace(_stub_cfg(tmp_path), use_3d_voxelizer=False)

    class _FakeVoxelizer:
        def __init__(self, **kwargs):
            pass

        def generate_combined(self, **kwargs):
            return np.zeros((10, 10, 5), dtype=np.int16)

    import voxcity.generator.voxelizer as vz
    monkeypatch.setattr(vz, 'Voxelizer', _FakeVoxelizer)

    art = pl.run_core(cfg)
    assert art.voxel_min_z is None
    assert art.mesh_vegetation_mask.dtype == bool
    assert art.mesh_vegetation_mask.shape == (10, 10)
    assert not art.mesh_vegetation_mask.any()
