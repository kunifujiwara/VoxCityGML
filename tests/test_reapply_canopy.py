"""nDSM canopy re-apply: the extras contract and the ``reapply_canopy`` overlay."""
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
    """run() must expose the z datum and mask, the mask in the model's frame.

    ``voxel_min_z`` is a scalar and frame-independent, so it crosses the
    assembly seam untouched.  The mask indexes ``voxels.classes`` 1:1, so it
    must cross it the same way the voxel grid does -- north-up inside
    ``run_core``, south-up on the assembled ``VoxCity`` (see
    ``pipeline._to_south_up``).
    """
    import voxcitygml.pipeline as pl

    _stub_pipeline(monkeypatch, tmp_path)
    city = pl.VoxCityGML(_stub_cfg(tmp_path)).run()
    assert "voxel_min_z" in city.extras
    assert isinstance(city.extras["voxel_min_z"], float)
    assert city.extras["voxel_min_z"] == -3.5
    mask = city.extras["mesh_vegetation_mask"]
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == city.voxels.classes.shape[:2]
    # ``fake_voxelize`` marks column (2, 3) of a 10-row grid in the north-up
    # frame, so the south-up model must carry it at row 10 - 1 - 2 = 7.  The
    # whole mask is pinned rather than that one cell: this way a conversion
    # that skipped the flip (mark left at row 2) or applied it twice fails
    # here, and so does one that smeared the mark across extra columns.
    expected = np.zeros((10, 10), dtype=bool)
    expected[7, 3] = True
    np.testing.assert_array_equal(mask, expected)


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


# =====================================================================
# reapply_canopy — overlaying a revised canopy onto an existing grid
# =====================================================================
#
# Every fixture below uses voxel_size 2.0 m, min_z 0.0 and a flat DEM at
# 0.0 m, so the z arithmetic of ``_apply_canopy`` reduces to
# ``z = rint(height / 2)`` over the half-open interval [z_start, z_end):
#
#   canopy_bottom=0.0, canopy_top=6.0  ->  z 0,1,2   (z_end = 3, excluded)
#   canopy_bottom=0.0, canopy_top=18.0 ->  z 0..8    (z_end = 9, excluded)
#   canopy_bottom=0.0, canopy_top=4.0  ->  z 0,1     (z_end = 2, excluded)
#
# canopy_bottom is passed explicitly throughout so the crowns do not depend
# on the trunk-height-ratio default.

_VS, _MIN_Z, _NZ = 2.0, 0.0, 12


def _make_city(voxel_grid, *, mask=None, min_z=_MIN_Z, meshsize=_VS,
               dem=None, canopy_top=None, canopy_bottom=None,
               land_cover=None, drop_extras=()):
    """A minimal but structurally real ``VoxCity`` around ``voxel_grid``.

    Field names/types are taken from ``voxcity.models`` rather than guessed;
    ``extras`` mirrors what ``VoxCityPipeline.assemble_voxcity`` writes
    (including its ``canopy_top`` / ``canopy_bottom`` aliases) so the
    component-grid bookkeeping is exercised the way the app sees it.
    """
    from voxcity.models import (VoxCity, VoxelGrid, BuildingGrid,
                                LandCoverGrid, DemGrid, CanopyGrid,
                                GridMetadata)

    n_rows, n_cols = voxel_grid.shape[:2]
    meta = GridMetadata(crs="EPSG:4326", bounds=(0.0, 0.0, 1.0, 1.0),
                        meshsize=meshsize)
    top = np.zeros((n_rows, n_cols)) if canopy_top is None else canopy_top
    bottom = canopy_bottom
    canopy = CanopyGrid(top=top, bottom=bottom, meta=meta)
    extras = {
        "canopy_top": canopy.top,
        "canopy_bottom": canopy.bottom,
        "voxel_min_z": min_z,
        "mesh_vegetation_mask": (np.zeros((n_rows, n_cols), dtype=bool)
                                 if mask is None else mask),
    }
    for key in drop_extras:
        extras.pop(key, None)
    return VoxCity(
        voxels=VoxelGrid(classes=voxel_grid, meta=meta),
        buildings=BuildingGrid(heights=np.zeros((n_rows, n_cols)),
                               min_heights=np.empty((n_rows, n_cols),
                                                    dtype=object),
                               ids=np.zeros((n_rows, n_cols)), meta=meta),
        land_cover=LandCoverGrid(
            classes=(np.zeros((n_rows, n_cols), dtype=np.int32)
                     if land_cover is None else land_cover), meta=meta),
        dem=DemGrid(elevation=(np.zeros((n_rows, n_cols)) if dem is None
                               else dem), meta=meta),
        tree_canopy=canopy,
        extras=extras,
    )


def _seeded_grid(n_rows=4, n_cols=4, n_z=_NZ):
    """Ground everywhere, a building column, and one land-cover voxel."""
    from voxcitygml.voxelizer3d import GROUND_CODE, BUILDING_CODE

    grid = np.zeros((n_rows, n_cols, n_z), dtype=np.int16)
    grid[:, :, 0] = GROUND_CODE
    grid[0, 0, 1:5] = BUILDING_CODE
    grid[3, 3, 1] = 6            # a land-cover class code
    return grid


def test_reapply_canopy_adds_trees_without_touching_other_classes():
    """Buildings, terrain and land cover must survive the overlay bit-exactly."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import GROUND_CODE, BUILDING_CODE, TREE_CODE

    grid = _seeded_grid()
    city = _make_city(grid)
    before_ground = (grid == GROUND_CODE).copy()
    before_building = (grid == BUILDING_CODE).copy()
    before_lc = (grid == 6).copy()

    reapply_canopy(city, np.full((4, 4), 6.0), np.zeros((4, 4)))

    assert np.array_equal(grid == GROUND_CODE, before_ground)
    assert np.array_equal(grid == BUILDING_CODE, before_building)
    assert np.array_equal(grid == 6, before_lc)
    assert np.any(grid == TREE_CODE), "no canopy was written at all"
    # z 0,1,2 -- z 0 is ground, so an ordinary column keeps 1 and 2 only.
    assert grid[2, 2, 0] == GROUND_CODE
    assert list(grid[2, 2, 1:4]) == [TREE_CODE, TREE_CODE, 0]
    # The building column is occupied at z 1..4: canopy is AIR-only, so it
    # must not have displaced a single building voxel.
    assert list(grid[0, 0, 1:5]) == [BUILDING_CODE] * 4
    # Mutation is in place: the caller's array object is the one updated.
    assert city.voxels.classes is grid


def test_reapply_canopy_preserves_mesh_vegetation_columns():
    """Masked columns keep their CityGML crown; unmasked ones get canopy."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = _seeded_grid()
    # An odd-shaped (non-contiguous, off-the-ground) mesh crown, the kind a
    # rectangular canopy column would flatten.
    grid[1, 1, 4] = TREE_CODE
    grid[1, 1, 6:8] = TREE_CODE
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True

    city = _make_city(grid, mask=mask)
    before_masked_column = grid[1, 1, :].copy()

    reapply_canopy(city, np.full((4, 4), 6.0), np.zeros((4, 4)))

    assert np.array_equal(grid[1, 1, :], before_masked_column), \
        "the mesh-vegetation column was modified"
    # ... and the fill still happened elsewhere.
    assert list(grid[2, 2, 1:3]) == [TREE_CODE, TREE_CODE]


def test_reapply_canopy_clears_stale_canopy_outside_the_mask():
    """A shorter canopy must shrink the crown, not union with the old one."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = _seeded_grid()
    city = _make_city(grid)

    reapply_canopy(city, np.full((4, 4), 18.0), np.zeros((4, 4)))
    assert grid[2, 2, 8] == TREE_CODE, "tall canopy did not reach z=8"

    reapply_canopy(city, np.full((4, 4), 4.0), np.zeros((4, 4)))
    assert grid[2, 2, 1] == TREE_CODE
    assert not np.any(grid[:, :, 2:] == TREE_CODE), \
        "stale canopy above the new crown was not cleared"


def test_reapply_canopy_is_idempotent_and_path_independent():
    """The result must be a function of the canopy, not of the prior grid.

    Two calls with the same canopy are compared, *and* the same canopy is
    applied to a grid that already carries a much taller stale crown.  The
    second half is where the teeth are: re-running an AIR-only fill over an
    already-populated grid unions old and new canopy, so a missing clear
    step makes the two grids differ.  (Double-applying an identical canopy
    on its own would pass even without clearing -- the union of a set with
    itself is that set.)
    """
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    top, bottom = np.full((4, 4), 6.0), np.zeros((4, 4))

    fresh = _seeded_grid()
    fresh[1, 1, 6:8] = TREE_CODE          # mesh crown, must be preserved
    reapply_canopy(_make_city(fresh, mask=mask), top, bottom)
    once = fresh.copy()

    reapply_canopy(_make_city(fresh, mask=mask), top, bottom)
    assert np.array_equal(fresh, once), "re-applying the same canopy changed it"

    dirty = _seeded_grid()
    dirty[1, 1, 6:8] = TREE_CODE
    dirty[2, 2, 1:9] = TREE_CODE          # a tall stale canopy column
    dirty[0, 2, 1:6] = TREE_CODE
    reapply_canopy(_make_city(dirty, mask=mask), top, bottom)
    assert np.array_equal(dirty, once), \
        "the outcome depended on the canopy already in the grid"


def test_reapply_canopy_updates_the_component_grids():
    """The 2.5-D canopy grids must track what was written into the voxels."""
    from voxcitygml import reapply_canopy

    city = _make_city(_seeded_grid(),
                      canopy_top=np.full((4, 4), 99.0),
                      canopy_bottom=np.full((4, 4), 1.0))
    top = np.full((4, 4), 6.0)
    reapply_canopy(city, top, np.full((4, 4), 2.0))

    assert np.allclose(city.tree_canopy.top, 6.0)
    assert np.allclose(city.tree_canopy.bottom, 2.0)
    # assemble_voxcity's extras aliases must not go stale.
    assert np.allclose(city.extras["canopy_top"], 6.0)
    assert np.allclose(city.extras["canopy_bottom"], 2.0)
    # The caller's array must not be aliased into the model.
    top[:] = 42.0
    assert np.allclose(city.tree_canopy.top, 6.0)


def test_reapply_canopy_derives_bottom_from_the_trunk_ratio():
    """Omitting canopy_bottom records the crown base actually voxelized."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    city = _make_city(_seeded_grid())
    reapply_canopy(city, np.full((4, 4), 20.0), trunk_height_ratio=0.5)

    assert np.allclose(city.tree_canopy.bottom, 10.0)
    # rint(10/2)=5 .. rint(20/2)=10, half-open -> z 5..9.
    column = city.voxels.classes[2, 2, :]
    assert not np.any(column[1:5] == TREE_CODE)
    assert np.all(column[5:10] == TREE_CODE)
    assert column[10] == 0


def test_reapply_canopy_without_mask_warns_and_still_applies():
    """Older models degrade to 'no vegetation to preserve', not to a crash."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = _seeded_grid()
    city = _make_city(grid, drop_extras=("mesh_vegetation_mask",))
    with pytest.warns(UserWarning, match="mesh_vegetation_mask"):
        reapply_canopy(city, np.full((4, 4), 6.0), np.zeros((4, 4)))
    assert np.any(grid == TREE_CODE)


def test_reapply_canopy_rejects_bad_input():
    """Shape mismatch and a missing z datum must each raise ValueError."""
    from voxcitygml import reapply_canopy

    city = _make_city(_seeded_grid())
    with pytest.raises(ValueError) as exc:
        reapply_canopy(city, np.zeros((4, 4)), np.zeros((3, 4)))
    assert "canopy_bottom" in str(exc.value)
    assert "(3, 4)" in str(exc.value) and "(4, 4)" in str(exc.value)

    with pytest.raises(ValueError, match="2-D"):
        reapply_canopy(city, np.zeros((4, 4, 4)))

    none_z = _make_city(_seeded_grid(), min_z=None)
    with pytest.raises(ValueError, match="voxel_min_z"):
        reapply_canopy(none_z, np.zeros((4, 4)))

    missing_z = _make_city(_seeded_grid(), drop_extras=("voxel_min_z",))
    with pytest.raises(ValueError, match="voxel_min_z"):
        reapply_canopy(missing_z, np.zeros((4, 4)))

    bad_mask = _make_city(_seeded_grid(), mask=np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="mesh_vegetation_mask"):
        reapply_canopy(bad_mask, np.zeros((4, 4)))


def test_reapply_canopy_honours_the_z_datum_and_terrain():
    """Crowns sit on the DEM, offset by the grid's own vertical datum.

    Pins the two-stage rounding of ``_apply_canopy``: the DEM is the only
    term shifted by ``min_z``; canopy heights are above ground.
    """
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = np.zeros((2, 2, 12), dtype=np.int16)
    dem = np.full((2, 2), 6.0)
    # ground_levels = rint((6 - (-4)) / 2) = 5; base 0 -> z_start 5;
    # top 8 -> rint(4) = 4 -> z_end 9.
    city = _make_city(grid, min_z=-4.0, dem=dem)
    reapply_canopy(city, np.full((2, 2), 8.0), np.zeros((2, 2)))

    assert not np.any(grid[0, 0, :5] == TREE_CODE)
    assert np.all(grid[0, 0, 5:9] == TREE_CODE)
    assert grid[0, 0, 9] == 0


def test_reapply_canopy_ignores_the_placeholder_xy_bounds():
    """The synthesised Grid3DParams' x/y bounds must never reach the overlay.

    ``reapply_canopy`` is a per-column overlay, so it fills gp's x/y bounds
    with a placeholder frame -- justified today by ``_apply_canopy`` reading
    only ``min_z`` / ``voxel_size`` / ``n_z``.  The trap is that
    ``Grid3DParams`` also carries ``xyz_to_indices``, the *truncating* mesh
    mapping; a future edit reaching for it would silently get plausible wrong
    rows/cols out of the placeholder frame rather than an error.  Poisoning
    the bounds with NaN turns "unused" from a comment into a checked contract:
    NaN would propagate into any index derived from them.
    """
    import voxcitygml.reapply as rp
    from voxcitygml import reapply_canopy

    top, bottom = np.full((4, 4), 6.0), np.zeros((4, 4))
    reference = _seeded_grid()
    reapply_canopy(_make_city(reference), top, bottom)

    real_params = rp.Grid3DParams

    def poisoned(**kwargs):
        kwargs.update(min_x=np.nan, max_x=np.nan, min_y=np.nan, max_y=np.nan)
        return real_params(**kwargs)

    poisoned_grid = _seeded_grid()
    try:
        rp.Grid3DParams = poisoned
        reapply_canopy(_make_city(poisoned_grid), top, bottom)
    finally:
        rp.Grid3DParams = real_params

    assert np.array_equal(poisoned_grid, reference), \
        "the canopy overlay depends on gp's x/y bounds after all"


def test_reapply_canopy_with_zero_canopy_removes_all_of_it():
    """An all-zero canopy must clear the overlay and leave everything else.

    The 'Use nDSM for Canopy' checkbox turned off is exactly this call, and
    it is the one path where the clear does the entire job -- ``_apply_canopy``
    takes its "no canopy anywhere" early return and writes nothing.
    """
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import GROUND_CODE, BUILDING_CODE, TREE_CODE

    grid = _seeded_grid()
    grid[1, 1, 6:8] = TREE_CODE          # a CityGML crown, must survive
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    city = _make_city(grid, mask=mask)

    reapply_canopy(city, np.full((4, 4), 8.0), np.zeros((4, 4)))
    assert np.any(grid[2, 2] == TREE_CODE)

    reapply_canopy(city, np.zeros((4, 4)), np.zeros((4, 4)))

    assert np.array_equal(np.where(grid == TREE_CODE)[0], np.array([1, 1])), \
        "only the masked mesh column may still hold TREE_CODE"
    assert list(grid[1, 1, 6:8]) == [TREE_CODE, TREE_CODE]
    assert np.all(grid[:, :, 0] == GROUND_CODE)
    assert list(grid[0, 0, 1:5]) == [BUILDING_CODE] * 4
    assert grid[3, 3, 1] == 6
    assert np.allclose(city.tree_canopy.top, 0.0)


def test_reapply_canopy_default_trunk_ratio_matches_the_voxelizer():
    """Omitting both canopy_bottom and the ratio must use 11.76/19.98.

    Pinned as a literal rather than by importing the constant, so the value
    itself is under test and not merely its own definition.
    """
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    city = _make_city(_seeded_grid())
    reapply_canopy(city, np.full((4, 4), 20.0))

    expected_base = 20.0 * 11.76 / 19.98          # 11.7718 m
    assert np.allclose(city.tree_canopy.bottom, expected_base)
    # rint(11.7718/2) = 6 .. rint(20/2) = 10, half-open -> z 6..9.
    column = city.voxels.classes[2, 2, :]
    assert not np.any(column[1:6] == TREE_CODE)
    assert np.all(column[6:10] == TREE_CODE)
    assert column[10] == 0


def test_reapply_canopy_resamples_a_coarser_dem():
    """A DEM at component-grid resolution is resampled onto the voxel grid."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = np.zeros((4, 4, _NZ), dtype=np.int16)
    # Flat, so the bilinear resample has an unambiguous answer: every voxel
    # column sits on 6 m of terrain -> ground level rint(6/2) = 3.
    city = _make_city(grid, dem=np.full((2, 2), 6.0))
    reapply_canopy(city, np.full((4, 4), 4.0), np.zeros((4, 4)))

    assert not np.any(grid[:, :, :3] == TREE_CODE)
    assert np.all(grid[:, :, 3:5] == TREE_CODE)
    assert not np.any(grid[:, :, 5:] == TREE_CODE)
    # The model's own DEM must not have been resized under the caller.
    assert city.dem.elevation.shape == (2, 2)


def test_reapply_canopy_resamples_a_coarser_canopy():
    """A canopy at component-grid resolution is resampled, not rejected.

    ``voxelize_citygml_meshes`` resizes canopy_top/canopy_bottom onto the
    voxel grid at build time; rejecting the same input here would make the
    re-apply path stricter than the path that produced the model.  The stored
    component grid keeps the caller's resolution, again as the build path
    does -- ``run_core`` hands ``assemble_voxcity`` the pre-resize arrays.
    """
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = np.zeros((4, 4, _NZ), dtype=np.int16)
    city = _make_city(grid)
    # Flat, so the bilinear resample has an unambiguous answer.
    reapply_canopy(city, np.full((2, 2), 6.0), np.zeros((2, 2)))

    assert np.all(grid[:, :, 0:3] == TREE_CODE)
    assert not np.any(grid[:, :, 3:] == TREE_CODE)
    # Stored at the caller's resolution, matching the other component grids.
    assert city.tree_canopy.top.shape == (2, 2)
    assert city.tree_canopy.bottom.shape == (2, 2)
    assert city.extras["canopy_top"].shape == (2, 2)


def test_reapply_canopy_restores_everything_when_the_overlay_raises():
    """An exception inside the fill must leave the model exactly as it was.

    The clear and the component-grid write both happen before the fill, so
    without a rollback a failure there leaves a canopy-stripped grid and a
    tree_canopy describing crowns that are no longer in it.
    """
    import voxcitygml.reapply as rp
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = _seeded_grid()
    grid[2, 2, 1:6] = TREE_CODE          # canopy from a previous run
    grid[1, 1, 6:8] = TREE_CODE          # a CityGML crown
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    city = _make_city(grid, mask=mask,
                      canopy_top=np.full((4, 4), 10.0),
                      canopy_bottom=np.full((4, 4), 3.0))
    before_grid = grid.copy()
    before_top = city.tree_canopy.top
    before_bottom = city.tree_canopy.bottom
    before_top_values = before_top.copy()

    real_apply = rp._apply_canopy

    def partial_then_raise(voxel_grid, gp, dem, top, bottom, ratio, **kwargs):
        # Write some canopy first, so the rollback has a partial fill to undo
        # and not merely a clear to reverse.
        voxel_grid[0, 3, 1:4] = TREE_CODE
        raise RuntimeError("overlay exploded")

    try:
        rp._apply_canopy = partial_then_raise
        with pytest.raises(RuntimeError, match="overlay exploded"):
            reapply_canopy(city, np.full((4, 4), 4.0), np.zeros((4, 4)))
    finally:
        rp._apply_canopy = real_apply

    assert np.array_equal(grid, before_grid), \
        "the voxel grid was left partially updated"
    assert city.tree_canopy.top is before_top
    assert city.tree_canopy.bottom is before_bottom
    assert np.allclose(city.tree_canopy.top, before_top_values)
    assert np.allclose(city.tree_canopy.bottom, 3.0)
    assert city.extras["canopy_top"] is before_top
    assert city.extras["canopy_bottom"] is before_bottom


def test_reapply_canopy_needs_no_flip_from_the_land_cover_frame():
    """A canopy derived from ``land_cover.classes`` is passed through as-is.

    This is the contract the *Frames* section of ``reapply_canopy`` states,
    and it is the one that was inverted before the assembly seam converted to
    south-up: the old docstring told callers to ``np.flipud`` a canopy built
    in the land-cover frame, which on today's models mirrors it north-south.
    Every grid on an assembled ``VoxCity`` now shares one frame, so the
    round trip land cover -> canopy -> TREE voxels must land on the *same*
    cells with no flip anywhere.

    ``reapply_canopy``'s arithmetic is frame-agnostic -- it only requires its
    arrays to agree -- so what this pins is that neither it nor anything it
    calls sneaks a flip in on the caller's behalf.
    """
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE, GROUND_CODE

    tree_class = 4                      # any land-cover class index
    n_rows, n_cols, band = 9, 4, 3      # rows 0..2 = the southern third

    lc = np.zeros((n_rows, n_cols), dtype=np.int32)
    lc[:band, :] = tree_class
    lc_tree = lc == tree_class

    # The fixture must be asymmetric, or the assertions below are vacuous: a
    # centred band is flip-invariant, which is exactly how the original
    # mirroring survived four reviews and a live A/B.
    assert lc_tree.any(), "no tree cells; the test would compare two empty sets"
    assert not (lc_tree & np.flipud(lc_tree)).any(), \
        "the land-cover band overlaps its own mirror; a flip would be undetectable"

    grid = np.zeros((n_rows, n_cols, _NZ), dtype=np.int16)
    grid[:, :, 0] = GROUND_CODE
    city = _make_city(grid, land_cover=lc)

    # Derived straight from the model's own land cover -- no flipud.
    canopy_top = np.where(lc_tree, 6.0, 0.0)
    reapply_canopy(city, canopy_top, np.zeros((n_rows, n_cols)))

    vox_tree = np.any(city.voxels.classes == TREE_CODE, axis=2)
    assert vox_tree.any(), "no canopy was written; the IoUs below are vacuous"

    direct = _iou(lc_tree, vox_tree)
    flipped = _iou(np.flipud(lc_tree), vox_tree)
    assert direct > flipped, (
        f"the canopy landed mirrored: direct IoU {direct:.4f} <= flipud "
        f"{flipped:.4f}; a caller-side flip is being applied somewhere")
    assert direct > 0.5, \
        f"weak alignment ({direct:.4f}); the canopy did not follow land cover"


def _iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def test_reapply_canopy_does_not_mutate_the_stored_mask():
    """extras['mesh_vegetation_mask'] is an input; re-applying must not edit it."""
    from voxcitygml import reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE

    grid = _seeded_grid()
    grid[1, 1, 6:8] = TREE_CODE
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    city = _make_city(grid, mask=mask)
    before = mask.copy()

    reapply_canopy(city, np.full((4, 4), 6.0), np.zeros((4, 4)))

    assert np.array_equal(mask, before)
    assert city.extras["mesh_vegetation_mask"] is mask
