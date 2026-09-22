import os

import numpy as np
import pytest

from voxcitygml.voxelizer3d import Grid3DParams, _frame_extent

RECT = [(139.770, 35.646), (139.770, 35.650), (139.775, 35.650), (139.775, 35.646)]
CLON = (RECT[0][0] + RECT[2][0]) / 2
CLAT = (RECT[0][1] + RECT[2][1]) / 2


def test_refined_params_keep_the_frame_and_datum():
    gp = Grid3DParams(n_rows=12, n_cols=10, n_z=8, min_x=-6.0, max_x=18.0,
                      min_y=-6.9, max_y=18.0, min_z=-6.0, max_z=10.0, voxel_size=2.0)
    r = gp.refined(2)
    assert (r.n_rows, r.n_cols, r.n_z) == (24, 20, 16)
    assert (r.min_x, r.max_x, r.min_y, r.max_y, r.min_z, r.max_z) == \
        (gp.min_x, gp.max_x, gp.min_y, gp.max_y, gp.min_z, gp.max_z)
    assert r.voxel_size == 1.0
    with pytest.raises(ValueError):
        gp.refined(0)


def test_refined_indices_nest_inside_coarse_cells():
    gp = Grid3DParams(n_rows=12, n_cols=10, n_z=8, min_x=-6.0, max_x=18.0,
                      min_y=-6.9, max_y=18.0, min_z=-6.0, max_z=10.0, voxel_size=2.0)
    r = gp.refined(4)
    rng = np.random.default_rng(0)
    x = rng.uniform(gp.min_x, gp.min_x + gp.n_cols * 2.0, 500)
    y = rng.uniform(gp.max_y - gp.n_rows * 2.0, gp.max_y, 500)
    z = rng.uniform(gp.min_z, gp.min_z + gp.n_z * 2.0, 500)
    for c, f in zip(gp.xyz_to_indices(x, y, z), r.xyz_to_indices(x, y, z)):
        assert np.array_equal(f // 4, c)


def test_frame_extent_matches_compute_grid_params_3d():
    from voxcitygml.models import CityGMLMeshCollection
    from voxcitygml.voxelizer3d import _compute_grid_params_3d
    gp, _ = _compute_grid_params_3d(RECT, CLON, CLAT, 5.0, CityGMLMeshCollection())
    _, min_x, max_x, min_y, max_y = _frame_extent(RECT, CLON, CLAT)
    assert (min_x, max_x, min_y, max_y) == (gp.min_x, gp.max_x, gp.min_y, gp.max_y)


def test_frame_transformer_inverse_round_trips():
    tr, min_x, max_x, min_y, max_y = _frame_extent(RECT, CLON, CLAT)
    rng = np.random.default_rng(1)
    x = rng.uniform(min_x, max_x, 50)
    y = rng.uniform(min_y, max_y, 50)
    lon, lat = tr.inverse(x, y)
    x2, y2 = tr.transform(lon, lat)
    assert np.allclose(x2, x, atol=1e-6) and np.allclose(y2, y, atol=1e-6)


import trimesh

from voxcitygml.models import Mesh3D, CityGMLMeshCollection
from voxcitygml.voxelizer3d import (_MESHLIB_VOXEL_AVAILABLE, BUILDING_CODE,
                                    GROUND_CODE, TREE_CODE, voxelize_citygml_meshes)

needs_meshlib = pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE,
                                   reason="building fill needs meshlib")


def test_meshlib_available_or_explicitly_opted_out():
    """The non-skippable guard (see ``tests/test_inclusive_voxelization.py``,
    where this pattern originates).  Every ``needs_meshlib``-marked test in
    this file exercises re-voxelization through actual building fill, and
    would happily vanish into "skipped" if meshlib were missing -- exactly
    the configuration where none of ``voxcitygml.refine``'s re-voxelization
    behaviour, nor ``grid_params=``'s alignment, gets checked at all.  This
    test has no skipif: it fails the run unless meshlib is present or a
    human explicitly opted out via VOXCITYGML_ALLOW_NO_MESHLIB=1."""
    if _MESHLIB_VOXEL_AVAILABLE:
        return
    if os.environ.get("VOXCITYGML_ALLOW_NO_MESHLIB"):
        pytest.skip(
            "meshlib unavailable; VOXCITYGML_ALLOW_NO_MESHLIB=1 explicitly "
            "accepts that every meshlib-dependent test in this module will "
            "now skip too -- this file verifies nothing about refined "
            "re-voxelization in this run.")
    pytest.fail(
        "meshlib is not installed, so every meshlib-dependent test in "
        "tests/test_refine.py is about to SKIP rather than run -- a green "
        "suite in that state does not mean voxcitygml.refine's re-"
        "voxelization (grid_params alignment, orientation flips, "
        "max_height_m clipping) holds; it means it was never checked.  "
        "Install meshlib, or set VOXCITYGML_ALLOW_NO_MESHLIB=1 to "
        "explicitly accept an unverified run.",
        pytrace=False)


def _box_building(transformer, x0, y0, sx, sy, z0, sz):
    """A closed box in the local metre frame, returned as Mesh3D (lat, lon, z)."""
    b = trimesh.creation.box(extents=[sx, sy, sz])
    b.apply_translation([x0 + sx / 2, y0 + sy / 2, z0 + sz / 2])
    v = np.asarray(b.vertices, dtype=np.float64)
    lon, lat = transformer.inverse(v[:, 0], v[:, 1])
    verts = np.column_stack([lat, lon, v[:, 2]])
    return Mesh3D(vertices=verts, faces=np.asarray(b.faces, dtype=np.int32),
                  feature_type="building")


def _base_params_and_collection():
    tr, min_x, max_x, min_y, max_y = _frame_extent(RECT, CLON, CLAT)
    vs = 5.0
    n_cols = int((max_x - min_x) / vs + 0.5)
    n_rows = int((max_y - min_y) / vs + 0.5)
    gp = Grid3DParams(n_rows=n_rows, n_cols=n_cols, n_z=12, min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y, min_z=-5.0, max_z=55.0, voxel_size=vs)
    bldg = _box_building(tr, min_x + 100.0, min_y + 120.0, 30.0, 20.0, 0.0, 25.0)
    return gp, CityGMLMeshCollection(buildings=[bldg])


@needs_meshlib
def test_explicit_grid_params_are_used_verbatim_and_refine_nests():
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    coarse = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size,
                                     dem_grid=dem, grid_params=gp)
    fine = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size / 2,
                                   dem_grid=dem, grid_params=gp.refined(2))
    assert coarse.shape == (gp.n_rows, gp.n_cols, gp.n_z)
    assert fine.shape == (2 * gp.n_rows, 2 * gp.n_cols, 2 * gp.n_z)
    nb0 = int((coarse == BUILDING_CODE).sum())
    nb2 = int((fine == BUILDING_CODE).sum())
    assert nb0 > 0 and 4 * nb0 <= nb2 <= 16 * nb0
    # every coarse building cell has at least one building child
    child_any = (fine == BUILDING_CODE).reshape(
        gp.n_rows, 2, gp.n_cols, 2, gp.n_z, 2).any(axis=(1, 3, 5))
    covered = child_any[coarse == BUILDING_CODE].mean()
    assert covered >= 0.9


def test_assembly_extras_carry_the_frame_centre():
    from voxcitygml.pipeline import assembly_extras
    import types
    cfg = types.SimpleNamespace(citygml_path="p")
    art = types.SimpleNamespace(
        citygml_paths=["p"], land_cover_source="OpenStreetMap",
        canopy_height_source="Static", dem_source=None, collection="COLL",
        voxel_min_z=-5.0, mesh_vegetation_mask=np.zeros((2, 2), bool),
        flatten_water_dem=True, water_dem_connectivity=4, water_dem_flattening={},
        center_lon=139.7725, center_lat=35.648)
    ex = assembly_extras(cfg, art)
    assert ex["center_lon"] == 139.7725 and ex["center_lat"] == 35.648
    assert ex["citygml_collection"] == "COLL" and ex["voxel_min_z"] == -5.0
    assert ex["mesh_vegetation_mask"].shape == (2, 2)


from voxcitygml.refine import (MeshSourceUnavailable, grid_params_from_model,
                               refine_voxel_grids)


class _Ns(dict):
    """Attribute access over a dict, for VoxCity look-alikes."""
    __getattr__ = dict.__getitem__


def _dem_ramp(gp):
    """South-up DEM: strictly increasing with the row index (south -> north
    in the south-up convention), so a dropped or duplicated flipud disagrees
    with the correct answer almost everywhere -- unlike an all-zero DEM,
    which cannot distinguish the two at all."""
    ramp = np.arange(gp.n_rows, dtype=np.float64) * 0.3
    return np.repeat(ramp[:, None], gp.n_cols, axis=1)


def _canopy_top(gp):
    """South-up canopy top: a single non-zero cell in a known off-centre row
    (row 5, far from both the middle row and the test building's footprint),
    so a dropped or duplicated flip moves the tree to a different row/block
    instead of merely changing a magnitude."""
    top = np.zeros((gp.n_rows, gp.n_cols))
    top[5, 5] = 8.0
    return top


def _model_from(gp, coll, voxel_grid_north_up, extras_overrides=None):
    rows, cols = gp.n_rows, gp.n_cols
    extras = {"rectangle_vertices": RECT, "center_lon": CLON, "center_lat": CLAT,
              "voxel_min_z": gp.min_z, "building_lod": 2, "land_cover_source": "OpenStreetMap",
              "citygml_collection": coll, "flatten_water_dem": True}
    extras.update(extras_overrides or {})
    south = np.ascontiguousarray(np.flipud(voxel_grid_north_up)).astype(np.int8)
    return _Ns(voxels=_Ns(classes=south),
               buildings=_Ns(meta=_Ns(meshsize=gp.voxel_size)),
               land_cover=_Ns(classes=np.zeros((rows, cols), np.int64)),
               dem=_Ns(elevation=_dem_ramp(gp)),
               tree_canopy=_Ns(top=_canopy_top(gp), bottom=None),
               extras=extras)


@needs_meshlib
def test_grid_params_from_model_reproduces_the_frame():
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    grid = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=dem,
                                   grid_params=gp)
    city = _model_from(gp, coll, grid)
    got = grid_params_from_model(city)
    assert (got.n_rows, got.n_cols, got.n_z) == (gp.n_rows, gp.n_cols, gp.n_z)
    assert got.min_x == pytest.approx(gp.min_x) and got.max_y == pytest.approx(gp.max_y)
    assert got.min_z == gp.min_z and got.voxel_size == gp.voxel_size


@needs_meshlib
def test_refine_voxel_grids_returns_aligned_south_up_grids():
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    grid = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=dem,
                                   grid_params=gp)
    city = _model_from(gp, coll, grid)
    (g2,) = refine_voxel_grids(city, (2,))
    base = city.voxels.classes
    assert g2.shape == tuple(2 * s for s in base.shape) and g2.dtype == base.dtype
    # south-up like the base: the building's row band doubles in place
    rows0 = np.where((base == BUILDING_CODE).any(axis=(1, 2)))[0]
    rows2 = np.where((g2 == BUILDING_CODE).any(axis=(1, 2)))[0]
    assert abs(rows2.min() / 2 - rows0.min()) <= 1 and abs(rows2.max() / 2 - rows0.max()) <= 1


@needs_meshlib
def test_refine_voxel_grids_accepts_an_explicit_collection_when_extras_lost_it():
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    grid = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=dem,
                                   grid_params=gp)
    city = _model_from(gp, coll, grid, {"citygml_collection": None})
    (g2,) = refine_voxel_grids(city, (2,), collection=coll)
    assert (g2 == BUILDING_CODE).any()


def test_refine_voxel_grids_without_meshes_or_paths_raises_named_error():
    gp, coll = _base_params_and_collection()
    city = _model_from(gp, coll, np.zeros((gp.n_rows, gp.n_cols, gp.n_z), np.int16),
                       {"citygml_collection": None, "citygml_paths": None})
    with pytest.raises(MeshSourceUnavailable, match="citygml_collection"):
        refine_voxel_grids(city, (2,))


@needs_meshlib
def test_refine_voxel_grids_max_height_m_shortens_the_column():
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    grid = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=dem,
                                   grid_params=gp)
    city = _model_from(gp, coll, grid)
    (g2,) = refine_voxel_grids(city, (2,), max_height_m=12.0)   # ceil(12/5) = 3 base cells
    assert g2.shape == (2 * gp.n_rows, 2 * gp.n_cols, 6)
    assert (g2 == BUILDING_CODE).any()


def test_grid_params_from_model_refuses_a_grid_that_does_not_match_the_frame():
    gp, coll = _base_params_and_collection()
    wrong = np.zeros((gp.n_rows + 1, gp.n_cols, gp.n_z), np.int16)
    city = _model_from(gp, coll, wrong)
    with pytest.raises(ValueError, match="frame"):
        grid_params_from_model(city)


def _ground_top_z(grid_south, gp):
    """Per-column physical z (m) of the top face of the tallest ground-surface
    cell in a south-up grid, or NaN where there is none.  ``refine_voxel_grids``
    always passes a ``land_cover_grid``, and ``_apply_land_cover`` recolours
    the topmost GROUND_CODE cell of every column to a positive land-cover
    code (see ``_ground_surface_index``'s docstring in voxelizer3d.py) -- so
    "ground surface" here, like there, means GROUND_CODE OR any positive
    code, not GROUND_CODE alone."""
    is_ground = (grid_south == GROUND_CODE) | (grid_south > 0)
    has_any = is_ground.any(axis=2)
    top_idx_from_end = np.argmax(is_ground[:, :, ::-1], axis=2)
    idx = is_ground.shape[2] - 1 - top_idx_from_end
    z = gp.min_z + (idx + 1) * gp.voxel_size
    return np.where(has_any, z, np.nan)


@needs_meshlib
def test_refine_voxel_grids_applies_the_dem_and_canopy_flips_correctly():
    """``refine.py`` flips the model's south-up DEM/canopy north-up before
    handing them to the voxelizer, then flips the result back south-up.  The
    OTHER tests in this file build ``_model_from`` on an all-zero DEM and
    canopy, so dropping or duplicating either flip would not fail any of
    them -- a flipped or unflipped all-zero raster is identical.  This test
    uses the module's asymmetric fixtures (``_dem_ramp``, ``_canopy_top``)
    instead: a ramp monotone in the row index, and an off-centre canopy
    spike, both of which disagree with themselves under a mirror almost
    everywhere.

    Verified this fails without the fix: with the DEM's ``np.flipud`` in
    ``refine.py`` removed, the max column-0 terrain-top disagreement below
    was 27.5 m (5.5 coarse voxels) against a base built with the SAME
    ramp -- far past the one-coarse-voxel quantization tolerance used here
    -- and the canopy block assertion failed outright (no TREE_CODE landed
    in the expected fine block). Restored before committing.
    """
    gp, coll = _base_params_and_collection()
    dem_north = np.ascontiguousarray(np.flipud(_dem_ramp(gp)))
    canopy_north = np.ascontiguousarray(np.flipud(_canopy_top(gp)))
    lc = np.zeros((gp.n_rows, gp.n_cols), np.int64)

    # Ground truth: voxelize the base grid directly, doing the same
    # north-up flip refine.py does (and passing the same all-zero
    # land_cover_grid it does, since a non-None land_cover_grid makes
    # ``_apply_land_cover`` recolour the topmost GROUND_CODE cell to a
    # positive code -- see ``_ground_top_z`` above), so this does not just
    # re-exercise refine.py's own flip on both sides of the comparison.
    base_grid = voxelize_citygml_meshes(
        coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=dem_north,
        canopy_top=canopy_north, land_cover_grid=lc,
        land_cover_source="OpenStreetMap", grid_params=gp)
    city = _model_from(gp, coll, base_grid)  # dem/canopy default to the same ramp/spike
    base_south = city.voxels.classes

    (fine,) = refine_voxel_grids(city, (2,))

    # Terrain: column 0 never touches the test building's footprint
    # (columns ~20-26), so its ground top is DEM-driven throughout.
    base_top = _ground_top_z(base_south, gp)
    fine_top = _ground_top_z(fine, gp.refined(2))
    base_col0 = base_top[:, 0]
    fine_col0_down = fine_top[0::2, 0][:len(base_col0)]
    diff = np.nanmax(np.abs(fine_col0_down - base_col0))
    assert diff <= gp.voxel_size + 1e-6, (
        f"refined terrain top disagrees with the base by {diff} m at column 0 "
        f"(tolerance {gp.voxel_size} m, one base voxel of quantization slack) "
        "-- looks like a dropped/duplicated DEM flip")

    # Canopy: base row 5 must carry TREE_CODE (that is where the spike is),
    # and the refined grid's corresponding 2x2 block (south-up rows 10-11,
    # cols 10-11) must too.
    assert (base_south[5, 5, :] == TREE_CODE).any()
    assert (fine[10:12, 10:12, :] == TREE_CODE).any(), (
        "no TREE_CODE in the refined block that should nest under the base "
        "canopy spike at row 5 -- looks like a dropped/duplicated canopy flip")
