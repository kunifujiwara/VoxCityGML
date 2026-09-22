import os

import numpy as np
import pytest

from voxcitygml.voxelizer3d import Grid3DParams, frame_extent

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
    _, min_x, max_x, min_y, max_y = frame_extent(RECT, CLON, CLAT)
    assert (min_x, max_x, min_y, max_y) == (gp.min_x, gp.max_x, gp.min_y, gp.max_y)


def test_frame_transformer_inverse_round_trips():
    tr, min_x, max_x, min_y, max_y = frame_extent(RECT, CLON, CLAT)
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


def _box_building(transformer, x0, y0, sx, sy, z0, sz, feature_type="building"):
    """A closed box in the local metre frame, returned as Mesh3D (lat, lon, z)."""
    b = trimesh.creation.box(extents=[sx, sy, sz])
    b.apply_translation([x0 + sx / 2, y0 + sy / 2, z0 + sz / 2])
    v = np.asarray(b.vertices, dtype=np.float64)
    lon, lat = transformer.inverse(v[:, 0], v[:, 1])
    verts = np.column_stack([lat, lon, v[:, 2]])
    return Mesh3D(vertices=verts, faces=np.asarray(b.faces, dtype=np.int32),
                  feature_type=feature_type)


def _base_params_and_collection():
    tr, min_x, max_x, min_y, max_y = frame_extent(RECT, CLON, CLAT)
    vs = 5.0
    n_cols = int((max_x - min_x) / vs + 0.5)
    n_rows = int((max_y - min_y) / vs + 0.5)
    gp = Grid3DParams(n_rows=n_rows, n_cols=n_cols, n_z=12, min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y, min_z=-5.0, max_z=55.0, voxel_size=vs)
    bldg = _box_building(tr, min_x + 100.0, min_y + 120.0, 30.0, 20.0, 0.0, 25.0)
    return gp, CityGMLMeshCollection(buildings=[bldg])


def _bridge_params_and_collection():
    """Same grid as ``_base_params_and_collection``, but with a 60x10x1 m
    bridge deck whose bottom sits 12 m above the (flat, elevation-0) ground,
    instead of a building. Bridges are voxelized as thin shells (surface +
    dilation + fill) rather than a watertight solid, so the air gap under
    the deck is not automatically preserved by construction -- worth its
    own fixture, not a variant of the building one."""
    tr, min_x, max_x, min_y, max_y = frame_extent(RECT, CLON, CLAT)
    vs = 5.0
    n_cols = int((max_x - min_x) / vs + 0.5)
    n_rows = int((max_y - min_y) / vs + 0.5)
    gp = Grid3DParams(n_rows=n_rows, n_cols=n_cols, n_z=12, min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y, min_z=-5.0, max_z=55.0, voxel_size=vs)
    deck_x0, deck_y0 = min_x + 100.0, min_y + 120.0
    deck = _box_building(tr, deck_x0, deck_y0, 60.0, 10.0, 12.0, 1.0, feature_type="bridge")
    return gp, CityGMLMeshCollection(bridges=[deck]), deck_x0, deck_y0


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
    cfg = types.SimpleNamespace(citygml_path="p", max_voxel_ram_mb=2048.0,
                                building_lod=None, dem_path=None, tree_citygml_path=None)
    art = types.SimpleNamespace(
        citygml_paths=["p"], land_cover_source="OpenStreetMap",
        canopy_height_source="Static", dem_source=None, collection="COLL",
        voxel_min_z=-5.0, mesh_vegetation_mask=np.zeros((2, 2), bool),
        flatten_water_dem=True, water_dem_connectivity=4, water_dem_flattening={},
        center_lon=139.7725, center_lat=35.648, buffered_rectangle=RECT)
    ex = assembly_extras(cfg, art)
    assert ex["center_lon"] == 139.7725 and ex["center_lat"] == 35.648
    assert ex["citygml_collection"] == "COLL" and ex["voxel_min_z"] == -5.0
    assert ex["mesh_vegetation_mask"].shape == (2, 2)
    assert ex["buffered_rectangle"] == RECT
    assert ex["max_voxel_ram_mb"] == 2048.0
    assert ex["citygml_building_lod"] is None
    assert ex["dem_path"] is None and ex["tree_citygml_path"] is None


from voxcitygml.refine import (MeshSourceUnavailable, _collection_for,
                               grid_params_from_model, refine_voxel_grids)


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


def _model_from(gp, coll, voxel_grid_north_up, extras_overrides=None, dem=None, canopy_top=None,
                land_cover=None):
    """``dem``/``canopy_top``/``land_cover`` default to the module's
    all-zero (land cover) or asymmetric (``_dem_ramp``, ``_canopy_top``)
    fixtures; pass explicit arrays (e.g. flat zeros, a striped raster) when
    a test's expected numbers were measured against a specific one and the
    default would change them."""
    rows, cols = gp.n_rows, gp.n_cols
    extras = {"rectangle_vertices": RECT, "center_lon": CLON, "center_lat": CLAT,
              "voxel_min_z": gp.min_z, "building_lod": 2, "land_cover_source": "OpenStreetMap",
              "citygml_collection": coll, "flatten_water_dem": True}
    extras.update(extras_overrides or {})
    south = np.ascontiguousarray(np.flipud(voxel_grid_north_up)).astype(np.int8)
    return _Ns(voxels=_Ns(classes=south),
               buildings=_Ns(meta=_Ns(meshsize=gp.voxel_size)),
               land_cover=_Ns(classes=np.zeros((rows, cols), np.int64) if land_cover is None
                              else land_cover),
               dem=_Ns(elevation=_dem_ramp(gp) if dem is None else dem),
               tree_canopy=_Ns(top=_canopy_top(gp) if canopy_top is None else canopy_top,
                               bottom=None),
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


def test_collection_for_reparse_forwards_the_pipelines_settings(tmp_path, monkeypatch):
    """No meshlib needed -- this pins ``_collection_for``'s re-parse kwargs
    directly, without voxelizing anything.  Three settings used to be
    dropped on the re-parse path (``dem_path``, ``tree_citygml_path``, and
    the buffered rectangle used to build the terrain solid), and
    ``building_lod`` was conflated with a same-named TAG some callers write
    that is not a parser preference; this is the test that would have
    caught all four."""
    citygml_dir = tmp_path / "citygml"
    citygml_dir.mkdir()
    buffered = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    extras = {
        "citygml_collection": None,          # lost on save/load
        "citygml_paths": [str(citygml_dir)],
        "rectangle_vertices": RECT,
        "buffered_rectangle": buffered,
        "citygml_building_lod": 2,
        "dem_path": "some_dem.tif",
        "tree_citygml_path": "some_trees_dir",
    }
    city = _Ns(extras=extras)

    captured = {}

    def fake_parse(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return CityGMLMeshCollection()

    monkeypatch.setattr("voxcitygml.citygml.parser.parse_citygml_directory", fake_parse)

    result = _collection_for(city, None)

    assert isinstance(result, CityGMLMeshCollection)
    assert captured["path"] == str(citygml_dir)
    assert captured["rectangle_vertices"] == buffered   # not rectangle_vertices
    assert captured["building_lod"] == 2
    assert captured["dem_path"] == "some_dem.tif"
    assert captured["tree_citygml_path"] == "some_trees_dir"
    assert captured["feature_types"] == ["terrain", "building", "bridge", "vegetation"]


def test_collection_for_reparse_records_a_none_lod_preference_faithfully(tmp_path, monkeypatch):
    """``citygml_building_lod`` must be distinguishable from an absent key:
    ``None`` here means the base run's parser preference genuinely was
    "highest available", not "unknown -- fall back to the legacy tag"."""
    citygml_dir = tmp_path / "citygml"
    citygml_dir.mkdir()
    extras = {
        "citygml_collection": None,
        "citygml_paths": [str(citygml_dir)],
        "rectangle_vertices": RECT,
        "citygml_building_lod": None,
        "building_lod": 3,   # a legacy tag that must NOT be used instead
    }
    city = _Ns(extras=extras)

    captured = {}

    def fake_parse(path, **kwargs):
        captured.update(kwargs)
        return CityGMLMeshCollection()

    monkeypatch.setattr("voxcitygml.citygml.parser.parse_citygml_directory", fake_parse)

    _collection_for(city, None)
    assert captured["building_lod"] is None


@needs_meshlib
def test_refine_voxel_grids_max_height_m_shortens_the_column():
    """``max_height_m`` clips BEFORE voxelizing, so a clip with no guard cell
    lets ``_apply_land_cover`` recolour the ARTIFICIAL ceiling as if it were
    the true ground surface -- a false land-cover skin that only the clipped
    levels would carry (level 0 in voxcitywind's stack is clipped AFTER
    voxelization, by ``clip_height``, and never sees it). The fix voxelizes
    one guard base cell above the clip and slices it back off, so a clipped
    run must agree with an unclipped run EXACTLY over the retained cells --
    not just share a shape and contain a building somewhere."""
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    grid = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=dem,
                                   grid_params=gp)
    city = _model_from(gp, coll, grid)
    k = 3   # ceil(12/5) = 3 base cells
    (clipped,) = refine_voxel_grids(city, (2,), max_height_m=12.0)
    (full,) = refine_voxel_grids(city, (2,))
    assert clipped.shape == (2 * gp.n_rows, 2 * gp.n_cols, 2 * k)
    assert np.array_equal(clipped, full[:, :, :2 * k])
    assert (clipped == BUILDING_CODE).any()


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
    was 27.5 m against a base built with the SAME ramp -- nearly 8x the
    3.5 m tolerance used below -- and the canopy block assertion failed
    outright (no TREE_CODE landed in the expected fine block). Restored
    before committing.
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
    # Tolerance chosen with real headroom: with the DEM pre-resampled at
    # grid_mode=True (refine_voxel_grids), this measures 2.5 m (one FINE
    # voxel) on this fixture -- down from 5.0 m (one base voxel, zero
    # margin against a 5.000001 m tolerance) before that fix, when the
    # voxelizer's own endpoint-aligned resize introduced a phase error on
    # top of the quantization. 3.5 m leaves 1.0 m of margin above the
    # measured value while staying far below the 27-30 m a dropped/
    # duplicated flip produces (see the docstring above).
    assert diff <= 3.5, (
        f"refined terrain top disagrees with the base by {diff} m at column 0 "
        "(tolerance 3.5 m) -- looks like a dropped/duplicated DEM flip")

    # Canopy: base row 5 must carry TREE_CODE (that is where the spike is),
    # and the refined grid's corresponding 2x2 block (south-up rows 10-11,
    # cols 10-11) must too.
    assert (base_south[5, 5, :] == TREE_CODE).any()
    assert (fine[10:12, 10:12, :] == TREE_CODE).any(), (
        "no TREE_CODE in the refined block that should nest under the base "
        "canopy spike at row 5 -- looks like a dropped/duplicated canopy flip")


@needs_meshlib
def test_refine_voxel_grids_keeps_air_under_a_bridge_deck_at_every_level():
    """The design calls the air under a bridge deck load-bearing: an
    obstruction analysis (wind, sunlight) needs the space UNDER a deck to
    stay open at every refinement level, not get swallowed by the bridge's
    own thin-shell fill. Measured directly rather than merely probed: of
    the cells from the grid floor up to (excluding) the deck's first solid
    cell, the AIR fraction is 1/2, 3/5 and 8/12 at factors 1, 2 and 4 --
    increasing with resolution (the deck's fixed 1 m thickness costs a
    shrinking share of ever-finer cells), and never zero."""
    gp, coll, deck_x0, deck_y0 = _bridge_params_and_collection()
    flat_dem = np.zeros((gp.n_rows, gp.n_cols))
    base_grid = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size,
                                        dem_grid=flat_dem, grid_params=gp)
    # Flat DEM, not the module's default ramp: the expected air/total counts
    # below were measured against elevation 0 everywhere.
    city = _model_from(gp, coll, base_grid, dem=flat_dem, canopy_top=np.zeros((gp.n_rows, gp.n_cols)))

    grids = refine_voxel_grids(city, (1, 2, 4))

    deck_cx, deck_cy = deck_x0 + 30.0, deck_y0 + 5.0
    expected = {1: (1, 2), 2: (3, 5), 4: (8, 12)}
    for f, grid in zip((1, 2, 4), grids):
        gpf = gp.refined(f)
        col = int((deck_cx - gpf.min_x) / gpf.voxel_size)
        row_north = int((gpf.max_y - deck_cy) / gpf.voxel_size)
        row_south = gpf.n_rows - 1 - row_north
        column = grid[row_south, col, :]
        building_idx = np.where(column == BUILDING_CODE)[0]
        assert building_idx.size > 0, f"no deck found under its own footprint at factor {f}"
        deck_bottom = int(building_idx.min())
        air_below = int(np.sum(column[:deck_bottom] == 0))
        assert air_below > 0, f"no air left under the deck at factor {f}"
        assert (air_below, deck_bottom) == expected[f], (
            f"factor {f}: expected {expected[f]} (air, total) cells below the "
            f"deck, got {(air_below, deck_bottom)}")


def _land_cover_skin(grid):
    """The (positive) land-cover code painted in each column, or -999 where
    none was (e.g. no ground surface at all). With a flat DEM and no
    buildings/vegetation there is at most one positive cell per column, so
    ``max`` over z picks it out unambiguously (every other class code is
    <= 0)."""
    has_positive = (grid > 0).any(axis=2)
    return np.where(has_positive, grid.max(axis=2), -999)


@needs_meshlib
def test_refine_voxel_grids_resamples_land_cover_by_nearest_repeat_at_factor4():
    """Land cover must resample by nearest-neighbour ``np.repeat`` nesting,
    not the endpoint-aligned ``zoom(order=0)`` the voxelizer's own
    ``_resize_int_grid`` would otherwise apply -- see
    ``_resample_land_cover``'s docstring for why a phase error here is
    worse than a class relabel (it moves the water mask feeding the DEM
    carve, not just a surface label). Factor 2 passes even without the
    fix on this fixture (coincidence); factor 4 is where the old path
    measurably displaced rows, so it is pinned here specifically."""
    gp, _ = _base_params_and_collection()
    coll = CityGMLMeshCollection()   # empty: no buildings to complicate the skin
    flat_dem = np.zeros((gp.n_rows, gp.n_cols))
    striped_lc = np.tile((np.arange(gp.n_rows) % 3)[:, None], (1, gp.n_cols)).astype(np.int64)

    base_grid = voxelize_citygml_meshes(
        coll, RECT, CLON, CLAT, gp.voxel_size, dem_grid=flat_dem,
        land_cover_grid=striped_lc, land_cover_source="OpenStreetMap", grid_params=gp)
    city = _model_from(gp, coll, base_grid, dem=flat_dem,
                       canopy_top=np.zeros((gp.n_rows, gp.n_cols)), land_cover=striped_lc)

    (fine,) = refine_voxel_grids(city, (4,))

    base_skin = _land_cover_skin(city.voxels.classes)
    fine_skin = _land_cover_skin(fine)
    expected = np.repeat(np.repeat(base_skin, 4, axis=0), 4, axis=1)
    assert fine_skin.shape == expected.shape
    mismatched = int(np.sum(fine_skin != expected))
    assert mismatched == 0, (
        f"{mismatched} of {fine_skin.size} refined land-cover cells are not "
        "the np.repeat nesting of the base's -- looks like the endpoint-"
        "aligned zoom path, not nearest-neighbour np.repeat")
