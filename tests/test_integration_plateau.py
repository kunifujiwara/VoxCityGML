"""End-to-end LOD2 integration test against a local PLATEAU dataset.

Skipped automatically when the dataset is not present (e.g. CI).

Set ``VOXCITYGML_PLATEAU_TEST_DATA`` to run it against a dataset stored
somewhere other than the default path below -- any PLATEAU dataset
directory (the one containing ``udx/``) whose coverage includes the
target rectangle will do.

The dataset-backed tests are marked ``slow`` because they parse the
intersecting CityGML tiles and voxelize them. The axis-aligned end-to-end
run takes roughly 35-125 s, most of it waiting on live OpenStreetMap land
cover; the rotated run (CityGML land cover, fully offline) and the
parse-cache round trip take a few seconds each. They run by default
because they are the proof the whole chain works; skip them explicitly
with ``pytest -m "not slow"``.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

BUILDING_CODE = -3

DATASET = os.environ.get(
    "VOXCITYGML_PLATEAU_TEST_DATA",
    r"D:\03_Data\citygml\plateau\13102_chuo-ku_pref_2023_citygml_2_op",
)

requires_dataset = pytest.mark.skipif(
    not os.path.isdir(DATASET),
    reason="local PLATEAU dataset not available",
)

# Measured on the reference rectangle below (Chuo-ku, 200 m, 2 m voxels):
# LOD2 scores 0.151, LOD1 scores 0.035. 0.08 sits between the two, more than
# 2x above the LOD1 figure and about half the LOD2 figure, so it tolerates the
# run-to-run variation from live OpenStreetMap land cover without going soft.
MIN_ROOF_SLOPE_FRACTION = 0.08


def roof_slope_fraction(classes: np.ndarray) -> float:
    """Fraction of adjacent building-column pairs whose roofs differ by 1 voxel.

    This is what separates true LOD2 roof geometry from a flat LOD1 extrusion.
    An extruded prism is flat-topped: every column inside one footprint tops out
    at the same z (delta 0), and where two parts of differing height meet, the
    roof steps by many voxels at once (a cliff). Only genuinely sloped or
    stepped roof geometry yields a large population of *single-voxel* steps.

    Deliberately measured on ``voxels.classes``: on a LOD2 model the 2.5-D
    component grids (``buildings.heights`` / ``min_heights``) carry only
    footprint and min/max height, so the roof detail exists nowhere else.
    """
    is_bldg = classes == BUILDING_CODE
    has_bldg = is_bldg.any(axis=2)
    # Highest occupied z per column (argmax over the reversed z axis).
    top = classes.shape[2] - 1 - np.argmax(is_bldg[:, :, ::-1], axis=2)
    top = top.astype(np.int64)

    deltas = []
    for axis in (0, 1):
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        both = has_bldg[lo] & has_bldg[hi]
        deltas.append(np.abs(top[lo] - top[hi])[both])

    d = np.concatenate(deltas)
    if d.size == 0:
        return 0.0
    return float((d == 1).mean())


# ---------------------------------------------------------------------------
# Negative control: the metric must reject flat geometry.
#
# Runs without the dataset, so CI still guards the discriminator itself even
# when the slow end-to-end test below is skipped.
# ---------------------------------------------------------------------------

def _flat_extrusion_grid():
    """A two-part flat-topped prism -- the shape LOD1 produces."""
    g = np.zeros((20, 20, 30), dtype=np.int16)
    g[4:16, 4:10, 0:12] = BUILDING_CODE     # part A, flat top at z=11
    g[4:16, 10:16, 0:20] = BUILDING_CODE    # part B, flat top at z=19 (cliff)
    return g


def _pitched_roof_grid():
    """A gabled roof: the ridge slopes one voxel per column."""
    g = np.zeros((20, 20, 30), dtype=np.int16)
    for j in range(4, 16):
        h = 12 + min(j - 4, 15 - j)         # rises to a ridge, then falls
        g[4:16, j, 0:h] = BUILDING_CODE
    return g


def test_roof_slope_fraction_rejects_flat_extrusion():
    """Guards the discriminator: a flat extrusion must score ~0, a pitched roof
    must clear the threshold. Without this, a metric that silently stopped
    discriminating would let the LOD2 assertion below pass on LOD1 output."""
    flat = roof_slope_fraction(_flat_extrusion_grid())
    pitched = roof_slope_fraction(_pitched_roof_grid())

    assert flat == 0.0, f"flat extrusion scored {flat}, expected 0"
    assert flat < MIN_ROOF_SLOPE_FRACTION < pitched, (
        f"threshold {MIN_ROOF_SLOPE_FRACTION} does not separate "
        f"flat={flat} from pitched={pitched}")


def test_roof_slope_fraction_handles_empty_grid():
    """No buildings must not raise (zero adjacent pairs)."""
    assert roof_slope_fraction(np.zeros((5, 5, 5), dtype=np.int16)) == 0.0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@requires_dataset
@pytest.mark.slow
def test_lod2_generate_voxcity_end_to_end(tmp_path):
    from voxcitygml import generate_voxcity, VoxelizerConfig
    from voxcitygml.voxelizer3d import BUILDING_CODE as PIPELINE_BUILDING_CODE
    from voxcitygml.citygml.coordinates import create_rectangle

    # -3 is the building class code the VoxCity app and its renderer rely on;
    # pin it here so a change in either repo surfaces as a test failure rather
    # than as silently mis-classified voxels downstream.
    assert PIPELINE_BUILDING_CODE == BUILDING_CODE == -3

    # A point inside PLATEAU tile 53393671 (Chuo-ku, Tokyo).
    rect = create_rectangle(139.7725, 35.6481, 200)
    cfg = VoxelizerConfig(
        citygml_path=DATASET,
        rectangle_vertices=rect,
        meshsize=2.0,
        building_lod=2,
        land_cover_source="OpenStreetMap",   # no GEE dependency
        canopy_height_source="Static",
        output_dir=str(tmp_path),
        save_output=False,
        # Parse the XML for real every run. With the cache on (the default)
        # this test would exercise the parse cache instead of the parser
        # after its first run -- and would deposit tens of megabytes into
        # the user's dataset directory as a side effect.
        use_parse_cache=False,
    )
    city = generate_voxcity(cfg)

    classes = city.voxels.classes
    assert classes.ndim == 3

    n_building = int(np.count_nonzero(classes == BUILDING_CODE))
    slope = roof_slope_fraction(classes)
    print(f"\nbuilding voxels: {n_building}, grid shape: {classes.shape}, "
          f"roof slope fraction: {slope:.3f}")
    assert n_building > 0, "expected building voxels in the grid"

    heights = city.buildings.heights
    assert np.any(heights > 0)

    # The feature's central claim: LOD2 gives true roof geometry, not a flat
    # extrusion. Re-running this config at building_lod=1 scores 0.035, so a
    # regression that quietly rebuilt the grid from the 2.5-D height grids
    # (as nDSM canopy refinement once did on the app side) fails here.
    assert slope > MIN_ROOF_SLOPE_FRACTION, (
        f"roof slope fraction {slope:.3f} <= {MIN_ROOF_SLOPE_FRACTION}: the "
        f"voxel grid looks like a flat LOD1 extrusion, not LOD2 roof geometry")


# ---------------------------------------------------------------------------
# End-to-end on a ROTATED rectangle
#
# The axis-aligned test above cannot see a rotation-frame bug: at theta ~ 0
# the rectangle-frame transformer degenerates to the plain local transformer,
# so every frame choice gives the same grid. This is the headline test for
# rotated-rectangle support.
# ---------------------------------------------------------------------------

@requires_dataset
@pytest.mark.slow
def test_rotated_rectangle_end_to_end(tmp_path):
    """A 30 deg rotated rectangle produces a valid LOD2 model with true roofs.

    Uses ``geodesic_rect`` -- the *production* construction, byte-for-byte the
    arithmetic of the app's ``/api/rectangle-from-dimensions`` -- so the input
    is exactly the shape the app sends. (Note the sign: the endpoint applies
    ``-radians(rotation_deg)`` to the local-frame corner offsets, so a naive
    ``+radians`` helper builds the mirror-image rectangle.)
    """
    from voxcitygml import generate_voxcity, VoxelizerConfig

    from .geo_helpers import geodesic_rect

    # 200 m wide (NW->NE, the column axis) x 150 m tall (NW->SW, the row axis),
    # turned 30 deg. Non-square on purpose: a row/col transpose in the rotated
    # frame -- the exact failure the old unrotated frame produced at 90 deg --
    # changes the shape, so the assertion below catches it.
    rect = geodesic_rect(139.7725, 35.6481, 200.0, 150.0, 30.0)
    cfg = VoxelizerConfig(
        citygml_path=DATASET,
        rectangle_vertices=rect,
        meshsize=2.0,
        building_lod=2,
        # Every assertion below is about geometry, and CityGML land cover is
        # parsed from the dataset itself: fully offline and deterministic,
        # unlike live Overpass (which rate-limits after a few consecutive
        # fetches and then stalls indefinitely rather than erroring). It also
        # makes this an end-to-end exercise of the CityGML land-cover
        # rasteriser on a rotated rectangle. Switching the source moved none
        # of the numbers below.
        land_cover_source="CityGML",
        canopy_height_source="Static",
        output_dir=str(tmp_path),
        save_output=False,
        use_parse_cache=False,   # keep this test exercising the parser
    )
    city = generate_voxcity(cfg)
    classes = city.voxels.classes

    n_building = int(np.count_nonzero(classes == BUILDING_CODE))
    slope = roof_slope_fraction(classes)
    ground = (classes != 0).any(axis=2)
    n_empty = int((~ground).sum())
    print(f"\nrotated 30 deg: grid shape {classes.shape}, "
          f"building voxels {n_building}, roof slope {slope:.3f}, "
          f"empty columns {n_empty}")

    # NOTE on what each assertion below actually catches. Reverting
    # ``_compute_grid_params_3d`` to the unrotated ``create_local_transformer``
    # was measured to give shape (115, 124, 37) -- caught by the shape
    # assertion -- while roof slope (0.150) and empty columns (0) were
    # *unaffected*. Those two are LOD2-quality and coverage guards, not
    # frame guards. The frame guard with teeth beyond shape is the 2-D/3-D
    # footprint IoU below.

    # Grid dims follow the rectangle's own sides, not the bbox of its corners
    # in an unrotated frame -- which at 30 deg spans 200*cos30 + 150*sin30 =
    # 248 m by 200*sin30 + 150*cos30 = 230 m, i.e. (115, 124) cells instead of
    # (75, 100). That is what the pre-fix code produced, and it is far outside
    # the +-2 tolerance below.
    assert abs(classes.shape[0] - 75) <= 2, classes.shape
    assert abs(classes.shape[1] - 100) <= 2, classes.shape

    # Measured 19,872 here vs 17,952 for the same rectangle at rotation 0 --
    # the 10% spread is different ground being covered, not a frame error.
    # The bound is deliberately loose: it only has to reject "the rotated
    # frame landed the buildings outside the grid".
    assert n_building > 1000, f"too few building voxels: {n_building}"

    # The two frames agree on real geometry, not just on grid dimensions.
    # ``buildings.heights`` is rasterized in the 2-D affine GridParams frame;
    # the voxel grid is built in the 3-D rotated-tmerc frame. If the two
    # disagreed in orientation the same buildings would land in different
    # cells. Measured: IoU 0.85 (the 2-D footprint is a strict subset of the
    # 3-D one -- 909 of 1068 cells; LOD2 mesh voxelization also catches
    # overhangs and sloped faces the 2.5-D raster has no cell for).
    #
    # This is what makes the test discriminating beyond grid shape. A
    # mis-oriented 3-D frame scores 0.04-0.12 on the same data (measured by
    # flipping the 3-D footprint up/down, left/right, and 180 deg), so 0.6
    # sits with 1.4x of margin below the real value and 5x above the best
    # mis-orientation.
    foot_2d = city.buildings.heights > 0
    foot_3d = (classes == BUILDING_CODE).any(axis=2)
    assert foot_2d.shape == foot_3d.shape, (foot_2d.shape, foot_3d.shape)
    iou = float((foot_2d & foot_3d).sum()) / float((foot_2d | foot_3d).sum())
    assert iou > 0.6, (
        f"2-D/3-D building footprint IoU {iou:.3f} -- the rasterized grids "
        f"and the voxel grid disagree about where the buildings are")

    # True LOD2 roofs survive rotation. Measured on this exact rectangle:
    #   rotation 30, LOD2 -> 0.145   (this test)
    #   rotation  0, LOD2 -> 0.161   (rotation costs ~10%, not the metric)
    #   rotation 30, LOD1 -> 0.032   (flat extrusion, as expected)
    # so the 0.08 threshold still sits between the two with 1.8x / 2.5x of
    # margin either side after rotation.
    assert slope > MIN_ROOF_SLOPE_FRACTION, (
        f"roof slope {slope:.4f} <= {MIN_ROOF_SLOPE_FRACTION} -- LOD2 geometry "
        f"missing, or the rotated frame mis-binned the roof columns")

    # Terrain reached every column. An empty vertical stripe is the signature
    # of a 2-D/3-D frame mismatch: the DEM is rasterized in the 2-D affine
    # frame and written into the 3-D voxel frame, so if the two disagree in
    # orientation the DEM covers only part of the voxel grid.
    assert ground.all(), f"{n_empty} empty columns -- frame mismatch?"


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------

def _assert_meshes_identical(cold_meshes, warm_meshes, label):
    """Cold (XML) and warm (cache) meshes must agree bitwise, matched by id.

    Counts and totals alone would pass on reordered or numerically altered
    geometry, so this compares the raw bytes of every array.
    """
    assert len(warm_meshes) == len(cold_meshes) > 0, (
        f"{label}: {len(cold_meshes)} cold vs {len(warm_meshes)} warm meshes")
    cold_by_id = {m.feature_id: m for m in cold_meshes}
    warm_by_id = {m.feature_id: m for m in warm_meshes}
    # Matching by id is only meaningful if ids are unique.
    assert len(cold_by_id) == len(cold_meshes), f"{label}: duplicate feature_ids"
    assert set(warm_by_id) == set(cold_by_id), f"{label}: feature_id sets differ"

    for fid, cold in cold_by_id.items():
        warm = warm_by_id[fid]
        for name in ('vertices', 'faces', 'normals', 'colors'):
            a, b = getattr(cold, name), getattr(warm, name)
            if a is None or b is None:
                assert a is b is None, f"{label}/{fid}: {name} presence differs"
                continue
            assert (a.dtype, a.shape) == (b.dtype, b.shape), (
                f"{label}/{fid}: {name} is {a.dtype}{a.shape} cold vs "
                f"{b.dtype}{b.shape} warm")
            assert a.tobytes() == b.tobytes(), (
                f"{label}/{fid}: {name} is not bitwise identical")
        assert cold.attributes.keys() == warm.attributes.keys(), (
            f"{label}/{fid}: attribute keys differ")
        for key, cold_val in cold.attributes.items():
            warm_val = warm.attributes[key]
            if isinstance(cold_val, np.ndarray):
                assert isinstance(warm_val, np.ndarray)
                assert cold_val.dtype == warm_val.dtype
                assert cold_val.tobytes() == warm_val.tobytes(), (
                    f"{label}/{fid}: attribute {key!r} not bitwise identical")
            else:
                # Type too: JSON round-tripping a np.bool_ as 'False' would be
                # truthy on the warm path and falsy on the cold one.
                assert type(cold_val) is type(warm_val), (
                    f"{label}/{fid}: attribute {key!r} changed type "
                    f"{type(cold_val)} -> {type(warm_val)}")
                assert cold_val == warm_val, (
                    f"{label}/{fid}: attribute {key!r} changed value")


@requires_dataset
@pytest.mark.slow
def test_parse_cache_round_trip_on_real_dataset(tmp_path):
    """Second parse must hit the cache: identical meshes, much faster.

    The unit tests stub the extractor, so this is the only proof the cache
    round-trips real PLATEAU XML.
    """
    from voxcitygml.citygml.coordinates import (
        create_rectangle, file_intersects_rectangle, rectangle_to_shapely,
    )
    from voxcitygml.citygml.parse_cache import CACHE_DIR_NAME, load_cached_meshes
    from voxcitygml.citygml.parser import parse_citygml_directory

    rect = create_rectangle(139.7725, 35.6481, 200)
    rect_polygon = rectangle_to_shapely(rect)

    # Work on a copy of the intersecting tiles so the real dataset's cache
    # state cannot make the first parse a hit (the test must own its fixture).
    # Tiles are selected with the same mesh-code filter the parser uses:
    # building tiles are 1/10 sub-meshes (53393671) while DEM tiles are whole
    # 2nd-level meshes (533936), so a literal filename prefix would copy no
    # terrain at all and every terrain assertion below would pass vacuously.
    dataset = tmp_path / "dataset"
    copied = {}
    for sub in ("udx/bldg", "udx/dem"):
        src_dir = os.path.join(DATASET, *sub.split("/"))
        dst_dir = dataset / sub
        dst_dir.mkdir(parents=True)
        names = [n for n in os.listdir(src_dir)
                 if n.endswith(".gml") and file_intersects_rectangle(n, rect_polygon)]
        for name in names:
            shutil.copy2(os.path.join(src_dir, name), dst_dir / name)
        copied[sub] = names
    assert copied["udx/bldg"], "no building tile intersects the test rectangle"
    assert copied["udx/dem"], "no DEM tile intersects the test rectangle"

    kwargs = dict(rectangle_vertices=rect,
                  feature_types=["building", "terrain"], building_lod=2)

    t0 = time.perf_counter()
    cold = parse_citygml_directory(str(dataset), **kwargs)
    t_cold = time.perf_counter() - t0
    cache_root = dataset / CACHE_DIR_NAME
    assert cache_root.is_dir(), "cache dir was not created"

    t0 = time.perf_counter()
    warm = parse_citygml_directory(str(dataset), **kwargs)
    t_warm = time.perf_counter() - t0
    print(f"\nparse cache: cold {t_cold:.2f}s -> warm {t_warm:.2f}s "
          f"({t_cold / t_warm:.1f}x)")

    # Both feature types must have been stored, otherwise the "warm" run
    # below is partly another cold parse and proves less than it looks.
    bldg_entries = sorted(cache_root.glob("udx/bldg/*.building.lod2.npz"))
    dem_entries = sorted(cache_root.glob("udx/dem/*.terrain.npz"))
    assert len(bldg_entries) == len(copied["udx/bldg"]), "building tile not cached"
    assert len(dem_entries) == len(copied["udx/dem"]), "DEM tile not cached"

    _assert_meshes_identical(cold.buildings, warm.buildings, "buildings")
    _assert_meshes_identical(cold.terrain, warm.terrain, "terrain")

    # ``triangle_coords`` is elided on store and rebuilt as ``vertices[faces]``
    # on load, so it is the one array a real-data test must check directly:
    # the filtered meshes compared above no longer carry it.
    with np.load(dem_entries[0], allow_pickle=False) as data:
        meta = json.loads(data["meta"].tobytes().decode("utf-8"))
        stored_keys = set(data.files)
    assert any("triangle_coords" in m["derived_attrs"] for m in meta["meshes"]), (
        "triangle_coords was not elided from the terrain cache entry")
    assert not [k for k in stored_keys if k.endswith("_triangle_coords")], (
        "triangle_coords was stored verbatim despite being elided")

    dem_gml = Path(dataset, "udx", "dem", copied["udx/dem"][0])
    reloaded = load_cached_meshes(dem_gml, "terrain", None, meta["source_epsg"])
    assert reloaded, "terrain cache entry did not load back"
    for mesh in reloaded:
        tri = mesh.attributes["triangle_coords"]
        assert tri.shape == (len(mesh.faces), 3, 3)
        assert np.array_equal(tri, mesh.vertices[mesh.faces]), (
            "rebuilt triangle_coords disagrees with vertices[faces]")

    # Generous bound: the XML parse is seconds, the npz load tens of ms. A 2x
    # margin absorbs CI noise while still failing if caching regresses.
    assert t_warm < t_cold / 2, (
        f"cache gave no speedup: {t_cold:.2f}s -> {t_warm:.2f}s")
