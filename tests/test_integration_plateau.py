"""End-to-end LOD2 integration test against a local PLATEAU dataset.

Skipped automatically when the dataset is not present (e.g. CI).

Set ``VOXCITYGML_PLATEAU_TEST_DATA`` to run it against a dataset stored
somewhere other than the default path below -- any PLATEAU dataset
directory (the one containing ``udx/``) whose coverage includes the
target rectangle will do.

The dataset-backed tests are marked ``slow`` because they parse the
intersecting CityGML tiles and voxelize them; each runs in roughly ten to
thirty seconds. **None of them touch the network** -- the ones that build a
model take CityGML land cover rather than OpenStreetMap, and the parse-cache
test never builds one -- so they are fully offline and deterministic. They
run by default because they are the proof the whole chain works; skip them
explicitly with ``pytest -m "not slow"``.

One test here is additionally marked ``network`` and is *deselected* by
default (see ``[tool.pytest.ini_options]`` in ``pyproject.toml``): it is
the remaining coverage of the live OpenStreetMap land-cover integration.
Run it with ``pytest -m network``.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

BUILDING_CODE = -3
TREE_CODE = -2

DATASET = os.environ.get(
    "VOXCITYGML_PLATEAU_TEST_DATA",
    r"D:\03_Data\citygml\plateau\13102_chuo-ku_pref_2023_citygml_2_op",
)

requires_dataset = pytest.mark.skipif(
    not os.path.isdir(DATASET),
    reason="local PLATEAU dataset not available",
)

# Recalibrated 2026-08-11 for the voxelizer-alignment fix (see
# docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md).
# Buildings now voxelize via a grid-aligned winding SDF on the raw mesh, with
# the surface shell kept only at >= 0.5 occupancy (surface-contact) instead
# of > 0.0 (any-corner-contact). That drops boundary cells that are less than
# half inside the mesh -- notably the single lone roof-skin face at a ridge
# or eave, which typically covers about 9 of the 27 sub-cells SAT samples
# (9/27 ~= 0.33 occupancy) and used to be kept at threshold 0 but is dropped
# at 0.5. Those dropped skin cells were exactly the single-voxel roof steps
# this metric counts, so the metric's absolute scale fell even though it
# still separates LOD2 from LOD1 by a comparable ratio -- the envelope is
# tighter and more accurate (round-to-nearest instead of round-up), not
# flatter.
#
# Measured on the reference rectangle (Chuo-ku, 200 m, 2 m voxels) and its
# 30 deg-rotated counterpart (200 m x 150 m), both post-fix:
#   unrotated LOD2: n=22,100 building voxels, slope=0.0741
#   unrotated LOD1: n=21,178 building voxels, slope=0.0163
#   rotated30 LOD2: n=16,081 building voxels, slope=0.0594
#   rotated30 LOD1: n=15,219 building voxels, slope=0.0093
#
# The worst case across rotation is the higher of the two LOD1 figures
# (unrotated, 0.0163) and the lower of the two LOD2 figures (rotated,
# 0.0594) -- the pairing that leaves the least room for a threshold in
# between. 0.03 sits inside that gap with margin on both sides:
#   0.03 / 0.0163 = 1.84x above the worst LOD1 figure
#   0.0594 / 0.03 = 1.98x below the worst LOD2 figure
# matching the old calibration's asymmetric-but->=1.8x-both-ways balance.
#
# Land cover still does not affect this: it only labels the topmost terrain
# voxels and never touches building geometry, which is all this metric
# measures. Every dataset test takes land cover from the CityGML dataset
# itself, so there is no run-to-run variation to absorb; the margin buys
# headroom against dataset and voxelizer changes instead.
MIN_ROOF_SLOPE_FRACTION = 0.03


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
        # Parsed from the dataset itself: fully offline and deterministic.
        # This used to be "OpenStreetMap", which made the default suite depend
        # on live Overpass -- and Overpass rate-limits after a few consecutive
        # fetches and then stalls *indefinitely* rather than erroring, so with
        # no timeout plugin installed this test could hang the whole run. The
        # OSM path keeps its own coverage in the opt-in `network` test below.
        # Switching the source moved none of the numbers this test asserts on
        # (see the comment on the roof-slope assertion).
        land_cover_source="CityGML",
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
    # extrusion. Re-running this config at building_lod=1 scores 0.0163, so a
    # regression that quietly rebuilt the grid from the 2.5-D height grids
    # (as nDSM canopy refinement once did on the app side) fails here.
    assert slope > MIN_ROOF_SLOPE_FRACTION, (
        f"roof slope fraction {slope:.3f} <= {MIN_ROOF_SLOPE_FRACTION}: the "
        f"voxel grid looks like a flat LOD1 extrusion, not LOD2 roof geometry")


# ---------------------------------------------------------------------------
# Live OpenStreetMap land cover (opt-in)
#
# The end-to-end tests above used to be the only thing exercising the
# voxcity/OSM land-cover integration, and paid for it by hanging whenever
# Overpass rate-limited. This keeps the coverage without the hazard: marked
# ``network`` (deselected by default) and reduced to the one call that is
# actually OSM-specific, so a single Overpass fetch covers it.
#
# Marked ``slow`` as well so the documented ``-m "not slow"`` override, which
# replaces the default ``-m "not network"``, cannot re-select it by accident.
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.slow
def test_openstreetmap_land_cover_grid_matches_target_grid(tmp_path):
    """Live OSM land cover comes back on the pipeline's own grid.

    Shape is the contract that matters: ``run_core`` silently
    ``_resize_int_grid``s any land-cover grid that disagrees with the DEM
    grid, so a source that returned the wrong shape would be resampled
    rather than rejected and the mislabelled ground would only show up by
    eye. Needs no PLATEAU dataset -- only Overpass.
    """
    from voxcitygml.grid_utils import compute_grid_params
    from voxcitygml.landcover.processor import get_land_cover_grid
    from voxcitygml.citygml.coordinates import create_rectangle

    rect = create_rectangle(139.7725, 35.6481, 200)
    meshsize = 2.0

    grid = get_land_cover_grid(rect, meshsize, "OpenStreetMap", str(tmp_path))

    expected = compute_grid_params(rect, meshsize).shape
    assert grid.shape == expected, (
        f"OSM land cover is {grid.shape}, target grid is {expected} -- "
        f"run_core would silently resample it")
    # Central Tokyo is not one uniform class; an all-one-value grid means the
    # fetch degraded to a fill rather than failing.
    assert len(np.unique(grid)) > 1, (
        f"OSM land cover is uniformly {np.unique(grid)} -- no real data")


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
    # assertion -- while roof slope and empty columns (0) were *unaffected*
    # (roof slope on this rectangle is now ~0.0594 post-2026-08-11 alignment
    # fix -- see the MIN_ROOF_SLOPE_FRACTION comment above for how that
    # figure was recalibrated; the frame-reversion side-experiment itself
    # was not re-run under the new voxelizer). Those two are LOD2-quality
    # and coverage guards, not frame guards. The frame guard with teeth
    # beyond shape is the 2-D/3-D footprint IoU below.

    # Grid dims follow the rectangle's own sides, not the bbox of its corners
    # in an unrotated frame -- which at 30 deg spans 200*cos30 + 150*sin30 =
    # 248 m by 200*sin30 + 150*cos30 = 230 m, i.e. (115, 124) cells instead of
    # (75, 100). That is what the pre-fix code produced, and it is far outside
    # the +-2 tolerance below.
    assert abs(classes.shape[0] - 75) <= 2, classes.shape
    assert abs(classes.shape[1] - 100) <= 2, classes.shape

    # Measured 16,081 here vs 15,192 for the same rectangle at rotation 0
    # (both post-2026-08-11 alignment fix) -- the ~6% spread is different
    # ground being covered, not a frame error. The bound is deliberately
    # loose: it only has to reject "the rotated frame landed the buildings
    # outside the grid".
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

    # True LOD2 roofs survive rotation. Measured on this exact rectangle,
    # post-2026-08-11 alignment fix (shell threshold 0.5):
    #   rotation 30, LOD2 -> 0.0594   (this test)
    #   rotation  0, LOD2 -> 0.0819   (rotation costs ~27%, not the metric)
    #   rotation 30, LOD1 -> 0.0093   (flat extrusion, as expected)
    # so the 0.03 threshold still sits between the two: 1.98x margin below
    # the LOD2 figure here, 3.23x above the LOD1 figure here. The tighter
    # global margin (1.84x) comes from the *unrotated* LOD1 figure -- see
    # the MIN_ROOF_SLOPE_FRACTION comment above for the full calibration.
    assert slope > MIN_ROOF_SLOPE_FRACTION, (
        f"roof slope {slope:.4f} <= {MIN_ROOF_SLOPE_FRACTION} -- LOD2 geometry "
        f"missing, or the rotated frame mis-binned the roof columns")

    # Terrain reached every column. An empty vertical stripe is the signature
    # of a 2-D/3-D frame mismatch: the DEM is rasterized in the 2-D affine
    # frame and written into the 3-D voxel frame, so if the two disagree in
    # orientation the DEM covers only part of the voxel grid.
    assert ground.all(), f"{n_empty} empty columns -- frame mismatch?"


# ---------------------------------------------------------------------------
# Canopy re-apply on real LOD2 data
#
# ``reapply_canopy`` exists because the app's nDSM canopy refinement used to
# end in ``regenerate_voxels(..., inplace=True)``, which rebuilds the whole
# voxel grid from the 2.5-D component grids -- i.e. re-runs the LOD1
# footprint-extrusion algorithm and destroys LOD2's mesh-voxelized roofs. The
# two tests below are the real-data proof that the replacement does not.
# ---------------------------------------------------------------------------

# Centre of the Chuo-ku reference rectangle (PLATEAU tile 53393671), shared by
# the end-to-end tests above. Measured: this tile carries *zero* CityGML
# vegetation, so the mesh-vegetation mask here is all-False and every column is
# the canopy overlay's to write -- which is what makes it the right place to
# test LOD2 preservation and the wrong place to test vegetation preservation.
LOD2_CENTRE = (139.7725, 35.6481)

# PLATEAU ships vegetation geometry for very few areas; in this dataset the only
# ``udx/veg`` tiles are 53393690, 53394600 and 53394611, all north of lat
# 35.658. 53393690 spans lat 35.658-35.667 / lon 139.750-139.7625; this point is
# the measured centroid of its vegetation meshes. A 200 m rectangle here was
# measured to capture 118 vegetation meshes *and* 199 buildings -- both are
# needed, since the test wants a working LOD2 model with vegetation in it.
VEG_CENTRE = (139.75336, 35.66510)


def _lod2_config(rect, tmp_path):
    """The shared LOD2 config: offline, deterministic, parser exercised."""
    from voxcitygml import VoxelizerConfig
    return VoxelizerConfig(
        citygml_path=DATASET,
        rectangle_vertices=rect,
        meshsize=2.0,
        building_lod=2,
        # Not "OpenStreetMap": live Overpass rate-limits after a few
        # consecutive fetches and then stalls *indefinitely* rather than
        # erroring, which would hang the suite. See the note above.
        land_cover_source="CityGML",
        canopy_height_source="Static",
        output_dir=str(tmp_path),
        save_output=False,
        # Parse the XML for real every run, as the tests above do. Turning
        # this on would cut ~13 s off the vegetation test -- 53393690's veg
        # tile is 12 MB -- which is exactly the tempting trade to refuse: the
        # cache is written *into the user's real dataset directory*, so the
        # price of those 13 s is tens of megabytes deposited beside their
        # data, and it stops exercising the parser after the first run. The
        # whole suite is ~41 s; this is not a budget worth optimising.
        use_parse_cache=False,
    )


def _sparse_canopy(shape):
    """A visibly different canopy: 12 m crowns on every third cell.

    Deliberately not a solid field. Scattered columns land on open ground as
    well as on buildings, so some of them can actually receive voxels (canopy
    is only written into cells that are AIR), and the pattern is nothing like
    the Static 10 m canopy it replaces.
    """
    canopy = np.zeros(shape, dtype=np.float64)
    canopy[::3, ::3] = 12.0
    return canopy


@requires_dataset
@pytest.mark.slow
def test_reapply_canopy_preserves_lod2_roofs(tmp_path):
    """Re-applying canopy must not flatten LOD2 geometry.

    This is the guard against anyone reintroducing ``regenerate_voxels`` on
    the LOD2 path. Measured by temporarily rewiring ``reapply_canopy`` to that
    rebuild, on this exact rectangle, post-2026-08-11 alignment fix:

        roof slope     0.0741 -> 0.0965
        building voxels 22,100 -> 21,623
        grid shape (100, 100, 37) -> (100, 100, 43)

    The "before" figures moved from the pre-fix reference (0.1512, 26,088)
    because they come from the mesh voxelizer this fix changed; the "after"
    figures are unchanged by the fix, because ``regenerate_voxels`` rebuilds
    from ``buildings.heights`` -- the 2.5-D component grid, extruded by
    ``voxcity.generator.update`` itself -- not through voxcitygml's mesh
    path at all.

    Note that 0.0965 is still *above* ``MIN_ROOF_SLOPE_FRACTION`` (0.03): the
    rebuild extrudes from ``buildings.heights``, which on a LOD2 model was
    rasterized per cell from the mesh and so retains a coarse staircase --
    it is nowhere near as flat as a true LOD1 run (0.0163).

    So ``assert slope > MIN_ROOF_SLOPE_FRACTION`` -- the exact assertion form
    used by ``test_lod2_generate_voxcity_end_to_end`` and
    ``test_rotated_rectangle_end_to_end`` above in this same file -- **passes
    on the rebuild**. Copying the neighbouring tests' idiom here is the wrong
    instinct, which is why every quantity below is instead pinned to *exact*
    equality with its pre-call value. Do not relax these into thresholds.
    """
    from voxcitygml import generate_voxcity, reapply_canopy
    from voxcitygml.voxelizer3d import TREE_CODE as PIPELINE_TREE_CODE
    from voxcitygml.citygml.coordinates import create_rectangle

    assert PIPELINE_TREE_CODE == TREE_CODE == -2

    rect = create_rectangle(*LOD2_CENTRE, 200)
    city = generate_voxcity(_lod2_config(rect, tmp_path))

    # A *copy*, not the live array. Everything below the re-apply must compare
    # against a genuine pre-call snapshot; holding the live reference is the
    # false negative described in the comment further down, and naming a live
    # reference `before` is how someone reintroduces it. Keeping this
    # structural rather than positional costs one 370 KB copy.
    before = city.voxels.classes.copy()

    slope_before = roof_slope_fraction(before)
    n_bldg_before = int((before == BUILDING_CODE).sum())
    buildings_before = before == BUILDING_CODE
    tree_before = before == TREE_CODE
    n_tree_before = int(tree_before.sum())

    new_canopy = _sparse_canopy(before.shape[:2])
    reapply_canopy(city, new_canopy)

    # Re-read off ``city`` rather than reusing any array captured above.
    # ``reapply_canopy`` edits in place, but ``regenerate_voxels(inplace=True)``
    # -- the regression this test exists to catch -- *rebinds* ``city.voxels``
    # to a freshly built grid and leaves the old array untouched. Measured: a
    # version of this test that asserted on the array it grabbed before the
    # call passed the slope check, the building count, the bitwise footprint
    # comparison *and* the shape guard under a real rebuild. Everything
    # compared against above is a snapshot, so this is correct whichever way
    # the call behaves.
    classes = city.voxels.classes

    # Measured and printed before any assertion, so a failing run reports the
    # collapsed roof-slope number rather than stopping at the shape mismatch
    # a rebuild also produces.
    slope_after = roof_slope_fraction(classes)
    n_tree_after = int((classes == TREE_CODE).sum())
    n_bldg_after = int((classes == BUILDING_CODE).sum())
    print(f"\nLOD2 preservation: roof slope {slope_before:.4f} -> "
          f"{slope_after:.4f}, building voxels {n_bldg_before} -> "
          f"{n_bldg_after}, tree voxels {n_tree_before} -> {n_tree_after}, "
          f"grid {before.shape} -> {classes.shape}")

    assert classes.shape == before.shape, (
        f"grid was reshaped {before.shape} -> {classes.shape} -- rebuilt?")

    # -- the re-apply must actually have done something ---------------------
    # Without this the assertions below would all pass on a no-op
    # ``reapply_canopy``, and the test would prove nothing at all.
    assert n_tree_after != n_tree_before, (
        f"tree voxel count unchanged at {n_tree_before} -- reapply_canopy "
        f"did nothing, so the preservation assertions below are vacuous")
    gained = (classes == TREE_CODE) & ~tree_before
    assert gained.any(), "no new tree voxels anywhere -- reapply_canopy did nothing"
    # The new crowns follow the *new* canopy, not the old one.
    rows, cols = np.nonzero(gained.any(axis=2))
    assert np.all(new_canopy[rows, cols] > 0), (
        "canopy voxels appeared in columns the new canopy leaves empty")

    # -- and the LOD2 geometry must be untouched ----------------------------
    assert slope_after == pytest.approx(slope_before, abs=1e-9), (
        f"LOD2 roof geometry changed ({slope_before:.4f} -> {slope_after:.4f})"
        f" -- was the grid rebuilt?")
    assert n_bldg_after == n_bldg_before, (
        f"building voxel count changed {n_bldg_before} -> {n_bldg_after} -- "
        f"the canopy overlay must never touch building geometry")
    np.testing.assert_array_equal(
        classes == BUILDING_CODE, buildings_before,
        err_msg="building voxels moved during the canopy re-apply")


@requires_dataset
@pytest.mark.slow
def test_reapply_canopy_preserves_citygml_vegetation(tmp_path):
    """CityGML vegetation crowns survive a canopy re-apply (fill-the-gaps).

    CityGML vegetation is voxelized as ``TREE_CODE`` -- the same class the
    canopy overlay writes -- so without ``extras['mesh_vegetation_mask']`` the
    overlay's "clear the stale canopy" step would delete real mesh geometry.
    Canopy must be written only where CityGML left nothing.

    Teeth, measured by sabotaging ``reapply_canopy``: forcing the mask to
    all-``False`` (i.e. ignoring it) moves 1,418 of the 25,000 voxels in the
    masked columns and fails assertion 1; making the whole call a no-op leaves
    zero unmasked columns with new canopy and fails assertion 2.
    """
    from voxcitygml import generate_voxcity, reapply_canopy
    from voxcitygml.citygml.coordinates import create_rectangle

    rect = create_rectangle(*VEG_CENTRE, 200)
    city = generate_voxcity(_lod2_config(rect, tmp_path))
    classes = city.voxels.classes

    mask = city.extras["mesh_vegetation_mask"]
    n_masked = int(mask.sum())
    print(f"\nvegetation preservation: grid {classes.shape}, "
          f"mesh-vegetation columns {n_masked}, "
          f"building voxels {int((classes == BUILDING_CODE).sum())}, "
          f"tree voxels {int((classes == TREE_CODE).sum())}")

    # Loudly, first: with an empty mask every assertion below passes without
    # exercising the preservation logic at all. If this ever trips, the
    # rectangle stopped covering vegetation or the mask stopped being captured
    # -- fix that, do not weaken the test.
    assert mask.shape == classes.shape[:2], (mask.shape, classes.shape)
    assert mask.any(), (
        f"no mesh-vegetation columns at {VEG_CENTRE} -- this test is vacuous. "
        f"Either extras['mesh_vegetation_mask'] is not being captured, or the "
        f"rectangle no longer covers PLATEAU tile 53393690's vegetation.")
    # The buildings matter too: this has to be a working LOD2 model, not just
    # a patch of trees.
    assert (classes == BUILDING_CODE).any(), "no buildings in the vegetation tile"
    slope = roof_slope_fraction(classes)
    assert slope > MIN_ROOF_SLOPE_FRACTION, (
        f"roof slope {slope:.4f} <= {MIN_ROOF_SLOPE_FRACTION} -- the "
        f"vegetation tile did not produce real LOD2 geometry, so this is not "
        f"the LOD2-plus-vegetation model the test needs")

    before = classes.copy()
    tree_before = before == TREE_CODE

    new_canopy = _sparse_canopy(classes.shape[:2])
    reapply_canopy(city, new_canopy)

    # Re-read: see the note in the LOD2 test above -- a rebuild rebinds
    # ``city.voxels`` rather than editing the array in place, so asserting on
    # the pre-call reference would pass on exactly that regression.
    classes = city.voxels.classes
    assert classes.shape == before.shape, (
        f"grid was reshaped {before.shape} -> {classes.shape} -- rebuilt? "
        f"reapply_canopy must overlay the existing grid, not rebuild it")

    # 1. Masked columns are bitwise untouched -- whole column, every class.
    np.testing.assert_array_equal(
        classes[mask], before[mask],
        err_msg="CityGML vegetation columns were modified by the canopy "
                "re-apply; the mesh-vegetation mask was ignored")

    # 2. Somewhere outside the mask, canopy was written. Otherwise the mask
    #    could simply be "everything", and preservation would be trivial.
    gained = ((classes == TREE_CODE) & ~tree_before).any(axis=2) & ~mask
    n_gained = int(gained.sum())
    print(f"  unmasked columns that gained canopy: {n_gained}")
    assert n_gained > 0, (
        "no unmasked column received canopy -- the overlay wrote nothing, so "
        "preservation of the masked columns proves nothing")


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
