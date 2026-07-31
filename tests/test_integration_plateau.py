"""End-to-end LOD2 integration test against a local PLATEAU dataset.

Skipped automatically when the dataset is not present (e.g. CI).

Set ``VOXCITYGML_PLATEAU_TEST_DATA`` to run it against a dataset stored
somewhere other than the default path below -- any PLATEAU dataset
directory (the one containing ``udx/``) whose coverage includes the
target rectangle will do.

The end-to-end test is marked ``slow`` and takes roughly 35-125 s, since
it parses the intersecting CityGML tiles and voxelizes them. It runs by
default because it is the proof the whole chain works; skip it explicitly
with ``pytest -m "not slow"``.
"""
import os

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
