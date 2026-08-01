"""The assembled VoxCity must honour voxcity's south-up axis contract.

voxcitygml works north-up internally. Before the assembly seam converted,
land_cover.classes was the only canonical grid and everything else was
mirrored relative to it -- measured on Chuo-ku as 0.0000 direct / 0.9549
flipud for land-cover Tree cells against voxel TREE columns.

Dataset-gated exactly like ``test_integration_plateau``: skipped when the
local PLATEAU dataset is absent, and marked ``slow`` because the fixture
below runs the real pipeline once (parse -> grids -> voxelize -> assemble).

Unlike the tests in that module this one does **not** pin
``land_cover_source``: the whole point is to compare the downloader's own
land-cover frame against the voxel grid, and the CityGML land-cover
rasteriser (which those tests use to stay offline) returns internal 1-based
VoxCity codes rather than source class indices, so it cannot be scored
against ``get_land_cover_classes``. Auto-selection picks
``OpenEarthMapJapan`` here, which fetches a raster over HTTPS -- so this
module is the one part of the suite that needs the network. It is not
marked ``network``: that marker is deselected by default and documented as
guarding against Overpass' indefinite rate-limit stall, a hazard the OEMJ
tile fetch does not share, and deselecting the axis contract by default
would let it silently never run.
"""
import numpy as np
import pytest

from tests.test_integration_plateau import DATASET, requires_dataset

pytestmark = [requires_dataset, pytest.mark.slow]

# The rectangle the frames in the module docstring were measured on:
# [SW, NW, NE, SE], ~236 m x 200 m in Chuo-ku, Tokyo (PLATEAU tile 53393671).
RECTANGLE = [
    [139.7712, 35.6472],
    [139.7712, 35.6490],
    [139.7738, 35.6490],
    [139.7738, 35.6472],
]

# Every artifact array the invariance tests below compare a converted grid
# against. Snapshotted before assembly (see ``_lod2_run``).
_WATCHED = (
    "voxel_grid",
    "land_cover_grid",
    "dem_grid",
    "building_height_grid",
    "building_id_grid",
    "canopy_top",
    "canopy_bottom",
    "mesh_vegetation_mask",
)


def _iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _frame_config(output_dir):
    """The measured configuration: LOD2, 2 m voxels, static canopy.

    ``canopy_height_source`` is pinned to ``Static`` rather than
    auto-selected: auto-selection returns the GEE canopy-height maps, and
    ``run_core`` calls ``auto_select_data_sources`` directly (not the
    Earth-Engine-aware wrapper in ``voxcity.generator.api``), so without a
    ``gee_project`` that choice would fail rather than degrade.
    """
    from voxcitygml import VoxelizerConfig

    return VoxelizerConfig(
        citygml_path=DATASET,
        rectangle_vertices=RECTANGLE,
        meshsize=2.0,
        building_lod=2,
        canopy_height_source="Static",
        output_dir=str(output_dir),
        save_output=False,
    )


@pytest.fixture(scope="session")
def _lod2_run(tmp_path_factory):
    """One real ``run_core`` and one real assembly, shared by every test here.

    The assembly goes through ``VoxCityGML.run()`` rather than calling
    ``assemble_voxcity`` directly, so the seam actually under test is the
    one exercised; ``run_core`` is stubbed out for that call only, to return
    the artifacts already built above instead of rebuilding them.
    """
    from voxcitygml import pipeline as pl

    cfg = _frame_config(tmp_path_factory.mktemp("frame_contract"))
    art = pl.run_core(cfg)

    # Snapshots taken *before* assembly. ``assemble_voxcity`` is a pure
    # constructor and ``_to_south_up`` allocates, so nothing should touch
    # ``art`` -- but if anything ever did, the invariance tests below would
    # be comparing a converted grid against converted data and would pass
    # vacuously. This makes that a failure instead of a silent no-op.
    before = {name: np.array(getattr(art, name), copy=True) for name in _WATCHED}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "run_core", lambda config: art)
        city = pl.VoxCityGML(cfg).run()

    for name, snapshot in before.items():
        current = getattr(art, name)
        assert current is not snapshot, f"snapshot of art.{name} is not a copy"
        assert np.array_equal(current, snapshot), (
            f"assembly mutated art.{name} in place -- every comparison "
            f"against the artifacts below would be vacuous")
    return city, art


@pytest.fixture(scope="session")
def lod2_city(_lod2_run):
    return _lod2_run[0]


@pytest.fixture(scope="session")
def lod2_artifacts(_lod2_run):
    return _lod2_run[1]


def test_land_cover_and_voxels_share_a_frame(lod2_city):
    """Tree land cover must align with TREE voxels directly, not flipped."""
    from voxcitygml.voxelizer3d import TREE_CODE
    from voxcity.utils.lc import get_land_cover_classes

    city = lod2_city
    names = list(dict.fromkeys(
        get_land_cover_classes(city.extras["land_cover_source"]).values()))
    tree_idx = names.index("Tree")

    vox_tree = (city.voxels.classes == TREE_CODE).any(axis=2)
    lc_tree = city.land_cover.classes == tree_idx
    assert vox_tree.any() and lc_tree.any(), "fixture has no trees; test is vacuous"

    direct, flipped = _iou(lc_tree, vox_tree), _iou(np.flipud(lc_tree), vox_tree)
    assert direct > flipped, (
        f"land cover is mirrored against the voxel grid: direct {direct:.4f} "
        f"<= flipud {flipped:.4f}")
    assert direct > 0.5, f"weak alignment ({direct:.4f}); frames may both be wrong"


def test_conversion_preserves_geometry(lod2_city, lod2_artifacts):
    """A flip must move the data, not rebuild or corrupt it."""
    from tests.test_integration_plateau import roof_slope_fraction, BUILDING_CODE
    city, art = lod2_city, lod2_artifacts
    # roof_slope_fraction counts adjacent-pair height differences, so it is
    # invariant under a vertical flip -- a real guard, not a tautology.
    assert roof_slope_fraction(city.voxels.classes) == \
        roof_slope_fraction(art.voxel_grid)
    assert (city.voxels.classes == BUILDING_CODE).sum() == \
        (art.voxel_grid == BUILDING_CODE).sum()
    assert np.array_equal(city.voxels.classes, np.flipud(art.voxel_grid))


def test_flipped_arrays_are_contiguous(lod2_city):
    """np.flipud returns a view; numba paths degrade silently on those."""
    for name, arr in [
        ("voxels.classes", lod2_city.voxels.classes),
        ("dem.elevation", lod2_city.dem.elevation),
        ("buildings.heights", lod2_city.buildings.heights),
        ("buildings.ids", lod2_city.buildings.ids),
        ("tree_canopy.top", lod2_city.tree_canopy.top),
    ]:
        assert np.asarray(arr).flags["C_CONTIGUOUS"], f"{name} is not contiguous"


def test_land_cover_is_not_flipped(lod2_city, lod2_artifacts):
    assert np.array_equal(lod2_city.land_cover.classes,
                          lod2_artifacts.land_cover_grid)
