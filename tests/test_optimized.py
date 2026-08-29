"""Functional tests for the optimized voxelizer3d module."""

import time
import warnings

import numpy as np

from voxcitygml.voxelizer3d import (
    Grid3DParams,
    _fill_air_to_dem_surface,
    _carve_water_to_dem_surface,
    _fill_interior,
    _apply_land_cover,
    _apply_canopy,
    _surface_voxelize_numba,
    _triangle_box_overlap_nb,
    _dilate6,
    BUILDING_CODE,
    GROUND_CODE,
    TREE_CODE,
)


def make_gp(n_rows=50, n_cols=50, n_z=40, voxel_size=2.0):
    return Grid3DParams(
        n_rows=n_rows, n_cols=n_cols, n_z=n_z,
        min_x=0.0, max_x=n_cols * voxel_size,
        min_y=0.0, max_y=n_rows * voxel_size,
        min_z=0.0, max_z=n_z * voxel_size,
        voxel_size=voxel_size,
    )


# ── Test 1: _fill_air_to_dem_surface ─────────────────────────────────

def test_fill_terrain():
    gp = make_gp()
    dem = np.full((gp.n_rows, gp.n_cols), 20.0)  # 20 m = voxel index 10
    grid = np.zeros((gp.n_rows, gp.n_cols, gp.n_z), dtype=np.int16)

    t0 = time.perf_counter()
    _fill_air_to_dem_surface(grid, gp, dem)
    dt = time.perf_counter() - t0

    expected_level = 9  # t = 10.0 on-lattice: containing voxel is ceil(10)-1
    # All cells below ground_level should be GROUND_CODE
    assert np.all(grid[:, :, :expected_level + 1] == GROUND_CODE), "Terrain fill below DEM failed"
    assert np.all(grid[:, :, expected_level + 1:] == 0), "Terrain fill above DEM should be empty"
    print(f"  [PASS] _fill_air_to_dem_surface  ({dt*1000:.1f} ms)")


# ── Test 2: _fill_interior (scipy) ──────────────────────────────────

def test_fill_interior():
    # Create a hollow cube shell
    shell = np.zeros((10, 10, 10), dtype=bool)
    shell[2:8, 2:8, 2] = True   # bottom
    shell[2:8, 2:8, 7] = True   # top
    shell[2:8, 2, 2:8] = True   # front
    shell[2:8, 7, 2:8] = True   # back
    shell[2, 2:8, 2:8] = True   # left
    shell[7, 2:8, 2:8] = True   # right

    t0 = time.perf_counter()
    filled = _fill_interior(shell)
    dt = time.perf_counter() - t0

    # The interior (3:7, 3:7, 3:7) should now be True
    assert np.all(filled[3:7, 3:7, 3:7]), "Interior of shell not filled"
    # Outside should remain False
    assert not filled[0, 0, 0], "Outside should not be filled"
    assert not filled[9, 9, 9], "Outside should not be filled"
    print(f"  [PASS] _fill_interior          ({dt*1000:.1f} ms)")


# ── Test 3: _apply_land_cover ───────────────────────────────────────

def test_apply_land_cover():
    gp = make_gp(n_rows=20, n_cols=20, n_z=30)
    dem = np.full((gp.n_rows, gp.n_cols), 10.0)
    grid = np.zeros((gp.n_rows, gp.n_cols, gp.n_z), dtype=np.int16)
    _fill_air_to_dem_surface(grid, gp, dem)

    lc = np.ones((gp.n_rows, gp.n_cols), dtype=np.int16) * 3  # some land cover code

    t0 = time.perf_counter()
    _apply_land_cover(grid, gp, lc, dem, "OpenStreetMap")
    dt = time.perf_counter() - t0

    ground_level = 4  # t = 5.0 on-lattice: containing voxel is ceil(5)-1
    # The ground level should now have the land cover code (3+1=4 for OSM)
    assert np.all(grid[:, :, ground_level] == 4), f"Land cover not applied correctly at z={ground_level}"
    print(f"  [PASS] _apply_land_cover       ({dt*1000:.1f} ms)")


# ── Test 4: _apply_canopy ──────────────────────────────────────────

def test_apply_canopy():
    gp = make_gp(n_rows=20, n_cols=20, n_z=30)
    dem = np.full((gp.n_rows, gp.n_cols), 10.0)
    grid = np.zeros((gp.n_rows, gp.n_cols, gp.n_z), dtype=np.int16)

    canopy_top = np.full((gp.n_rows, gp.n_cols), 12.0)   # 12m above ground
    canopy_bottom = np.full((gp.n_rows, gp.n_cols), 4.0)  # trunk starts at 4m

    t0 = time.perf_counter()
    _apply_canopy(grid, gp, dem, canopy_top, canopy_bottom, trunk_height_ratio=None)
    dt = time.perf_counter() - t0

    # Check some tree voxels exist
    assert np.any(grid == TREE_CODE), "No canopy voxels placed"
    # grid is a bare np.zeros array -- no ground voxel was ever written --
    # so _ground_surface_index falls back to the DEM's containing voxel
    # ceil(t)-1, and _apply_canopy anchors the crown one above that:
    # ceil(t)-1+1 == ceil(t).  t = 10.0/2.0 = 5.0 lands exactly on-lattice,
    # so this still evaluates to 5, the same as the old round(t) -- by
    # coincidence of the on-lattice case, not because the formula is round().
    ground_level = int(np.ceil(round(10.0 / gp.voxel_size, 9)))
    # Tree canopy should be above trunk height
    z_start = ground_level + int(round(4.0 / gp.voxel_size))
    z_end = ground_level + int(round(12.0 / gp.voxel_size))
    assert np.all(grid[0, 0, z_start:z_end] == TREE_CODE), "Canopy region incorrect"
    print(f"  [PASS] _apply_canopy           ({dt*1000:.1f} ms)")


# ── Test 5: Numba triangle-box overlap ──────────────────────────────

def test_triangle_box_overlap_nb():
    # Triangle that clearly overlaps the box at origin
    t0 = time.perf_counter()

    # Warm up JIT
    _triangle_box_overlap_nb(
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0,
    )

    # Triangle inside box
    assert _triangle_box_overlap_nb(
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0,
    ), "Triangle inside box should overlap"

    # Triangle far away
    assert not _triangle_box_overlap_nb(
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        10.0, 10.0, 10.0, 11.0, 10.0, 10.0, 10.5, 11.0, 10.0,
    ), "Triangle far away should not overlap"

    dt = time.perf_counter() - t0
    print(f"  [PASS] _triangle_box_overlap_nb ({dt*1000:.1f} ms)")


# ── Test 6: Numba surface voxelization kernel ───────────────────────

def test_surface_voxelize_numba():
    gp = make_gp(n_rows=20, n_cols=20, n_z=20, voxel_size=2.0)
    vs = gp.voxel_size

    # A simple quad (two triangles) at z=10, covering x=[10,30], y=[10,30]
    verts = np.array([
        [10.0, 10.0, 10.0],
        [30.0, 10.0, 10.0],
        [30.0, 30.0, 10.0],
        [10.0, 30.0, 10.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)

    half = np.array([vs / 2.0 + vs * 1e-6] * 3, dtype=np.float64)

    # Warm up JIT on first call
    t0 = time.perf_counter()
    surface = _surface_voxelize_numba(
        verts, faces,
        gp.min_x, gp.max_y, gp.min_z, gp.voxel_size,
        0, gp.n_rows - 1, 0, gp.n_cols - 1, 0, gp.n_z - 1,
        gp.n_rows, gp.n_cols, gp.n_z,
        half,
    )
    dt_first = time.perf_counter() - t0

    assert np.any(surface), "Surface voxelization produced no voxels"

    # Second call should be fast (JIT cached)
    t0 = time.perf_counter()
    surface2 = _surface_voxelize_numba(
        verts, faces,
        gp.min_x, gp.max_y, gp.min_z, gp.voxel_size,
        0, gp.n_rows - 1, 0, gp.n_cols - 1, 0, gp.n_z - 1,
        gp.n_rows, gp.n_cols, gp.n_z,
        half,
    )
    dt_cached = time.perf_counter() - t0

    assert np.array_equal(surface, surface2), "Results should be deterministic"
    print(f"  [PASS] _surface_voxelize_numba  (JIT: {dt_first*1000:.0f} ms, cached: {dt_cached*1000:.1f} ms)")


# ── Test 7: Larger-scale performance benchmark ─────────────────────

def test_perf_terrain_large():
    gp = make_gp(n_rows=200, n_cols=200, n_z=100, voxel_size=2.0)
    dem = np.random.uniform(20.0, 80.0, (gp.n_rows, gp.n_cols))
    grid = np.zeros((gp.n_rows, gp.n_cols, gp.n_z), dtype=np.int16)

    t0 = time.perf_counter()
    _fill_air_to_dem_surface(grid, gp, dem)
    dt = time.perf_counter() - t0

    assert np.any(grid == GROUND_CODE), "No terrain filled"
    print(f"  [PASS] terrain 200x200x100     ({dt*1000:.1f} ms)")


# ── Test 8: _carve_water_to_dem_surface ─────────────────────────────
#
# The counterpart of _fill_air_to_dem_surface for water columns: that
# conform is raise-only by design, so nothing else in the pipeline can
# lower ground the terrain solid planted above a flattened water DEM.
#
# ``land_cover_source="CityGML"`` is used wherever the conversion itself
# is not what is under test: its codes are already the standard 1-based
# ones (water == 9), so the fixtures say what they mean.  The conversion
# path gets its own test below.

CARVE_VS = 1.0
CARVE_DEM_Z = 3.0
#: Containing voxel of CARVE_DEM_Z at voxel_size 1.0 / min_z 0.0:
#: ceil(3.0) - 1 == 2, the one rounding rule the module uses everywhere.
CARVE_SURFACE = 2
WATER = 9
NOT_WATER = 3


def carve_gp(n_rows=6, n_cols=6, n_z=10):
    return make_gp(n_rows=n_rows, n_cols=n_cols, n_z=n_z,
                   voxel_size=CARVE_VS)


def carve_fixture(n_rows=6, n_cols=6, n_z=10, ground_top=7):
    """Grid with GROUND up to ``ground_top`` and a flat DEM at 3.0 m."""
    gp = carve_gp(n_rows, n_cols, n_z)
    grid = np.zeros((gp.n_rows, gp.n_cols, gp.n_z), dtype=np.int16)
    grid[:, :, :ground_top + 1] = GROUND_CODE
    dem = np.full((gp.n_rows, gp.n_cols), CARVE_DEM_Z)
    return gp, grid, dem


def test_carve_water_lowers_ground_to_dem_surface():
    gp, grid, dem = carve_fixture()
    lc = np.full((gp.n_rows, gp.n_cols), WATER, dtype=np.int16)

    _carve_water_to_dem_surface(grid, gp, dem, lc, "CityGML")

    assert np.all(grid[:, :, :CARVE_SURFACE + 1] == GROUND_CODE), (
        "ground at and below the DEM surface must survive the carve")
    assert np.all(grid[:, :, CARVE_SURFACE + 1:] == 0), (
        "every ground voxel above the DEM surface must be carved to air")


def test_carve_leaves_non_water_columns_alone():
    gp, grid, dem = carve_fixture()
    before = grid.copy()
    lc = np.full((gp.n_rows, gp.n_cols), NOT_WATER, dtype=np.int16)

    _carve_water_to_dem_surface(grid, gp, dem, lc, "CityGML")

    assert np.array_equal(grid, before), (
        "a grid with no water land cover must come back byte-identical")


def test_carve_never_touches_building_or_tree_voxels():
    """Bridge piers and tree voxels stand in the carved water.

    Bridges voxelize as BUILDING_CODE with overwrite=True and their piers
    legitimately punch through the ground in water columns; carving
    anything but GROUND_CODE would strand that geometry.
    """
    gp, grid, dem = carve_fixture()
    grid[2, 2, 4:8] = BUILDING_CODE   # pier through the carve zone
    grid[3, 3, 6] = TREE_CODE
    lc = np.full((gp.n_rows, gp.n_cols), WATER, dtype=np.int16)

    _carve_water_to_dem_surface(grid, gp, dem, lc, "CityGML")

    assert np.all(grid[2, 2, 4:8] == BUILDING_CODE), "pier was carved away"
    assert grid[3, 3, 6] == TREE_CODE, "tree voxel was carved away"
    # ...and the GROUND around them is still carved.
    assert np.all(grid[2, 2, 3:4] == 0)
    assert np.all(grid[3, 3, 3:6] == 0) and grid[3, 3, 7] == 0
    assert np.all(grid[:, :, :CARVE_SURFACE + 1] == GROUND_CODE)


def test_carve_skips_nonfinite_dem_cells():
    """A non-finite DEM cell leaves its column exactly as the solid made it.

    The grid outcome ALONE does not pin the sanitisation: ``finite`` masks
    the cell out of ``carve`` regardless of what garbage the integer cast
    produced from ``np.ceil(nan)``, so feeding the raw DEM to the cast
    gives a byte-identical grid.  Promoting RuntimeWarning to an error is
    what makes this test fail if the non-finite cells are ever handed to
    ``astype()`` again -- the same reason
    ``test_conform_skips_nonfinite_dem_cells`` does it for the conform.
    """
    gp, grid, dem = carve_fixture()
    dem[1, 4] = np.nan
    dem[2, 5] = np.inf
    lc = np.full((gp.n_rows, gp.n_cols), WATER, dtype=np.int16)
    before_nan = grid[1, 4].copy()
    before_inf = grid[2, 5].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _carve_water_to_dem_surface(grid, gp, dem, lc, "CityGML")

    assert np.array_equal(grid[1, 4], before_nan), (
        "the NaN column must be left untouched")
    assert np.array_equal(grid[2, 5], before_inf), (
        "the inf column must be left untouched")
    assert np.all(grid[0, :, CARVE_SURFACE + 1:] == 0), (
        "finite columns must still be carved")


def test_carve_never_empties_a_column():
    """A DEM sitting exactly on the grid floor must not wipe the column.

    ``ceil(0) - 1 == -1`` -- the conform's "below the floor, fill
    nothing" sentinel.  Reused unclamped here it INVERTS, because
    ``z > -1`` selects every voxel, and the whole water column is carved
    to air.  That hole would appear after ``_fill_air_to_dem_surface``'s
    bare-column warning has already run, so nothing would report it.
    """
    gp, grid, _ = carve_fixture()
    dem = np.full((gp.n_rows, gp.n_cols), gp.min_z)
    lc = np.full((gp.n_rows, gp.n_cols), WATER, dtype=np.int16)

    _carve_water_to_dem_surface(grid, gp, dem, lc, "CityGML")

    bare = int((~(grid != 0).any(axis=2)).sum())
    assert bare == 0, f"{bare} column(s) were carved bare"
    assert np.all(grid[:, :, 0] == GROUND_CODE), (
        "the floor voxel must survive a DEM clamped to the grid floor")


def test_carve_reads_land_cover_south_up():
    """The land cover arrives SOUTH-up and must be flipped internally.

    ``dem_grid`` and ``voxel_grid`` are north-up; ``land_cover_grid`` is
    the lone south-up array, exactly as ``_apply_land_cover`` receives it.
    A uniform water grid cannot see a missing ``np.flipud``; this
    asymmetric one can.

    The water patch is asymmetric on BOTH axes, so it separates ``flipud``
    from the identity *and* from ``fliplr`` / a transpose -- a row-only
    patch would pass under any of those.
    """
    gp, grid, dem = carve_fixture()
    lc = np.full((gp.n_rows, gp.n_cols), NOT_WATER, dtype=np.int16)
    # South-up rows 0..2 == north-up rows 3..5; columns are not flipped.
    lc[:3, :2] = WATER

    _carve_water_to_dem_surface(grid, gp, dem, lc, "CityGML")

    assert np.all(grid[3:, :2, CARVE_SURFACE + 1:] == 0), (
        "north-up rows 3..5, cols 0..1 are the water patch and must carve")
    assert np.all(grid[:3, :, :8] == GROUND_CODE), (
        "north-up rows 0..2 are dry land and must be untouched -- "
        "carving them means the south-up flip is missing")
    assert np.all(grid[3:, 2:, :8] == GROUND_CODE), (
        "columns 2.. are dry in every row -- carving them means the flip "
        "hit the wrong axis")


def test_carve_converts_source_specific_land_cover_codes():
    """A 0-based source's raw water code reaches the carve as water.

    The raw index is derived by inverting ``convert_land_cover`` for the
    source rather than hardcoded, so this stays honest if the upstream
    mapping is renumbered.
    """
    from voxcity.utils.lc import convert_land_cover

    source = "OpenEarthMapJapan"
    # 0..8 are the source's own classes; convert_land_cover passes
    # anything outside its mapping through unchanged, so probing wider
    # would "find" standard code 9 sitting at raw 9 and mean nothing.
    probe = np.arange(0, 9, dtype=np.int32)
    raw_water = [int(c) for c, std in
                 zip(probe, convert_land_cover(probe, land_cover_source=source))
                 if int(std) == WATER]
    assert len(raw_water) == 1, (
        f"expected exactly one {source} code to mean water, got {raw_water}")
    raw_dry = next(int(c) for c, std in
                   zip(probe, convert_land_cover(probe, land_cover_source=source))
                   if int(std) not in (WATER, 0))

    gp, grid, dem = carve_fixture()
    lc = np.full((gp.n_rows, gp.n_cols), raw_dry, dtype=np.int32)
    lc[:, 2] = raw_water[0]

    _carve_water_to_dem_surface(grid, gp, dem, lc, source)

    assert np.all(grid[:, 2, CARVE_SURFACE + 1:] == 0), (
        f"{source} raw code {raw_water[0]} must convert to water and carve")
    assert np.all(grid[:, 0, :8] == GROUND_CODE), (
        f"{source} raw code {raw_dry} is not water and must not carve")


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running optimized voxelizer3d tests...\n")

    test_fill_terrain()
    test_fill_interior()
    test_apply_land_cover()
    test_apply_canopy()
    test_triangle_box_overlap_nb()
    test_surface_voxelize_numba()
    test_perf_terrain_large()

    print("\nAll tests passed!")
