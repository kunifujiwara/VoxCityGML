"""Functional tests for the optimized voxelizer3d module."""

import time
import numpy as np

from voxcitygml.voxelizer3d import (
    Grid3DParams,
    _fill_air_to_dem_surface,
    _fill_interior,
    _apply_land_cover,
    _apply_canopy,
    _surface_voxelize_numba,
    _triangle_box_overlap_nb,
    _dilate6,
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
