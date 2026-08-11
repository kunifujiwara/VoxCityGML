"""Regression tests for building voxelization alignment.

An axis-aligned box whose faces lie exactly on the 2 m grid must voxelize to
exactly its analytic cell set -- no displacement, no dilation.  These tests
pin the 2026-08-11 diagnosis: the old levelset path filled 294 cells where
the answer is 180 (corner-sampled SDF stamped as centre-sampled).
"""
import numpy as np
import pytest
import trimesh

from voxcitygml.voxelizer3d import (
    _MESHLIB_VOXEL_AVAILABLE,
    Grid3DParams,
    _voxelize_meshlib_winding,
)

# _voxelize_building_solid does not exist until Task 3; import it lazily so
# earlier tasks get test FAILURES, not a module-collection error that would
# also block the winding tests.


def building_solid(*args, **kw):
    from voxcitygml.voxelizer3d import _voxelize_building_solid
    return _voxelize_building_solid(*args, **kw)

pytestmark = pytest.mark.skipif(
    not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")

MS = 2.0


def make_gp():
    # 24 x 24 x 20 m grid; rows span [max_y - n_rows*vs, max_y] = [-6, 18].
    # min_y is deliberately NOT congruent with max_y modulo voxel_size:
    # every row computation in voxelizer3d.py anchors at max_y
    # (_stamp_meshlib_mask, Grid3DParams.box_center, _bbox_to_index_range),
    # and production grids keep max_y as the raw rectangle coordinate while
    # n_rows is rounded (_compute_grid_params_3d), so (max_y - min_y) is
    # generally not a whole number of voxels.  A snap anchored at min_y
    # passes on a congruent grid but leaves a half-voxel y-phase error in
    # production — this grid makes that bug fail the tests.
    return Grid3DParams(n_rows=12, n_cols=12, n_z=10,
                        min_x=-6.0, max_x=18.0, min_y=-6.9, max_y=18.0,
                        min_z=-6.0, max_z=14.0, voxel_size=MS)


def box_mesh(dx=0.0, dy=0.0, dz=0.0, extents=(12.0, 12.0, 10.0)):
    b = trimesh.creation.box(extents=list(extents))
    b.apply_translation([extents[0] / 2 + dx, extents[1] / 2 + dy,
                         extents[2] / 2 + dz])
    return np.asarray(b.vertices, float), np.asarray(b.faces)


def filled(grid):
    return set(zip(*np.nonzero(grid == -3)))


def expected_box_cells(gp, dx=0.0, dy=0.0, dz=0.0, extents=(12.0, 12.0, 10.0)):
    """Centre-inside cells of the translated box, in (row, col, z) indices."""
    out = set()
    for row in range(gp.n_rows):
        for col in range(gp.n_cols):
            for zi in range(gp.n_z):
                x = gp.min_x + (col + 0.5) * MS
                y = gp.max_y - (row + 0.5) * MS
                z = gp.min_z + (zi + 0.5) * MS
                if (dx <= x <= dx + extents[0] and dy <= y <= dy + extents[1]
                        and dz <= z <= dz + extents[2]):
                    out.add((row, col, zi))
    return out


@pytest.mark.parametrize("off", [0.0, 0.7, 1.3])
def test_winding_aligned_box_exact(off):
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh(dx=off, dy=off)
    ok = _voxelize_meshlib_winding(v, f, gp, grid, -3, True, align_origin=True)
    assert ok
    assert filled(grid) == expected_box_cells(gp, dx=off, dy=off)


@pytest.mark.parametrize("off", [0.0, 0.7, 1.3])
def test_building_path_box_exact(off):
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh(dx=off, dy=off)
    building_solid(v, f, gp, grid, -3, True,
                   occupancy_threshold=0.0,
                   occupancy_subdivisions=3,
                   shell_threshold=0.5)
    got = filled(grid)
    want = expected_box_cells(gp, dx=off, dy=off)
    # The aligned winding fill supplies every centre-inside cell.  (Volume
    # overlap >= 0.5 implies centre-inside, but NOT the converse: a corner
    # cell can be centre-inside at only ~0.42 overlap — 0.65 x 0.65 at
    # off=0.7 — which is why the >= 0.5 bound below is applied to the
    # EXTRA cells only, never demanded of `want`.)  The shell
    # measures SURFACE-CONTACT occupancy — the fraction of 3x3x3 sub-cells
    # touching a triangle — so a lone flat face scores ~1/3 and is dropped
    # at threshold 0.5; only multi-face cells (corners/edges) can survive,
    # and those are centre-inside anyway.  The envelope bound still holds:
    # never lose a centre-inside cell, never add a cell the box covers by
    # less than half its volume.
    assert want <= got
    for cell in got - want:
        row, col, zi = cell
        # every extra cell must overlap the box by >= 0.5 along each axis
        x0 = gp.min_x + col * MS
        y0 = gp.max_y - (row + 1) * MS
        z0 = gp.min_z + zi * MS
        fx = max(0.0, min(x0 + MS, off + 12.0) - max(x0, off)) / MS
        fy = max(0.0, min(y0 + MS, off + 12.0) - max(y0, off)) / MS
        fz = max(0.0, min(z0 + MS, 10.0) - max(z0, 0.0)) / MS
        assert fx * fy * fz >= 0.5 - 1e-9, (
            f"cell {cell} kept with overlap {fx*fy*fz:.2f} < 0.5")


def test_aligned_box_no_dilation():
    """The historic failure: aligned box produced 294 cells, one extra layer
    on every +x/+y/+z face.  Exactly 180, exactly placed."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh()
    building_solid(v, f, gp, grid, -3, True,
                   occupancy_threshold=0.0,
                   occupancy_subdivisions=3,
                   shell_threshold=0.5)
    assert filled(grid) == expected_box_cells(gp)
    assert len(filled(grid)) == 180


def test_shell_threshold_discriminates_surface_contact():
    """The shell keeps a cell only when enough sub-cells TOUCH geometry.

    A 0.3 m sliver crosses one sub-slab of its cell (contact 9/27 = 0.33):
    kept at threshold 0.0, dropped at 0.5.  The shell's 6-connected anchor
    guard needs a neighbouring filled voxel (voxelizer3d.py, `_dilate6`),
    so the layer below is pre-filled — without an anchor the sliver
    disappears at ANY threshold and the test would prove nothing.
    """
    v, f = box_mesh(extents=(12.0, 12.0, 0.3))
    for thr, expect_cells in ((0.0, True), (0.5, False)):
        gp = make_gp()
        grid = np.zeros((12, 12, 10), np.int32)
        grid[:, :, 2] = -1                 # anchor layer under the sliver
        building_solid(v, f, gp, grid, -3, True, shell_threshold=thr)
        assert (len(filled(grid)) > 0) == expect_cells, f"threshold {thr}"
    # A >= half-voxel slab (1.2 m) survives threshold 0.5 regardless: its
    # centre-inside cells come from the winding fill, not the shell.
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v2, f2 = box_mesh(extents=(12.0, 12.0, 1.2))
    building_solid(v2, f2, gp, grid, -3, True, shell_threshold=0.5)
    assert len(filled(grid)) > 0
