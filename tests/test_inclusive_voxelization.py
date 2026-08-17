"""Inclusive voxelization: shell anchor rules and gap-free thin volumes.

Pins the 2026-08-17 inclusive-voxelization design
(docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md):

- ``_overlay_surface_shell`` ``anchor="connected"`` keeps thin features
  connected *through the shell* to any filled voxel, still drops
  disconnected fragments, and keeps the whole shell when no anchor exists
  at all (per-category export grids contain no terrain to anchor on).
- Building defaults are inclusive: shell threshold 0.0 + connected anchor
  produce gap-free thin walls (the Plateau LOD2 "comb" bug).
- ``VoxelizerConfig.voxelization_mode`` resolves mode -> mechanism knobs,
  with explicit threshold values overriding the mode.
"""
import numpy as np
import pytest
import trimesh

from voxcitygml.voxelizer3d import (
    _MESHLIB_VOXEL_AVAILABLE,
    Grid3DParams,
    _overlay_surface_shell,
)

MS = 2.0


def make_gp():
    # Same deliberately y-incongruent grid as tests/test_voxelizer_alignment.py:
    # (max_y - min_y) is not a whole number of voxels, matching production.
    return Grid3DParams(n_rows=12, n_cols=12, n_z=10,
                        min_x=-6.0, max_x=18.0, min_y=-6.9, max_y=18.0,
                        min_z=-6.0, max_z=14.0, voxel_size=MS)


def box(min_corner, extents):
    b = trimesh.creation.box(extents=list(extents))
    b.apply_translation([min_corner[i] + extents[i] / 2 for i in range(3)])
    return np.asarray(b.vertices, float), np.asarray(b.faces)


def filled(grid, code=-3):
    return set(zip(*np.nonzero(grid == code)))


# Thin wall used throughout: 0.5 m thick on a 2 m grid, crossing the cell
# boundary at x=4 so each of columns 4 and 5 sees a single face
# (~9/27 = 0.33 surface contact).  Extents chosen so no face lies exactly on
# a cell boundary: x in [3.9, 4.4], y in [0.3, 11.7], z in [0.3, 9.7].
# Cells the wall crosses: rows 3..8, cols {4, 5}, zi 3..7.
# No voxel-column centre (x = ..., 3, 5, ...) lies inside the wall, so the
# winding fill contributes nothing — the shell must supply every voxel.
WALL = ((3.9, 0.3, 0.3), (0.5, 11.4, 9.4))


def wall_cells():
    return [(row, col, zi)
            for row in range(3, 9) for col in (4, 5) for zi in range(3, 8)]


def test_adjacent_anchor_drops_upper_thin_wall():
    """Documents the pre-fix rule: only shell voxels 6-adjacent to a filled
    voxel survive, so a tall thin wall keeps just its bottom slice."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1                     # ground layer under the wall
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="adjacent")
    got = filled(grid)
    assert (5, 4, 3) in got                # bottom slice: adjacent to ground
    assert (5, 4, 6) not in got            # upper wall: dropped by adjacency


def test_connected_anchor_keeps_full_thin_wall():
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="connected")
    got = filled(grid)
    for cell in wall_cells():
        assert cell in got, f"gap at {cell}"


def test_connected_anchor_drops_disconnected_fragment():
    """One mesh containing an anchored wall AND a floating cube far away:
    the wall survives the flood, the fragment does not."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1
    v1, f1 = box(*WALL)
    v2, f2 = box((-3.7, 0.3, 10.3), (1.4, 1.4, 1.4))   # one cell: (8, 1, 8)
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="connected")
    got = filled(grid)
    assert (5, 4, 5) in got                # wall kept
    assert (8, 1, 8) not in got            # floating fragment dropped


def test_connected_anchor_without_any_seed_keeps_whole_shell():
    """No filled voxel anywhere (per-category export grids have no terrain):
    dropping the whole feature would be worse than keeping it unanchored."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)   # completely empty: no anchors
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="connected")
    got = filled(grid)
    for cell in wall_cells():
        assert cell in got, f"gap at {cell}"


def test_adjacent_anchor_without_any_seed_keeps_nothing():
    """Current behavior, unchanged in tight mode."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="adjacent")
    assert filled(grid) == set()


def test_unknown_anchor_raises():
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    v, f = box(*WALL)
    with pytest.raises(ValueError):
        _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="loose")
