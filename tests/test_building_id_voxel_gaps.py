"""``building_id_grid`` must cover every column the voxeliser fills.

The 2-D id grid and the 3-D voxel grid are built by two independent
rasterisations of the same meshes: ``meshes_to_building_grids`` claims a cell
only when the cell *centre* passes a barycentric inside-triangle test, while
``voxelize_citygml_meshes`` marks every voxel the mesh *touches* (SAT
any-touch shell unioned with an SDF/winding interior fill). The 3-D fill is
therefore up to one cell wider on each side.

Consumers that intersect the two -- per-building highlighting, landmark
marking, per-building surface statistics, carve/delete -- silently drop that
fringe: on real PLATEAU LoD2 tiles it is 7.5% of building columns at 2 m and
14.9% at 5 m, and 30-45% for small buildings. Every dropped column is
adjacent to a claimed one (a pure 1-cell fringe, no interior holes), so
propagating the nearest id closes it.
"""
import numpy as np
import pytest

from voxcitygml.buildings.processor import fill_building_id_gaps

BUILDING = -3
AIR = 0
GROUND = -1


def _voxels(columns, shape):
    """A voxel grid with BUILDING voxels in the given (row, col) columns."""
    g = np.full(shape, AIR, dtype=np.int16)
    for r, c in columns:
        g[r, c, :2] = BUILDING
    return g


def test_fills_a_column_that_has_building_voxels_but_no_id():
    bid = np.zeros((4, 4), dtype=np.int32)
    bid[1, 1] = 7
    # (1,2) is the fringe: the voxeliser filled it, the rasteriser missed it.
    vox = _voxels([(1, 1), (1, 2)], (4, 4, 3))

    out = fill_building_id_gaps(bid, vox)

    assert out[1, 2] == 7
    assert out[1, 1] == 7          # untouched
    assert out.dtype == bid.dtype


def test_leaves_columns_without_building_voxels_at_zero():
    """Air and terrain columns must never acquire an id -- that would make a
    building appear to own ground it has no voxels in."""
    bid = np.zeros((4, 4), dtype=np.int32)
    bid[0, 0] = 5
    vox = _voxels([(0, 0)], (4, 4, 3))

    out = fill_building_id_gaps(bid, vox)

    assert out[3, 3] == 0
    assert int((out != 0).sum()) == 1


def test_never_overwrites_an_existing_id():
    """A column the rasteriser already attributed keeps its own id, even when
    a neighbour is nearer in the EDT sense."""
    bid = np.zeros((3, 5), dtype=np.int32)
    bid[1, 1] = 4
    bid[1, 3] = 9
    vox = _voxels([(1, 1), (1, 2), (1, 3)], (3, 5, 3))

    out = fill_building_id_gaps(bid, vox)

    assert out[1, 1] == 4
    assert out[1, 3] == 9
    assert out[1, 2] in (4, 9)     # the gap takes one of its neighbours


def test_is_a_no_op_when_every_building_column_already_has_an_id():
    bid = np.zeros((3, 3), dtype=np.int32)
    bid[1, 1] = 2
    vox = _voxels([(1, 1)], (3, 3, 3))

    out = fill_building_id_gaps(bid, vox)

    assert np.array_equal(out, bid)


def test_a_grid_with_no_ids_at_all_is_returned_unchanged():
    """Guard against the degenerate EDT case: with no non-zero id anywhere
    there is no nearest source to propagate, and scipy's returned indices are
    meaningless. Building voxels with no ids at all must stay at 0 rather
    than being attributed to a fabricated id."""
    bid = np.zeros((3, 3), dtype=np.int32)
    vox = _voxels([(1, 1)], (3, 3, 3))

    out = fill_building_id_gaps(bid, vox)

    assert int((out != 0).sum()) == 0


def test_does_not_mutate_its_input():
    bid = np.zeros((3, 3), dtype=np.int32)
    bid[0, 0] = 1
    vox = _voxels([(0, 0), (0, 1)], (3, 3, 3))
    before = bid.copy()

    fill_building_id_gaps(bid, vox)

    assert np.array_equal(bid, before)


def test_a_shape_mismatch_is_refused_rather_than_broadcast():
    """The two grids come from separately-computed frames. Silently
    broadcasting a mismatch would mis-attribute every column."""
    bid = np.zeros((3, 3), dtype=np.int32)
    vox = _voxels([(1, 1)], (4, 3, 3))

    with pytest.raises(ValueError):
        fill_building_id_gaps(bid, vox)


def test_the_1_cell_fringe_of_a_solid_block_is_fully_recovered():
    """The real-world shape: a block whose voxel footprint is one cell larger
    on every side than its rasterised footprint."""
    bid = np.zeros((7, 7), dtype=np.int32)
    bid[2:5, 2:5] = 3                                  # rasterised core
    cols = [(r, c) for r in range(1, 6) for c in range(1, 6)]
    vox = _voxels(cols, (7, 7, 3))                     # voxelised, 1 wider

    out = fill_building_id_gaps(bid, vox)

    assert int((out == 3).sum()) == 25                 # 5x5, fringe recovered
    assert int((out != 0).sum()) == 25                 # nothing else claimed
