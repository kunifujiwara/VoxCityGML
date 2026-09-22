import numpy as np
import pytest

from voxcitygml.voxelizer3d import Grid3DParams, _frame_extent

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
    _, min_x, max_x, min_y, max_y = _frame_extent(RECT, CLON, CLAT)
    assert (min_x, max_x, min_y, max_y) == (gp.min_x, gp.max_x, gp.min_y, gp.max_y)


def test_frame_transformer_inverse_round_trips():
    tr, min_x, max_x, min_y, max_y = _frame_extent(RECT, CLON, CLAT)
    rng = np.random.default_rng(1)
    x = rng.uniform(min_x, max_x, 50)
    y = rng.uniform(min_y, max_y, 50)
    lon, lat = tr.inverse(x, y)
    x2, y2 = tr.transform(lon, lat)
    assert np.allclose(x2, x, atol=1e-6) and np.allclose(y2, y, atol=1e-6)
