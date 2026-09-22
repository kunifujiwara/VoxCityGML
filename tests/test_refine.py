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


import trimesh

from voxcitygml.models import Mesh3D, CityGMLMeshCollection
from voxcitygml.voxelizer3d import (_MESHLIB_VOXEL_AVAILABLE, BUILDING_CODE,
                                    voxelize_citygml_meshes)

needs_meshlib = pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE,
                                   reason="building fill needs meshlib")


def _box_building(transformer, x0, y0, sx, sy, z0, sz):
    """A closed box in the local metre frame, returned as Mesh3D (lat, lon, z)."""
    b = trimesh.creation.box(extents=[sx, sy, sz])
    b.apply_translation([x0 + sx / 2, y0 + sy / 2, z0 + sz / 2])
    v = np.asarray(b.vertices, dtype=np.float64)
    lon, lat = transformer.inverse(v[:, 0], v[:, 1])
    verts = np.column_stack([lat, lon, v[:, 2]])
    return Mesh3D(vertices=verts, faces=np.asarray(b.faces, dtype=np.int32),
                  feature_type="building")


def _base_params_and_collection():
    tr, min_x, max_x, min_y, max_y = _frame_extent(RECT, CLON, CLAT)
    vs = 5.0
    n_cols = int((max_x - min_x) / vs + 0.5)
    n_rows = int((max_y - min_y) / vs + 0.5)
    gp = Grid3DParams(n_rows=n_rows, n_cols=n_cols, n_z=12, min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y, min_z=-5.0, max_z=55.0, voxel_size=vs)
    bldg = _box_building(tr, min_x + 100.0, min_y + 120.0, 30.0, 20.0, 0.0, 25.0)
    return gp, CityGMLMeshCollection(buildings=[bldg])


@needs_meshlib
def test_explicit_grid_params_are_used_verbatim_and_refine_nests():
    gp, coll = _base_params_and_collection()
    dem = np.zeros((gp.n_rows, gp.n_cols))
    coarse = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size,
                                     dem_grid=dem, grid_params=gp)
    fine = voxelize_citygml_meshes(coll, RECT, CLON, CLAT, gp.voxel_size / 2,
                                   dem_grid=dem, grid_params=gp.refined(2))
    assert coarse.shape == (gp.n_rows, gp.n_cols, gp.n_z)
    assert fine.shape == (2 * gp.n_rows, 2 * gp.n_cols, 2 * gp.n_z)
    nb0 = int((coarse == BUILDING_CODE).sum())
    nb2 = int((fine == BUILDING_CODE).sum())
    assert nb0 > 0 and 4 * nb0 <= nb2 <= 16 * nb0
    # every coarse building cell has at least one building child
    child_any = (fine == BUILDING_CODE).reshape(
        gp.n_rows, 2, gp.n_cols, 2, gp.n_z, 2).any(axis=(1, 3, 5))
    covered = child_any[coarse == BUILDING_CODE].mean()
    assert covered >= 0.9


def test_assembly_extras_carry_the_frame_centre():
    from voxcitygml.pipeline import assembly_extras
    import types
    cfg = types.SimpleNamespace(citygml_path="p")
    art = types.SimpleNamespace(
        citygml_paths=["p"], land_cover_source="OpenStreetMap",
        canopy_height_source="Static", dem_source=None, collection="COLL",
        voxel_min_z=-5.0, mesh_vegetation_mask=np.zeros((2, 2), bool),
        flatten_water_dem=True, water_dem_connectivity=4, water_dem_flattening={},
        center_lon=139.7725, center_lat=35.648)
    ex = assembly_extras(cfg, art)
    assert ex["center_lon"] == 139.7725 and ex["center_lat"] == 35.648
    assert ex["citygml_collection"] == "COLL" and ex["voxel_min_z"] == -5.0
    assert ex["mesh_vegetation_mask"].shape == (2, 2)
