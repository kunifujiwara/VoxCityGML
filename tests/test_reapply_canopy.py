"""nDSM re-apply support: extras carry the z datum and the mesh-vegetation mask."""
import numpy as np
import pytest

from voxcitygml.models import CityGMLMeshCollection, Mesh3D
# The stub config is identical to the one the pipeline-core tests use.
from tests.test_pipeline_core import _config as _stub_cfg


def _stub_pipeline(monkeypatch, tmp_path, *, use_3d=True):
    """Monkeypatch every heavy pipeline stage, mirroring test_pipeline_core.

    The one deliberate difference: ``fake_voxelize`` honours the ``info_out``
    contract, so the assertions exercise the threading of those values from
    the voxelizer through ``run_core`` into ``extras``.
    """
    import voxcitygml.pipeline as pl

    def fake_parse(path, **kwargs):
        col = CityGMLMeshCollection()
        col.buildings = [Mesh3D(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
            feature_type='building', feature_id='b1')]
        return col

    monkeypatch.setattr(pl, 'parse_citygml_directory', fake_parse)
    monkeypatch.setattr(pl, 'merge_terrain_meshes', lambda t: t)
    monkeypatch.setattr(pl, 'terrain_meshes_to_dem_grid',
                        lambda *a, **k: np.zeros((10, 10)))
    monkeypatch.setattr(pl, 'get_land_cover_grid',
                        lambda *a, **k: np.zeros((10, 10), dtype=np.int32))
    monkeypatch.setattr(pl, 'meshes_to_building_grids',
                        lambda *a, **k: (np.zeros((10, 10)),
                                         np.empty((10, 10), dtype=object),
                                         np.zeros((10, 10))))
    monkeypatch.setattr(pl, 'get_canopy_grids',
                        lambda *a, **k: (np.zeros((10, 10)),
                                         np.zeros((10, 10))))

    def fake_voxelize(collection, rectangle, center_lon, center_lat, meshsize,
                      *, info_out=None, **kwargs):
        if info_out is not None:
            info_out["voxel_min_z"] = -3.5
            mask = np.zeros((10, 10), dtype=bool)
            mask[2, 3] = True
            info_out["mesh_vegetation_mask"] = mask
        return np.zeros((10, 10, 5), dtype=np.int16)

    monkeypatch.setattr(pl, 'voxelize_citygml_meshes', fake_voxelize)
    monkeypatch.setattr(pl, 'resolve_citygml_paths', lambda p: [str(tmp_path)])


def test_apply_canopy_reports_mesh_tree_mask():
    """_apply_canopy must surface the already_has_tree array it computes."""
    from voxcitygml.voxelizer3d import _apply_canopy, Grid3DParams, TREE_CODE

    gp = Grid3DParams(n_rows=4, n_cols=4, n_z=10,
                      min_x=0.0, max_x=8.0, min_y=0.0, max_y=8.0,
                      min_z=0.0, max_z=20.0, voxel_size=2.0)
    grid = np.zeros((4, 4, 10), dtype=np.int16)
    grid[1, 1, 0:3] = TREE_CODE          # a mesh-derived tree column
    dem = np.zeros((4, 4))
    canopy_top = np.full((4, 4), 6.0)
    out = {}
    _apply_canopy(grid, gp, dem, canopy_top, None, None, mesh_tree_mask_out=out)

    mask = out["mesh_tree_mask"]
    assert mask.shape == (4, 4)
    assert mask[1, 1]
    assert mask.sum() == 1, "only the pre-seeded mesh column may be marked"
    # Guard the capture *point*, not just the value: this run wrote canopy
    # voxels into the other 15 columns, so a mask captured after the write
    # would have sum() == 16.  Assert the write actually happened, otherwise
    # the sum()==1 check above proves nothing.
    assert np.any(grid[0, 0] == TREE_CODE), "canopy fill did not run"


def test_apply_canopy_reports_mask_even_when_no_canopy():
    """The early return for 'no canopy anywhere' must still yield a mask."""
    from voxcitygml.voxelizer3d import _apply_canopy, Grid3DParams, TREE_CODE

    gp = Grid3DParams(n_rows=4, n_cols=4, n_z=10,
                      min_x=0.0, max_x=8.0, min_y=0.0, max_y=8.0,
                      min_z=0.0, max_z=20.0, voxel_size=2.0)
    grid = np.zeros((4, 4, 10), dtype=np.int16)
    grid[3, 2, 0:2] = TREE_CODE
    out = {}
    _apply_canopy(grid, gp, np.zeros((4, 4)), np.zeros((4, 4)), None, None,
                  mesh_tree_mask_out=out)

    mask = out["mesh_tree_mask"]
    assert mask.shape == (4, 4)
    assert mask[3, 2]
    assert mask.sum() == 1


def test_extras_carry_voxel_min_z_and_mask(monkeypatch, tmp_path):
    """run() must expose the z datum and mask so canopy can be re-applied."""
    import voxcitygml.pipeline as pl

    _stub_pipeline(monkeypatch, tmp_path)
    city = pl.VoxCityGML(_stub_cfg(tmp_path)).run()
    assert "voxel_min_z" in city.extras
    assert isinstance(city.extras["voxel_min_z"], float)
    assert city.extras["voxel_min_z"] == -3.5
    mask = city.extras["mesh_vegetation_mask"]
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.shape == city.voxels.classes.shape[:2]
    assert mask[2, 3]


def test_extras_sane_without_3d_voxelizer(monkeypatch, tmp_path):
    """The legacy voxcity Voxelizer path has no z datum and no mesh trees."""
    from dataclasses import replace
    import voxcitygml.pipeline as pl

    _stub_pipeline(monkeypatch, tmp_path)
    cfg = replace(_stub_cfg(tmp_path), use_3d_voxelizer=False)

    class _FakeVoxelizer:
        def __init__(self, **kwargs):
            pass

        def generate_combined(self, **kwargs):
            return np.zeros((10, 10, 5), dtype=np.int16)

    import voxcity.generator.voxelizer as vz
    monkeypatch.setattr(vz, 'Voxelizer', _FakeVoxelizer)

    art = pl.run_core(cfg)
    assert art.voxel_min_z is None
    assert art.mesh_vegetation_mask.dtype == bool
    assert art.mesh_vegetation_mask.shape == (10, 10)
    assert not art.mesh_vegetation_mask.any()
