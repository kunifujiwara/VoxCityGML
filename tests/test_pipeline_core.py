"""Tests for the unified pipeline core (heavy stages monkeypatched)."""
import numpy as np
import pytest

import voxcitygml.pipeline as pl
from voxcitygml.models import (
    VoxelizerConfig, CityGMLMeshCollection, Mesh3D,
)


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Monkeypatch every heavy stage of the pipeline with tiny stubs."""
    calls = {}

    def fake_parse(path, **kwargs):
        calls['parse_kwargs'] = kwargs
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

    def fake_voxelize(collection, rectangle, center_lon, center_lat,
                      meshsize, **kwargs):
        calls['voxelize_kwargs'] = kwargs
        return np.zeros((10, 10, 5), dtype=np.int16)

    monkeypatch.setattr(pl, 'voxelize_citygml_meshes', fake_voxelize)
    monkeypatch.setattr(pl, 'resolve_citygml_paths', lambda p: [str(tmp_path)])
    return calls


def _config(tmp_path):
    return VoxelizerConfig(
        citygml_path=str(tmp_path),
        center_lon=139.77, center_lat=35.65,
        size_meters=100, meshsize=5.0,
        land_cover_source='OpenStreetMap',
        canopy_height_source='Static',
        save_output=False,
        terrain_underground_depth=7.5,
    )


def test_run_core_returns_artifacts(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    art = pl.run_core(cfg)
    assert art.voxel_grid.shape == (10, 10, 5)
    assert art.dem_grid.shape == (10, 10)
    assert len(art.collection.buildings) == 1
    assert len(art.rectangle) == 4


def test_run_core_passes_underground_depth(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    pl.run_core(cfg)
    assert stub_pipeline['voxelize_kwargs']['underground_depth'] == 7.5


def test_run_raises_when_no_buildings(stub_pipeline, monkeypatch, tmp_path):
    monkeypatch.setattr(pl, 'parse_citygml_directory',
                        lambda path, **kwargs: CityGMLMeshCollection())
    cfg = _config(tmp_path)
    with pytest.raises(ValueError, match="[Nn]o.*buildings"):
        pl.run_core(cfg)


def test_run_assembles_voxcity(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    city = pl.VoxCityGML(cfg).run()
    # a real voxcity.models.VoxCity comes back from assemble_voxcity
    assert city.voxels.classes.shape == (10, 10, 5)
