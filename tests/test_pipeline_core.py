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
    # run_core must not mutate cfg; the resolved value lives on the
    # artifacts instead.
    assert cfg.land_cover_source == 'OpenStreetMap'
    assert art.land_cover_source == 'OpenStreetMap'


def test_run_core_passes_underground_depth(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    pl.run_core(cfg)
    assert stub_pipeline['voxelize_kwargs']['underground_depth'] == 7.5


def test_run_core_forwards_parse_kwargs(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    pl.run_core(cfg)
    parse_kwargs = stub_pipeline['parse_kwargs']
    assert 'tree_citygml_path' in parse_kwargs
    assert 'dem_path' in parse_kwargs


def test_run_core_forwards_use_parse_cache(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    cfg.use_parse_cache = False
    pl.run_core(cfg)
    assert stub_pipeline['parse_kwargs']['use_parse_cache'] is False


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


def test_run_and_export_uses_resolved_center(monkeypatch, tmp_path):
    """run_and_export must route through run_core and use its resolved
    center_lon/center_lat (not cfg's, which are wrong for
    rectangle_vertices configs)."""
    import voxcitygml.pipeline_export as pe

    sentinel_lon, sentinel_lat = 12345.0, 6789.0
    fake_artifacts = pl.PipelineArtifacts(
        collection=CityGMLMeshCollection(),
        rectangle=[(0, 0), (0, 0), (0, 0), (0, 0)],
        buffered_rectangle=[(0, 0), (0, 0), (0, 0), (0, 0)],
        center_lon=sentinel_lon,
        center_lat=sentinel_lat,
        citygml_paths=[str(tmp_path)],
        land_cover_source='OpenStreetMap',
        canopy_height_source='Static',
        dem_grid=np.zeros((10, 10)),
        land_cover_grid=np.zeros((10, 10), dtype=np.int32),
        building_height_grid=np.zeros((10, 10)),
        building_min_height_grid=np.empty((10, 10), dtype=object),
        building_id_grid=np.zeros((10, 10)),
        canopy_top=np.zeros((10, 10)),
        canopy_bottom=np.zeros((10, 10)),
        voxel_grid=np.zeros((10, 10, 5), dtype=np.int16),
    )

    captured = {}

    def fake_export_voxels_obj(voxel_grid, collection, rectangle, **kwargs):
        captured['voxel_kwargs'] = kwargs
        return ('v.obj', object())

    monkeypatch.setattr(pe, 'run_core', lambda cfg: fake_artifacts)
    monkeypatch.setattr(pe, 'export_voxels_obj', fake_export_voxels_obj)
    monkeypatch.setattr(pe, 'export_meshes_obj', lambda *a, **k: ('m.obj', {}))
    monkeypatch.setattr(pe, 'export_per_category_voxels_obj',
                        lambda *a, **k: ('p.obj', None))
    monkeypatch.setattr(pe, 'export_landcover_obj', lambda *a, **k: 'l.obj')

    cfg = _config(tmp_path)
    result = pe.run_and_export(cfg)

    assert captured['voxel_kwargs']['center_lon'] == sentinel_lon
    assert captured['voxel_kwargs']['center_lat'] == sentinel_lat
    assert result == ('m.obj', 'v.obj', 'p.obj', 'l.obj')
