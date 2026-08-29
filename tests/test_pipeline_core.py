"""Tests for the unified pipeline core (heavy stages monkeypatched)."""
from dataclasses import replace

import numpy as np
import pytest

import voxcitygml.pipeline as pl
from voxcitygml.models import (
    VoxelizerConfig, CityGMLMeshCollection, Mesh3D,
)
#: The anchors the shell rasterizer actually implements -- imported, not
#: restated, so this stays a real check rather than a second opinion.
from voxcitygml.voxelizer3d import SHELL_ANCHORS

#: Centre coordinates only ``run_core`` can supply, so a caller that used
#: ``cfg``'s instead is caught.
_SENTINEL_LON, _SENTINEL_LAT = 12345.0, 6789.0


def _assert_concrete_voxel_kwargs(kwargs, where):
    """Fail if the voxelizer seam got policy instead of resolved values.

    ``VoxelizerConfig.occupancy_threshold`` / ``.building_shell_threshold``
    are ``Optional[float]`` whose ``None`` means "``voxelization_mode``
    decides".  Only ``cfg.resolved_voxel_params()`` turns that into a
    number.  A caller that forwards the raw attribute instead ships ``None``
    into ``if occupancy_threshold > 0.0:`` and dies with a ``TypeError`` on
    the first real run — invisible to any test whose stub merely tolerates
    whatever it is handed.  So the stubs assert here instead.

    Both are documented 0–1 occupancy fractions, so the range is checked
    too: type alone would wave through an argument-order slip that landed
    ``occupancy_subdivisions=3`` in a threshold slot.
    """
    for name in ("occupancy_threshold", "building_shell_threshold"):
        assert name in kwargs, f"{where} did not pass {name}"
        value = kwargs[name]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{where} passed {name}={value!r}; it must resolve the config "
            f"through cfg.resolved_voxel_params() rather than forward the "
            f"raw Optional attribute")
        assert 0.0 <= value <= 1.0, (
            f"{where} passed {name}={value!r}; occupancy is a 0-1 fraction, "
            f"so this is a wrong value in the right slot (or the right "
            f"value in the wrong slot)")
    assert kwargs.get("shell_anchor") in SHELL_ANCHORS, (
        f"{where} passed shell_anchor={kwargs.get('shell_anchor')!r}; "
        f"expected one of {SHELL_ANCHORS}")


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
                      meshsize, *, info_out=None, **kwargs):
        calls['voxelize_kwargs'] = kwargs
        _assert_concrete_voxel_kwargs(kwargs, 'run_core')
        # Honour the info_out contract: run_core subscripts these keys, so a
        # stub that omitted them would diverge from the real function.
        if info_out is not None:
            info_out['voxel_min_z'] = -2.0
            info_out['mesh_vegetation_mask'] = np.zeros((10, 10), dtype=bool)
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


@pytest.mark.parametrize('mode', ['inclusive', 'tight'])
def test_run_core_passes_resolved_voxel_params(stub_pipeline, tmp_path, mode):
    """Both modes must reach the voxelizer as the resolver's own values.

    Pinned to ``cfg.resolved_voxel_params()`` rather than to literals so
    retuning a mode in models.py cannot silently desync the pipeline from
    the policy it is supposed to implement.
    """
    cfg = replace(_config(tmp_path), voxelization_mode=mode)
    pl.run_core(cfg)
    kwargs = stub_pipeline['voxelize_kwargs']
    expected = cfg.resolved_voxel_params()
    assert kwargs['occupancy_threshold'] == expected.occupancy_threshold
    assert kwargs['building_shell_threshold'] == expected.building_shell_threshold
    assert kwargs['shell_anchor'] == expected.shell_anchor
    # Guard against a resolver that quietly agrees with itself: the two
    # modes must actually differ at this seam.
    assert kwargs['shell_anchor'] == (
        'connected' if mode == 'inclusive' else 'adjacent')


def test_run_core_forwards_explicit_threshold_override(stub_pipeline, tmp_path):
    """Explicit thresholds must survive the plumbing, beating the mode.

    ``occupancy_threshold`` is covered here as well as
    ``building_shell_threshold``: both modes resolve it to 0.0, so a user
    override is the ONLY way it is ever non-zero — and
    examples/run_building_gvi.py depends on exactly that.  A plumbing bug
    that pinned it to the mode value would be invisible otherwise.
    """
    cfg = replace(_config(tmp_path),
                  building_shell_threshold=0.5, occupancy_threshold=0.25)
    assert cfg.voxelization_mode == 'inclusive'
    pl.run_core(cfg)
    kwargs = stub_pipeline['voxelize_kwargs']
    assert kwargs['building_shell_threshold'] == 0.5
    assert kwargs['occupancy_threshold'] == 0.25
    # The anchor has no per-field override; it still follows the mode.
    assert kwargs['shell_anchor'] == 'connected'


def test_run_core_forwards_parse_kwargs(stub_pipeline, tmp_path):
    cfg = _config(tmp_path)
    pl.run_core(cfg)
    parse_kwargs = stub_pipeline['parse_kwargs']
    assert 'tree_citygml_path' in parse_kwargs
    assert 'dem_path' in parse_kwargs


def test_run_core_forwards_use_parse_cache(stub_pipeline, tmp_path):
    # Constructor kwarg, not attribute assignment: VoxelizerConfig is a plain
    # mutable dataclass, so setting the attribute afterwards would still pass
    # if the field were dropped from models.py.
    assert VoxelizerConfig(citygml_path=str(tmp_path)).use_parse_cache is True
    cfg = _config(tmp_path)
    cfg = replace(cfg, use_parse_cache=False)
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


@pytest.fixture
def stub_export(monkeypatch, tmp_path):
    """Stub every exporter so ``run_and_export`` runs in milliseconds.

    Returns the dict the stubs record their kwargs into.
    """
    import voxcitygml.pipeline_export as pe

    fake_artifacts = pl.PipelineArtifacts(
        collection=CityGMLMeshCollection(),
        rectangle=[(0, 0), (0, 0), (0, 0), (0, 0)],
        buffered_rectangle=[(0, 0), (0, 0), (0, 0), (0, 0)],
        center_lon=_SENTINEL_LON,
        center_lat=_SENTINEL_LAT,
        citygml_paths=[str(tmp_path)],
        land_cover_source='OpenStreetMap',
        canopy_height_source='Static',
        dem_source='CityGML Terrain',
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

    def fake_export_per_category_voxels_obj(*args, **kwargs):
        captured['per_cat_kwargs'] = kwargs
        _assert_concrete_voxel_kwargs(kwargs, 'run_and_export')
        return ('p.obj', None)

    monkeypatch.setattr(pe, 'run_core', lambda cfg: fake_artifacts)
    monkeypatch.setattr(pe, 'export_voxels_obj', fake_export_voxels_obj)
    monkeypatch.setattr(pe, 'export_meshes_obj', lambda *a, **k: ('m.obj', {}))
    monkeypatch.setattr(pe, 'export_per_category_voxels_obj',
                        fake_export_per_category_voxels_obj)
    monkeypatch.setattr(pe, 'export_landcover_obj', lambda *a, **k: 'l.obj')
    return captured


def test_run_and_export_uses_resolved_center(stub_export, tmp_path):
    """run_and_export must route through run_core and use its resolved
    center_lon/center_lat (not cfg's, which are wrong for
    rectangle_vertices configs)."""
    import voxcitygml.pipeline_export as pe

    cfg = _config(tmp_path)
    result = pe.run_and_export(cfg)

    assert stub_export['voxel_kwargs']['center_lon'] == _SENTINEL_LON
    assert stub_export['voxel_kwargs']['center_lat'] == _SENTINEL_LAT
    assert result == ('m.obj', 'v.obj', 'p.obj', 'l.obj')


@pytest.mark.parametrize('mode', ['inclusive', 'tight'])
def test_run_and_export_passes_resolved_voxel_params(stub_export, tmp_path, mode):
    """The per-category OBJ export must voxelize buildings with the SAME
    resolved knobs the main grid used, or exported building voxels stop
    matching ``voxelize_citygml_meshes`` (the 2026-08-11 invariant)."""
    import voxcitygml.pipeline_export as pe

    cfg = replace(_config(tmp_path), voxelization_mode=mode)
    pe.run_and_export(cfg)
    kwargs = stub_export['per_cat_kwargs']
    expected = cfg.resolved_voxel_params()
    assert kwargs['occupancy_threshold'] == expected.occupancy_threshold
    assert kwargs['building_shell_threshold'] == expected.building_shell_threshold
    assert kwargs['shell_anchor'] == expected.shell_anchor


def test_run_and_export_forwards_explicit_threshold_override(stub_export,
                                                             tmp_path):
    import voxcitygml.pipeline_export as pe

    cfg = replace(_config(tmp_path),
                  building_shell_threshold=0.5, occupancy_threshold=0.25)
    pe.run_and_export(cfg)
    kwargs = stub_export['per_cat_kwargs']
    assert kwargs['building_shell_threshold'] == 0.5
    assert kwargs['occupancy_threshold'] == 0.25
    assert kwargs['shell_anchor'] == 'connected'


def test_voxelizer_config_water_flatten_fields_default_on():
    cfg = VoxelizerConfig(citygml_path="x")
    assert cfg.flatten_water_dem is True
    assert cfg.water_dem_connectivity == 4


def test_voxelizer_config_rejects_bad_connectivity():
    with pytest.raises(ValueError, match="water_dem_connectivity"):
        VoxelizerConfig(citygml_path="x", water_dem_connectivity=6)


def test_voxelizer_config_accepts_connectivity_8():
    cfg = VoxelizerConfig(citygml_path="x", water_dem_connectivity=8)
    assert cfg.water_dem_connectivity == 8
