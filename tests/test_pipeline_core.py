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
#: The converter the water CARVE uses.  Imported so the agreement test
#: below asks the carve itself what water is, rather than restating a
#: per-source table that could drift away from it.
from voxcitygml.voxelizer3d import _convert_land_cover

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


def _positional(args, kwargs, name, index):
    """Read an argument a stub may receive either way.

    ``run_core`` passes these positionally today; a refactor to keywords
    must not silently turn an ordering assertion into ``None``.
    """
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


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
    def fake_building_grids(*args, **kwargs):
        # Recorded so ordering tests can see WHICH dem_grid this stage was
        # handed, not merely that it ran.
        calls['building_dem'] = _positional(args, kwargs, 'dem_grid', 4)
        return (np.zeros((10, 10)),
                np.empty((10, 10), dtype=object),
                np.zeros((10, 10)))

    def fake_canopy_grids(*args, **kwargs):
        calls['canopy_dem'] = _positional(args, kwargs, 'dem_grid', 5)
        return np.zeros((10, 10)), np.zeros((10, 10))

    monkeypatch.setattr(pl, 'meshes_to_building_grids', fake_building_grids)
    monkeypatch.setattr(pl, 'get_canopy_grids', fake_canopy_grids)

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


@pytest.mark.parametrize('flatten', [True, False])
def test_run_core_forwards_flatten_water_dem(stub_pipeline, tmp_path, flatten):
    """``cfg.flatten_water_dem`` must actually reach the voxelizer.

    The field's default and validation are pinned above, but neither
    notices if ``run_core`` simply never forwards it -- and the voxelizer
    defaults the kwarg to True, so an unwired opt-out carves anyway and
    every test still passes.  Assert the value on the call itself.
    """
    cfg = replace(_config(tmp_path), flatten_water_dem=flatten)
    pl.run_core(cfg)
    assert stub_pipeline['voxelize_kwargs']['flatten_water_dem'] is flatten


#: Water in the OpenStreetMap source's own numbering.  voxcity's flattener
#: reaches the standard code 9 for OSM by adding 1, so 8 is water here --
#: confirmed against ``voxcity.utils.lc.get_land_cover_classes``.
_OSM_WATER = 8

#: Rows 0-2 of the north-up DEM: the water body, carrying TWO levels -- the
#: intra-river cliff this feature exists to remove.
_WATER_ROWS = 3
#: The water body spans only the WESTERN columns, so the fixture is
#: asymmetric along columns as well as rows (see ``_asymmetric_water_grids``).
_WATER_COLS = 6
_WATER_TOP_LEVEL = 8.0
_WATER_BODY_MIN = 3.0


def _asymmetric_water_grids():
    """A DEM / land-cover pair whose water body sits in the world NORTH-WEST.

    Deliberately asymmetric along ROWS.  The two grids live in opposite row
    frames -- ``dem_grid`` is north-up, ``land_cover_grid`` south-up
    (``landcover/processor.py``) -- so a flattener that forgets to flip one
    of them flattens the vertically MIRRORED set of rows: no exception,
    plausible output, wrong rivers.  A uniform or vertically symmetric
    fixture cannot see that; this one can, because the mirrored rows carry
    different values than the real ones.

    Asymmetric along COLUMNS too, and that is a separate guarantee: on a
    column-uniform fixture ``fliplr`` is the identity, so a spurious EXTRA
    flip on top of the correct one would survive every assertion.  Here the
    water covers only columns 0-5, so a mirrored mask flattens columns 4-9
    instead and the eastern columns give it away.
    """
    dem = np.zeros((10, 10), dtype=np.float64)
    dem[0, :] = _WATER_TOP_LEVEL
    dem[1:_WATER_ROWS, :] = _WATER_BODY_MIN
    for r in range(_WATER_ROWS, 10):          # land: 20.0 .. 26.0, distinct
        dem[r, :] = 20.0 + (r - _WATER_ROWS)
    # SOUTH-up land cover: its row 9 is the NORTHERN edge, so the water rows
    # are 7-9 here and land on DEM rows 0-2 only after a flipud.
    lc = np.zeros((10, 10), dtype=np.int32)
    lc[10 - _WATER_ROWS:, :_WATER_COLS] = _OSM_WATER
    return dem, lc


def _assert_cliff_intact(dem):
    """The unflattened river: both original water levels still present."""
    assert np.all(dem[0, :] == _WATER_TOP_LEVEL)
    assert np.all(dem[1:_WATER_ROWS, :] == _WATER_BODY_MIN)


@pytest.fixture
def water_pipeline(stub_pipeline, monkeypatch):
    """``stub_pipeline`` with a real river in it.  Returns the calls dict."""
    dem, lc = _asymmetric_water_grids()
    monkeypatch.setattr(pl, 'terrain_meshes_to_dem_grid',
                        lambda *a, **k: dem.copy())
    monkeypatch.setattr(pl, 'get_land_cover_grid',
                        lambda *a, **k: lc.copy())
    return stub_pipeline


def test_water_dem_flattening_respects_the_land_cover_frame(water_pipeline,
                                                            tmp_path):
    """The NORTHERN water body flattens; the southern land rows do not.

    This is the whole point of the adapter.  voxcity's shared rule labels
    the water mask and indexes the DEM with it directly, so it assumes one
    frame; voxcitygml hands it two.  Dropping the ``np.flipud`` fails both
    halves of this test at once.
    """
    cfg = _config(tmp_path)
    art = pl.run_core(cfg)
    dem = art.dem_grid

    assert np.all(dem[:_WATER_ROWS, :_WATER_COLS] == _WATER_BODY_MIN), (
        "the north-western water body did not collapse to its own minimum "
        "-- the land-cover mask was probably applied in the wrong frame")
    # The eastern half of those same rows is dry land and must be untouched;
    # this is what a spurious left-right flip trips over.
    assert np.all(dem[0, _WATER_COLS:] == _WATER_TOP_LEVEL)
    assert np.all(dem[1:_WATER_ROWS, _WATER_COLS:] == _WATER_BODY_MIN)
    for r in range(_WATER_ROWS, 10):
        assert np.all(dem[r, :] == 20.0 + (r - _WATER_ROWS)), (
            f"land row {r} was flattened; only water cells may change")


def test_water_dem_flattening_reaches_artifacts_and_extras(water_pipeline,
                                                           tmp_path):
    cfg = _config(tmp_path)
    art = pl.run_core(cfg)

    assert art.flatten_water_dem is True
    assert art.water_dem_connectivity == 4
    info = art.water_dem_flattening
    assert set(info) == {"applied", "reason", "water_body_count",
                         "water_cell_count", "water_dem_min_values"}, (
        f"info dict drifted from voxcity's contract: {sorted(info)}")
    assert info["applied"] is True
    assert info["reason"] == "applied"
    assert info["water_body_count"] == 1
    assert info["water_cell_count"] == _WATER_ROWS * _WATER_COLS
    assert info["water_dem_min_values"] == [_WATER_BODY_MIN]

    city = pl.VoxCityGML(cfg).run()
    assert city.extras["flatten_water_dem"] is True
    assert city.extras["water_dem_connectivity"] == 4
    assert city.extras["water_dem_flattening"]["applied"] is True
    # C3: the DEM the model carries must be the flattened one, or the
    # voxelizer's carve targets a level the stored DEM never agreed to.
    assert np.all(
        city.dem.elevation[-_WATER_ROWS:, :_WATER_COLS] == _WATER_BODY_MIN), (
        "assemble_voxcity received the pre-flattening DEM "
        "(south-up, so the northern water rows are the LAST ones)")


def test_water_dem_flattening_can_be_opted_out(water_pipeline, tmp_path):
    cfg = replace(_config(tmp_path), flatten_water_dem=False)
    art = pl.run_core(cfg)
    assert art.flatten_water_dem is False
    assert art.water_dem_flattening["applied"] is False
    assert art.water_dem_flattening["reason"] == "disabled"
    _assert_cliff_intact(art.dem_grid)


def test_flatten_runs_before_building_and_canopy_rasterisation(water_pipeline,
                                                               tmp_path):
    """C2: both stages read ``dem_grid[r, c]`` as their ground datum.

    Flattening after them would leave building and vegetation heights
    measured against a DEM that no longer exists.
    """
    cfg = _config(tmp_path)
    art = pl.run_core(cfg)

    for stage in ('building_dem', 'canopy_dem'):
        seen = water_pipeline[stage]
        assert seen is not None, f"{stage} was never recorded"
        assert np.all(seen[:_WATER_ROWS, :_WATER_COLS] == _WATER_BODY_MIN), (
            f"{stage} was handed the pre-flattening DEM")
        np.testing.assert_array_equal(seen, art.dem_grid)

    # The voxelizer is the other half of the fix: its carve conforms voxel
    # ground to whatever DEM it is given, so handing it the pre-flatten grid
    # while storing the flattened one is exactly the C3 desync.
    np.testing.assert_array_equal(
        water_pipeline['voxelize_kwargs']['dem_grid'], art.dem_grid)


def test_older_voxcity_degrades_instead_of_crashing(water_pipeline,
                                                    monkeypatch, tmp_path):
    """C4: pyproject pins ``voxcity>=1.3.2``, which predates the public name.

    ``from module import missing_name`` raises ImportError, so deleting the
    attribute reproduces an older install exactly.
    """
    import voxcity.generator.pipeline as vp
    monkeypatch.delattr(vp, 'flatten_water_dem_by_component')

    cfg = _config(tmp_path)
    art = pl.run_core(cfg)
    assert art.water_dem_flattening["applied"] is False
    assert art.water_dem_flattening["reason"] == "voxcity_flattener_unavailable"
    _assert_cliff_intact(art.dem_grid)


#: Every land-cover source voxcitygml supports.  ``CityGML`` is the one
#: the PLATEAU integration path actually uses, and the one whose codes
#: voxcity's converter does NOT know about.
_LAND_COVER_SOURCES = [
    "OpenStreetMap", "CityGML", "Urbanwatch", "ESA WorldCover",
    "ESRI 10m Annual Land Cover", "Dynamic World V1", "OpenEarthMapJapan",
]


def _carve_water_codes(source):
    """The raw codes THE CARVE calls water, asked of the carve's converter.

    Not a hand-written table: the whole point of the agreement test is that
    one side is derived from the carve and the other is measured off the
    flatten, so a table restating either would defeat it.
    """
    probe = np.arange(-1, 16, dtype=np.int64).reshape(1, -1)
    return probe[0][_convert_land_cover(probe.copy(), source)[0] == 9]


def _dry_code(source):
    """A raw code this source does NOT call water."""
    water = set(_carve_water_codes(source).tolist())
    return next(c for c in range(16) if c not in water)


def _assert_flatten_sees_the_water(source, code, connectivity=4):
    """The flatten must flatten exactly the cells the carve calls water."""
    lc = np.full((6, 6), _dry_code(source), dtype=np.int64)
    lc[:3, :] = code                    # SOUTH-up: the southern half
    dem = np.arange(36, dtype=np.float64).reshape(6, 6)

    flattened, info = pl._flatten_water_dem(
        dem, lc, source, enabled=True, connectivity=connectivity)

    # South-up rows 0-2 are north-up rows 3-5.  Pinning the whole grid, not
    # just a count, catches an empty mask, a mirrored mask and an
    # over-broad mask with one assertion.
    expected = dem.copy()
    expected[3:, :] = dem[3:, :].min()
    np.testing.assert_array_equal(
        flattened, expected,
        err_msg=(f"{source} raw water code {code}: the flatten did not "
                 f"flatten the cells the carve calls water"))
    assert info["water_cell_count"] == 18, (
        f"{source} raw water code {code}: flatten counted "
        f"{info['water_cell_count']} water cells, carve sees 18")
    assert info["water_body_count"] == 1
    assert info["applied"] is True
    assert info["reason"] == "applied"


@pytest.mark.parametrize('source', _LAND_COVER_SOURCES)
def test_carve_and_flatten_agree_on_what_water_is(source):
    """The two halves of the fix must share one definition of water.

    They convert land cover with DIFFERENT functions: the carve uses
    voxcitygml's ``_convert_land_cover``, which has a CityGML branch
    (those codes are already 1-based); voxcity's has no such branch and
    its unknown-source else adds 1, turning CityGML water (9) into 10 so
    the ``== 9`` mask comes back EMPTY.  The flatten then no-ops on the
    exact source the PLATEAU path uses while reporting
    ``no_water_cells`` -- a false answer rather than an error, and one no
    OpenStreetMap fixture can see, because OSM is the single source where
    the two converters happen to agree.

    Parametrised over every source so this dies once and stays dead, in
    either package and in either direction.
    """
    codes = _carve_water_codes(source)
    assert codes.size, f"no water code found for {source}"
    for code in codes:
        _assert_flatten_sees_the_water(source, int(code))


def test_citygml_source_flattens_through_the_whole_pipeline(stub_pipeline,
                                                            monkeypatch,
                                                            tmp_path):
    """The above at ``run_core`` level, on the PLATEAU pairing.

    ``land_cover_source='CityGML'`` with LOD2 CityGML terrain is what every
    PLATEAU integration test in this repo runs; the unit test above would
    still pass if ``run_core`` stopped routing through the adapter.
    """
    dem, lc = _asymmetric_water_grids()
    lc = np.where(lc == _OSM_WATER, 9, 0).astype(np.int32)   # CityGML codes
    monkeypatch.setattr(pl, 'terrain_meshes_to_dem_grid',
                        lambda *a, **k: dem.copy())
    monkeypatch.setattr(pl, 'get_land_cover_grid', lambda *a, **k: lc.copy())

    cfg = replace(_config(tmp_path), land_cover_source='CityGML')
    art = pl.run_core(cfg)

    assert art.water_dem_flattening["applied"] is True, (
        f"CityGML water was invisible to the flatten: "
        f"{art.water_dem_flattening}")
    assert art.water_dem_flattening["water_cell_count"] == (
        _WATER_ROWS * _WATER_COLS)
    assert np.all(art.dem_grid[:_WATER_ROWS, :_WATER_COLS] == _WATER_BODY_MIN)


def test_flatten_works_without_voxcitys_standard_passthrough(monkeypatch):
    """``convert_land_cover``'s ``"Standard"`` branch is as new as the flattener.

    ``pyproject`` pins only ``voxcity>=1.3.2``, so it may be absent -- and
    absent it is not an error but an UNKNOWN source, whose else-branch adds
    1 and silently reintroduces the empty-mask bug.  Emulate that install
    and require the flatten to still be correct, rather than merely to warn.
    """
    import voxcity.utils.lc as vlc
    real = vlc.convert_land_cover

    def older_convert(input_array, land_cover_source=None, **kwargs):
        if land_cover_source == "Standard":      # did not exist yet
            return np.asarray(input_array).copy() + 1
        return real(input_array, land_cover_source=land_cover_source,
                    **kwargs)

    monkeypatch.setattr(vlc, 'convert_land_cover', older_convert)
    # CityGML is the source that depends on the standardisation entirely.
    _assert_flatten_sees_the_water("CityGML", 9)


def test_a_run_that_flattens_nothing_says_why(stub_pipeline, tmp_path, capsys):
    """A broken water mask must not be indistinguishable from a dry AOI.

    The stub land cover is all zeros, so nothing is water and the flatten
    reports ``no_water_cells``.  That is the same message a genuinely
    broken mask produces, and printing NOTHING on this path is what let
    the CityGML no-op hide for two commits: the fix has to be audible.
    """
    pl.run_core(_config(tmp_path))
    out = capsys.readouterr().out
    assert "no DEM flattening applied" in out
    assert "no_water_cells" in out


def test_water_dem_connectivity_reaches_the_flattener():
    """The knob must change the ANSWER, not merely be recorded.

    Two water blocks that touch only at a corner: separate bodies under
    4-connectivity, one body under 8.  Every other assertion in this file
    reads the value back off the config or the artifact, so a flattener
    call with ``connectivity`` hardcoded to 4 survives all of them.
    """
    # SOUTH-up land cover; blocks touch diagonally at (7,1)/(8,2).
    lc = np.zeros((10, 10), dtype=np.int32)
    lc[6:8, 0:2] = _OSM_WATER          # -> north-up rows 2-3, cols 0-1
    lc[8:10, 2:4] = _OSM_WATER         # -> north-up rows 0-1, cols 2-3
    dem = np.full((10, 10), 50.0)
    dem[2:4, 0:2] = [[10.0, 11.0], [11.0, 11.0]]     # southern block
    dem[0:2, 2:4] = [[20.0, 21.0], [21.0, 21.0]]     # northern block

    flat4, info4 = pl._flatten_water_dem(dem, lc, 'OpenStreetMap',
                                         enabled=True, connectivity=4)
    assert info4["water_body_count"] == 2
    assert sorted(info4["water_dem_min_values"]) == [10.0, 20.0]
    assert np.all(flat4[2:4, 0:2] == 10.0)
    assert np.all(flat4[0:2, 2:4] == 20.0), (
        "the corner-touching blocks were merged under 4-connectivity")

    flat8, info8 = pl._flatten_water_dem(dem, lc, 'OpenStreetMap',
                                         enabled=True, connectivity=8)
    assert info8["water_body_count"] == 1
    assert info8["water_dem_min_values"] == [10.0]
    assert np.all(flat8[2:4, 0:2] == 10.0)
    assert np.all(flat8[0:2, 2:4] == 10.0), (
        "connectivity=8 never reached the flattener")


def test_run_core_forwards_water_dem_connectivity(stub_pipeline, monkeypatch,
                                                  tmp_path):
    """...and ``run_core`` must hand the config's value down, not a default."""
    lc = np.zeros((10, 10), dtype=np.int32)
    lc[6:8, 0:2] = _OSM_WATER
    lc[8:10, 2:4] = _OSM_WATER
    dem = np.full((10, 10), 50.0)
    dem[2:4, 0:2] = 10.0
    dem[0:2, 2:4] = 20.0
    monkeypatch.setattr(pl, 'terrain_meshes_to_dem_grid',
                        lambda *a, **k: dem.copy())
    monkeypatch.setattr(pl, 'get_land_cover_grid', lambda *a, **k: lc.copy())

    art4 = pl.run_core(replace(_config(tmp_path), water_dem_connectivity=4))
    assert art4.water_dem_flattening["water_body_count"] == 2
    assert np.all(art4.dem_grid[0:2, 2:4] == 20.0)

    art8 = pl.run_core(replace(_config(tmp_path), water_dem_connectivity=8))
    assert art8.water_dem_flattening["water_body_count"] == 1
    assert np.all(art8.dem_grid[0:2, 2:4] == 10.0)


def test_voxelizer_config_rejects_bad_connectivity():
    with pytest.raises(ValueError, match="water_dem_connectivity"):
        VoxelizerConfig(citygml_path="x", water_dem_connectivity=6)


def test_voxelizer_config_accepts_connectivity_8():
    cfg = VoxelizerConfig(citygml_path="x", water_dem_connectivity=8)
    assert cfg.water_dem_connectivity == 8
