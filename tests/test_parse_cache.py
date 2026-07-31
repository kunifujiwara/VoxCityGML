"""Unit tests for the per-file parse cache (store/load/validation)."""
import os
import stat

import numpy as np
import pytest

from voxcitygml.models import Mesh3D
from voxcitygml.citygml import parse_cache
from voxcitygml.citygml.parse_cache import (
    load_cached_meshes, store_cached_meshes, reset_store_failures, _cache_path,
)


@pytest.fixture(autouse=True)
def _clear_store_latch():
    """Store failures latch writes off process-wide; isolate every test."""
    reset_store_failures()
    yield
    reset_store_failures()


def _src(tmp_path, name="53393671_bldg_6697_op.gml", content="<x/>"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _mesh(seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    defaults = dict(
        vertices=rng.random((5, 3)),
        faces=np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
        feature_type="building",
        feature_id=f"bldg_{seed}",
        attributes={"height": 12.5, "name": "A"},
    )
    defaults.update(kwargs)
    return Mesh3D(**defaults)


def _assert_meshes_equal(a, b):
    assert len(a) == len(b)
    for m1, m2 in zip(a, b):
        np.testing.assert_array_equal(m1.vertices, m2.vertices)
        np.testing.assert_array_equal(m1.faces, m2.faces)
        assert m1.feature_type == m2.feature_type
        assert m1.feature_id == m2.feature_id
        assert set(m1.attributes) == set(m2.attributes)
        for k, v in m1.attributes.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(v, m2.attributes[k])
            else:
                assert m2.attributes[k] == v


def test_miss_returns_none(tmp_path):
    src = _src(tmp_path)
    assert load_cached_meshes(src, "building", 2) is None


def test_store_then_load_round_trip(tmp_path):
    src = _src(tmp_path)
    meshes = [_mesh(0), _mesh(1)]
    store_cached_meshes(src, "building", 2, meshes)
    assert _cache_path(src, "building", 2).exists()
    loaded = load_cached_meshes(src, "building", 2)
    _assert_meshes_equal(meshes, loaded)


def test_ndarray_attribute_round_trips_as_array(tmp_path):
    """Terrain meshes carry attributes['triangle_coords'] (N,3,3) ndarray."""
    src = _src(tmp_path, name="dem.gml")
    tri = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    meshes = [_mesh(0, feature_type="terrain",
                    attributes={"triangle_coords": tri, "kind": "TIN"})]
    store_cached_meshes(src, "terrain", None, meshes)
    loaded = load_cached_meshes(src, "terrain", None)
    got = loaded[0].attributes["triangle_coords"]
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, tri)
    assert loaded[0].attributes["kind"] == "TIN"


def test_normals_and_empty_list_round_trip(tmp_path):
    src = _src(tmp_path)
    with_normals = [_mesh(0, normals=np.ones((5, 3)))]
    store_cached_meshes(src, "building", 2, with_normals)
    loaded = load_cached_meshes(src, "building", 2)
    np.testing.assert_array_equal(loaded[0].normals, np.ones((5, 3)))

    src2 = _src(tmp_path, name="empty.gml")
    store_cached_meshes(src2, "vegetation", None, [])
    assert load_cached_meshes(src2, "vegetation", None) == []


def test_source_change_invalidates(tmp_path):
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    # Change content -> different size (mtime alone can be too coarse on FAT)
    src.write_text("<x>changed</x>", encoding="utf-8")
    assert load_cached_meshes(src, "building", 2) is None


def test_same_size_mtime_change_invalidates(tmp_path):
    """An edit that preserves byte count must still invalidate (mtime clause)."""
    src = _src(tmp_path, content="<x/>")
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    assert load_cached_meshes(src, "building", 2) is not None
    src.write_text("<y/>", encoding="utf-8")  # same size, different content
    os.utime(src, ns=(1_000_000_000, 1_234_567_891))
    assert load_cached_meshes(src, "building", 2) is None


def test_same_mtime_size_change_invalidates(tmp_path):
    """Size is the backstop when mtime is unchanged (coarse clocks, restores)."""
    src = _src(tmp_path, content="<x/>")
    mtime_ns = os.stat(src).st_mtime_ns
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    src.write_text("<x>much longer</x>", encoding="utf-8")
    os.utime(src, ns=(mtime_ns, mtime_ns))  # pretend the clock never moved
    assert os.stat(src).st_mtime_ns == mtime_ns
    assert load_cached_meshes(src, "building", 2) is None


def test_cache_version_bump_invalidates(tmp_path, monkeypatch):
    """CACHE_VERSION is what retires caches after an extractor change."""
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    assert load_cached_meshes(src, "building", 2) is not None
    monkeypatch.setattr(parse_cache, "CACHE_VERSION",
                        parse_cache.CACHE_VERSION + 1)
    assert load_cached_meshes(src, "building", 2) is None


def test_building_lod_keys_separately(tmp_path):
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 1, [_mesh(1)])
    store_cached_meshes(src, "building", 2, [_mesh(2)])
    store_cached_meshes(src, "building", None, [_mesh(3)])
    assert load_cached_meshes(src, "building", 1)[0].feature_id == "bldg_1"
    assert load_cached_meshes(src, "building", 2)[0].feature_id == "bldg_2"
    assert load_cached_meshes(src, "building", None)[0].feature_id == "bldg_3"


def test_derived_triangle_coords_elided_and_rebuilt(tmp_path):
    """triangle_coords == vertices[faces]: don't store it, rebuild on load."""
    src = _src(tmp_path, name="dem.gml")
    v = np.arange(27, dtype=np.float64).reshape(9, 3)
    f = np.arange(9, dtype=np.int32).reshape(3, 3)
    tri = v[f]
    store_cached_meshes(src, "terrain", None, [
        _mesh(0, vertices=v, faces=f, feature_type="terrain",
              attributes={"triangle_coords": tri, "kind": "TIN"})])

    with np.load(_cache_path(src, "terrain", None), allow_pickle=False) as data:
        assert "a0_triangle_coords" not in data.files, "derived array stored"

    loaded = load_cached_meshes(src, "terrain", None)
    got = loaded[0].attributes["triangle_coords"]
    assert got.tobytes() == tri.tobytes(), "rebuild must be bitwise identical"
    assert got.dtype == tri.dtype and got.shape == tri.shape
    assert loaded[0].attributes["kind"] == "TIN"


def test_independent_array_of_derived_shape_is_still_stored(tmp_path):
    """Same shape as vertices[faces] but different values -> must be stored."""
    src = _src(tmp_path, name="dem.gml")
    v = np.arange(27, dtype=np.float64).reshape(9, 3)
    f = np.arange(9, dtype=np.int32).reshape(3, 3)
    other = v[f] + 1.0  # right shape, wrong values
    store_cached_meshes(src, "terrain", None, [
        _mesh(0, vertices=v, faces=f, feature_type="terrain",
              attributes={"triangle_coords": other})])
    with np.load(_cache_path(src, "terrain", None), allow_pickle=False) as data:
        assert "a0_triangle_coords" in data.files
    loaded = load_cached_meshes(src, "terrain", None)
    np.testing.assert_array_equal(loaded[0].attributes["triangle_coords"], other)


def test_out_of_range_faces_do_not_break_store(tmp_path):
    """A malformed faces array must fall back to storing, not fail the store."""
    src = _src(tmp_path, name="dem.gml")
    v = np.arange(9, dtype=np.float64).reshape(3, 3)
    f = np.array([[0, 1, 99]], dtype=np.int32)  # 99 is out of range
    tri = np.zeros((1, 3, 3))
    store_cached_meshes(src, "terrain", None, [
        _mesh(0, vertices=v, faces=f, feature_type="terrain",
              attributes={"triangle_coords": tri})])
    assert _cache_path(src, "terrain", None).exists()
    loaded = load_cached_meshes(src, "terrain", None)
    np.testing.assert_array_equal(loaded[0].attributes["triangle_coords"], tri)


def test_source_epsg_mismatch_invalidates(tmp_path):
    """Vertices are cached post-reprojection, so the CRS is part of validity."""
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2, [_mesh(0)], "EPSG:25832")
    assert load_cached_meshes(src, "building", 2, "EPSG:25832") is not None
    assert load_cached_meshes(src, "building", 2, "EPSG:32633") is None
    assert load_cached_meshes(src, "building", 2, None) is None


def test_corrupt_cache_falls_back_to_none(tmp_path):
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    _cache_path(src, "building", 2).write_bytes(b"not an npz")
    assert load_cached_meshes(src, "building", 2) is None


@pytest.mark.skipif(os.name == "nt",
                    reason="chmod on directories is a no-op on Windows NTFS")
def test_readonly_cache_dir_store_does_not_raise(tmp_path):
    src = _src(tmp_path)
    cache_dir = _cache_path(src, "building", 2).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, stat.S_IREAD | stat.S_IEXEC)
    try:
        store_cached_meshes(src, "building", 2, [_mesh(0)])  # must not raise
    finally:
        os.chmod(cache_dir, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)


@pytest.mark.parametrize("value", [
    object(),          # no encoder at all
    np.bool_(False),   # would become the truthy string 'False' if stringified
    np.int64(7),       # would become '7'
])
def test_unserializable_attribute_skips_cache(tmp_path, value):
    """A hit must never disagree with a miss, so odd types are not cached."""
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2,
                        [_mesh(0, attributes={"weird": value})])  # no raise
    assert not _cache_path(src, "building", 2).exists()
    assert load_cached_meshes(src, "building", 2) is None


def test_store_failures_latch_writes_off(tmp_path):
    """A read-only dataset must not warn once per file, forever."""
    missing = tmp_path / "does_not_exist.gml"  # os.stat fails every time
    for _ in range(parse_cache._MAX_STORE_FAILURE_WARNINGS):
        store_cached_meshes(missing, "building", 2, [_mesh(0)])
    assert parse_cache._stores_disabled

    # Latched off: even a perfectly good store is now skipped.
    good = _src(tmp_path)
    store_cached_meshes(good, "building", 2, [_mesh(0)])
    assert not _cache_path(good, "building", 2).exists()

    reset_store_failures()
    store_cached_meshes(good, "building", 2, [_mesh(0)])
    assert _cache_path(good, "building", 2).exists()


def test_plateau_layout_cache_sits_beside_dataset(tmp_path):
    """udx/<type>/x.gml -> <dataset>/.voxcitygml_cache/udx/<type>/x...npz"""
    dataset = tmp_path / "13102_dataset"
    bldg_dir = dataset / "udx" / "bldg"
    bldg_dir.mkdir(parents=True)
    src = bldg_dir / "53393671_bldg_6697_op.gml"
    src.write_text("<x/>", encoding="utf-8")
    cp = _cache_path(src, "building", 2)
    assert cp == (dataset / ".voxcitygml_cache" / "udx" / "bldg"
                  / "53393671_bldg_6697_op.gml.building.lod2.npz")


def test_nested_udx_uses_innermost_dataset_root(tmp_path):
    """A parent folder named 'udx' must not pull the cache out of the dataset."""
    dataset = tmp_path / "udx" / "13102_dataset"
    bldg_dir = dataset / "udx" / "bldg"
    bldg_dir.mkdir(parents=True)
    src = bldg_dir / "53393671_bldg_6697_op.gml"
    src.write_text("<x/>", encoding="utf-8")
    cp = _cache_path(src, "building", 2)
    assert cp == (dataset / ".voxcitygml_cache" / "udx" / "bldg"
                  / "53393671_bldg_6697_op.gml.building.lod2.npz")


def test_meta_is_stored_as_utf8_bytes_not_utf32(tmp_path):
    """np.array(str) would be UTF-32: 4x the bytes on a hot path."""
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    with np.load(_cache_path(src, "building", 2), allow_pickle=False) as data:
        assert data["meta"].dtype == np.uint8


def test_non_ascii_attributes_round_trip_unescaped(tmp_path):
    """PLATEAU attributes are Japanese; \\uXXXX escaping would inflate meta."""
    src = _src(tmp_path)
    attrs = {"usage": "業務施設", "name": "東京都庁舎"}
    store_cached_meshes(src, "building", 2, [_mesh(0, attributes=attrs)])
    loaded = load_cached_meshes(src, "building", 2)
    assert loaded[0].attributes == attrs
    with np.load(_cache_path(src, "building", 2), allow_pickle=False) as data:
        meta_bytes = data["meta"].tobytes()
    assert "業務施設".encode("utf-8") in meta_bytes, "meta must not be \\u-escaped"


# ---------------------------------------------------------------------------
# Integration with _parse_single_file
# ---------------------------------------------------------------------------
import voxcitygml.citygml.parser as parser_mod
from voxcitygml.citygml.parser import _parse_single_file


@pytest.fixture
def fake_extractor(monkeypatch, tmp_path):
    """Minimal real XML file + stubbed building extractor with call counter."""
    src = tmp_path / "53393671_bldg_6697_op.gml"
    src.write_text("<CityModel/>", encoding="utf-8")
    calls = {"n": 0}

    def fake_extract(root, ns, prefer_lod=None, max_lod=4):
        calls["n"] += 1
        return [_mesh(seed=prefer_lod or 0)]

    monkeypatch.setattr(parser_mod, "extract_buildings_from_root", fake_extract)
    monkeypatch.setattr(parser_mod, "detect_crs_from_root", lambda root: None)
    return src, calls


def test_second_parse_hits_cache(fake_extractor):
    src, calls = fake_extractor
    first = _parse_single_file(src, "building", None, None, building_lod=2)
    second = _parse_single_file(src, "building", None, None, building_lod=2)
    assert calls["n"] == 1, "extractor must not run on cache hit"
    _assert_meshes_equal(first, second)


def test_lod_change_reparses(fake_extractor):
    src, calls = fake_extractor
    _parse_single_file(src, "building", None, None, building_lod=1)
    _parse_single_file(src, "building", None, None, building_lod=2)
    assert calls["n"] == 2, "different building_lod must not share a cache key"


def test_use_cache_false_bypasses(fake_extractor):
    src, calls = fake_extractor
    _parse_single_file(src, "building", None, None, building_lod=2,
                       use_cache=False)
    _parse_single_file(src, "building", None, None, building_lod=2,
                       use_cache=False)
    assert calls["n"] == 2
    assert not _cache_path(src, "building", 2).exists()


def test_corrupt_gml_still_reports_failure(fake_extractor, monkeypatch):
    """The Task-3 (parse-failures) contract must survive the restructure."""
    src, _ = fake_extractor

    def boom(root, ns, prefer_lod=None, max_lod=4):
        raise ValueError("broken file")

    monkeypatch.setattr(parser_mod, "extract_buildings_from_root", boom)
    failures = []
    result = _parse_single_file(src, "building", None, None, building_lod=3,
                                failures=failures)
    assert result == []
    assert len(failures) == 1 and "broken file" in failures[0]


def test_empty_extraction_is_cached_not_reparsed(fake_extractor, monkeypatch):
    """`[]` is a real cached value; only None means miss (cheapest tiles!)."""
    src, calls = fake_extractor

    def empty(root, ns, prefer_lod=None, max_lod=4):
        calls["n"] += 1
        return []

    monkeypatch.setattr(parser_mod, "extract_buildings_from_root", empty)
    assert _parse_single_file(src, "building", None, None, building_lod=2) == []
    assert _parse_single_file(src, "building", None, None, building_lod=2) == []
    assert calls["n"] == 1, "an empty result must be a cache hit, not a re-parse"


def test_rectangle_filter_applied_after_cache_hit(fake_extractor):
    """Cached meshes are unfiltered: a later, different rectangle must apply."""
    from shapely.geometry import Polygon as _Poly

    src, calls = fake_extractor
    # A name without a leading mesh code, so the filename pre-filter (which
    # would short-circuit before extraction) always passes.
    plain = src.parent / "trees.gml"
    plain.write_text("<CityModel/>", encoding="utf-8")

    unfiltered = _parse_single_file(plain, "building", None, None,
                                    building_lod=2)
    assert len(unfiltered) == 1

    # Rectangle far from the mesh (_mesh vertices are in [0, 1)).
    far = _Poly([(10, 10), (10, 11), (11, 11), (11, 10)])
    assert _parse_single_file(plain, "building", far, None, building_lod=2) == []
    # Rectangle covering the mesh -> the cached mesh comes back.
    near = _Poly([(-1, -1), (-1, 2), (2, 2), (2, -1)])
    assert len(_parse_single_file(plain, "building", near, None,
                                  building_lod=2)) == 1
    assert calls["n"] == 1, "filtering must not re-parse"


def test_unknown_feature_type_is_rejected_before_caching(fake_extractor):
    """An unknown type must not parse, and must not poison the cache with []."""
    src, calls = fake_extractor
    assert _parse_single_file(src, "sculpture", None, None) == []
    assert calls["n"] == 0, "must not parse an unsupported feature type"
    assert not _cache_path(src, "sculpture", None).exists()


# ---------------------------------------------------------------------------
# Projected-CRS (reprojection) path -- no other test exercises it, since
# PLATEAU is EPSG:6697 which resolves to None (geographic, never reprojected).
# ---------------------------------------------------------------------------

@pytest.fixture
def utm_extractor(monkeypatch, tmp_path):
    """Extractor returning UTM-32N metres (Munich), as a European dataset would."""
    src = tmp_path / "munich_tile.gml"
    src.write_text("<CityModel/>", encoding="utf-8")
    calls = {"n": 0}
    utm = np.array([[691000.0, 5334000.0, 10.0],
                    [691010.0, 5334010.0, 12.0],
                    [691020.0, 5334000.0, 11.0]])

    def fake_extract(root, ns, prefer_lod=None, max_lod=4):
        calls["n"] += 1
        return [_mesh(0, vertices=utm.copy(),
                      faces=np.array([[0, 1, 2]], dtype=np.int32))]

    monkeypatch.setattr(parser_mod, "extract_buildings_from_root", fake_extract)
    return src, calls


def test_reprojection_happens_before_caching(utm_extractor):
    """Cached vertices must already be WGS84 (lat, lon, z), not raw UTM."""
    src, calls = utm_extractor
    first = _parse_single_file(src, "building", None, None, building_lod=2,
                               source_epsg="EPSG:25832")
    second = _parse_single_file(src, "building", None, None, building_lod=2,
                                source_epsg="EPSG:25832")
    assert calls["n"] == 1, "second call must be served from the cache"
    # Munich is ~48.1 N, 11.6 E -- proves the cached array is reprojected.
    lat, lon = second[0].vertices[0][:2]
    assert 47.0 < lat < 49.0, f"latitude not reprojected: {lat}"
    assert 10.0 < lon < 13.0, f"longitude not reprojected: {lon}"
    _assert_meshes_equal(first, second)


def test_source_epsg_change_reparses(utm_extractor):
    """Dataset CRS detection can change without the GML file changing."""
    src, calls = utm_extractor
    a = _parse_single_file(src, "building", None, None, building_lod=2,
                           source_epsg="EPSG:25832")
    b = _parse_single_file(src, "building", None, None, building_lod=2,
                           source_epsg="EPSG:32633")
    assert calls["n"] == 2, "a different source_epsg must not reuse the entry"
    # Different CRS -> genuinely different coordinates, not a stale replay.
    assert a[0].vertices[0][1] != b[0].vertices[0][1]
