"""Unit tests for the per-file parse cache (store/load/validation)."""
import os
import stat

import numpy as np
import pytest

from voxcitygml.models import Mesh3D
from voxcitygml.citygml.parse_cache import (
    load_cached_meshes, store_cached_meshes, _cache_path,
)


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


def test_building_lod_keys_separately(tmp_path):
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 1, [_mesh(1)])
    store_cached_meshes(src, "building", 2, [_mesh(2)])
    store_cached_meshes(src, "building", None, [_mesh(3)])
    assert load_cached_meshes(src, "building", 1)[0].feature_id == "bldg_1"
    assert load_cached_meshes(src, "building", 2)[0].feature_id == "bldg_2"
    assert load_cached_meshes(src, "building", None)[0].feature_id == "bldg_3"


def test_corrupt_cache_falls_back_to_none(tmp_path):
    src = _src(tmp_path)
    store_cached_meshes(src, "building", 2, [_mesh(0)])
    _cache_path(src, "building", 2).write_bytes(b"not an npz")
    assert load_cached_meshes(src, "building", 2) is None


def test_readonly_cache_dir_store_does_not_raise(tmp_path):
    src = _src(tmp_path)
    cache_dir = _cache_path(src, "building", 2).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, stat.S_IREAD | stat.S_IEXEC)
    try:
        store_cached_meshes(src, "building", 2, [_mesh(0)])  # must not raise
    finally:
        os.chmod(cache_dir, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)


def test_unserializable_attribute_does_not_raise(tmp_path):
    src = _src(tmp_path)
    meshes = [_mesh(0, attributes={"weird": object()})]
    store_cached_meshes(src, "building", 2, meshes)  # must not raise
    # json default=str stringifies object(); load must still work
    loaded = load_cached_meshes(src, "building", 2)
    if loaded is not None:  # stored with stringified attribute
        assert isinstance(loaded[0].attributes["weird"], str)


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
