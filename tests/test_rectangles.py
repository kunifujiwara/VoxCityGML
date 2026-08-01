"""Tests for rectangle resolution (center+size vs explicit vertices)."""
import pytest
from pyproj import Geod
from shapely.geometry import Polygon

from voxcitygml.models import VoxelizerConfig, resolve_rectangles
from voxcitygml.citygml.coordinates import create_rectangle


def test_center_size_matches_create_rectangle():
    cfg = VoxelizerConfig(center_lon=139.77, center_lat=35.65,
                          size_meters=500, buffer_meters=50)
    rect, buffered, center_lon, center_lat = resolve_rectangles(cfg)
    assert rect == create_rectangle(139.77, 35.65, 500)
    assert buffered == create_rectangle(139.77, 35.65, 600)
    assert center_lon == pytest.approx(139.77)
    assert center_lat == pytest.approx(35.65)


def test_explicit_vertices_used_verbatim():
    verts = create_rectangle(139.77, 35.65, 400)
    cfg = VoxelizerConfig(rectangle_vertices=verts, buffer_meters=50)
    rect, buffered, center_lon, center_lat = resolve_rectangles(cfg)
    assert rect == [tuple(v) for v in verts]
    # centre is the vertex centroid
    assert center_lon == pytest.approx(sum(v[0] for v in verts) / 4)
    assert center_lat == pytest.approx(sum(v[1] for v in verts) / 4)


def test_buffered_rectangle_contains_original():
    verts = create_rectangle(139.77, 35.65, 400)
    cfg = VoxelizerConfig(rectangle_vertices=verts, buffer_meters=50)
    rect, buffered, _, _ = resolve_rectangles(cfg)
    assert Polygon(buffered).buffer(1e-9).contains(Polygon(rect))


def test_buffered_rectangle_side_length():
    # A 400 m rect buffered by 50 m on every side must measure 500 m on a
    # side (the containment test alone would also pass for any factor >= 1,
    # so pin down the actual magnitude here).
    verts = create_rectangle(139.77, 35.65, 400)
    cfg = VoxelizerConfig(rectangle_vertices=verts, buffer_meters=50)
    _, buffered, _, _ = resolve_rectangles(cfg)
    geod = Geod(ellps="WGS84")
    sw, nw, ne, se = buffered
    _, _, d_ns = geod.inv(sw[0], sw[1], nw[0], nw[1])
    _, _, d_ew = geod.inv(sw[0], sw[1], se[0], se[1])
    assert d_ns == pytest.approx(500.0, abs=0.1)
    assert d_ew == pytest.approx(500.0, abs=0.1)


def test_missing_inputs_raise():
    cfg = VoxelizerConfig(size_meters=0)
    with pytest.raises(ValueError):
        resolve_rectangles(cfg)


def test_negative_buffer_raises():
    verts = create_rectangle(139.77, 35.65, 400)
    cfg = VoxelizerConfig(rectangle_vertices=verts, buffer_meters=-1)
    with pytest.raises(ValueError):
        resolve_rectangles(cfg)


def test_vertex_with_extra_component_normalized():
    verts = create_rectangle(139.77, 35.65, 400)
    verts_3d = [(lon, lat, 0.0) for lon, lat in verts]
    cfg = VoxelizerConfig(rectangle_vertices=verts_3d, buffer_meters=50)
    rect, buffered, _, _ = resolve_rectangles(cfg)
    assert rect == [tuple(v) for v in verts]
    assert all(len(v) == 2 for v in rect)
    assert all(len(v) == 2 for v in buffered)


def test_vertex_with_too_few_components_raises():
    cfg = VoxelizerConfig(rectangle_vertices=[(1.0,), (2.0, 3.0),
                                              (4.0, 5.0), (6.0, 7.0)],
                          buffer_meters=50)
    with pytest.raises(ValueError):
        resolve_rectangles(cfg)
