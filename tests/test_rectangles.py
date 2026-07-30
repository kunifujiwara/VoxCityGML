"""Tests for rectangle resolution (center+size vs explicit vertices)."""
import pytest
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


def test_missing_inputs_raise():
    cfg = VoxelizerConfig()  # neither vertices nor a real center/size
    cfg.center_lon = 0.0
    cfg.center_lat = 0.0
    cfg.rectangle_vertices = None
    # center (0,0) with default size is technically valid; only None-vertices
    # plus zero size must fail
    cfg.size_meters = 0
    with pytest.raises(ValueError):
        resolve_rectangles(cfg)
