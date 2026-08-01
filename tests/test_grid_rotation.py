"""Affine GridParams: rotated rectangles map correctly; axis-aligned reduces
to the historical bbox formulas."""
import math

import numpy as np
import pytest

from voxcitygml.grid_utils import compute_grid_params
from voxcitygml.citygml.coordinates import create_rectangle


def _rotate_rect(center_lon, center_lat, half_w_deg, half_h_deg, theta_deg):
    """Build a rotated rectangle [SW, NW, NE, SE] in pure lon/lat plane
    (adequate for unit tests; degree-space rotation)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    out = []
    for dx, dy in [(-half_w_deg, -half_h_deg), (-half_w_deg, half_h_deg),
                   (half_w_deg, half_h_deg), (half_w_deg, -half_h_deg)]:
        out.append((center_lon + c * dx - s * dy,
                    center_lat + s * dx + c * dy))
    return out


def test_axis_aligned_reduces_to_bbox_formulas():
    rect = create_rectangle(139.77, 35.65, 500)
    gp = compute_grid_params(rect, 5.0)
    lons = np.array([139.7695, 139.7712, 139.77])
    lats = np.array([35.6489, 35.6478, 35.65])
    row, col = gp.lonlat_to_rowcol(lons, lats)
    # Historical formulas (bbox arithmetic)
    exp_col = (lons - gp.min_lon) / gp.pixel_width - 0.5
    exp_row = (gp.max_lat - lats) / gp.pixel_height - 0.5
    np.testing.assert_allclose(col, exp_col, rtol=0, atol=1e-6)
    np.testing.assert_allclose(row, exp_row, rtol=0, atol=1e-6)


def test_rowcol_lonlat_round_trip_rotated():
    rect = _rotate_rect(139.77, 35.65, 0.003, 0.002, 30.0)
    gp = compute_grid_params(rect, 5.0)
    rows = np.array([0.0, 3.2, gp.n_rows - 1.0])
    cols = np.array([0.0, 7.7, gp.n_cols - 1.0])
    lon, lat = gp.rowcol_to_lonlat(rows, cols)
    r2, c2 = gp.lonlat_to_rowcol(lon, lat)
    np.testing.assert_allclose(r2, rows, atol=1e-9)
    np.testing.assert_allclose(c2, cols, atol=1e-9)


def test_corners_map_to_grid_corners_rotated():
    rect = _rotate_rect(139.77, 35.65, 0.003, 0.002, 30.0)
    sw, nw, ne, se = rect
    gp = compute_grid_params(rect, 5.0)
    # NW is the (row 0, col 0) *corner*, i.e. cell-centre coords (-0.5, -0.5)
    r, c = gp.lonlat_to_rowcol(np.array([nw[0]]), np.array([nw[1]]))
    assert abs(r[0] + 0.5) < 1e-6 and abs(c[0] + 0.5) < 1e-6
    # SE is the far corner (n_rows-0.5, n_cols-0.5).  _rotate_rect builds an
    # exact parallelogram, so SE == NW + n_cols*e_col + n_rows*e_row and the
    # affine map lands on it to floating-point precision (measured ~1e-14).
    r, c = gp.lonlat_to_rowcol(np.array([se[0]]), np.array([se[1]]))
    assert abs(r[0] - (gp.n_rows - 0.5)) < 1e-9
    assert abs(c[0] - (gp.n_cols - 0.5)) < 1e-9


def test_cell_centres_match_scalar_method_rotated():
    rect = _rotate_rect(139.77, 35.65, 0.002, 0.002, 17.0)
    gp = compute_grid_params(rect, 10.0)
    PX, PY = gp.cell_centres()
    assert PX.shape == (gp.n_rows, gp.n_cols)
    lon, lat = gp.cell_centre(2, 3)
    assert abs(PX[2, 3] - lon) < 1e-12 and abs(PY[2, 3] - lat) < 1e-12


def test_grid_size_still_matches_voxcity_rotated():
    from voxcity.geoprocessor.raster.core import compute_grid_geometry
    rect = _rotate_rect(139.77, 35.65, 0.003, 0.002, 45.0)
    gp = compute_grid_params(rect, 5.0)
    geom = compute_grid_geometry(rect, 5.0)
    assert (gp.n_rows, gp.n_cols) == tuple(geom["grid_size"])


def test_90_degree_rotation_transposes_the_frame():
    """Orientation pin (spec): rotating the rectangle 90° swaps the roles of
    rows and columns — a fixed geographic point must land at the transposed
    cell. Catches row/col swaps and sign errors that magnitude checks miss."""
    base = _rotate_rect(139.77, 35.65, 0.002, 0.002, 0.0)      # square
    rot90 = _rotate_rect(139.77, 35.65, 0.002, 0.002, 90.0)
    gp0 = compute_grid_params(base, 10.0)
    gp90 = compute_grid_params(rot90, 10.0)
    assert (gp90.n_rows, gp90.n_cols) == (gp0.n_cols, gp0.n_rows)
    # Probe a fixed geographic point off-centre in both frames.
    lon, lat = np.array([139.7712]), np.array([35.6489])
    r0, c0 = gp0.lonlat_to_rowcol(lon, lat)
    r90, c90 = gp90.lonlat_to_rowcol(lon, lat)
    # 90° CCW rotation of the rectangle: new rows run along old cols.
    # new_row == old_col and new_col == (n_rows-1) - old_row (centre-symmetric).
    np.testing.assert_allclose(r90, c0, atol=1e-6)
    np.testing.assert_allclose(c90, (gp0.n_rows - 1) - r0, atol=1e-6)
