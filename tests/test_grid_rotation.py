"""Affine GridParams: rotated rectangles map correctly; axis-aligned reduces
to the historical bbox formulas."""
import math

import numpy as np
import pytest

from voxcitygml.grid_utils import compute_grid_params, GridParams
from voxcitygml.citygml.coordinates import create_rectangle


def _geodesic_rect(center_lon, center_lat, width_m, height_m, rotation_deg):
    """Rectangle built exactly as the app's /api/rectangle-from-dimensions
    does: each corner placed by ``Geod.fwd`` at the bearing/distance of its
    rotated local-frame offset.  This is the production construction, and
    unlike ``_rotate_rect`` it is *not* an exact parallelogram in degree
    space — even at rotation 0 it carries ~1e-4 relative skew."""
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    hw, hh = width_m / 2.0, height_m / 2.0
    a = -math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for x, y in [(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)]:
        rx = x * ca - y * sa
        ry = x * sa + y * ca
        lon, lat, _ = geod.fwd(center_lon, center_lat,
                               math.degrees(math.atan2(rx, ry)),
                               math.hypot(rx, ry))
        out.append((lon, lat))
    return out


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


# ---------------------------------------------------------------------
# Degenerate input must be rejected at construction, not become NaN
# ---------------------------------------------------------------------

def test_zero_area_rectangle_raises():
    """All four vertices coincident: both side vectors have zero length."""
    rect = [(139.77, 35.65)] * 4
    with pytest.raises(ValueError, match="[Dd]egenerate"):
        compute_grid_params(rect, 5.0)


def test_collinear_rectangle_raises():
    """Four distinct but collinear vertices: the 2x2 basis is singular.

    Without the guard this silently produced row=nan, col=nan (the
    division by ``det`` warns but does not raise), which would propagate
    straight into the DEM / building / canopy rasterizers.
    """
    rect = [(139.77, 35.65), (139.77, 35.66),
            (139.77, 35.67), (139.77, 35.68)]
    with pytest.raises(ValueError, match="[Cc]ollinear"):
        compute_grid_params(rect, 5.0)


def test_degenerate_guard_does_not_fire_on_real_rectangles():
    """The guard is scale-free and must never reject legitimate input."""
    for rot in (0.0, 15.0, 30.0, 45.0, 90.0, 137.0):
        for w, h in ((900, 900), (1500, 600), (4000, 4000)):
            gp = compute_grid_params(
                _geodesic_rect(139.77, 35.65, w, h, rot), 5.0)
            assert gp.n_rows > 0 and gp.n_cols > 0


# ---------------------------------------------------------------------
# The six affine fields are required and immutable
# ---------------------------------------------------------------------

def test_affine_fields_are_required():
    """Omitting the basis must fail loudly, not silently give det == 0."""
    with pytest.raises(TypeError):
        GridParams(
            n_rows=10, n_cols=10,
            min_lon=0.0, max_lon=1.0, min_lat=0.0, max_lat=1.0,
            pixel_width=0.1, pixel_height=0.1,
        )


def test_grid_params_is_frozen():
    gp = compute_grid_params(create_rectangle(139.77, 35.65, 500), 5.0)
    with pytest.raises(Exception):       # dataclasses.FrozenInstanceError
        gp.e_col_lon = 0.0


# ---------------------------------------------------------------------
# Production-shaped input: geodesically built rectangles are not exact
# parallelograms in degree space, even at nominal rotation 0
# ---------------------------------------------------------------------

def test_geodesic_rectangle_round_trip_and_corners():
    """Closest case to production: the affine frame must round-trip and
    pin NW/SW/NE exactly despite the rectangle's inherent ~1e-4 skew."""
    rect = _geodesic_rect(139.77, 35.65, 900.0, 900.0, 0.0)
    sw, nw, ne, se = rect
    gp = compute_grid_params(rect, 5.0)

    rows = np.array([0.0, 12.5, gp.n_rows - 1.0])
    cols = np.array([0.0, 33.25, gp.n_cols - 1.0])
    lon, lat = gp.rowcol_to_lonlat(rows, cols)
    r2, c2 = gp.lonlat_to_rowcol(lon, lat)
    np.testing.assert_allclose(r2, rows, atol=1e-8)
    np.testing.assert_allclose(c2, cols, atol=1e-8)

    # The three vertices that *define* the frame land exactly.
    for vertex, exp in ((nw, (-0.5, -0.5)),
                        (sw, (gp.n_rows - 0.5, -0.5)),
                        (ne, (-0.5, gp.n_cols - 0.5))):
        r, c = gp.lonlat_to_rowcol(np.array([vertex[0]]),
                                   np.array([vertex[1]]))
        assert abs(r[0] - exp[0]) < 1e-9
        assert abs(c[0] - exp[1]) < 1e-9

    # SE is *implied* by the frame, so it carries the geodesic-vs-affine
    # residual.  Sub-cell, but not zero — pin the magnitude so a future
    # change that inflates it is noticed.
    r, c = gp.lonlat_to_rowcol(np.array([se[0]]), np.array([se[1]]))
    assert abs(r[0] - (gp.n_rows - 0.5)) < 0.1
    assert abs(c[0] - (gp.n_cols - 0.5)) < 0.1


@pytest.mark.parametrize("rotation", [0.0, 15.0, 30.0, 45.0, 90.0])
def test_geodesic_rectangle_grid_size_matches_voxcity(rotation):
    from voxcity.geoprocessor.raster.core import compute_grid_geometry
    rect = _geodesic_rect(139.77, 35.65, 1500.0, 600.0, rotation)
    gp = compute_grid_params(rect, 5.0)
    geom = compute_grid_geometry(rect, 5.0)
    assert (gp.n_rows, gp.n_cols) == tuple(geom["grid_size"])


# ---------------------------------------------------------------------
# Building rasterizer orientation, on a NON-SQUARE grid
#
# create_rectangle only ever builds squares, so every integration test
# has n_rows == n_cols and a row/col transpose in the building rasterizer
# would go unnoticed.  These two pin the orientation.
# ---------------------------------------------------------------------

def _synthetic_gp(n_rows, n_cols):
    """Axis-aligned frame with 1 degree cells, NW origin at (0, 0).

    Cell (r, c) centre is exactly (lon, lat) = (c + 0.5, -(r + 0.5)),
    which makes the expected rasterisation hand-checkable.
    """
    return GridParams(
        n_rows=n_rows, n_cols=n_cols,
        min_lon=0.0, max_lon=float(n_cols),
        min_lat=float(-n_rows), max_lat=0.0,
        pixel_width=1.0, pixel_height=1.0,
        origin_lon=0.0, origin_lat=0.0,
        e_col_lon=1.0, e_col_lat=0.0,
        e_row_lon=0.0, e_row_lat=-1.0,
    )


def test_rasterise_triangle_non_square_grid_orientation():
    """One triangle on a 3x7 grid, hand-computed cell set.

    The expected set is asymmetric under transpose (and the transposed
    rows would be out of range on a 3-row grid), so a row/col swap in
    ``_rasterise_triangle_to_cells`` cannot pass this.
    """
    from voxcitygml.buildings.processor import _rasterise_triangle_to_cells

    gp = _synthetic_gp(n_rows=3, n_cols=7)
    assert gp.n_rows != gp.n_cols

    # Right triangle: lon >= 4, lat <= 0, (lon - 4) + (-lat) <= 3.5
    v0 = np.array([4.0, 0.0, 10.0])
    v1 = np.array([7.5, 0.0, 10.0])
    v2 = np.array([4.0, -3.5, 10.0])

    cells = _rasterise_triangle_to_cells(v0, v1, v2, gp)

    expected = {(0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (2, 4)}
    assert set(cells) == expected
    for zlo, zhi in cells.values():
        assert zlo == pytest.approx(10.0)
        assert zhi == pytest.approx(10.0)


def test_meshes_to_building_grids_non_square_orientation():
    """End-to-end through the public rasteriser on a 1500 x 600 m box.

    Places one small triangle over a known cell far off the diagonal;
    a transposed index would clip to the wrong end of the grid.
    """
    from voxcitygml.models import Mesh3D
    from voxcitygml.buildings.processor import meshes_to_building_grids

    rect = _geodesic_rect(139.77, 35.65, 1500.0, 600.0, 0.0)
    gp = compute_grid_params(rect, 5.0)
    assert gp.n_rows != gp.n_cols

    target_row, target_col = 10, 250
    assert target_row < gp.n_rows and target_col < gp.n_cols
    # Deliberately off the diagonal and out of range if transposed.
    assert target_col >= gp.n_rows

    lon0, lat0 = gp.cell_centre(target_row, target_col)
    k = 0.35
    offsets = [
        (k * gp.e_col_lon, k * gp.e_col_lat),
        (-k * gp.e_col_lon + k * gp.e_row_lon,
         -k * gp.e_col_lat + k * gp.e_row_lat),
        (-k * gp.e_col_lon - k * gp.e_row_lon,
         -k * gp.e_col_lat - k * gp.e_row_lat),
    ]
    # Mesh vertices are stored (lat, lon, z)
    verts = np.array([[lat0 + dlat, lon0 + dlon, 25.0]
                      for dlon, dlat in offsets], dtype=np.float64)
    mesh = Mesh3D(vertices=verts, faces=np.array([[0, 1, 2]], dtype=np.int32))

    dem = np.zeros(gp.shape, dtype=np.float64)
    height_grid, _min_h, id_grid = meshes_to_building_grids(
        [mesh], [], rect, 5.0, dem,
    )

    assert height_grid.shape == gp.shape
    nz = np.argwhere(height_grid > 0)
    assert len(nz) > 0, "triangle produced no cells"
    assert nz[:, 0].min() >= target_row - 2
    assert nz[:, 0].max() <= target_row + 2
    assert nz[:, 1].min() >= target_col - 2
    assert nz[:, 1].max() <= target_col + 2
    assert height_grid[target_row, target_col] == pytest.approx(25.0)
    assert id_grid[target_row, target_col] == 1
