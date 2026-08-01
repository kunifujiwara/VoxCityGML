"""
Shared grid-sizing utilities compatible with voxcity.

Cell-count sizing (rows/cols) is delegated to
``voxcity.geoprocessor.raster.core.compute_grid_geometry`` so the two
codebases can never drift apart. That function computes:

1.  Geodesic distances (WGS-84 ellipsoid) along N-S and E-W sides.
2.  ``int(distance / meshsize + 0.5)`` rounding for cell count.

On top of the delegated ``n_rows``/``n_cols``, this module builds an
**affine frame** that maps (lon, lat) ↔ (row, col).  The frame is
anchored at the NW vertex and its basis vectors run along the
rectangle's own sides::

    e_col = (NE − NW) / n_cols      # one column step
    e_row = (SW − NW) / n_rows      # one row step

    lon(row, col) = NW.lon + (col + 0.5) * e_col.lon + (row + 0.5) * e_row.lon
    lat(row, col) = NW.lat + (col + 0.5) * e_col.lat + (row + 0.5) * e_row.lat

This works for **rotated** rectangles.  For an axis-aligned rectangle it
reduces algebraically to the historical bbox arithmetic, because then
``e_col = (pixel_width, 0)``, ``e_row = (0, −pixel_height)`` and
``NW = (min_lon, max_lat)``::

    lon(col) = min_lon + (col + 0.5) * pixel_width
    lat(row) = max_lat − (row + 0.5) * pixel_height

The bbox / pixel fields are still populated (``pixel_width =
(max_lon − min_lon) / n_cols`` etc.) but they are **metadata only** —
they describe the axis-aligned bounding box of the rectangle and must
not be used for coordinate mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from voxcity.geoprocessor.raster.core import compute_grid_geometry


# -----------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class GridParams:
    """Affine frame mapping (lon, lat) ↔ grid (row, col).

    The frame is anchored at the NW vertex (the (row 0, col 0) *corner*
    under the north-up convention) with basis vectors along the
    rectangle's sides::

        e_col = (NE − NW) / n_cols      e_row = (SW − NW) / n_rows

    Rows therefore run NW→SW and columns run NW→NE, whatever the
    rectangle's rotation.  For an axis-aligned rectangle this reduces
    exactly to the historical bbox arithmetic.  The min/max/pixel fields
    are retained as metadata (bounding box of the rectangle) but are no
    longer used for mapping.

    Note that the frame is defined by **three** vertices (NW, NE, SW);
    SE is implied to be ``NW + n_cols·e_col + n_rows·e_row``.  A
    geodesically-constructed rectangle is not an exact parallelogram in
    degree space, so its true SE corner sits slightly off that point
    (measured: ~0.02 cells for a 900 m box, ~0.36 cells for a 4 km box
    at 5 m resolution).  That is the inherent cost of modelling a
    geodesic quadrilateral with an affine frame, not a defect of the
    inversion.

    All fields are required and the dataclass is frozen: the six affine
    values are meaningless unless assigned together and consistently, so
    partial or mutated construction is rejected rather than silently
    producing a singular frame.
    """
    n_rows: int                  # number of rows (along SW→NW side)
    n_cols: int                  # number of columns (along SW→SE side)
    min_lon: float               # bbox: western extent
    max_lon: float               # bbox: eastern extent
    min_lat: float               # bbox: southern extent
    max_lat: float               # bbox: northern extent
    pixel_width: float           # bbox width / n_cols  (metadata)
    pixel_height: float          # bbox height / n_rows (metadata)
    origin_lon: float            # NW vertex
    origin_lat: float
    e_col_lon: float             # lon/lat step per column (NW→NE / n_cols)
    e_col_lat: float
    e_row_lon: float             # lon/lat step per row    (NW→SW / n_rows)
    e_row_lat: float

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)

    # ------ forward mapping: (row, col) → (lon, lat) ------
    def rowcol_to_lonlat(self, row, col):
        """Vectorized cell-centre (lon, lat) for float (row, col)."""
        row = np.asarray(row, dtype=np.float64)
        col = np.asarray(col, dtype=np.float64)
        lon = (self.origin_lon + (col + 0.5) * self.e_col_lon
               + (row + 0.5) * self.e_row_lon)
        lat = (self.origin_lat + (col + 0.5) * self.e_col_lat
               + (row + 0.5) * self.e_row_lat)
        return lon, lat

    def cell_centre(self, row: int, col: int) -> Tuple[float, float]:
        """Return (lon, lat) of a single cell centre."""
        lon, lat = self.rowcol_to_lonlat(row, col)
        return float(lon), float(lat)

    def cell_centres(
        self,
        rows: Optional[np.ndarray] = None,
        cols: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(PX, PY) lon/lat arrays of cell centres.

        Covers the full grid by default, or the meshgrid of the given
        row/col index arrays.  Result shape is ``(len(rows), len(cols))``.
        """
        if rows is None:
            rows = np.arange(self.n_rows)
        if cols is None:
            cols = np.arange(self.n_cols)
        rr = (np.asarray(rows, dtype=np.float64) + 0.5)[:, None]
        cc = (np.asarray(cols, dtype=np.float64) + 0.5)[None, :]
        PX = self.origin_lon + cc * self.e_col_lon + rr * self.e_row_lon
        PY = self.origin_lat + cc * self.e_col_lat + rr * self.e_row_lat
        return PX, PY

    # ------ inverse mapping: (lon, lat) → (row, col) ------
    def lonlat_to_rowcol(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map (lon, lat) to floating-point (row, col).

        Inverts the affine frame with Cramer's rule on the 2×2 basis
        ``[e_col, e_row]``.
        """
        det = self.e_col_lon * self.e_row_lat - self.e_col_lat * self.e_row_lon
        dx = np.asarray(lon, dtype=np.float64) - self.origin_lon
        dy = np.asarray(lat, dtype=np.float64) - self.origin_lat
        col = (dx * self.e_row_lat - dy * self.e_row_lon) / det - 0.5
        row = (-dx * self.e_col_lat + dy * self.e_col_lon) / det - 0.5
        return row, col

    def lonlat_to_rowcol_int(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map (lon, lat) → nearest integer (row, col), clipped to grid."""
        row_f, col_f = self.lonlat_to_rowcol(lon, lat)
        row = np.clip(np.round(row_f).astype(np.intp), 0, self.n_rows - 1)
        col = np.clip(np.round(col_f).astype(np.intp), 0, self.n_cols - 1)
        return row, col


# -----------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------

# |sin θ| between the two side directions must exceed this.  The check is
# a *ratio*, so it is scale-free (independent of rectangle size, meshsize
# and latitude).  For reference, real geodesically-built rectangles measure
# |sin θ| ≥ 0.979 even at 45° rotation, so this leaves ~9 orders of
# magnitude of headroom and only fires on genuinely collinear input.
_MIN_SIDE_SIN = 1e-9


def check_non_degenerate(sw, nw, ne) -> None:
    """Reject rectangles whose affine frame would be singular.

    The mapping inverts the 2×2 basis built from the side vectors NW→NE
    and NW→SW.  If either side has zero length, or the two sides are
    collinear, that basis is singular and every subsequent lon/lat ↔
    row/col conversion silently yields ``inf``/``nan`` — which then
    propagates into the DEM, building and canopy rasterizers.  Nothing
    upstream rejects such input (``resolve_rectangles`` only validates
    vertex count/shape, and voxcity's ``compute_grid_geometry`` floors
    its dimensions with ``max(1, …)``), so it is caught here, at
    construction, rather than as a downstream NaN.
    """
    col_lon, col_lat = ne[0] - nw[0], ne[1] - nw[1]   # NW→NE
    row_lon, row_lat = sw[0] - nw[0], sw[1] - nw[1]   # NW→SW

    len_col = math.hypot(col_lon, col_lat)
    len_row = math.hypot(row_lon, row_lat)
    if len_col == 0.0 or len_row == 0.0:
        raise ValueError(
            "Degenerate rectangle: zero-length side "
            f"(|NW→NE| = {len_col}, |NW→SW| = {len_row}). "
            "The rectangle has zero area — check that the four vertices "
            "are distinct and in [SW, NW, NE, SE] order."
        )

    cross = col_lon * row_lat - col_lat * row_lon
    sin_theta = abs(cross) / (len_col * len_row)
    if sin_theta <= _MIN_SIDE_SIN:
        raise ValueError(
            "Degenerate rectangle: adjacent sides are collinear "
            f"(|sin θ| = {sin_theta:.3e} ≤ {_MIN_SIDE_SIN:.0e}), so the "
            "grid frame is singular and would map every point to NaN. "
            "Check that the four vertices form a real quadrilateral in "
            "[SW, NW, NE, SE] order."
        )


def compute_grid_params(
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
) -> GridParams:
    """Compute grid parameters **exactly matching voxcity's algorithm**.

    Cell-count sizing (``n_rows``/``n_cols``) is delegated to
    ``voxcity.geoprocessor.raster.core.compute_grid_geometry`` so this
    module can never drift from voxcity's own grid sizing.

    The returned frame is affine (NW origin + side-vector basis), so
    rotated rectangles are handled correctly.

    Parameters
    ----------
    rectangle_vertices : [(lon, lat), …]
        Rectangle corners in VoxCity order: [SW, NW, NE, SE].
    meshsize : float
        Target cell size in metres.

    Returns
    -------
    GridParams
    """
    # SE is not used: the affine frame is defined by NW (origin) plus the
    # two side vectors NW→NE and NW→SW.  See the GridParams docstring.
    sw, nw, ne, _se = [tuple(v[:2]) for v in rectangle_vertices]
    check_non_degenerate(sw, nw, ne)

    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    if geom is None:
        raise ValueError(
            "compute_grid_geometry returned None for the given "
            "rectangle_vertices/meshsize (insufficient inputs)"
        )
    # grid_size[0] is along side_1 (SW→NW → rows),
    # grid_size[1] is along side_2 (SW→SE → cols).
    n_rows, n_cols = geom["grid_size"]

    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    pixel_width  = (max_lon - min_lon) / n_cols
    pixel_height = (max_lat - min_lat) / n_rows

    return GridParams(
        n_rows=n_rows,
        n_cols=n_cols,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        origin_lon=nw[0],
        origin_lat=nw[1],
        e_col_lon=(ne[0] - nw[0]) / n_cols,
        e_col_lat=(ne[1] - nw[1]) / n_cols,
        e_row_lon=(sw[0] - nw[0]) / n_rows,
        e_row_lat=(sw[1] - nw[1]) / n_rows,
    )
