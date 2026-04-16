"""
Shared grid-sizing utilities compatible with voxcity.

All grid dimensions and pixel sizes are computed using the same algorithm
as ``voxcity.geoprocessor.raster.core.calculate_grid_size``:

1.  Geodesic distances (WGS-84 ellipsoid) along N-S and E-W sides.
2.  ``int(distance / meshsize + 0.5)`` rounding for cell count.
3.  ``pixel_width = (max_lon − min_lon) / n_cols``
    ``pixel_height = (max_lat − min_lat) / n_rows``
    so cells tile the rectangle exactly.

Cell centres (north-up, row 0 = north):
    lon(col) = min_lon + (col + 0.5) * pixel_width
    lat(row) = max_lat - (row + 0.5) * pixel_height
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pyproj import Geod


# -----------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------

@dataclass
class GridParams:
    """Everything needed to map between (lon, lat) and grid (row, col)."""
    n_rows: int                  # number of rows  (N-S direction)
    n_cols: int                  # number of columns (E-W direction)
    min_lon: float               # western edge of rectangle
    max_lon: float               # eastern edge
    min_lat: float               # southern edge
    max_lat: float               # northern edge
    pixel_width: float           # degrees per column
    pixel_height: float          # degrees per row

    # ------ derived convenience ------
    def cell_centre(self, row: int, col: int) -> Tuple[float, float]:
        """Return (lon, lat) of a cell centre."""
        lon = self.min_lon + (col + 0.5) * self.pixel_width
        lat = self.max_lat - (row + 0.5) * self.pixel_height
        return lon, lat

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)

    def lonlat_to_rowcol(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map arrays of (lon, lat) to floating-point (row, col)."""
        col = (lon - self.min_lon) / self.pixel_width - 0.5
        row = (self.max_lat - lat) / self.pixel_height - 0.5
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

def compute_grid_params(
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
) -> GridParams:
    """Compute grid parameters **exactly matching voxcity's algorithm**.

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
    sw, nw, ne, se = rectangle_vertices

    geod = Geod(ellps="WGS84")

    # N-S distance (side_1): SW → NW
    _, _, dist_ns = geod.inv(sw[0], sw[1], nw[0], nw[1])
    # E-W distance (side_2): SW → SE
    _, _, dist_ew = geod.inv(sw[0], sw[1], se[0], se[1])

    n_rows = max(1, int(dist_ns / meshsize + 0.5))   # same as voxcity
    n_cols = max(1, int(dist_ew / meshsize + 0.5))

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
    )
