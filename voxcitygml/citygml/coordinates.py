"""
Coordinate handling and transformations.
"""

import logging
import math
import re
from typing import List, Tuple, Optional, Union
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box

from ..models import Mesh3D

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformer helpers
# ---------------------------------------------------------------------------

def setup_transformer(from_crs: Union[str, int], to_crs: Union[str, int]) -> Transformer:
    """Set up a coordinate transformer between two CRS."""
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)


def swap_coordinates_3d(coords: np.ndarray) -> np.ndarray:
    """Swap lat/lon to lon/lat for 3D coordinates (PLATEAU stores lat, lon, z)."""
    swapped = coords.copy()
    swapped[:, 0], swapped[:, 1] = coords[:, 1].copy(), coords[:, 0].copy()
    return swapped


# ---------------------------------------------------------------------------
# CRS reprojection (projected → WGS84 lat/lon)
# ---------------------------------------------------------------------------

_REPROJECT_CACHE: dict = {}  # source_epsg → Transformer


def _get_reprojector(source_epsg: str) -> Transformer:
    """Return (and cache) a Transformer from *source_epsg* to WGS84."""
    if source_epsg not in _REPROJECT_CACHE:
        _REPROJECT_CACHE[source_epsg] = Transformer.from_crs(
            source_epsg, 'EPSG:4326', always_xy=True,
        )
    return _REPROJECT_CACHE[source_epsg]


def reproject_vertices(vertices: np.ndarray,
                       source_epsg: str) -> np.ndarray:
    """Reproject (X, Y, Z) vertices from *source_epsg* to WGS84 **(lat, lon, z)**.

    The returned array has the same shape as the input, but columns 0/1
    are (latitude, longitude) — matching the PLATEAU CityGML convention
    so that all downstream code (swap_coordinates_3d, etc.) works unchanged.

    Parameters
    ----------
    vertices : (N, 3) float64  – (Easting, Northing, Z) in *source_epsg*.
    source_epsg : str            – e.g. ``'EPSG:25832'``.

    Returns
    -------
    (N, 3) float64  – (lat, lon, Z).
    """
    if len(vertices) == 0:
        return vertices
    t = _get_reprojector(source_epsg)
    lon, lat = t.transform(vertices[:, 0], vertices[:, 1])
    return np.column_stack([lat, lon, vertices[:, 2]])


# ---------------------------------------------------------------------------
# Rectangle creation (VoxCity-compatible vertex order)
# ---------------------------------------------------------------------------

def create_rectangle(center_lon: float, center_lat: float,
                     size_meters: float) -> List[Tuple[float, float]]:
    """Create rectangle vertices [SW, NW, NE, SE] from centre and size.

    Uses geodesic calculations (WGS-84 ellipsoid) so that the rectangle
    measures exactly *size_meters* × *size_meters* when later checked
    by ``compute_grid_params`` (which also uses ``pyproj.Geod``).

    Following voxcity's ``center_location_map_cityname``, the E-W extent
    is computed at the **south** latitude of the rectangle.  This ensures
    that the geodesic SW→SE distance equals *size_meters*, producing a
    square grid (e.g. 800 × 800 for 4 km at 5 m resolution).
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    # N-S offsets (bearing 0 = north, 180 = south)
    _, south_lat, _ = geod.fwd(center_lon, center_lat, 180, size_meters / 2)
    _, north_lat, _ = geod.fwd(center_lon, center_lat, 0, size_meters / 2)

    # E-W offsets computed at the south latitude (matches grid_utils SW→SE)
    west_lon, _, _ = geod.fwd(center_lon, south_lat, 270, size_meters / 2)
    east_lon, _, _ = geod.fwd(center_lon, south_lat, 90, size_meters / 2)

    return [
        (west_lon, south_lat),    # SW
        (west_lon, north_lat),    # NW
        (east_lon, north_lat),    # NE
        (east_lon, south_lat),    # SE
    ]


def rectangle_to_shapely(rectangle_vertices: List[Tuple[float, float]]) -> ShapelyPolygon:
    """Convert rectangle vertices to a Shapely polygon."""
    coords = list(rectangle_vertices) + [rectangle_vertices[0]]
    return ShapelyPolygon(coords)


# ---------------------------------------------------------------------------
# Mesh-code / file filtering
# ---------------------------------------------------------------------------

def decode_mesh_code(mesh_str: str) -> Optional[Tuple[float, float, float, float]]:
    """Decode Japanese mesh code → (lat_sw, lon_sw, lat_ne, lon_ne) or None."""
    if len(mesh_str) < 6:
        return None
    try:
        mesh6 = mesh_str[:6]
        code = int(mesh6)
        N1 = code // 10000
        M1 = (code // 100) % 100
        row_2nd = (code // 10) % 10
        col_2nd = code % 10

        lat_sw_1 = (N1 * 40.0) / 60.0
        lon_sw_1 = 100.0 + M1

        dlat_2nd = (40.0 / 60.0) / 8.0
        dlon_2nd = 1.0 / 8.0

        lat_sw = lat_sw_1 + row_2nd * dlat_2nd
        lon_sw = lon_sw_1 + col_2nd * dlon_2nd
        lat_ne = lat_sw + dlat_2nd
        lon_ne = lon_sw + dlon_2nd

        if len(mesh_str) >= 8:
            row_10 = int(mesh_str[6])
            col_10 = int(mesh_str[7])
            dlat_10 = dlat_2nd / 10.0
            dlon_10 = dlon_2nd / 10.0
            lat_sw = lat_sw + row_10 * dlat_10
            lon_sw = lon_sw + col_10 * dlon_10
            lat_ne = lat_sw + dlat_10
            lon_ne = lon_sw + dlon_10

        return (lat_sw, lon_sw, lat_ne, lon_ne)
    except Exception:
        return None


def get_mesh_code_from_filename(filename: str) -> Optional[str]:
    """Extract mesh code from PLATEAU filename."""
    match = re.match(r'^(\d{6,8})', filename)
    return match.group(1) if match else None


def file_intersects_rectangle(filename: str, rect_polygon: ShapelyPolygon) -> bool:
    """Check if a CityGML file's mesh area intersects the filter rectangle."""
    mesh_code = get_mesh_code_from_filename(filename)
    if not mesh_code:
        return True
    bounds = decode_mesh_code(mesh_code)
    if not bounds:
        return True
    lat_sw, lon_sw, lat_ne, lon_ne = bounds
    return rect_polygon.intersects(shapely_box(lon_sw, lat_sw, lon_ne, lat_ne))


def mesh_intersects_rectangle(mesh: Mesh3D,
                              rect_polygon: ShapelyPolygon,
                              prepared_rect=None) -> bool:
    """Check if a mesh's 2D bbox intersects the rectangle.

    Vertices are expected as (lat, lon, z) — standard PLATEAU order.
    After CRS reprojection, non-PLATEAU data is also stored this way.
    """
    if len(mesh.vertices) == 0:
        return False
    # Vertices are (lat, lon, z)
    min_lat = mesh.vertices[:, 0].min()
    max_lat = mesh.vertices[:, 0].max()
    min_lon = mesh.vertices[:, 1].min()
    max_lon = mesh.vertices[:, 1].max()
    mesh_bbox = shapely_box(min_lon, min_lat, max_lon, max_lat)
    if prepared_rect is not None:
        return prepared_rect.intersects(mesh_bbox)
    return rect_polygon.intersects(mesh_bbox)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def transform_to_local_meters(vertices: np.ndarray,
                              center_lon: float,
                              center_lat: float,
                              transformer: Transformer = None) -> np.ndarray:
    """Transform WGS84 (lon, lat, z) to local metre coordinates."""
    if transformer is None:
        proj_string = (
            f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
            "+k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
        )
        transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    x_m, y_m = transformer.transform(vertices[:, 0], vertices[:, 1])
    return np.column_stack([x_m, y_m, vertices[:, 2]])


def create_local_transformer(center_lon: float, center_lat: float) -> Transformer:
    """Create a reusable transformer for WGS84 → local metres."""
    proj_string = (
        f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
        "+k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)


class _RectangleFrameTransformer:
    """Wraps the local tmerc transformer with a rotation by −θ so the
    target rectangle is axis-aligned in the working frame.

    Only the forward ``transform(lons, lats)`` is provided — nothing in
    the package uses the inverse direction.
    """

    __slots__ = ("_base", "_cos", "_sin")

    def __init__(self, base: Transformer, cos_t: float, sin_t: float):
        self._base = base
        self._cos = cos_t
        self._sin = sin_t

    def transform(self, lons, lats):
        x, y = self._base.transform(lons, lats)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return (self._cos * x + self._sin * y,
                -self._sin * x + self._cos * y)


def create_rectangle_frame_transformer(center_lon: float, center_lat: float,
                                       rectangle_vertices):
    """Local metric transformer aligned to the rectangle's own axes.

    θ is the bearing of the **NW→NE** side in the tmerc frame; the returned
    transformer rotates by −θ, so a rotated rectangle becomes axis-aligned
    and every downstream bbox / row-col computation is valid unchanged.  For
    an axis-aligned rectangle θ ≈ 0 and this degenerates to (numerically) the
    plain local transformer.

    **NW→NE, not SW→SE, deliberately.**  The 2-D ``GridParams`` frame is
    anchored on NW with ``e_col`` along NW→NE and ``e_row`` along NW→SW.  A
    geodesic quadrilateral is not a true parallelogram, so NW→NE and SW→SE
    are not parallel — deriving θ from the other pair would make the 2-D and
    3-D frames exact on different corners and leave a sub-cell seam between
    the rasterized DEM / building grids and the voxel grid.  Sharing the
    vertex pair keeps them consistent by construction.

    Parameters
    ----------
    rectangle_vertices : sequence of 4 (lon, lat[, ...]) in [SW, NW, NE, SE]
        order — the package-wide convention (see ``create_rectangle``).
    """
    base = create_local_transformer(center_lon, center_lat)
    _sw, nw, ne, _se = [tuple(v[:2]) for v in rectangle_vertices]
    xs, ys = base.transform([nw[0], ne[0]], [nw[1], ne[1]])
    theta = math.atan2(ys[1] - ys[0], xs[1] - xs[0])
    return _RectangleFrameTransformer(base, math.cos(theta), math.sin(theta))


def transform_geographic_to_local_simple(vertices: np.ndarray,
                                         return_params: bool = False):
    """Simple geographic → local metres (approximate, fast)."""
    centroid = np.mean(vertices, axis=0)
    if centroid[0] > 90:
        lon_idx, lat_idx = 0, 1
    else:
        lat_idx, lon_idx = 0, 1
    lat_center = centroid[lat_idx]
    lon_to_m = np.cos(np.radians(lat_center)) * 111_320
    lat_to_m = 110_540

    local = np.zeros_like(vertices, dtype=np.float64)
    local[:, lon_idx] = (vertices[:, lon_idx] - centroid[lon_idx]) * lon_to_m
    local[:, lat_idx] = (vertices[:, lat_idx] - centroid[lat_idx]) * lat_to_m
    local[:, 2] = vertices[:, 2] - centroid[2]

    if return_params:
        return local, {
            'centroid': centroid, 'lat_idx': lat_idx, 'lon_idx': lon_idx,
            'lat_to_m': lat_to_m, 'lon_to_m': lon_to_m,
        }
    return local
