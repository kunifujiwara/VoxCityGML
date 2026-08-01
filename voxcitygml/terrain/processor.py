"""
Terrain processing: CityGML TIN → DEM grid → voxelised subsurface.

Steps
-----
1. Collect all terrain TIN triangles from parsed CityGML meshes.
2. Build a regular 2-D grid (meshsize resolution) that covers the
   target rectangle.
3. For every grid cell, find the enclosing terrain triangle and
   interpolate the elevation (barycentric interpolation).
4. Return a *north-up* DEM grid (row 0 = north, columns increase east)
   matching voxcity conventions.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import Delaunay

from ..models import Mesh3D
from ..citygml.coordinates import swap_coordinates_3d
from ..grid_utils import compute_grid_params, GridParams


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def terrain_meshes_to_dem_grid(
    terrain_meshes: List[Mesh3D],
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
) -> np.ndarray:
    """Convert CityGML terrain meshes to a north-up DEM elevation grid.

    Grid sizing follows voxcity's algorithm (geodesic distances, round-
    half-up cell counts, evenly spaced pixels) so that all grids
    produced by the pipeline share the same geometry.

    Parameters
    ----------
    terrain_meshes : list[Mesh3D]
        Terrain meshes extracted from CityGML (vertices in lat, lon, z).
    rectangle_vertices : list[(lon, lat)]
        Target rectangle [SW, NW, NE, SE].
    meshsize : float
        Grid resolution in metres.

    Returns
    -------
    np.ndarray
        2-D float64 array, shape ``(n_rows, n_cols)``, in *north-up*
        orientation (row 0 = north).  Values are elevations in metres.
    """
    gp = compute_grid_params(rectangle_vertices, meshsize)

    if not terrain_meshes:
        return np.zeros(gp.shape, dtype=np.float64)

    # Merge all terrain mesh vertices (lat, lon, z) → (lon, lat, z)
    all_pts = _collect_terrain_points(terrain_meshes)  # (N, 3) lon, lat, z

    # Build query grid using affine cell centres (rotation-aware)
    grid_lon, grid_lat = gp.cell_centres()
    query_pts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

    # Interpolate elevation on the TIN
    dem_flat = _interpolate_on_tin(all_pts[:, :2], all_pts[:, 2], query_pts)
    dem_grid = dem_flat.reshape(gp.n_rows, gp.n_cols)

    return dem_grid


def _collect_terrain_points(meshes: List[Mesh3D]) -> np.ndarray:
    """Merge terrain vertices and swap to (lon, lat, z)."""
    parts = []
    for m in meshes:
        if len(m.vertices) == 0:
            continue
        # CityGML PLATEAU stores vertices as (lat, lon, z) → swap to (lon, lat, z)
        swapped = swap_coordinates_3d(m.vertices)
        parts.append(swapped)
    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    all_pts = np.vstack(parts)
    # Remove exact duplicates to speed up Delaunay
    _, idx = np.unique(np.round(all_pts[:, :2], decimals=10), axis=0, return_index=True)
    return all_pts[idx]


def _interpolate_on_tin(
    xy: np.ndarray,        # (N, 2) terrain sample positions
    z: np.ndarray,          # (N,)   terrain elevations
    query_xy: np.ndarray,   # (M, 2) grid cell positions
) -> np.ndarray:
    """Interpolate elevation at query points using Delaunay triangulation.

    For points outside the convex hull, nearest-neighbour elevation is used.
    """
    n = len(xy)
    if n == 0:
        return np.zeros(len(query_xy), dtype=np.float64)

    if n < 3:
        # Not enough points for triangulation – use mean elevation
        return np.full(len(query_xy), z.mean(), dtype=np.float64)

    tri = Delaunay(xy)
    simplex_idx = tri.find_simplex(query_xy)

    result = np.full(len(query_xy), np.nan, dtype=np.float64)

    # Points inside the triangulation
    inside = simplex_idx >= 0
    if inside.any():
        s_idx = simplex_idx[inside]
        tri_verts = tri.simplices[s_idx]              # (K, 3) vertex indices
        bary = _barycentric(xy, tri_verts, query_xy[inside])
        z_corners = z[tri_verts]                       # (K, 3)
        result[inside] = np.sum(bary * z_corners, axis=1)

    # Points outside – nearest-neighbour
    outside = ~inside
    if outside.any():
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        _, nn_idx = tree.query(query_xy[outside])
        result[outside] = z[nn_idx]

    return result


def _barycentric(xy: np.ndarray, tri_verts: np.ndarray,
                 pts: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates for *pts* inside triangles."""
    a = xy[tri_verts[:, 0]]
    b = xy[tri_verts[:, 1]]
    c = xy[tri_verts[:, 2]]

    v0 = b - a
    v1 = c - a
    v2 = pts - a

    dot00 = np.sum(v0 * v0, axis=1)
    dot01 = np.sum(v0 * v1, axis=1)
    dot02 = np.sum(v0 * v2, axis=1)
    dot11 = np.sum(v1 * v1, axis=1)
    dot12 = np.sum(v1 * v2, axis=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-30)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v

    return np.column_stack([w, u, v])
