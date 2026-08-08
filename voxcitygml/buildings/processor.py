"""
Building & bridge mesh -> height-grid conversion for voxelisation.

For each building / bridge mesh extracted from CityGML, this module
rasterises three aligned 2-D grids that the VoxCity voxeliser expects:

* **building_height_grid** - maximum roof height per cell (metres above
  ground).
* **building_min_height_grid** - object-dtype array of [[min_h, max_h], ...]
  lists per cell (allows multi-storey / piloti segments).
* **building_id_grid** - integer building ID per cell (for per-building
  post-processing such as material assignment).

Grid cell centres are computed using the same voxcity-compatible formula
as the DEM / land-cover grids (see grid_utils.compute_grid_params).
Triangle rasterisation is performed directly in lon/lat space (not
projected metres) so that every cell index aligns exactly with the
other pipeline grids.
"""

from typing import List, Tuple, Dict
import numpy as np

from ..models import Mesh3D
from ..citygml.coordinates import swap_coordinates_3d
from ..grid_utils import compute_grid_params, GridParams


# -----------------------------------------------------------------------
# Triangle rasterisation (lon/lat space)
# -----------------------------------------------------------------------

def _rasterise_triangle_to_cells(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    gp: GridParams,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Rasterise one triangle and return per-cell z ranges.

    Triangle vertices are (lon, lat, z).  Grid mapping uses the same
    affine GridParams frame as the DEM / land-cover grids.

    Parameters
    ----------
    v0, v1, v2 : (3,) arrays - [lon, lat, z]
    gp          : GridParams - shared grid geometry.

    Returns
    -------
    dict  { (row, col): (z_min, z_max) }
    """
    n_rows, n_cols = gp.n_rows, gp.n_cols

    # Triangle vertices in continuous grid coordinates
    tri_lon = np.array([v0[0], v1[0], v2[0]])
    tri_lat = np.array([v0[1], v1[1], v2[1]])
    gy, gx = gp.lonlat_to_rowcol(tri_lon, tri_lat)

    # Bounding box in grid-index space
    min_gc = int(max(0, np.floor(gx.min())))
    max_gc = int(min(n_cols - 1, np.floor(gx.max() + 0.5)))
    min_gr = int(max(0, np.floor(gy.min())))
    max_gr = int(min(n_rows - 1, np.floor(gy.max() + 0.5)))

    if min_gc > max_gc or min_gr > max_gr:
        return {}

    # Barycentric denominator in (lon, lat) space
    denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) +
             (v2[0] - v1[0]) * (v0[1] - v2[1]))

    if abs(denom) < 1e-20:
        # Degenerate in XY (wall triangle or truly degenerate).
        result: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for v in (v0, v1, v2):
            row_f, col_f = gp.lonlat_to_rowcol(
                np.array([v[0]]), np.array([v[1]]),
            )
            c = int(np.round(col_f[0]))
            r = int(np.round(row_f[0]))
            if 0 <= r < n_rows and 0 <= c < n_cols:
                z = v[2]
                prev = result.get((r, c))
                if prev is None:
                    result[(r, c)] = (z, z)
                else:
                    result[(r, c)] = (min(prev[0], z), max(prev[1], z))
        return result

    inv_denom = 1.0 / denom

    # Build arrays of cell-centre (lon, lat) for the bounding box
    cols = np.arange(min_gc, max_gc + 1)
    rows = np.arange(min_gr, max_gr + 1)
    PX, PY = gp.cell_centres(rows=rows, cols=cols)   # lon, lat grids

    # Vectorised barycentric coordinates
    W0 = ((v1[1] - v2[1]) * (PX - v2[0]) +
          (v2[0] - v1[0]) * (PY - v2[1])) * inv_denom
    W1 = ((v2[1] - v0[1]) * (PX - v2[0]) +
          (v0[0] - v2[0]) * (PY - v2[1])) * inv_denom
    W2 = 1.0 - W0 - W1

    TOL = -0.01
    inside = (W0 >= TOL) & (W1 >= TOL) & (W2 >= TOL)

    hit_rows, hit_cols_local = np.nonzero(inside)
    if len(hit_rows) == 0:
        return {}

    abs_rows = hit_rows + min_gr
    abs_cols = hit_cols_local + min_gc

    # Clamp barycentric weights for z interpolation
    cw0 = np.clip(W0[hit_rows, hit_cols_local], 0.0, 1.0)
    cw1 = np.clip(W1[hit_rows, hit_cols_local], 0.0, 1.0)
    cw2 = 1.0 - cw0 - cw1
    cw2 = np.clip(cw2, 0.0, None)
    s = cw0 + cw1 + cw2
    mask_s = s > 0
    cw0[mask_s] /= s[mask_s]
    cw1[mask_s] /= s[mask_s]
    cw2[mask_s] /= s[mask_s]

    z_interp = cw0 * v0[2] + cw1 * v1[2] + cw2 * v2[2]

    result: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for idx in range(len(abs_rows)):
        r = int(abs_rows[idx])
        c = int(abs_cols[idx])
        z = float(z_interp[idx])
        prev = result.get((r, c))
        if prev is None:
            result[(r, c)] = (z, z)
        else:
            result[(r, c)] = (min(prev[0], z), max(prev[1], z))

    return result


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def meshes_to_building_grids(
    building_meshes: List[Mesh3D],
    bridge_meshes: List[Mesh3D],
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
    dem_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise CityGML building & bridge meshes into voxcity-compatible grids.

    Triangle rasterisation is performed in lon/lat space using the same
    GridParams as the DEM, ensuring pixel-perfect alignment with the
    terrain and land-cover grids.

    Parameters
    ----------
    building_meshes : list[Mesh3D]
        Building meshes from CityGML (vertices in lat, lon, z).
    bridge_meshes : list[Mesh3D]
        Bridge meshes (treated like buildings for voxelisation).
    rectangle_vertices : list[(lon, lat)]
        Target rectangle [SW, NW, NE, SE].
    meshsize : float
        Grid resolution in metres.
    dem_grid : np.ndarray
        DEM elevation grid (north-up).

    Returns
    -------
    building_height_grid : np.ndarray   (rows, cols) float64
    building_min_height_grid : np.ndarray  (rows, cols) object of lists
    building_id_grid : np.ndarray       (rows, cols) int32
    """
    gp = compute_grid_params(rectangle_vertices, meshsize)
    all_meshes = list(building_meshes) + list(bridge_meshes)

    n_rows, n_cols = dem_grid.shape

    # Validate grid-param shape matches DEM (should be identical now)
    if gp.shape != (n_rows, n_cols):
        print(f"  WARNING: GridParams shape {gp.shape} != DEM shape "
              f"{(n_rows, n_cols)}.  Using GridParams shape.")
        n_rows, n_cols = gp.shape

    building_height_grid = np.zeros((n_rows, n_cols), dtype=np.float64)
    building_id_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    building_min_height_grid = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            building_min_height_grid[i, j] = []

    if not all_meshes:
        return building_height_grid, building_min_height_grid, building_id_grid

    print(f"Rasterising {len(all_meshes)} building/bridge meshes to "
          f"grid ({n_rows} x {n_cols})...")

    for mesh_idx, mesh in enumerate(all_meshes):
        bid = mesh_idx + 1
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        # Convert (lat, lon, z) -> (lon, lat, z)
        pts_ll = swap_coordinates_3d(mesh.vertices)

        # Accumulate per-cell z ranges from every triangle
        cell_zranges: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for face in mesh.faces:
            tri = pts_ll[face]  # (3, 3) - [lon, lat, z]
            tri_cells = _rasterise_triangle_to_cells(
                tri[0], tri[1], tri[2], gp,
            )
            for (r, c), (zlo, zhi) in tri_cells.items():
                prev = cell_zranges.get((r, c))
                if prev is None:
                    cell_zranges[(r, c)] = (zlo, zhi)
                else:
                    cell_zranges[(r, c)] = (min(prev[0], zlo),
                                            max(prev[1], zhi))

        # Write cell z-ranges into the output grids
        for (r, c), (z_min_abs, z_max_abs) in cell_zranges.items():
            ground_z = dem_grid[r, c]
            h_max = max(0.0, z_max_abs - ground_z)
            h_min = max(0.0, z_min_abs - ground_z)

            if h_max <= 0:
                continue

            if h_max > building_height_grid[r, c]:
                building_height_grid[r, c] = h_max
                building_id_grid[r, c] = bid

            building_min_height_grid[r, c].append([h_min, h_max])

    n_occupied = np.count_nonzero(building_height_grid)
    print(f"  {n_occupied} grid cells contain building/bridge data")
    return building_height_grid, building_min_height_grid, building_id_grid


def fill_building_id_gaps(
    building_id_grid: np.ndarray,
    voxel_grid: np.ndarray,
    building_code: int = -3,
) -> np.ndarray:
    """Extend ``building_id_grid`` to every column the voxeliser filled.

    The two grids are built by independent rasterisations of the same meshes.
    ``meshes_to_building_grids`` above claims a cell only when the cell
    *centre* passes a barycentric inside-triangle test, whereas
    ``voxelizer3d.voxelize_citygml_meshes`` marks every voxel the mesh
    *touches* -- an SAT any-touch surface shell unioned with an SDF/winding
    interior fill. The 3-D footprint is therefore up to one cell wider on each
    side, and two smaller rules here widen the gap further: a cell whose mesh
    does not rise above the DEM is skipped entirely (``h_max <= 0``), and a
    contested cell goes to the *tallest* mesh alone.

    Anything that intersects the two grids -- per-building highlighting,
    landmark marking, per-building surface statistics, carve/delete -- drops
    that fringe silently. On real PLATEAU LoD2 tiles it is 7.5% of building
    columns at 2 m and 14.9% at 5 m, and 30-45% for small buildings.

    Every dropped column is adjacent to a claimed one (measured: a pure
    1-cell fringe, no interior holes), so propagating the nearest claimed id
    recovers it -- to 0.04% residual at 2 m and 0.8% at 5 m. The residual is
    irreducible here: one id per cell cannot name both owners of a column two
    buildings share.

    Columns with no building voxels are left at 0, so a building never
    acquires ground it has no voxels in.

    Both grids must be in the same orientation and share their first two axes;
    in ``pipeline.run_core`` both are north-up before assembly flips them.

    Parameters
    ----------
    building_id_grid : (R, C) int array
        Per-cell building id, ``0`` meaning unclaimed.
    voxel_grid : (R, C, Z) array
        The 3-D voxel grid whose ``building_code`` columns define coverage.
    building_code : int
        Voxel class marking a building (also used for bridges).

    Returns
    -------
    (R, C) array, same dtype as *building_id_grid* -- a copy; the input is
    never mutated.
    """
    from scipy.ndimage import distance_transform_edt

    if voxel_grid.shape[:2] != building_id_grid.shape:
        raise ValueError(
            "building_id_grid and voxel_grid must share their first two axes; "
            f"got {building_id_grid.shape} and {voxel_grid.shape[:2]}. These "
            "grids come from separately-computed frames, so a mismatch would "
            "mis-attribute every column rather than a few."
        )

    has_id = building_id_grid > 0
    # No source to propagate from. scipy's returned indices are meaningless
    # when the background is empty, so filling here would fabricate ids.
    if not has_id.any():
        return building_id_grid.copy()

    needs_id = np.any(voxel_grid == building_code, axis=2) & ~has_id
    if not needs_id.any():
        return building_id_grid.copy()

    # For every unclaimed cell, the (row, col) of the nearest claimed one.
    _distances, indices = distance_transform_edt(~has_id, return_indices=True)

    out = building_id_grid.copy()
    out[needs_id] = building_id_grid[indices[0][needs_id], indices[1][needs_id]]
    return out
