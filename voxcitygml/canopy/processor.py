"""
Canopy-height grid acquisition.

Downloads canopy height data using ``voxcity.generator.grids.get_canopy_height_grid``
and optionally merges it with vegetation height information extracted
from CityGML.

Two grids are produced:

* **canopy_top** – tree canopy top height above ground (metres).
* **canopy_bottom** – crown base height above ground (metres).
  If not available, estimated as ``top × trunk_height_ratio``.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..models import Mesh3D
from ..citygml.coordinates import swap_coordinates_3d
from ..grid_utils import compute_grid_params


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def get_canopy_grids(
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
    canopy_height_source: str,
    land_cover_source: str,
    land_cover_grid: np.ndarray,
    dem_grid: np.ndarray,
    output_dir: str = "output",
    vegetation_meshes: Optional[List[Mesh3D]] = None,
    trunk_height_ratio: Optional[float] = None,
    static_tree_height: float = 10.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Download canopy height and merge with CityGML vegetation data.

    Parameters
    ----------
    rectangle_vertices, meshsize, canopy_height_source, land_cover_source,
    land_cover_grid, dem_grid, output_dir :
        Standard pipeline parameters (see ``VoxelizerConfig``).
    vegetation_meshes :
        Optional CityGML vegetation meshes for direct height extraction.
    trunk_height_ratio :
        Ratio of trunk height to total tree height (default 11.76/19.98).
    static_tree_height :
        Height to assign when *canopy_height_source* is ``"Static"``.

    Returns
    -------
    canopy_top : np.ndarray   (rows, cols) float64
    canopy_bottom : np.ndarray (rows, cols) float64
    """
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98

    n_rows, n_cols = dem_grid.shape

    # ------------------------------------------------------------------
    # 1. Download canopy height from remote source
    # ------------------------------------------------------------------
    if canopy_height_source == "Static":
        canopy_top_remote = _static_canopy(
            land_cover_grid, land_cover_source, static_tree_height,
        )
    else:
        canopy_top_remote, canopy_bottom_remote = _download_canopy(
            rectangle_vertices, meshsize, canopy_height_source,
            land_cover_source, land_cover_grid, output_dir, **kwargs,
        )

    # Ensure shape matches DEM grid
    if canopy_top_remote.shape != (n_rows, n_cols):
        canopy_top_remote = _resize_grid(canopy_top_remote, n_rows, n_cols)

    # voxcity grids are south-up (row 0 = south); flip to north-up
    # to match DEM and 3-D voxel grid orientation (row 0 = north).
    canopy_top_remote = np.flipud(canopy_top_remote)

    canopy_bottom_remote = canopy_top_remote * trunk_height_ratio

    # ------------------------------------------------------------------
    # 2. Merge CityGML vegetation heights (if available)
    # ------------------------------------------------------------------
    if vegetation_meshes:
        canopy_top_gml = _vegetation_meshes_to_grid(
            vegetation_meshes, rectangle_vertices, meshsize, dem_grid,
        )
        canopy_bottom_gml = canopy_top_gml * trunk_height_ratio

        # CityGML vegetation takes priority; fill gaps with remote
        mask = (canopy_top_gml == 0) & (canopy_top_remote != 0)
        canopy_top_gml[mask] = canopy_top_remote[mask]
        canopy_bottom_gml[mask] = canopy_bottom_remote[mask]
        canopy_top = canopy_top_gml
        canopy_bottom = canopy_bottom_gml
    else:
        canopy_top = canopy_top_remote
        canopy_bottom = canopy_bottom_remote

    canopy_bottom = np.minimum(canopy_bottom, canopy_top)

    print(f"  Canopy grid shape: {canopy_top.shape}, "
          f"max height: {canopy_top.max():.1f} m")
    return canopy_top, canopy_bottom


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _static_canopy(land_cover_grid, land_cover_source, height):
    """Assign a fixed canopy height to tree-class cells."""
    from voxcity.utils.lc import get_land_cover_classes

    canopy = np.zeros_like(land_cover_grid, dtype=np.float64)
    try:
        classes = get_land_cover_classes(land_cover_source)
        cls_to_int = {name: i for i, name in enumerate(classes.values())}
        tree_labels = ["Tree", "Trees", "Tree Canopy"]
        tree_indices = [cls_to_int[l] for l in tree_labels if l in cls_to_int]
        if tree_indices:
            canopy[np.isin(land_cover_grid, tree_indices)] = height
    except Exception:
        pass
    return canopy


def _download_canopy(rect_verts, meshsize, source, lc_source,
                     lc_grid, output_dir, **kwargs):
    """Download canopy height from a remote source via voxcity."""
    from voxcity.generator.grids import get_canopy_height_grid

    print(f"Downloading canopy height ({source})...")
    top, bottom = get_canopy_height_grid(
        rect_verts, meshsize, source, output_dir, **kwargs,
    )
    if bottom is None:
        bottom = top * (11.76 / 19.98)
    return top, bottom


def _vegetation_meshes_to_grid(
    veg_meshes: List[Mesh3D],
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
    dem_grid: np.ndarray,
) -> np.ndarray:
    """Convert CityGML vegetation meshes to a canopy-height grid."""
    n_rows, n_cols = dem_grid.shape
    canopy = np.zeros((n_rows, n_cols), dtype=np.float64)

    gp = compute_grid_params(rectangle_vertices, meshsize)

    for mesh in veg_meshes:
        if len(mesh.vertices) == 0:
            continue

        # Use the stored height attribute if available
        height = mesh.attributes.get('height')
        pts = swap_coordinates_3d(mesh.vertices)

        rows, cols = gp.lonlat_to_rowcol_int(pts[:, 0], pts[:, 1])
        zs = pts[:, 2]

        valid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        rows, cols, zs = rows[valid], cols[valid], zs[valid]

        if len(zs) == 0:
            continue

        cells = set(zip(rows.tolist(), cols.tolist()))
        for r, c in cells:
            mask = (rows == r) & (cols == c)
            z_max = float(zs[mask].max())
            ground_z = dem_grid[r, c]

            if height is not None:
                h = float(height)
            else:
                h = max(0.0, z_max - ground_z)

            if h > canopy[r, c]:
                canopy[r, c] = h

    return canopy


def _resize_grid(grid: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Resize a 2-D grid to target shape using nearest-neighbour."""
    from scipy.ndimage import zoom
    row_factor = target_rows / grid.shape[0]
    col_factor = target_cols / grid.shape[1]
    return zoom(grid, (row_factor, col_factor), order=0)
