"""Overlay revised layers onto a finished VoxCity model, in place.

Everything here operates on an *assembled* ``voxcity.models.VoxCity`` rather
than on the raw arrays the voxelizer deals in — which is why it lives beside
``voxelizer3d`` rather than inside it.  The z arithmetic, the voxel class
codes and the crown-base rule are imported from there, never restated.
"""

from __future__ import annotations

import logging as _logging
import warnings
from typing import Optional

import numpy as np

from .voxelizer3d import (
    Grid3DParams,
    TREE_CODE,
    _apply_canopy,
    _resize_float_grid,
    _canopy_base_heights,
)

_log = _logging.getLogger(__name__)

__all__ = ["reapply_canopy"]


def reapply_canopy(
    city,
    canopy_top: np.ndarray,
    canopy_bottom: Optional[np.ndarray] = None,
    trunk_height_ratio: Optional[float] = None,
) -> None:
    """Overlay a revised canopy onto an existing VoxCity's voxel grid, in place.

    Unlike ``voxcity.generator.update.regenerate_voxels``, this does **not**
    rebuild the grid from the 2.5-D component grids — so mesh-voxelized LOD2
    roof/wall geometry survives.  Buildings, bridges, terrain and land cover
    are never touched: the previous canopy is removed and the new one is
    written into cells that are AIR afterwards.

    Columns recorded in ``extras['mesh_vegetation_mask']`` keep their CityGML
    crown geometry (fill-the-gaps); canopy is written only elsewhere.  PLATEAU
    ships vegetation geometry for very few areas, so that mask is usually
    all-``False`` and every column is canopy's to write.

    Args:
        city: a ``voxcity.models.VoxCity`` produced by this package with
            ``use_3d_voxelizer=True``.  ``city.voxels.classes``,
            ``city.tree_canopy`` and the ``canopy_top`` / ``canopy_bottom``
            aliases in ``city.extras`` are all updated in place.
        canopy_top: (n_rows, n_cols) crown-top height **above ground** (m),
            north-up like ``city.voxels.classes``.  Cells ``<= 0`` get no
            canopy.
        canopy_bottom: matching crown-base heights above ground (m).  Derived
            from ``trunk_height_ratio`` when omitted.
        trunk_height_ratio: crown base as a fraction of crown top, used only
            when ``canopy_bottom`` is ``None``.  Defaults to the voxelizer's
            own ratio.

    Raises:
        ValueError: on a shape mismatch, or when the model carries no
            ``voxel_min_z`` (its vertical datum) to place crowns against.
    """
    voxel_grid = city.voxels.classes
    if voxel_grid.ndim != 3:
        raise ValueError(
            f"city.voxels.classes must be a 3-D (rows, cols, z) array; "
            f"got shape {voxel_grid.shape}")
    n_rows, n_cols, n_z = voxel_grid.shape
    grid_shape = (n_rows, n_cols)

    canopy_top = np.asarray(canopy_top, dtype=np.float64)
    if canopy_top.shape != grid_shape:
        raise ValueError(
            f"canopy_top shape {canopy_top.shape} does not match the voxel "
            f"grid's (rows, cols) {grid_shape}")
    if canopy_bottom is not None:
        canopy_bottom = np.asarray(canopy_bottom, dtype=np.float64)
        if canopy_bottom.shape != grid_shape:
            raise ValueError(
                f"canopy_bottom shape {canopy_bottom.shape} does not match "
                f"the voxel grid's (rows, cols) {grid_shape}")

    min_z = city.extras.get("voxel_min_z")
    if min_z is None:
        raise ValueError(
            "extras['voxel_min_z'] is missing or None: the voxel grid's "
            "vertical datum is unknown, so canopy cannot be placed against "
            "it.  The model predates canopy re-apply support, or was built "
            "with use_3d_voxelizer=False (the legacy voxcity Voxelizer "
            "exposes no such datum).  Regenerate it with a current "
            "VoxCityGML.")
    min_z = float(min_z)

    mask = city.extras.get("mesh_vegetation_mask")
    if mask is None:
        warnings.warn(
            "extras['mesh_vegetation_mask'] is missing; treating every "
            "column as canopy-derived.  Any CityGML vegetation crowns in the "
            "grid will be replaced by the canopy overlay.  Regenerate the "
            "model with a current VoxCityGML to preserve them.",
            UserWarning, stacklevel=2)
        mask = np.zeros(grid_shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != grid_shape:
            raise ValueError(
                f"extras['mesh_vegetation_mask'] shape {mask.shape} does not "
                f"match the voxel grid's (rows, cols) {grid_shape}")

    meshsize = float(city.voxels.meta.meshsize)
    # `_apply_canopy` reads exactly min_z, voxel_size and n_z off gp (it is a
    # per-column overlay: rows/cols come from the arrays, never from x/y
    # bounds).  The bounds below are therefore a consistent local frame for
    # completeness, not values the overlay depends on.
    gp = Grid3DParams(
        n_rows=n_rows, n_cols=n_cols, n_z=n_z,
        min_x=0.0, max_x=n_cols * meshsize,
        min_y=0.0, max_y=n_rows * meshsize,
        min_z=min_z, max_z=min_z + n_z * meshsize,
        voxel_size=meshsize,
    )

    dem = np.asarray(city.dem.elevation, dtype=np.float64)
    if dem.shape != grid_shape:
        # The voxelizer resamples the DEM onto the voxel grid the same way
        # (voxelize_citygml_meshes), so crowns land where they did at build
        # time even when the component grids are a different resolution.
        dem = _resize_float_grid(dem, n_rows, n_cols)

    # -- 1. Clear the canopy that is already there --------------------------
    # `_apply_canopy` writes into AIR cells only, so re-running it over a
    # populated grid would *union* the new crowns with the old rather than
    # replace them.  Outside the mesh-vegetation mask, every TREE_CODE voxel
    # came from a previous canopy overlay and was AIR before it — clearing it
    # back to AIR is the exact inverse.  Inside the mask the voxels are
    # CityGML crown geometry and are left alone.
    stale = voxel_grid == TREE_CODE
    if mask.any():
        stale &= ~mask[:, :, None]
    n_cleared = int(np.count_nonzero(stale))
    voxel_grid[stale] = 0

    # -- 2. Keep the 2.5-D component grids describing the voxels ------------
    base = _canopy_base_heights(canopy_top, canopy_bottom, trunk_height_ratio)
    _write_through(city.tree_canopy, "top", canopy_top, city.extras,
                   "canopy_top")
    _write_through(city.tree_canopy, "bottom", base, city.extras,
                   "canopy_bottom")

    # -- 3. Write the revised canopy ----------------------------------------
    _apply_canopy(voxel_grid, gp, dem, canopy_top, canopy_bottom,
                  trunk_height_ratio, mesh_tree_mask=mask)

    _log.info(
        "  [reapply_canopy] cleared %d stale canopy voxels, preserved %d "
        "mesh-vegetation column(s)", n_cleared, int(np.count_nonzero(mask)))


def _write_through(obj, attr: str, values: np.ndarray, extras: dict,
                   extras_key: str) -> None:
    """Store ``values`` on ``obj.attr``, in place where that is possible.

    ``assemble_voxcity`` aliases the canopy grids into ``extras`` as well, so
    writing into the existing float array keeps both views current; a rebind
    is only taken when the array cannot receive the values (absent, wrong
    shape, integer dtype), and then the extras alias is refreshed too.
    """
    existing = getattr(obj, attr, None)
    if (isinstance(existing, np.ndarray) and existing.shape == values.shape
            and existing.dtype.kind == "f"):
        existing[:] = values
    else:
        # Copy so the caller's array is not aliased into the model, where a
        # later mutation of it would silently desync the grids from the voxels.
        setattr(obj, attr, values.copy())
    if extras_key in extras:
        extras[extras_key] = getattr(obj, attr)
