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

    **Frames.**  ``canopy_top`` / ``canopy_bottom`` must be **north-up**: row 0
    is the north edge, matching ``city.voxels.classes``, ``city.dem.elevation``,
    ``city.tree_canopy.top`` and ``extras['mesh_vegetation_mask']``.
    ``city.land_cover.classes`` is the odd one out — it is south-up, and the
    voxelizer ``np.flipud``\\ s it before use (``_apply_land_cover``), as does
    ``canopy/processor.py`` for downloaded canopy rasters.  A canopy built in
    the land-cover frame, or indexed against it, must therefore be flipped
    before it is passed here.  **Nothing checks this** — orientation is not
    recoverable from an array — and a south-up canopy yields a north-south
    mirrored result that looks entirely plausible.

    **Atomicity.**  Either the whole update lands or none of it does: if the
    overlay raises, the cleared canopy voxels and the previous component grids
    are restored before the exception propagates.

    Grids that disagree with the voxel grid's ``(n_rows, n_cols)`` are
    resampled onto it rather than rejected — the same bilinear resize
    ``voxelize_citygml_meshes`` applies to canopy and DEM at build time.  What
    is *stored* on ``city.tree_canopy`` keeps the caller's resolution, again
    mirroring the build path, so the 2.5-D component grids stay consistent
    with each other.

    Args:
        city: a ``voxcity.models.VoxCity`` produced by this package with
            ``use_3d_voxelizer=True``.  ``city.voxels.classes``,
            ``city.tree_canopy`` and the ``canopy_top`` / ``canopy_bottom``
            aliases in ``city.extras`` are all updated in place.
        canopy_top: 2-D crown-top height **above ground** (m), north-up like
            ``city.voxels.classes`` (see *Frames*).  Cells ``<= 0`` get no
            canopy.  Resampled onto the voxel grid if it is a different shape.
        canopy_bottom: matching crown-base heights above ground (m).  Derived
            from ``trunk_height_ratio`` when omitted — and the derived array
            is **written back onto** ``city.tree_canopy.bottom``, replacing a
            ``None`` "unknown" marker, so the component grid describes the
            crowns that are really in the voxel grid.  Note the consequence:
            a later ``voxcity.generator.update`` call that revises only the
            canopy *top* will reuse this stored bottom rather than re-deriving
            one from the new top.
        trunk_height_ratio: crown base as a fraction of crown top, used only
            when ``canopy_bottom`` is ``None``.  Defaults to the voxelizer's
            own ratio.

    Raises:
        ValueError: when ``canopy_bottom`` does not match ``canopy_top``, when
            ``extras['mesh_vegetation_mask']`` does not match the voxel grid
            (the two index each other 1:1, so a mismatch means the extras do
            not belong to this model — not a resolution to resample away), or
            when the model carries no ``voxel_min_z`` (its vertical datum) to
            place crowns against.  The model is unchanged in every case.
    """
    voxel_grid = city.voxels.classes
    if voxel_grid.ndim != 3:
        raise ValueError(
            f"city.voxels.classes must be a 3-D (rows, cols, z) array; "
            f"got shape {voxel_grid.shape}")
    n_rows, n_cols, n_z = voxel_grid.shape
    grid_shape = (n_rows, n_cols)

    canopy_top = np.asarray(canopy_top, dtype=np.float64)
    if canopy_top.ndim != 2:
        raise ValueError(
            f"canopy_top must be a 2-D (rows, cols) array; got shape "
            f"{canopy_top.shape}")
    if canopy_bottom is not None:
        canopy_bottom = np.asarray(canopy_bottom, dtype=np.float64)
        # The pairwise constraint is the meaningful one: these two describe the
        # same crowns cell for cell.  Agreement with the *voxel* grid is not
        # required — a mismatch there is a resolution difference, resampled
        # below the way the build path resamples it.
        if canopy_bottom.shape != canopy_top.shape:
            raise ValueError(
                f"canopy_bottom shape {canopy_bottom.shape} does not match "
                f"canopy_top {canopy_top.shape}")

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

    # Everything the overlay reads is resampled onto the voxel grid, exactly
    # as `voxelize_citygml_meshes` does for the DEM and the canopy at build
    # time, so crowns land where they did then even if the 2.5-D component
    # grids are a different resolution.  What gets *stored* on
    # `city.tree_canopy` below is the caller's array at the caller's
    # resolution -- the build path likewise stores the pre-resize grids.
    dem = _to_voxel_resolution(
        np.asarray(city.dem.elevation, dtype=np.float64), grid_shape,
        "city.dem.elevation")
    top_voxels = _to_voxel_resolution(canopy_top, grid_shape, "canopy_top")
    bottom_voxels = (None if canopy_bottom is None else
                     _to_voxel_resolution(canopy_bottom, grid_shape,
                                          "canopy_bottom"))

    # -- 1. Clear the canopy that is already there --------------------------
    # `_apply_canopy` writes into AIR cells only, so re-running it over a
    # populated grid would *union* the new crowns with the old rather than
    # replace them.  Outside the mesh-vegetation mask, every TREE_CODE voxel
    # came from a previous canopy overlay and was AIR before it — clearing it
    # back to AIR is the exact inverse.  Inside the mask the voxels are
    # CityGML crown geometry and are left alone.
    #
    # Steps 1-3 are one transaction: the clear and the component-grid write
    # both precede the fill, so an exception in the fill would otherwise leave
    # the grid canopy-stripped and `tree_canopy` describing crowns that are no
    # longer in it.  The snapshot is two 2-D float arrays plus the `stale`
    # index we already hold -- no second copy of the voxel grid.
    snapshot = _snapshot_canopy_grids(city)
    stale = voxel_grid == TREE_CODE
    if mask.any():
        stale &= ~mask[:, :, None]
    voxel_grid[stale] = 0
    try:
        # -- 2. Keep the 2.5-D component grids describing the voxels --------
        base = _canopy_base_heights(canopy_top, canopy_bottom,
                                    trunk_height_ratio)
        _write_through(city.tree_canopy, "top", canopy_top, city.extras,
                       "canopy_top")
        _write_through(city.tree_canopy, "bottom", base, city.extras,
                       "canopy_bottom")

        # -- 3. Write the revised canopy ------------------------------------
        _apply_canopy(voxel_grid, gp, dem, top_voxels, bottom_voxels,
                      trunk_height_ratio, mesh_tree_mask=mask)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt through here would
        # leave the same half-written model as a ValueError would.
        _restore_canopy_voxels(voxel_grid, stale, mask)
        _restore_canopy_grids(city, snapshot)
        raise

    # Counting `stale` is a second full pass over the grid for a log line the
    # caller may not even be listening to; only pay for it if they are.
    if _log.isEnabledFor(_logging.INFO):
        _log.info(
            "  [reapply_canopy] cleared %d stale canopy voxels, preserved %d "
            "mesh-vegetation column(s)",
            int(np.count_nonzero(stale)), int(np.count_nonzero(mask)))


def _to_voxel_resolution(grid: np.ndarray, grid_shape, name: str) -> np.ndarray:
    """``grid`` resampled onto the voxel grid's (rows, cols), if it isn't already."""
    if grid.shape == grid_shape:
        return grid
    _log.info("  [reapply_canopy] resampled %s from %s onto the voxel grid %s",
              name, grid.shape, grid_shape)
    return _resize_float_grid(grid, grid_shape[0], grid_shape[1])


#: Sentinel for "this extras key was absent", distinct from a stored None.
_ABSENT = object()


def _snapshot_canopy_grids(city) -> dict:
    """Everything outside the voxel grid that the update is about to overwrite."""
    snapshot = {}
    for attr in ("top", "bottom"):
        value = getattr(city.tree_canopy, attr, None)
        # Both the object (in case of a rebind) and its contents (in case of
        # an in-place write-through) are needed to put things back.
        snapshot[attr] = (value,
                          value.copy() if isinstance(value, np.ndarray)
                          else None)
    for key in ("canopy_top", "canopy_bottom"):
        snapshot[key] = city.extras.get(key, _ABSENT)
    return snapshot


def _restore_canopy_grids(city, snapshot: dict) -> None:
    for attr in ("top", "bottom"):
        value, contents = snapshot[attr]
        if contents is not None:
            value[:] = contents
        setattr(city.tree_canopy, attr, value)
    for key in ("canopy_top", "canopy_bottom"):
        previous = snapshot[key]
        if previous is _ABSENT:
            city.extras.pop(key, None)
        else:
            city.extras[key] = previous


def _restore_canopy_voxels(voxel_grid: np.ndarray, stale: np.ndarray,
                           mask: np.ndarray) -> None:
    """Undo the clear, and any canopy a failed overlay managed to write.

    Outside the mask the grid held exactly ``stale`` before this call, so
    dropping every TREE_CODE voxel there and putting ``stale`` back is an
    exact restore whether the overlay wrote nothing or wrote part of a fill.
    Masked columns were never touched by either step.
    """
    written = voxel_grid == TREE_CODE
    if mask.any():
        written &= ~mask[:, :, None]
    voxel_grid[written] = 0
    voxel_grid[stale] = TREE_CODE


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
