"""Re-voxelize an LOD2 model at finer meshsizes: the mesh provider for
voxcitywind's adaptive resolution (level-stack contract in
voxcitywind/levels.py).

An LOD2 grid came from ``voxelize_citygml_meshes`` on a ``Grid3DParams``; the
2.5-D rasters beside it are summaries and cannot rebuild roofs. So refining
means running the SAME voxelizer again on ``Grid3DParams.refined(f)`` -- same
frame, same vertical datum, cell size halved -- over the same meshes.

Mesh source, in order: the ``collection`` argument; ``extras['citygml_collection']``
(live only -- it does not survive save/load); a one-off re-parse of
``extras['citygml_paths']`` when those directories still exist. Otherwise
``MeshSourceUnavailable`` is raised and the caller degrades to upsampling.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
from scipy.ndimage import zoom

from .models import CityGMLMeshCollection, _MODE_PARAMS
from .voxelizer3d import Grid3DParams, frame_extent, voxelize_citygml_meshes


class MeshSourceUnavailable(RuntimeError):
    """No meshes to re-voxelize from: no live collection and no readable paths."""


def _resample_grid_mode(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Resample a float raster to ``(target_rows, target_cols)`` with
    ``grid_mode=True`` (cell-centre aligned), so the voxelizer's own
    ``_resize_float_grid`` (``grid_mode=False``, endpoint aligned) becomes a
    no-op on the result -- see ``refine_voxel_grids`` for the measured phase
    error this avoids. A no-op when the shape already matches."""
    if arr.shape == (target_rows, target_cols):
        return arr
    factor = (target_rows / arr.shape[0], target_cols / arr.shape[1])
    # mode="nearest": extends the edge cell rather than padding with a
    # constant (scipy's default), which would distort values right at the
    # raster boundary -- exactly where a DEM/canopy raster has no true
    # neighbour to interpolate from.
    return zoom(arr, factor, order=1, grid_mode=True, mode="nearest")


def _resample_land_cover(lc: np.ndarray, factor: int, target_rows: int, target_cols: int) -> np.ndarray:
    """Nearest-neighbour upsample land cover by the exact integer ``factor``,
    instead of handing the base-resolution array to the voxelizer's own
    ``_resize_int_grid`` (``zoom(..., order=0)``, ``grid_mode=False`` --
    the same endpoint-aligned phase error the DEM/canopy resample above
    avoids). ``np.repeat`` nesting was measured exactly equal to a
    ``grid_mode=True`` nearest-neighbour resample and is cheaper; it is also
    orientation-agnostic for an integer factor (every child of a coarse
    cell gets that cell's own value, regardless of which end of the axis
    index 0 is), so unlike the DEM/canopy this needs no flip bookkeeping.

    A phase error here is not merely a class relabel: ``build_cell_types``
    maps every positive land-cover code to SOLID alike, so swapping one
    land class for its neighbour is inert for the solver -- but this same
    raster also drives the water mask feeding
    ``_carve_water_to_dem_surface``, so a displaced channel edge becomes a
    GROUND-HEIGHT change, not just a mislabelled surface. Measured on an
    89x91 striped raster against the ``np.repeat`` nesting the level
    contract implies: factor 2 was clean by coincidence, factor 4
    displaced 14196 of 129584 cells (11%), with 39 of 356 rows shifted by
    one fine row -- and factors up to 8 are reachable (four levels).
    """
    out = np.repeat(np.repeat(lc, factor, axis=0), factor, axis=1)
    if out.shape != (target_rows, target_cols):
        raise RuntimeError(
            f"np.repeat land-cover resample gave {out.shape}, expected "
            f"{(target_rows, target_cols)}")
    return out


def _centre(extras):
    """The frame anchor: ``extras['center_lon'/'center_lat']`` when recorded,
    else the target rectangle's four-vertex mean.

    The fallback is exact, not approximate: ``resolve_rectangles`` (the
    pipeline's own centre resolution) DEFINES the centre as that same
    four-vertex mean whenever ``rectangle_vertices`` is given, which is the
    only case this fallback is reached in (``center_lon``/``center_lat`` are
    always recorded by ``assembly_extras`` otherwise). Measured against the
    true resolved centre anyway, across latitudes 1.3-55 degrees, sizes
    500 m-4 km and rotations 0-73 degrees: worst-case drift 0.057 m (0.011
    base cells at a 5 m meshsize), and the derived row/column counts never
    changed. Not worth flagging or logging.
    """
    lon, lat = extras.get("center_lon"), extras.get("center_lat")
    if lon is not None and lat is not None:
        return float(lon), float(lat)
    rect = extras["rectangle_vertices"]
    lon = sum(v[0] for v in rect) / len(rect)
    lat = sum(v[1] for v in rect) / len(rect)
    return float(lon), float(lat)


def _clip_cells(nz0: int, meshsize: float, max_height_m) -> int:
    """Base cells to keep: the whole column, or ``max_height_m`` rounded UP to
    a whole base cell, clamped to [1, nz0]. Same rule as the other two
    providers; non-finite input is refused."""
    if max_height_m is None:
        return int(nz0)
    h = float(max_height_m)
    if not np.isfinite(h):
        raise ValueError(f"max_height_m must be finite: {max_height_m!r}")
    return max(1, min(int(np.ceil(h / float(meshsize))), int(nz0)))


def grid_params_from_model(city) -> Grid3DParams:
    """Rebuild the Grid3DParams the model's voxel grid was made on, from its
    shape, meshsize, ``voxel_min_z`` and the rectangle frame. Refuses a grid
    whose shape does not match the frame -- a model not built by the 3-D
    voxelizer, or one edited since."""
    extras = city.extras or {}
    rect = extras.get("rectangle_vertices")
    if rect is None:
        raise ValueError("grid_params_from_model needs extras['rectangle_vertices']")
    min_z = extras.get("voxel_min_z")
    if min_z is None:
        raise ValueError(
            "grid_params_from_model needs extras['voxel_min_z']: this model was not "
            "built by the 3-D mesh voxelizer")
    lon, lat = _centre(extras)
    _, min_x, max_x, min_y, max_y = frame_extent(rect, lon, lat)
    vs = float(city.buildings.meta.meshsize)
    rows, cols, n_z = np.asarray(city.voxels.classes).shape
    want_cols = max(1, int((max_x - min_x) / vs + 0.5))
    want_rows = max(1, int((max_y - min_y) / vs + 0.5))
    if (rows, cols) != (want_rows, want_cols):
        raise ValueError(
            f"voxel grid is {(rows, cols)} but the rectangle frame at {vs} m gives "
            f"{(want_rows, want_cols)}: the grid does not match its frame")
    return Grid3DParams(n_rows=rows, n_cols=cols, n_z=int(n_z),
                        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                        min_z=float(min_z), max_z=float(min_z) + n_z * vs, voxel_size=vs)


#: Distinguishes "key absent" (an older model, predating ``citygml_building_lod``)
#: from "key present with value None" (this run's parser preference WAS
#: "highest available") -- ``extras.get(..., None)`` cannot tell those apart.
_UNSET = object()


def _collection_for(city, collection) -> CityGMLMeshCollection:
    if collection is not None:
        return collection
    extras = city.extras or {}
    live = extras.get("citygml_collection")
    if live is not None:
        return live
    paths = extras.get("citygml_paths") or (
        [extras["citygml_path"]] if extras.get("citygml_path") else [])
    paths = [p for p in paths if p and os.path.isdir(str(p))]
    if not paths:
        raise MeshSourceUnavailable(
            "no mesh source: extras['citygml_collection'] is absent (it does not "
            "survive save/load) and no extras['citygml_paths'] directory exists")
    from .citygml.parser import parse_citygml_directory

    # ``citygml_building_lod`` is the parser PREFERENCE this run used
    # (``assembly_extras``, recorded faithfully including ``None`` =
    # "highest available"). ``building_lod`` is only a fallback for models
    # that predate it: some callers (e.g. VoxCityApp) write it as a TAG
    # recording what LOD was actually parsed, not a preference, so it is
    # used only when the real preference was never recorded.
    lod = extras.get("citygml_building_lod", _UNSET)
    if lod is _UNSET:
        lod = extras.get("building_lod")
    lod_pref = int(lod) if lod else None

    # The buffered rectangle used at the base run, when recorded, so the
    # re-parse sees the same buffer the terrain solid was built with at the
    # outermost row/column; falls back to the target rectangle otherwise.
    rect = extras.get("buffered_rectangle") or extras.get("rectangle_vertices")

    coll = parse_citygml_directory(
        str(paths[0]), rectangle_vertices=rect,
        feature_types=["terrain", "building", "bridge", "vegetation"],
        building_lod=lod_pref,
        dem_path=extras.get("dem_path"),
        tree_citygml_path=extras.get("tree_citygml_path"))
    for extra in paths[1:]:
        coll.merge(parse_citygml_directory(
            str(extra), rectangle_vertices=rect,
            feature_types=["terrain", "building", "bridge", "vegetation"],
            building_lod=lod_pref))
    return coll


def refine_voxel_grids(city, factors: Sequence[int],
                       collection: Optional[CityGMLMeshCollection] = None,
                       max_height_m=None) -> List[np.ndarray]:
    """One voxel grid per factor, each exactly ``factor`` times the base shape
    (z limited to ``k0 * factor`` cells when ``max_height_m`` is given),
    south-up like ``city.voxels.classes`` and in its dtype."""
    extras = city.extras or {}
    gp_full = grid_params_from_model(city)
    k0 = _clip_cells(gp_full.n_z, gp_full.voxel_size, max_height_m)
    clipping = k0 != gp_full.n_z
    if clipping:
        # Voxelize one extra base cell ABOVE the clip (guaranteed to exist:
        # k0 < gp_full.n_z here, so k0 + 1 <= gp_full.n_z) and slice it back
        # off below, rather than handing the voxelizer a grid whose top IS
        # k0.  Without the guard cell, any column whose terrain/DEM rises
        # above the clip gets its topmost retained cell -- an artificial
        # ceiling, not the true ground -- recoloured by `_apply_land_cover`
        # as if it were the real surface: a false land-cover skin.  Measured
        # on this module's own building fixture at max_height_m=12: 10.3% of
        # retained cells differed from an unclipped run (all GROUND_CODE ->
        # a positive land-cover code in the top layer) with a 0-cell guard;
        # a 1-cell guard measured exactly zero divergence (a 2-cell guard
        # also measured zero, so 1 is the minimum that works). This matters
        # because voxcitywind.levels clips level 0 AFTER voxelization
        # (`clip_height`), so level 0 never sees this skin while the fine
        # levels built here would -- turning the fetch margins solid at
        # z=k0-1 on the fine levels and not at level 0 (`pad_streamwise`
        # zeroes every positive class there).
        gp0 = Grid3DParams(n_rows=gp_full.n_rows, n_cols=gp_full.n_cols, n_z=k0 + 1,
                           min_x=gp_full.min_x, max_x=gp_full.max_x, min_y=gp_full.min_y,
                           max_y=gp_full.max_y, min_z=gp_full.min_z,
                           max_z=gp_full.min_z + (k0 + 1) * gp_full.voxel_size,
                           voxel_size=gp_full.voxel_size)
    else:
        gp0 = gp_full
    coll = _collection_for(city, collection)
    lon, lat = _centre(extras)
    rect = extras["rectangle_vertices"]
    base = np.asarray(city.voxels.classes)
    # Inclusive mode, always -- not the base grid's own mode: the extras
    # this provider reads (``assembly_extras``) do not record
    # ``voxelization_mode``, so a base grid built in "tight" mode still
    # gets refined in inclusive mode here. Read the mode table directly
    # rather than constructing a whole ``VoxelizerConfig`` for its three
    # resolved constants.
    vox = _MODE_PARAMS["inclusive"]
    dem_n = np.ascontiguousarray(np.flipud(np.asarray(city.dem.elevation, dtype=np.float64)))
    # voxcity.models.CanopyGrid always has both attributes (bottom may be None).
    top = city.tree_canopy.top
    bottom = city.tree_canopy.bottom
    top_n = None if top is None else np.ascontiguousarray(np.flipud(np.asarray(top, dtype=np.float64)))
    bottom_n = None if bottom is None else np.ascontiguousarray(np.flipud(np.asarray(bottom, dtype=np.float64)))
    lc_s = np.asarray(city.land_cover.classes)          # south-up: the voxelizer flips it
    out = []
    for f in factors:
        f = int(f)
        if f < 1:
            raise ValueError(f"factors must be >= 1: {factors}")
        gp = gp0.refined(f)
        # Pre-resample the DEM/canopy rasters ourselves, with
        # ``grid_mode=True`` (cell-centre aligned), instead of handing the
        # base-resolution array to the voxelizer and letting its own
        # ``_resize_float_grid`` (``zoom(..., order=1)``, ``grid_mode=False``
        # -- endpoint aligned, not centre aligned) do it: at the target
        # shape this call already matches, that resize is a no-op, so the
        # array actually used is the one built here. Measured phase error
        # from the endpoint-aligned resize on a 0.3 m/cell ramp: 0.074 m at
        # f=2, 0.111 m at f=4; grid_mode=True measured exactly zero. Terrain
        # genuinely comes from meshes in THIS provider, so interpolating the
        # DEM/canopy rasters onto the finer grid is the right thing to do
        # here -- unlike the polygon-only level provider, where interpolating
        # a DEM was measured to break its vertical datum and a step
        # (nearest-neighbour) resize is used instead. Do NOT change
        # ``_resize_float_grid`` itself: the main single-level pipeline
        # depends on its current (grid_mode=False) behaviour. Land cover
        # gets its OWN pre-resample (``_resample_land_cover``, nearest via
        # ``np.repeat``) rather than this bilinear one: interpolating class
        # codes is meaningless, and this raster also drives the water mask
        # feeding ``_carve_water_to_dem_surface`` -- see
        # ``_resample_land_cover``'s docstring for the measured displacement
        # and why it matters more than a class relabel would.
        dem_f = _resample_grid_mode(dem_n, gp.n_rows, gp.n_cols)
        top_f = None if top_n is None else _resample_grid_mode(top_n, gp.n_rows, gp.n_cols)
        bottom_f = (None if bottom_n is None
                   else _resample_grid_mode(bottom_n, gp.n_rows, gp.n_cols))
        lc_f = _resample_land_cover(lc_s, f, gp.n_rows, gp.n_cols)
        # The no-op claim above holds only because the voxelizer's own
        # resize target is exactly (gp.n_rows, gp.n_cols) today. Pin that
        # rather than trust it silently: if it ever changed, a shape
        # mismatch here would otherwise become a second, undetected
        # interpolation pass instead of a loud failure.
        for name, arr in (("dem", dem_f), ("canopy_top", top_f),
                          ("canopy_bottom", bottom_f), ("land_cover", lc_f)):
            if arr is not None and arr.shape != (gp.n_rows, gp.n_cols):
                raise RuntimeError(
                    f"pre-resampled {name} raster is {arr.shape}, expected "
                    f"{(gp.n_rows, gp.n_cols)}: the voxelizer's own resize "
                    "target no longer matches this provider's assumption")
        grid = voxelize_citygml_meshes(
            coll, rect, lon, lat, gp.voxel_size,
            dem_grid=dem_f, land_cover_grid=lc_f, canopy_top=top_f, canopy_bottom=bottom_f,
            land_cover_source=extras.get("land_cover_source") or "OpenStreetMap",
            occupancy_threshold=vox.occupancy_threshold,
            building_shell_threshold=vox.building_shell_threshold,
            shell_anchor=vox.shell_anchor,
            flatten_water_dem=bool(extras.get("flatten_water_dem", True)),
            max_voxel_ram_mb=extras.get("max_voxel_ram_mb"),
            grid_params=gp)
        if grid.shape != (gp.n_rows, gp.n_cols, gp.n_z):
            raise RuntimeError(f"voxelizer returned {grid.shape}, expected "
                               f"{(gp.n_rows, gp.n_cols, gp.n_z)}")
        if clipping:
            # Drop the guard layer's f fine cells now that land cover has
            # already been (mis)applied to it instead of to the true clip.
            grid = grid[:, :, :k0 * f]
        south = np.ascontiguousarray(np.flipud(grid))
        out.append(south.astype(base.dtype, copy=False))
    return out
