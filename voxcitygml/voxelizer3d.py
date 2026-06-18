"""
3D mesh voxelizer for CityGML data.

This voxelizer places all CityGML meshes (terrain, buildings, bridges,
vegetation) into a shared 3D space and voxelizes them on a single grid.
Land cover and canopy data are then overlaid on the voxel grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from numba import njit, prange
from scipy.ndimage import binary_fill_holes, zoom

from .models import Mesh3D, CityGMLMeshCollection
from .citygml.coordinates import swap_coordinates_3d, create_local_transformer
from .watertight import make_watertight_mesh
from .terrain_solid import build_terrain_solid

import logging as _logging
_log = _logging.getLogger(__name__)

# ── MeshLib availability ─────────────────────────────────────────────
_MESHLIB_VOXEL_AVAILABLE = False
try:
    import meshlib.mrmeshpy as _mr
    import meshlib.mrmeshnumpy as _mrnp
    # Verify the APIs we need actually exist
    _ = _mr.MeshToVolumeParams
    _ = _mr.MeshToDistanceVolumeParams
    _ = _mrnp.meshFromFacesVerts
    _ = _mrnp.getNumpy3Darray
    _MESHLIB_VOXEL_AVAILABLE = True
except (ImportError, AttributeError):
    _log.info(
        "meshlib not available or missing voxel APIs – "
        "falling back to Numba Z-scanline voxelizer."
    )

GROUND_CODE = -1
TREE_CODE = -2
BUILDING_CODE = -3


def _bbox_to_index_range(gp: "Grid3DParams", bmin: np.ndarray, bmax: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    vs = gp.voxel_size

    c0 = int(np.floor((bmin[0] - gp.min_x) / vs)) - 1
    c1 = int(np.floor((bmax[0] - gp.min_x) / vs)) + 1

    r0 = int(np.floor((gp.max_y - bmax[1]) / vs)) - 1
    r1 = int(np.floor((gp.max_y - bmin[1]) / vs)) + 1

    z0 = int(np.floor((bmin[2] - gp.min_z) / vs)) - 1
    z1 = int(np.floor((bmax[2] - gp.min_z) / vs)) + 1

    r0 = int(np.clip(r0, 0, gp.n_rows - 1))
    r1 = int(np.clip(r1, 0, gp.n_rows - 1))
    c0 = int(np.clip(c0, 0, gp.n_cols - 1))
    c1 = int(np.clip(c1, 0, gp.n_cols - 1))
    z0 = int(np.clip(z0, 0, gp.n_z - 1))
    z1 = int(np.clip(z1, 0, gp.n_z - 1))

    if r0 > r1:
        r0, r1 = r1, r0
    if c0 > c1:
        c0, c1 = c1, c0
    if z0 > z1:
        z0, z1 = z1, z0

    return r0, r1, c0, c1, z0, z1


def _dilate6(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    out = mask.copy()
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


@dataclass
class Grid3DParams:
    n_rows: int
    n_cols: int
    n_z: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    voxel_size: float

    def xyz_to_indices(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        col = ((x - self.min_x) / self.voxel_size).astype(np.intp)
        row = ((self.max_y - y) / self.voxel_size).astype(np.intp)
        zi = ((z - self.min_z) / self.voxel_size).astype(np.intp)
        return row, col, zi

    def box_center(self, row: int, col: int, zi: int) -> np.ndarray:
        x = self.min_x + (col + 0.5) * self.voxel_size
        y = self.max_y - (row + 0.5) * self.voxel_size
        z = self.min_z + (zi + 0.5) * self.voxel_size
        return np.array([x, y, z], dtype=np.float64)


def voxelize_citygml_meshes(
    collection: CityGMLMeshCollection,
    rectangle_vertices: List[Tuple[float, float]],
    center_lon: float,
    center_lat: float,
    meshsize: float,
    dem_grid: Optional[np.ndarray] = None,
    land_cover_grid: Optional[np.ndarray] = None,
    canopy_top: Optional[np.ndarray] = None,
    canopy_bottom: Optional[np.ndarray] = None,
    land_cover_source: str = "OpenStreetMap",
    trunk_height_ratio: Optional[float] = None,
    max_voxel_ram_mb: Optional[float] = None,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
) -> np.ndarray:
    """Voxelize CityGML meshes on a shared 3D grid.

    This fills terrain as a solid volume, voxelizes buildings/bridges
    as solids, and then overlays land cover and canopy voxels.

    Buildings and Bridges:
        All buildings and bridges passed in the collection are voxelized in full,
        including parts that extend outside the target rectangle. This ensures
        complete geometry representation even at areal boundaries.

    Args:
        occupancy_threshold: Minimum volume overlap fraction (0.0–1.0) for a
            boundary voxel to be kept.  0.0 (default) keeps every voxel that
            has *any* geometric contact (current behaviour).  E.g. 0.5 means
            a voxel must be at least 50 % filled by geometry.
        occupancy_subdivisions: Number of sub-divisions per axis when
            estimating the volume fraction (default 3 → 27 sub-samples).
    """
    gp, transformer = _compute_grid_params_3d(
        rectangle_vertices,
        center_lon,
        center_lat,
        meshsize,
        collection,
    )

    voxel_grid = _allocate_voxel_grid(gp, max_voxel_ram_mb=max_voxel_ram_mb)

    # Resize DEM early – it's used by land cover / canopy overlay later too.
    if dem_grid is not None:
        dem_grid = _resize_float_grid(dem_grid, gp.n_rows, gp.n_cols)

    # Terrain: build watertight extrusion solid from terrain meshes, then
    # voxelize via the same MeshLib / Numba paths used for buildings.
    # Falls back to per-column DEM fill when no terrain meshes exist.
    terrain_filled = False
    if collection.terrain:
        terrain_filled = _voxelize_terrain_solid(
            collection.terrain,
            transformer,
            gp,
            voxel_grid,
        )

    if not terrain_filled and dem_grid is not None:
        _fill_terrain_from_dem(voxel_grid, gp, dem_grid)
    elif terrain_filled and dem_grid is not None:
        # The terrain solid may leave gaps (rivers, missing tiles, failed
        # Boolean union).  Fill columns that are still empty using the DEM
        # as a safety-net — only writes to cells that are currently AIR (0).
        _fill_terrain_gaps_from_dem(voxel_grid, gp, dem_grid)

    # Buildings (watertight cascade → Z-scanline interior fill)
    _voxelize_mesh_group(
        collection.buildings,
        transformer,
        gp,
        voxel_grid,
        class_code=BUILDING_CODE,
        overwrite=True,
        occupancy_threshold=occupancy_threshold,
        occupancy_subdivisions=occupancy_subdivisions,
    )

    # Bridges – thin shell structures; always use surface + dilation + fill
    # rather than the watertight/Z-scanline path which under-fills thin slabs.
    _voxelize_mesh_group(
        collection.bridges,
        transformer,
        gp,
        voxel_grid,
        class_code=BUILDING_CODE,
        overwrite=True,
        occupancy_threshold=occupancy_threshold,
        occupancy_subdivisions=occupancy_subdivisions,
        force_surface=True,
    )

    # Vegetation
    _voxelize_mesh_group(
        collection.vegetation,
        transformer,
        gp,
        voxel_grid,
        class_code=TREE_CODE,
        overwrite=False,
        occupancy_threshold=occupancy_threshold,
        occupancy_subdivisions=occupancy_subdivisions,
    )

    # Land cover overlay (topmost terrain voxel)
    if land_cover_grid is not None and dem_grid is not None:
        land_cover_grid = _resize_int_grid(land_cover_grid, gp.n_rows, gp.n_cols)
        _apply_land_cover(voxel_grid, gp, land_cover_grid, dem_grid, land_cover_source)

    # Canopy overlay
    if canopy_top is not None and dem_grid is not None:
        canopy_top = _resize_float_grid(canopy_top, gp.n_rows, gp.n_cols)
        canopy_bottom = _resize_float_grid(canopy_bottom, gp.n_rows, gp.n_cols) if canopy_bottom is not None else None
        _apply_canopy(
            voxel_grid,
            gp,
            dem_grid,
            canopy_top,
            canopy_bottom,
            trunk_height_ratio=trunk_height_ratio,
        )

    return voxel_grid


def _compute_grid_params_3d(
    rectangle_vertices: List[Tuple[float, float]],
    center_lon: float,
    center_lat: float,
    meshsize: float,
    collection: CityGMLMeshCollection,
) -> Tuple[Grid3DParams, object]:
    transformer = create_local_transformer(center_lon, center_lat)

    rect_lon = [v[0] for v in rectangle_vertices]
    rect_lat = [v[1] for v in rectangle_vertices]
    rx, ry = transformer.transform(rect_lon, rect_lat)
    min_x, max_x = float(min(rx)), float(max(rx))
    min_y, max_y = float(min(ry)), float(max(ry))

    all_z = []
    for meshes in [collection.terrain, collection.buildings, collection.bridges, collection.vegetation]:
        for mesh in meshes:
            if len(mesh.vertices) == 0:
                continue
            all_z.append(mesh.vertices[:, 2])

    if all_z:
        z_min = float(np.min(np.concatenate(all_z)))
        z_max = float(np.max(np.concatenate(all_z)))
    else:
        z_min = 0.0
        z_max = meshsize

    z_min -= meshsize
    z_max += meshsize

    n_cols = max(1, int((max_x - min_x) / meshsize + 0.5))
    n_rows = max(1, int((max_y - min_y) / meshsize + 0.5))
    n_z = max(1, int((z_max - z_min) / meshsize + 0.5))

    gp = Grid3DParams(
        n_rows=n_rows,
        n_cols=n_cols,
        n_z=n_z,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=z_min,
        max_z=z_max,
        voxel_size=float(meshsize),
    )
    return gp, transformer


def _allocate_voxel_grid(gp: Grid3DParams, max_voxel_ram_mb: Optional[float]) -> np.ndarray:
    bytes_per = np.dtype(np.int16).itemsize
    est_mb = gp.n_rows * gp.n_cols * gp.n_z * bytes_per / (1024 ** 2)
    _log.info("3D voxel grid: (%d, %d, %d) ~%.1f MB", gp.n_rows, gp.n_cols, gp.n_z, est_mb)
    if max_voxel_ram_mb is not None and est_mb > max_voxel_ram_mb:
        raise MemoryError(
            f"Estimated voxel grid memory {est_mb:.1f} MB exceeds limit {max_voxel_ram_mb} MB."
        )
    return np.zeros((gp.n_rows, gp.n_cols, gp.n_z), dtype=np.int16)


# ── Terrain solid voxelization ─────────────────────────────────────────

def _voxelize_terrain_solid(
    terrain_meshes: List[Mesh3D],
    transformer,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
) -> bool:
    """Build a watertight terrain extrusion solid and voxelize it.

    Pipeline (mirrors citygml_mesher.solid):
        1. Merge terrain tiles
        2. Remove degenerate triangles
        3. Weld duplicate vertices
        4. Fix non-manifold edges / split pinch points
        5. Extrude boundary downward + bottom cap → watertight solid
        6. Voxelize solid via MeshLib level-set / winding or Numba Z-scanline

    Returns *True* if voxels were successfully written, *False* otherwise.
    """
    # Transform terrain vertices to local metres
    local_meshes: List[Mesh3D] = []
    for mesh in terrain_meshes:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        verts_ll = swap_coordinates_3d(mesh.vertices)
        x_m, y_m = transformer.transform(verts_ll[:, 0], verts_ll[:, 1])
        local_verts = np.column_stack([x_m, y_m, verts_ll[:, 2]])
        local_meshes.append(Mesh3D(
            vertices=local_verts,
            faces=mesh.faces.copy(),
            feature_type=mesh.feature_type,
            feature_id=mesh.feature_id,
        ))

    if not local_meshes:
        return False

    # Bottom Z: one voxel below the grid floor
    bottom_z = gp.min_z

    solid, stats = build_terrain_solid(
        local_meshes,
        bottom_z=bottom_z,
        weld_tolerance=gp.voxel_size * 1e-3,
        grid_bounds=(gp.min_x, gp.max_x, gp.min_y, gp.max_y),
        voxel_size=gp.voxel_size,
        verbose=True,
    )
    if solid is None or len(solid.faces) == 0:
        _log.warning("Terrain solid construction failed – falling back to DEM.")
        return False

    _log.info(
        "Terrain solid ready: %d verts, %d faces, watertight=%s",
        len(solid.vertices), len(solid.faces), stats.is_watertight,
    )

    verts = solid.vertices.copy()
    faces = solid.faces

    # Shift the solid down by half a voxel to compensate for the MeshLib
    # level-set SDF centre-sampling bias: a voxel is marked "inside" when
    # its centre lies inside the solid, which consistently places the
    # topmost ground voxel ~1 index too high.
    verts[:, 2] -= 0.5 * gp.voxel_size

    # ── MeshLib path ──────────────────────────────────────────────────
    if _MESHLIB_VOXEL_AVAILABLE:
        if stats.is_watertight:
            ok = _voxelize_meshlib_levelset(
                verts, faces, gp, voxel_grid, GROUND_CODE, overwrite=False,
            )
            if ok:
                _log.info("Terrain voxelized via MeshLib level-set.")
                return True
        # Fallback: winding number (works even if not perfectly watertight)
        ok = _voxelize_meshlib_winding(
            verts, faces, gp, voxel_grid, GROUND_CODE, overwrite=False,
        )
        if ok:
            _log.info("Terrain voxelized via MeshLib winding number.")
            return True

    # ── Legacy Numba path ─────────────────────────────────────────────
    _voxelize_single_mesh(
        verts, faces, gp, voxel_grid,
        class_code=GROUND_CODE,
        overwrite=False,
        seal_surface=False,  # solid is already closed, no dilation needed
    )
    _log.info("Terrain voxelized via Numba Z-scanline.")
    return True


def _fill_terrain_from_dem(voxel_grid: np.ndarray, gp: Grid3DParams, dem_grid: np.ndarray) -> None:
    ground_levels = np.rint((dem_grid - gp.min_z) / gp.voxel_size).astype(np.intp)
    ground_levels = np.clip(ground_levels, -1, gp.n_z - 1)
    # Build a z-index array and compare against ground_levels via broadcasting
    z_indices = np.arange(gp.n_z, dtype=np.intp)
    mask = z_indices[np.newaxis, np.newaxis, :] <= ground_levels[:, :, np.newaxis]
    # Only fill where ground_level >= 0
    valid = ground_levels >= 0
    mask &= valid[:, :, np.newaxis]
    voxel_grid[mask] = GROUND_CODE


def _fill_terrain_gaps_from_dem(
    voxel_grid: np.ndarray, gp: Grid3DParams, dem_grid: np.ndarray,
) -> None:
    """Fill columns that have no ground voxels using DEM elevation.

    After terrain-solid voxelization, some columns may remain empty due
    to river channels, missing tiles, or a failed Boolean union with the
    base box.  This function identifies those gap columns and fills them
    up to the DEM surface, leaving columns already populated by the
    terrain solid untouched.
    """
    has_ground = np.any(voxel_grid == GROUND_CODE, axis=2)
    n_gaps = int((~has_ground).sum())
    if n_gaps == 0:
        return

    ground_levels = np.rint((dem_grid - gp.min_z) / gp.voxel_size).astype(np.intp)
    ground_levels = np.clip(ground_levels, -1, gp.n_z - 1)

    z_indices = np.arange(gp.n_z, dtype=np.intp)
    fill_mask = z_indices[np.newaxis, np.newaxis, :] <= ground_levels[:, :, np.newaxis]
    valid = ground_levels >= 0
    fill_mask &= valid[:, :, np.newaxis]
    # Only fill columns that have NO existing ground voxels
    fill_mask &= (~has_ground)[:, :, np.newaxis]

    voxel_grid[fill_mask] = GROUND_CODE
    _log.info("  Terrain DEM gap-fill: filled %d empty columns", n_gaps)


# ── MeshLib-based voxelization ────────────────────────────────────────

def _meshlib_mesh_from_numpy(verts: np.ndarray, faces: np.ndarray):
    """Convert numpy verts & faces to a MeshLib Mesh in a shifted local frame.

    MeshLib's internal Vector3f is float32, so verts are cast to float32. To
    avoid precision loss when the local-metre frame is far from zero (~10 cm
    error at ±1e5 m), the bbox-min is subtracted first so float32 values stay
    near the origin where precision is sub-millimeter.

    Returns ``(mesh, shift)`` where *shift* is a float64 (3,) array that the
    caller must add to any MeshLib-frame coordinate (e.g. SDF origin, bbox
    min) to recover the original world coordinates.
    """
    verts_f64 = np.ascontiguousarray(verts, dtype=np.float64)
    shift = verts_f64.min(axis=0)
    verts_local = (verts_f64 - shift).astype(np.float32, copy=False)
    mesh = _mrnp.meshFromFacesVerts(
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(verts_local),
    )
    return mesh, shift


def _voxelize_meshlib_levelset(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
) -> bool:
    """Voxelize a *watertight* mesh via MeshLib's OpenVDB signed level set.

    ``meshToVolume(Signed)`` computes a narrow-band signed distance field
    using OpenVDB's spatial hash tree.  Interior voxels have distance ≤ 0.
    This replaces the Z-scanline ray-parity approach with an O(F log F)
    algorithm that also encodes the surface shell in one pass.

    Returns True on success, False if the mesh cannot be voxelized this way.
    """
    try:
        ml_mesh, shift = _meshlib_mesh_from_numpy(verts, faces)
        vs = float(gp.voxel_size)

        params = _mr.MeshToVolumeParams()
        params.type = _mr.MeshToVolumeParams.Type.Signed
        params.surfaceOffset = 3  # narrow-band half-width in voxels
        params.voxelSize = _mr.Vector3f.diagonal(vs)
        out_xf = _mr.AffineXf3f()
        params.outXf = out_xf

        vdb_vol = _mr.meshToVolume(_mr.MeshPart(ml_mesh), params)

        # Convert VDB → numpy 3-D array of signed distances
        simple_vol = _mr.vdbVolumeToSimpleVolume(vdb_vol)
        sdf = _mrnp.getNumpy3Darray(simple_vol)  # shape (dx, dy, dz)

        # Origin of the VDB grid in world (local-metre) coordinates: the
        # shifted-frame origin from MeshLib plus the pre-shift applied in
        # _meshlib_mesh_from_numpy.
        origin = np.array([out_xf.b.x, out_xf.b.y, out_xf.b.z], dtype=np.float64) + shift

        inside_mask = sdf <= 0.0  # negative = inside the solid

        # The narrow-band SDF may leave far-interior voxels at the
        # background value (+).  Flood-fill guarantees a solid.
        inside_mask = binary_fill_holes(inside_mask)

        # Map SDF voxel indices → main voxel-grid indices
        _stamp_meshlib_mask(
            inside_mask, origin, vs, gp, voxel_grid, class_code, overwrite,
        )
        return True
    except Exception as exc:
        _log.debug("MeshLib levelset voxelization failed: %s", exc)
        return False


def _voxelize_meshlib_winding(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
) -> bool:
    """Voxelize a mesh (possibly open) via MeshLib's Generalized Winding Number.

    ``meshToDistanceVolume`` with ``SignDetectionMode.HoleWindingRule``
    robustly classifies inside/outside even for meshes with holes, gaps,
    or self-intersections.  This replaces the surface + dilation +
    flood-fill fallback.

    Returns True on success.
    """
    try:
        ml_mesh, shift = _meshlib_mesh_from_numpy(verts, faces)
        vs = float(gp.voxel_size)

        box = ml_mesh.computeBoundingBox()
        expansion = _mr.Vector3f.diagonal(3 * vs)

        params = _mr.MeshToDistanceVolumeParams()
        params.vol.origin = box.min - expansion
        params.vol.voxelSize = _mr.Vector3f.diagonal(vs)
        dim_f = (box.max + expansion - params.vol.origin) / vs
        params.vol.dimensions = _mr.Vector3i(
            int(dim_f.x) + 1, int(dim_f.y) + 1, int(dim_f.z) + 1,
        )
        params.dist.signMode = _mr.SignDetectionMode.HoleWindingRule
        params.dist.maxDistSq = (3 * vs) ** 2

        simple_vol = _mr.meshToDistanceVolume(_mr.MeshPart(ml_mesh), params)
        sdf = _mrnp.getNumpy3Darray(simple_vol)  # may have NaN outside band

        # SDF origin in world (local-metre) coordinates: shifted-frame
        # origin plus the pre-shift applied in _meshlib_mesh_from_numpy.
        origin = np.array(
            [params.vol.origin.x, params.vol.origin.y, params.vol.origin.z],
            dtype=np.float64,
        ) + shift

        # Inside = distance ≤ 0 (NaN is outside the narrow band)
        inside_mask = np.nan_to_num(sdf, nan=1.0) <= 0.0

        # The narrow band (maxDistSq) only computes SDF within a few
        # voxels of the surface.  Far-interior voxels are NaN → False.
        # Flood-fill closes the enclosed interior cheaply.
        inside_mask = binary_fill_holes(inside_mask)

        _stamp_meshlib_mask(
            inside_mask, origin, vs, gp, voxel_grid, class_code, overwrite,
        )
        return True
    except Exception as exc:
        _log.debug("MeshLib winding voxelization failed: %s", exc)
        return False


def _stamp_meshlib_mask(
    inside_mask: np.ndarray,
    origin: np.ndarray,
    vs: float,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
) -> None:
    """Write a MeshLib boolean mask into the shared voxel grid.

    *origin* is the (x, y, z) position of the (0,0,0) corner of
    *inside_mask* in local-metre space.  The MeshLib SDF array is
    indexed as (ix, iy, iz) with +X, +Y, +Z axis order, whereas the
    main voxel grid is (row, col, z) with row=0 at max_y (north).
    """
    src_dx, src_dy, src_dz = inside_mask.shape
    if src_dx == 0 or src_dy == 0 or src_dz == 0:
        return

    # For each active voxel in the SDF grid, compute the corresponding
    # (row, col, z) in the main grid.
    ix, iy, iz = np.nonzero(inside_mask)
    if len(ix) == 0:
        return

    # SDF grid centre coords in local metres
    x_m = origin[0] + (ix.astype(np.float64) + 0.5) * vs
    y_m = origin[1] + (iy.astype(np.float64) + 0.5) * vs
    z_m = origin[2] + (iz.astype(np.float64) + 0.5) * vs

    # Main-grid indices
    col = ((x_m - gp.min_x) / vs).astype(np.intp)
    row = ((gp.max_y - y_m) / vs).astype(np.intp)
    zi  = ((z_m - gp.min_z) / vs).astype(np.intp)

    # Clip to grid bounds
    valid = (
        (row >= 0) & (row < gp.n_rows) &
        (col >= 0) & (col < gp.n_cols) &
        (zi >= 0)  & (zi < gp.n_z)
    )
    row, col, zi = row[valid], col[valid], zi[valid]

    if overwrite:
        voxel_grid[row, col, zi] = class_code
    else:
        empty = voxel_grid[row, col, zi] == 0
        voxel_grid[row[empty], col[empty], zi[empty]] = class_code


def _overlay_surface_shell(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: "Grid3DParams",
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
) -> None:
    """Stamp the triangle surface shell onto the voxel grid via SAT.

    This guarantees that every voxel touching an original mesh triangle
    is marked, even when the SDF discretisation misses thin walls whose
    thickness is smaller than the voxel size.

    When *occupancy_threshold* > 0, boundary voxels whose volume overlap
    fraction is below the threshold are discarded.  This controls how
    "inclusive" the surface shell is.
    """
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    r0, r1, c0, c1, z0, z1 = _bbox_to_index_range(gp, vmin, vmax)
    if r0 > r1 or c0 > c1 or z0 > z1:
        return
    nr, nc, nz = r1 - r0 + 1, c1 - c0 + 1, z1 - z0 + 1
    half_eps = gp.voxel_size * 1e-6
    half = np.array([gp.voxel_size / 2.0 + half_eps] * 3, dtype=np.float64)
    verts_f64 = np.ascontiguousarray(verts, dtype=np.float64)
    faces_ip = np.ascontiguousarray(faces, dtype=np.intp)
    surface = _surface_voxelize_numba(
        verts_f64, faces_ip,
        gp.min_x, gp.max_y, gp.min_z, gp.voxel_size,
        r0, r1, c0, c1, z0, z1,
        nr, nc, nz, half,
    )
    if occupancy_threshold > 0.0:
        surface = _filter_surface_by_occupancy(
            surface, verts_f64, faces_ip, gp,
            r0, c0, z0, nr, nc, nz,
            occupancy_threshold, occupancy_subdivisions,
        )

    # Only keep surface voxels that are 6-connected neighbours of an
    # already-filled voxel in the main grid.  This prevents stray /
    # disconnected mesh fragments from creating floating artifacts.
    subgrid = voxel_grid[r0:r1 + 1, c0:c1 + 1, z0:z1 + 1]
    existing = subgrid != 0  # any non-empty voxel counts as anchor
    adjacent = _dilate6(existing)
    surface &= adjacent

    if overwrite:
        subgrid[surface] = class_code
    else:
        mask = surface & (subgrid == 0)
        subgrid[mask] = class_code


# ── Dispatcher ────────────────────────────────────────────────────────

def _voxelize_mesh_group(
    meshes: List[Mesh3D],
    transformer,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
    force_surface: bool = False,
) -> None:
    """Voxelize a group of meshes (buildings, bridges, or vegetation).

    All meshes in the collection are voxelized in full, including parts
    that extend outside the voxelization grid.  For buildings and bridges,
    this ensures complete geometry representation even at areal boundaries.

    Strategy (ordered by preference):

    **MeshLib path** (when ``meshlib`` is installed):
        1. Watertight cascade → ``meshToVolume(Signed)`` (OpenVDB level-set).
        2. Fallback           → ``meshToDistanceVolume(HoleWindingRule)``
           (robust winding number, works for open/self-intersecting meshes).

    **Legacy Numba path** (when ``meshlib`` is not available):
        1. Watertight cascade → Z-scanline ray-parity interior fill.
        2. Fallback           → surface SAT rasterization + dilation +
           flood-fill.

    Args:
        force_surface: If True, always use surface + dilation + fill
            (legacy) or winding-number (MeshLib) instead of the closed-
            mesh path.  Recommended for thin shell structures (bridges).
    """
    if not meshes:
        return

    for mesh in meshes:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        verts_ll = swap_coordinates_3d(mesh.vertices)
        x_m, y_m = transformer.transform(verts_ll[:, 0], verts_ll[:, 1])
        verts = np.column_stack([x_m, y_m, verts_ll[:, 2]])

        faces = mesh.faces

        # ── Buildings / Bridges (solid via watertight cascade) ────────
        if class_code == BUILDING_CODE and not force_surface:
            # Try MeshLib signed level-set (fastest, requires watertight)
            if _MESHLIB_VOXEL_AVAILABLE:
                wt = make_watertight_mesh(verts, faces, voxel_size=gp.voxel_size)
                if wt.is_watertight and len(wt.faces) > 0 and len(wt.vertices) > 0:
                    ok = _voxelize_meshlib_levelset(
                        wt.vertices, wt.faces, gp, voxel_grid,
                        class_code, overwrite,
                    )
                    if ok:
                        # Union with watertight-mesh surface shell so that
                        # thin walls (< 1 voxel) are never missed.
                        # Use wt (not raw verts) to avoid stray-triangle
                        # artifacts from the uncleaned original mesh.
                        _overlay_surface_shell(
                            wt.vertices, wt.faces, gp, voxel_grid,
                            class_code, overwrite,
                            occupancy_threshold=occupancy_threshold,
                            occupancy_subdivisions=occupancy_subdivisions,
                        )
                        continue
                # Watertight failed → winding number (open-mesh robust)
                ok = _voxelize_meshlib_winding(
                    verts, faces, gp, voxel_grid,
                    class_code, overwrite,
                )
                if ok:
                    _overlay_surface_shell(
                        verts, faces, gp, voxel_grid,
                        class_code, overwrite,
                        occupancy_threshold=occupancy_threshold,
                        occupancy_subdivisions=occupancy_subdivisions,
                    )
                    continue

            # Legacy Numba path
            wt = make_watertight_mesh(verts, faces, voxel_size=gp.voxel_size)
            if wt.is_watertight and len(wt.faces) > 0 and len(wt.vertices) > 0:
                ok = _voxelize_by_occupancy(
                    wt.vertices, wt.faces, gp, voxel_grid,
                    class_code, overwrite,
                    occupancy_threshold=occupancy_threshold,
                    occupancy_subdivisions=occupancy_subdivisions,
                )
                if ok:
                    continue
            _voxelize_single_mesh(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                seal_surface=True,
                occupancy_threshold=occupancy_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
            )

        # ── Bridges (force_surface) ──────────────────────────────────
        # Bridges are thin-shell structures (deck slabs, guard rails)
        # with negligible interior volume.  The MeshLib winding-number
        # SDF never goes negative inside a thin shell, so it produces
        # empty output.  Always use the legacy surface + dilation +
        # flood-fill path which was designed for this case.
        elif class_code == BUILDING_CODE and force_surface:
            _voxelize_single_mesh(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                seal_surface=True,
                occupancy_threshold=occupancy_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
            )

        # ── Vegetation / other (surface only) ────────────────────────
        else:
            if _MESHLIB_VOXEL_AVAILABLE:
                ok = _voxelize_meshlib_winding(
                    verts, faces, gp, voxel_grid,
                    class_code, overwrite,
                )
                if ok:
                    # Union with surface shell so leaves / branches thinner
                    # than one voxel are not lost by the narrow-band SDF.
                    _overlay_surface_shell(
                        verts, faces, gp, voxel_grid,
                        class_code, overwrite,
                        occupancy_threshold=occupancy_threshold,
                        occupancy_subdivisions=occupancy_subdivisions,
                    )
                    continue
            _voxelize_single_mesh(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                seal_surface=False,
                occupancy_threshold=occupancy_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
            )


def _voxelize_by_occupancy(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
) -> bool:
    """Mark voxels interior to a watertight mesh using Z-scanline ray parity.

    For each (row, col) grid column, cast a ray in the +Z direction through
    all mesh triangles.  Sort intersection Z values and use even–odd parity
    to classify voxel Z-indices as inside or outside.

    Falls back to surface voxelization if the mesh is not a valid volume.

    Returns True on success, False if the mesh is not watertight.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()
    if not mesh.is_volume:
        return False

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    r0, r1, c0, c1, z0, z1 = _bbox_to_index_range(gp, vmin, vmax)
    if r0 > r1 or c0 > c1 or z0 > z1:
        return True

    vs = gp.voxel_size
    nr, nc, nz = r1 - r0 + 1, c1 - c0 + 1, z1 - z0 + 1

    tri_verts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    tri_faces = np.ascontiguousarray(mesh.faces, dtype=np.intp)

    inside_3d = _z_scanline_fill(
        tri_verts, tri_faces,
        gp.min_x, gp.max_y, gp.min_z, vs,
        r0, c0, z0, nr, nc, nz,
    )

    # Also compute surface shell via triangle-box overlap and union with
    # interior.  This guarantees boundary voxels are always present even
    # if ray-parity fails for some columns (edge/vertex degeneracy).
    half_eps = vs * 1e-6
    half = np.array([vs / 2.0 + half_eps] * 3, dtype=np.float64)
    surface = _surface_voxelize_numba(
        tri_verts, tri_faces,
        gp.min_x, gp.max_y, gp.min_z, vs,
        r0, r1, c0, c1, z0, z1,
        nr, nc, nz, half,
    )
    if occupancy_threshold > 0.0:
        surface = _filter_surface_by_occupancy(
            surface, tri_verts, tri_faces, gp,
            r0, c0, z0, nr, nc, nz,
            occupancy_threshold, occupancy_subdivisions,
        )
    inside_3d |= surface

    # ---- Write to voxel grid --------------------------------------------
    subgrid = voxel_grid[r0:r1 + 1, c0:c1 + 1, z0:z1 + 1]
    if overwrite:
        subgrid[inside_3d] = class_code
    else:
        mask = inside_3d & (subgrid == 0)
        subgrid[mask] = class_code

    return True


@njit(cache=True)
def _ray_z_triangle(
    ox: float, oy: float,
    v0x: float, v0y: float, v0z: float,
    v1x: float, v1y: float, v1z: float,
    v2x: float, v2y: float, v2z: float,
) -> float:
    """Intersect a +Z ray at (ox, oy) with a triangle, return Z or NaN."""
    # Edges in XY
    e1x = v1x - v0x;  e1y = v1y - v0y
    e2x = v2x - v0x;  e2y = v2y - v0y
    det = e1x * e2y - e1y * e2x
    if abs(det) < 1e-16:
        return np.nan
    inv_det = 1.0 / det
    dx = ox - v0x;  dy = oy - v0y
    u = (dx * e2y - dy * e2x) * inv_det
    if u < 0.0 or u > 1.0:
        return np.nan
    v = (e1x * dy - e1y * dx) * inv_det
    if v < 0.0 or u + v > 1.0:
        return np.nan
    # Z at intersection
    e1z = v1z - v0z;  e2z = v2z - v0z
    return v0z + u * e1z + v * e2z


@njit(cache=True, parallel=True)
def _z_scanline_fill(
    verts: np.ndarray,
    faces: np.ndarray,
    min_x: float, max_y: float, min_z: float, vs: float,
    r0: int, c0: int, z0: int,
    nr: int, nc: int, nz: int,
) -> np.ndarray:
    """Fill inside a watertight mesh using Z-scanline parity, parallelised over rows."""
    inside = np.zeros((nr, nc, nz), dtype=np.bool_)
    n_faces = len(faces)

    # Pre-compute per-triangle XY bounding boxes for fast culling
    tri_xmin = np.empty(n_faces, dtype=np.float64)
    tri_xmax = np.empty(n_faces, dtype=np.float64)
    tri_ymin = np.empty(n_faces, dtype=np.float64)
    tri_ymax = np.empty(n_faces, dtype=np.float64)
    for fi in range(n_faces):
        f0 = faces[fi, 0]; f1 = faces[fi, 1]; f2 = faces[fi, 2]
        x0 = verts[f0, 0]; x1 = verts[f1, 0]; x2 = verts[f2, 0]
        y0 = verts[f0, 1]; y1 = verts[f1, 1]; y2 = verts[f2, 1]
        tri_xmin[fi] = min(x0, x1, x2)
        tri_xmax[fi] = max(x0, x1, x2)
        tri_ymin[fi] = min(y0, y1, y2)
        tri_ymax[fi] = max(y0, y1, y2)

    # Small deterministic jitter to avoid exact edge/vertex alignment
    jx = vs * 0.00137
    jy = vs * 0.00098
    eps = vs * 1e-6  # Tolerance for merging coincident Z-hits

    for ri in prange(nr):
        ray_y = max_y - (r0 + ri + 0.5) * vs + jy
        # Scratch buffers (reused per column)
        z_hits = np.empty(n_faces, dtype=np.float64)
        z_dedup = np.empty(n_faces, dtype=np.float64)
        for ci in range(nc):
            ray_x = min_x + (c0 + ci + 0.5) * vs + jx
            n_hits = 0
            for fi in range(n_faces):
                # Quick XY bounding-box cull
                if ray_x < tri_xmin[fi] or ray_x > tri_xmax[fi]:
                    continue
                if ray_y < tri_ymin[fi] or ray_y > tri_ymax[fi]:
                    continue
                f0 = faces[fi, 0]; f1 = faces[fi, 1]; f2 = faces[fi, 2]
                z_val = _ray_z_triangle(
                    ray_x, ray_y,
                    verts[f0, 0], verts[f0, 1], verts[f0, 2],
                    verts[f1, 0], verts[f1, 1], verts[f1, 2],
                    verts[f2, 0], verts[f2, 1], verts[f2, 2],
                )
                if not np.isnan(z_val):
                    z_hits[n_hits] = z_val
                    n_hits += 1

            if n_hits < 2:
                continue

            # Sort Z-hits (insertion sort – n_hits is typically small)
            for i in range(1, n_hits):
                key = z_hits[i]
                j = i - 1
                while j >= 0 and z_hits[j] > key:
                    z_hits[j + 1] = z_hits[j]
                    j -= 1
                z_hits[j + 1] = key

            # Deduplicate near-coincident hits (edge/vertex degeneracy).
            # Group consecutive hits within eps; keep one if odd count,
            # remove all if even (they cancel in parity).
            n_dedup = 0
            i = 0
            while i < n_hits:
                j = i + 1
                while j < n_hits and z_hits[j] - z_hits[i] < eps:
                    j += 1
                group_size = j - i
                if group_size % 2 == 1:
                    z_dedup[n_dedup] = z_hits[i]
                    n_dedup += 1
                # Even group: discard all (they cancel out in parity)
                i = j

            if n_dedup < 2:
                continue

            # Even-odd parity: between hits[0]→hits[1] is inside,
            # hits[2]→hits[3] is inside, etc.
            for pair in range(n_dedup // 2):
                z_lo = z_dedup[pair * 2]
                z_hi = z_dedup[pair * 2 + 1]
                zi_lo = int(np.floor((z_lo - min_z) / vs)) - z0
                zi_hi = int(np.floor((z_hi - min_z) / vs)) - z0
                zi_lo = max(0, zi_lo)
                zi_hi = min(nz - 1, zi_hi)
                for zi in range(zi_lo, zi_hi + 1):
                    inside[ri, ci, zi] = True

    return inside


def _voxelize_single_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    seal_surface: bool,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
) -> None:
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    r0, r1, c0, c1, z0, z1 = _bbox_to_index_range(gp, vmin, vmax)
    if r0 > r1 or c0 > c1 or z0 > z1:
        return

    sub_rows = r1 - r0 + 1
    sub_cols = c1 - c0 + 1
    sub_z = z1 - z0 + 1

    half_eps = gp.voxel_size * 1e-6
    half = np.array([gp.voxel_size / 2.0 + half_eps] * 3, dtype=np.float64)

    # Call the Numba-compiled surface voxelization kernel
    surface = _surface_voxelize_numba(
        verts.astype(np.float64),
        faces.astype(np.intp),
        gp.min_x, gp.max_y, gp.min_z, gp.voxel_size,
        r0, r1, c0, c1, z0, z1,
        sub_rows, sub_cols, sub_z,
        half,
    )

    if occupancy_threshold > 0.0:
        surface = _filter_surface_by_occupancy(
            surface,
            verts.astype(np.float64),
            faces.astype(np.intp),
            gp, r0, c0, z0,
            sub_rows, sub_cols, sub_z,
            occupancy_threshold, occupancy_subdivisions,
        )

    if seal_surface:
        surface = _dilate6(surface)

    occupancy = _fill_interior(surface)

    subgrid = voxel_grid[r0:r1 + 1, c0:c1 + 1, z0:z1 + 1]
    if overwrite:
        subgrid[occupancy] = class_code
    else:
        mask = occupancy & (subgrid == 0)
        subgrid[mask] = class_code


@njit(cache=True)
def _triangle_box_overlap_nb(
    box_center_0: float, box_center_1: float, box_center_2: float,
    half_0: float, half_1: float, half_2: float,
    tv0: float, tv1: float, tv2: float,
    tv3: float, tv4: float, tv5: float,
    tv6: float, tv7: float, tv8: float,
) -> bool:
    """Numba-JIT triangle-box overlap (SAT, Akenine-Möller)."""
    # Translate to box-center-at-origin
    v0x = tv0 - box_center_0
    v0y = tv1 - box_center_1
    v0z = tv2 - box_center_2
    v1x = tv3 - box_center_0
    v1y = tv4 - box_center_1
    v1z = tv5 - box_center_2
    v2x = tv6 - box_center_0
    v2y = tv7 - box_center_1
    v2z = tv8 - box_center_2

    # Edge vectors
    e0x = v1x - v0x; e0y = v1y - v0y; e0z = v1z - v0z
    e1x = v2x - v1x; e1y = v2y - v1y; e1z = v2z - v1z
    e2x = v0x - v2x; e2y = v0y - v2y; e2z = v0z - v2z

    # AABB overlap
    for i in range(3):
        if i == 0:
            mn = min(v0x, v1x, v2x); mx = max(v0x, v1x, v2x); h = half_0
        elif i == 1:
            mn = min(v0y, v1y, v2y); mx = max(v0y, v1y, v2y); h = half_1
        else:
            mn = min(v0z, v1z, v2z); mx = max(v0z, v1z, v2z); h = half_2
        if mn > h or mx < -h:
            return False

    # Triangle normal
    nx = e0y * e1z - e0z * e1y
    ny = e0z * e1x - e0x * e1z
    nz = e0x * e1y - e0y * e1x
    d = -(nx * v0x + ny * v0y + nz * v0z)
    r = abs(nx) * half_0 + abs(ny) * half_1 + abs(nz) * half_2
    if abs(d) > r:
        return False

    # 9 edge cross-product axes
    edges_x = (e0x, e1x, e2x)
    edges_y = (e0y, e1y, e2y)
    edges_z = (e0z, e1z, e2z)
    for j in range(3):
        ex = edges_x[j]; ey = edges_y[j]; ez = edges_z[j]
        for i in range(3):
            if i == 0:
                ax = 0.0; ay = -ez; az = ey
            elif i == 1:
                ax = ez; ay = 0.0; az = -ex
            else:
                ax = -ey; ay = ex; az = 0.0
            if ax * ax + ay * ay + az * az < 1e-20:
                continue
            p0 = ax * v0x + ay * v0y + az * v0z
            p1 = ax * v1x + ay * v1y + az * v1z
            p2 = ax * v2x + ay * v2y + az * v2z
            mn = min(p0, p1, p2)
            mx = max(p0, p1, p2)
            r = abs(ax) * half_0 + abs(ay) * half_1 + abs(az) * half_2
            if mn > r or mx < -r:
                return False

    return True


@njit(cache=True, parallel=True)
def _surface_voxelize_numba(
    verts: np.ndarray,
    faces: np.ndarray,
    min_x: float, max_y: float, min_z: float, voxel_size: float,
    r0: int, r1: int, c0: int, c1: int, z0: int, z1: int,
    sub_rows: int, sub_cols: int, sub_z: int,
    half: np.ndarray,
) -> np.ndarray:
    """Rasterize triangle faces onto a voxel sub-grid (parallel over faces)."""
    surface = np.zeros((sub_rows, sub_cols, sub_z), dtype=np.bool_)
    h0 = half[0]; h1 = half[1]; h2 = half[2]

    for fi in prange(len(faces)):
        f0 = faces[fi, 0]; f1 = faces[fi, 1]; f2 = faces[fi, 2]
        tv0 = verts[f0, 0]; tv1 = verts[f0, 1]; tv2 = verts[f0, 2]
        tv3 = verts[f1, 0]; tv4 = verts[f1, 1]; tv5 = verts[f1, 2]
        tv6 = verts[f2, 0]; tv7 = verts[f2, 1]; tv8 = verts[f2, 2]

        # Triangle AABB
        tmin_x = min(tv0, tv3, tv6); tmax_x = max(tv0, tv3, tv6)
        tmin_y = min(tv1, tv4, tv7); tmax_y = max(tv1, tv4, tv7)
        tmin_z = min(tv2, tv5, tv8); tmax_z = max(tv2, tv5, tv8)

        # Convert triangle AABB to grid indices
        tc0 = int(np.floor((tmin_x - min_x) / voxel_size)) - 1
        tc1 = int(np.floor((tmax_x - min_x) / voxel_size)) + 1
        tr0 = int(np.floor((max_y - tmax_y) / voxel_size)) - 1
        tr1 = int(np.floor((max_y - tmin_y) / voxel_size)) + 1
        tz0 = int(np.floor((tmin_z - min_z) / voxel_size)) - 1
        tz1 = int(np.floor((tmax_z - min_z) / voxel_size)) + 1

        # Clip to subgrid bounds
        lr0 = max(r0, tr0); lr1 = min(r1, tr1)
        lc0 = max(c0, tc0); lc1 = min(c1, tc1)
        lz0 = max(z0, tz0); lz1 = min(z1, tz1)

        if lr0 > lr1 or lc0 > lc1 or lz0 > lz1:
            continue

        for r in range(lr0, lr1 + 1):
            cy = max_y - (r + 0.5) * voxel_size
            for c in range(lc0, lc1 + 1):
                cx = min_x + (c + 0.5) * voxel_size
                for z in range(lz0, lz1 + 1):
                    cz = min_z + (z + 0.5) * voxel_size
                    if _triangle_box_overlap_nb(
                        cx, cy, cz, h0, h1, h2,
                        tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8,
                    ):
                        surface[r - r0, c - c0, z - z0] = True

    return surface


@njit(cache=True, parallel=True)
def _compute_occupancy_fraction(
    surface: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    min_x: float, max_y: float, min_z: float, vs: float,
    r0: int, c0: int, z0: int,
    nr: int, nc: int, nz: int,
    subdivisions: int,
) -> np.ndarray:
    """Compute volume overlap fraction for each surface voxel via sub-voxel sampling.

    Each marked surface voxel is subdivided into ``subdivisions**3`` sub-cubes.
    For every sub-cube we test whether at least one mesh triangle overlaps it
    (using the SAT triangle-box test).  The returned fraction is the ratio of
    overlapping sub-cubes to the total number of sub-cubes.
    """
    fractions = np.zeros((nr, nc, nz), dtype=np.float64)
    n_faces = len(faces)
    sub_vs = vs / subdivisions
    sub_half = sub_vs / 2.0
    total_subs = float(subdivisions * subdivisions * subdivisions)

    # Pre-compute per-triangle AABBs for fast culling
    tri_xmin = np.empty(n_faces, dtype=np.float64)
    tri_xmax = np.empty(n_faces, dtype=np.float64)
    tri_ymin = np.empty(n_faces, dtype=np.float64)
    tri_ymax = np.empty(n_faces, dtype=np.float64)
    tri_zmin = np.empty(n_faces, dtype=np.float64)
    tri_zmax = np.empty(n_faces, dtype=np.float64)
    for fi in range(n_faces):
        i0 = faces[fi, 0]; i1 = faces[fi, 1]; i2 = faces[fi, 2]
        tri_xmin[fi] = min(verts[i0, 0], verts[i1, 0], verts[i2, 0])
        tri_xmax[fi] = max(verts[i0, 0], verts[i1, 0], verts[i2, 0])
        tri_ymin[fi] = min(verts[i0, 1], verts[i1, 1], verts[i2, 1])
        tri_ymax[fi] = max(verts[i0, 1], verts[i1, 1], verts[i2, 1])
        tri_zmin[fi] = min(verts[i0, 2], verts[i1, 2], verts[i2, 2])
        tri_zmax[fi] = max(verts[i0, 2], verts[i1, 2], verts[i2, 2])

    for ri in prange(nr):
        for ci in range(nc):
            for zi in range(nz):
                if not surface[ri, ci, zi]:
                    continue

                # Absolute voxel bounds
                abs_c = c0 + ci
                abs_r = r0 + ri
                abs_z = z0 + zi
                vox_min_x = min_x + abs_c * vs
                vox_max_x = vox_min_x + vs
                vox_max_y = max_y - abs_r * vs
                vox_min_y = vox_max_y - vs
                vox_min_z = min_z + abs_z * vs
                vox_max_z = vox_min_z + vs

                count = 0
                for si in range(subdivisions):
                    sx = vox_min_x + (si + 0.5) * sub_vs
                    for sj in range(subdivisions):
                        sy = vox_min_y + (sj + 0.5) * sub_vs
                        for sk in range(subdivisions):
                            sz = vox_min_z + (sk + 0.5) * sub_vs

                            hit = False
                            for fi in range(n_faces):
                                # Cull triangles outside the parent voxel
                                if tri_xmax[fi] < vox_min_x or tri_xmin[fi] > vox_max_x:
                                    continue
                                if tri_ymax[fi] < vox_min_y or tri_ymin[fi] > vox_max_y:
                                    continue
                                if tri_zmax[fi] < vox_min_z or tri_zmin[fi] > vox_max_z:
                                    continue

                                i0 = faces[fi, 0]; i1 = faces[fi, 1]; i2 = faces[fi, 2]
                                if _triangle_box_overlap_nb(
                                    sx, sy, sz,
                                    sub_half, sub_half, sub_half,
                                    verts[i0, 0], verts[i0, 1], verts[i0, 2],
                                    verts[i1, 0], verts[i1, 1], verts[i1, 2],
                                    verts[i2, 0], verts[i2, 1], verts[i2, 2],
                                ):
                                    hit = True
                                    break
                            if hit:
                                count += 1

                fractions[ri, ci, zi] = count / total_subs

    return fractions


def _filter_surface_by_occupancy(
    surface: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    r0: int, c0: int, z0: int,
    nr: int, nc: int, nz: int,
    occupancy_threshold: float,
    occupancy_subdivisions: int = 3,
) -> np.ndarray:
    """Remove surface voxels whose volume overlap fraction is below *occupancy_threshold*."""
    if occupancy_threshold <= 0.0:
        return surface
    fractions = _compute_occupancy_fraction(
        surface,
        np.ascontiguousarray(verts, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.intp),
        gp.min_x, gp.max_y, gp.min_z, gp.voxel_size,
        r0, c0, z0, nr, nc, nz,
        occupancy_subdivisions,
    )
    return surface & (fractions >= occupancy_threshold)


def _fill_interior(surface: np.ndarray) -> np.ndarray:
    """Fill the interior of a closed surface shell using scipy's binary_fill_holes."""
    if surface.size == 0:
        return surface
    return binary_fill_holes(surface)


def _apply_land_cover(
    voxel_grid: np.ndarray,
    gp: Grid3DParams,
    land_cover_grid: np.ndarray,
    dem_grid: np.ndarray,
    land_cover_source: str,
) -> None:
    land_cover = np.flipud(_convert_land_cover(land_cover_grid, land_cover_source))

    # Find the actual topmost GROUND_CODE voxel in each column so that
    # land cover follows the real voxelized surface (e.g. the flat base-
    # box surface in gap-filled areas) rather than the interpolated DEM.
    is_ground = (voxel_grid == GROUND_CODE)
    has_ground = is_ground.any(axis=2)
    # argmax on reversed Z gives index-from-top of first ground voxel
    rev = np.flip(is_ground, axis=2)
    first_from_top = np.argmax(rev, axis=2)
    actual_z = (gp.n_z - 1 - first_from_top).astype(np.intp)

    # Fall back to DEM height for columns without ground voxels
    dem_levels = np.rint((dem_grid - gp.min_z) / gp.voxel_size).astype(np.intp)
    ground_levels = np.where(has_ground, actual_z, dem_levels)
    ground_levels = np.clip(ground_levels, 0, gp.n_z - 1)

    # Skip cells with code 0 — they would otherwise overwrite the ground
    # voxel below (only OpenStreetMap codes are pre-shifted by +1; CityGML
    # and generic sources may legitimately produce 0 for "unknown").
    valid = (ground_levels >= 0) & (ground_levels < gp.n_z) & (land_cover != 0)
    rows, cols = np.where(valid)
    voxel_grid[rows, cols, ground_levels[rows, cols]] = land_cover[rows, cols]


def _apply_canopy(
    voxel_grid: np.ndarray,
    gp: Grid3DParams,
    dem_grid: np.ndarray,
    canopy_top: np.ndarray,
    canopy_bottom: Optional[np.ndarray],
    trunk_height_ratio: Optional[float],
) -> None:
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98

    top_arr = canopy_top.astype(np.float64)
    has_tree = top_arr > 0
    if not np.any(has_tree):
        _log.info("  [canopy] No cells with canopy_top > 0 – skipping.")
        return

    n_tree_cells = int(np.count_nonzero(has_tree))

    if canopy_bottom is not None:
        base_arr = canopy_bottom.astype(np.float64)
        base_arr = np.minimum(base_arr, top_arr)
    else:
        base_arr = top_arr * trunk_height_ratio

    ground_levels = np.rint((dem_grid - gp.min_z) / gp.voxel_size).astype(np.intp)
    z_starts = np.clip(ground_levels + np.rint(base_arr / gp.voxel_size).astype(np.intp), 0, gp.n_z)
    z_ends = np.clip(ground_levels + np.rint(top_arr / gp.voxel_size).astype(np.intp), 0, gp.n_z)

    # Skip column fill for grid cells that already contain 3-D mesh-
    # voxelized tree voxels (TREE_CODE).  Those cells already have the
    # correct crown shape from CityGML vegetation meshes; overwriting
    # with a rectangular column would destroy the spheroid/ellipsoid form.
    already_has_tree = np.any(voxel_grid == TREE_CODE, axis=2)

    valid = has_tree & (z_ends > z_starts) & ~already_has_tree
    n_skipped_mesh = int(np.count_nonzero(has_tree & (z_ends > z_starts) & already_has_tree))
    rs, cs = np.where(valid)
    # Vectorize column writes: build flat indices for all canopy voxels at once
    n_placed = 0
    if len(rs) > 0:
        zs_list = []
        rs_list = []
        cs_list = []
        for i in range(len(rs)):
            r, c = rs[i], cs[i]
            zs = np.arange(z_starts[r, c], z_ends[r, c], dtype=np.intp)
            zs_list.append(zs)
            rs_list.append(np.full(len(zs), r, dtype=np.intp))
            cs_list.append(np.full(len(zs), c, dtype=np.intp))
        all_r = np.concatenate(rs_list)
        all_c = np.concatenate(cs_list)
        all_z = np.concatenate(zs_list)
        mask = voxel_grid[all_r, all_c, all_z] == 0
        voxel_grid[all_r[mask], all_c[mask], all_z[mask]] = TREE_CODE
        n_placed = int(mask.sum())

    _log.info(
        "  [canopy] %d cells with canopy > 0, %d valid after z-range check, "
        "%d tree voxels placed%s",
        n_tree_cells, len(rs), n_placed,
        f", {n_skipped_mesh} cells skipped (3D mesh)" if n_skipped_mesh else "",
    )


def _convert_land_cover(land_cover_grid: np.ndarray, land_cover_source: str) -> np.ndarray:
    if land_cover_source == "OpenStreetMap":
        return land_cover_grid + 1
    if land_cover_source == "CityGML":
        # Already 1-based internal codes from citygml_landcover rasterizer
        return land_cover_grid
    from voxcity.utils.lc import convert_land_cover
    return convert_land_cover(land_cover_grid, land_cover_source=land_cover_source)


def _resize_int_grid(grid: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    factor = (target_rows / grid.shape[0], target_cols / grid.shape[1])
    return zoom(grid, factor, order=0).astype(grid.dtype)


def _resize_float_grid(grid: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    factor = (target_rows / grid.shape[0], target_cols / grid.shape[1])
    return zoom(grid, factor, order=1)

