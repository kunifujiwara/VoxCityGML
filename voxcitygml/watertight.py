"""Watertight conversion helpers for voxelization.

Transplanted from reference/citygml_mesher/solid/watertight.py.

Cascade (auto mode, MeshLib available):
  1. merge  –  vertex deduplication (fast, often enough for LOD2)
  2. repair –  MeshLib hole-fill / self-intersection fix
  3. meshlib_double_offset  –  shrink-then-expand (best quality)
  4. meshlib_offset  –  offsetMesh with HoleWindingRule
  5. voxel  –  voxelisation + marching cubes
  6. hull   –  convex hull (loses concave detail)

Without MeshLib the cascade is merge → repair(trimesh) → voxel → hull.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import logging
import numpy as np


# ─── MeshLib availability ───────────────────────────────────────────────────────

_MESHLIB_AVAILABLE = False
_mrmesh = None
_mrmeshnumpy = None

try:
    import meshlib.mrmeshpy as _mrmesh_mod
    import meshlib.mrmeshnumpy as _mrmeshnumpy_mod

    _mrmesh = _mrmesh_mod
    _mrmeshnumpy = _mrmeshnumpy_mod
    _MESHLIB_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning(
        "meshlib not installed – MeshLib-based repair/offset methods will be skipped. "
        "The watertight cascade will use: merge → trimesh repair → voxel → hull. "
        "Install meshlib for better results: pip install meshlib"
    )


# ─── Result dataclass ────────────────────────────────────────────────

@dataclass
class WatertightResult:
    vertices: np.ndarray
    faces: np.ndarray
    is_watertight: bool
    method: str
    error: Optional[str] = None


# ─── MeshLib <-> numpy helpers ────────────────────────────────────────

def _numpy_to_meshlib(vertices: np.ndarray, faces: np.ndarray):
    verts32 = np.ascontiguousarray(vertices, dtype=np.float32)
    faces32 = np.ascontiguousarray(faces, dtype=np.int32)
    return _mrmeshnumpy.meshFromFacesVerts(faces32, verts32)


def _meshlib_to_numpy(ml_mesh) -> Tuple[np.ndarray, np.ndarray]:
    v = _mrmeshnumpy.getNumpyVerts(ml_mesh).astype(np.float64)
    f = _mrmeshnumpy.getNumpyFaces(ml_mesh.topology).astype(np.int32)
    return v, f


# ─── Validation ───────────────────────────────────────────────────────

def _validate_watertight(
    vertices: np.ndarray, faces: np.ndarray,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Check mesh is watertight AND has consistent winding (is_volume).

    Using ``is_volume`` instead of ``is_watertight`` ensures face normals
    are consistently oriented – required for reliable ray-based
    containment tests.

    Returns ``(ok, vertices, faces)`` where the arrays may have been
    updated with corrected normals / face winding.
    """
    try:
        import trimesh
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # is_volume = is_watertight + consistent winding
        if tm.is_volume:
            return True, vertices, faces
        # Try fixing normals and re-check
        if tm.is_watertight:
            tm.fix_normals()
            if tm.is_volume:
                fixed_v = np.asarray(tm.vertices, dtype=np.float64)
                fixed_f = np.asarray(tm.faces, dtype=np.int32)
                return True, fixed_v, fixed_f
        return False, vertices, faces
    except Exception:
        return False, vertices, faces


# ─── Individual conversion steps ──────────────────────────────────────

def _merge_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    precision: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate vertices using numpy float64 rounding.

    Always uses numpy-based deduplication on original float64 coordinates,
    because MeshLib's uniteCloseVertices works on float32 which can miss
    exact duplicates present in the float64 source (CityGML polygons
    store the same coordinate text for shared edges but each polygon
    gets its own vertex array during extraction).
    """
    rounded = np.round(vertices, decimals=int(precision))
    merged_v, inv = np.unique(rounded, axis=0, return_inverse=True)
    merged_f = inv[faces].astype(np.int32)
    merged_v = merged_v.astype(np.float64)

    # Also try MeshLib topology cleanup with a slightly wider tolerance
    # to catch coordinates that differ by more than the rounding step.
    if _MESHLIB_AVAILABLE:
        try:
            ml = _numpy_to_meshlib(merged_v, merged_f)
            bbox = ml.computeBoundingBox()
            diag = (bbox.max - bbox.min).length()
            _mrmesh.uniteCloseVertices(ml, diag * 1e-5)
            merged_v, merged_f = _meshlib_to_numpy(ml)
        except Exception:
            pass

    # Drop degenerate triangles.
    a, b, c = merged_f[:, 0], merged_f[:, 1], merged_f[:, 2]
    keep = (a != b) & (b != c) & (a != c)
    merged_f = merged_f[keep]
    return merged_v, merged_f


def _meshlib_repair(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """MeshLib-based repair: merge, fix self-intersections, fix degeneracies, fill holes."""
    ml = _numpy_to_meshlib(vertices, faces)

    # 1) Merge close vertices
    bbox = ml.computeBoundingBox()
    diag = (bbox.max - bbox.min).length()
    _mrmesh.uniteCloseVertices(ml, diag * 1e-7)

    # 2) Fix self-intersections
    try:
        settings = _mrmesh.FixSelfIntersectionSettings()
        settings.method = _mrmesh.FixSelfIntersectionMethod.CutAndFill
        _mrmesh.localFixSelfIntersections(ml, settings)
    except Exception:
        pass

    # 3) Fix degeneracies
    try:
        params = _mrmesh.FixMeshDegeneraciesParams()
        bbox2 = ml.computeBoundingBox()
        params.maxDeviation = (bbox2.max - bbox2.min).length() * 1e-5
        _mrmesh.fixMeshDegeneracies(ml, params)
    except Exception:
        pass

    # 4) Fill every hole
    try:
        hole_edges = ml.topology.findHoleRepresentiveEdges()
        for i in range(hole_edges.size()):
            edge = hole_edges[i]
            hp = _mrmesh.FillHoleParams()
            hp.metric = _mrmesh.getUniversalMetric(ml)
            _mrmesh.fillHole(ml, edge, hp)
    except Exception:
        pass

    return _meshlib_to_numpy(ml)


def _trimesh_repair(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trimesh-based repair (fallback when MeshLib is not available)."""
    import trimesh

    tm = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=True)
    try:
        tm.merge_vertices(merge_tex=True, merge_norm=True)
    except Exception:
        pass
    try:
        tm.remove_degenerate_faces()
    except Exception:
        pass
    try:
        tm.remove_duplicate_faces()
    except Exception:
        pass
    try:
        tm.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(tm)
    except Exception:
        pass
    try:
        tm.fix_normals()
    except Exception:
        pass
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int32)


def _meshlib_double_offset(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    voxel_size: Optional[float],
    offset_factor: float,
    approx_num_voxels: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """MeshLib doubleOffsetMesh: shrink by *offset* then expand back."""
    ml = _numpy_to_meshlib(vertices, faces)

    vs = float(voxel_size) if voxel_size is not None else float(
        _mrmesh.suggestVoxelSize(ml, approx_num_voxels)
    )
    if not np.isfinite(vs) or vs <= 0:
        vs = float(voxel_size) if voxel_size is not None else 1.0

    offset = abs(vs * float(offset_factor))
    params = _mrmesh.OffsetParameters()
    params.voxelSize = vs

    result = _mrmesh.doubleOffsetMesh(ml, -offset, offset, params)

    # Keep only the largest connected component.
    try:
        mp = _mrmesh.MeshPart(result)
        largest = _mrmesh.MeshComponents.getLargestComponent(mp)
        if 0 < largest.count() < result.topology.numValidFaces():
            filtered = _mrmesh.Mesh()
            filtered.addPartByMask(_mrmesh.MeshPart(mp.mesh, largest), _mrmesh.FaceMap())
            result = filtered
    except Exception:
        pass

    return _meshlib_to_numpy(result)


def _meshlib_offset(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    voxel_size: Optional[float],
    approx_num_voxels: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """MeshLib offsetMesh with HoleWindingRule for open meshes."""
    ml = _numpy_to_meshlib(vertices, faces)

    vs = float(voxel_size) if voxel_size is not None else float(
        _mrmesh.suggestVoxelSize(ml, approx_num_voxels)
    )
    if not np.isfinite(vs) or vs <= 0:
        vs = float(voxel_size) if voxel_size is not None else 1.0

    params = _mrmesh.OffsetParameters()
    params.voxelSize = vs

    hole_edges = ml.topology.findHoleRepresentiveEdges()
    if hole_edges.size() > 0:
        params.signDetectionMode = _mrmesh.SignDetectionMode.HoleWindingRule

    result = _mrmesh.offsetMesh(ml, 0.0, params)
    return _meshlib_to_numpy(result)


def _voxel_marching(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    voxel_size: Optional[float],
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize → flood fill interior → marching cubes."""
    import trimesh

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    bounds = tm.bounds
    extent = bounds[1] - bounds[0]
    max_ext = float(np.max(extent)) if np.all(np.isfinite(extent)) else 1.0
    pitch = float(voxel_size) if voxel_size is not None else (max_ext / float(max(8, resolution)))
    if not np.isfinite(pitch) or pitch <= 0:
        pitch = 1.0

    vg = tm.voxelized(pitch)
    mc = vg.marching_cubes
    return np.asarray(mc.vertices, dtype=np.float64), np.asarray(mc.faces, dtype=np.int32)


def _convex_hull(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convex hull (guaranteed watertight, loses concavity)."""
    import trimesh
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hull = tm.convex_hull
    return np.asarray(hull.vertices, dtype=np.float64), np.asarray(hull.faces, dtype=np.int32)


# ─── Main public API ─────────────────────────────────────────────────

def make_watertight_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    method: str = "auto",
    voxel_size: Optional[float] = None,
    resolution: int = 64,
    precision: int = 6,
    offset_factor: float = 0.1,
    approx_num_voxels: float = 5e6,
) -> WatertightResult:
    """Attempt to produce a watertight solid from raw CityGML geometry.

    The *auto* cascade tries fast methods first and escalates to heavier
    methods until a watertight result is obtained (same strategy as the
    reference citygml_mesher/solid/watertight.py).

    Parameters
    ----------
    vertices, faces : ndarray
        Triangle mesh in local metres.
    voxel_size : float | None
        If given, used as MeshLib voxelSize. Otherwise MeshLib will suggest one.
    offset_factor : float
        Offset as fraction of voxel_size for doubleOffsetMesh (default 0.1).
    approx_num_voxels : float
        Target voxel count when voxel_size is None (for MeshLib suggestVoxelSize).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    # Build ordered list of (name, callable -> (verts, faces))
    # Each callable may raise; the loop catches and tries the next.
    attempts: list[Tuple[str, object]] = []

    method = (method or "auto").lower()

    if method == "auto":
        # 1) merge
        attempts.append(("merge", lambda v, f: _merge_vertices(v, f, precision=precision)))

        # 2) repair (MeshLib preferred)
        if _MESHLIB_AVAILABLE:
            attempts.append(("repair", lambda v, f: _meshlib_repair(v, f)))
        else:
            attempts.append(("repair", lambda v, f: _trimesh_repair(v, f)))

        # 3-4) MeshLib heavy methods
        if _MESHLIB_AVAILABLE:
            attempts.append((
                "meshlib_double_offset",
                lambda v, f: _meshlib_double_offset(
                    v, f,
                    voxel_size=voxel_size,
                    offset_factor=offset_factor,
                    approx_num_voxels=approx_num_voxels,
                ),
            ))
            attempts.append((
                "meshlib_offset",
                lambda v, f: _meshlib_offset(
                    v, f,
                    voxel_size=voxel_size,
                    approx_num_voxels=approx_num_voxels,
                ),
            ))

        # 5) voxel marching cubes
        attempts.append((
            "voxel",
            lambda v, f: _voxel_marching(v, f, voxel_size=voxel_size, resolution=resolution),
        ))

        # Note: convex hull is intentionally excluded from auto cascade.
        # For concave structures (bridges, complex buildings), hull loses
        # essential shape detail.  When all methods above fail, the caller
        # falls back to surface voxelization which preserves concavity.

    elif method == "merge":
        attempts.append(("merge", lambda v, f: _merge_vertices(v, f, precision=precision)))
    elif method == "repair":
        if _MESHLIB_AVAILABLE:
            attempts.append(("repair", lambda v, f: _meshlib_repair(v, f)))
        else:
            attempts.append(("repair", lambda v, f: _trimesh_repair(v, f)))
    elif method == "meshlib":
        attempts.append((
            "meshlib_double_offset",
            lambda v, f: _meshlib_double_offset(
                v, f, voxel_size=voxel_size, offset_factor=offset_factor,
                approx_num_voxels=approx_num_voxels,
            ),
        ))
    elif method == "meshlib_offset":
        attempts.append((
            "meshlib_offset",
            lambda v, f: _meshlib_offset(v, f, voxel_size=voxel_size, approx_num_voxels=approx_num_voxels),
        ))
    elif method == "voxel":
        attempts.append((
            "voxel",
            lambda v, f: _voxel_marching(v, f, voxel_size=voxel_size, resolution=resolution),
        ))
    elif method == "hull":
        attempts.append(("hull", lambda v, f: _convex_hull(v, f)))
    else:
        return WatertightResult(vertices, faces, False, method="none", error=f"Unknown method: {method}")

    # ---- Run cascade (progressive: each step builds on the previous) ----
    # Compute original mesh area and centroid for quality gate.
    try:
        import trimesh as _tm
        _orig_tm = _tm.Trimesh(vertices=vertices, faces=faces, process=False)
        _orig_area = _orig_tm.area
        _orig_centroid = np.mean(vertices, axis=0)
        _orig_bounds = _orig_tm.bounds  # (2, 3) array: [min_corner, max_corner]
        _orig_diag = float(np.linalg.norm(_orig_bounds[1] - _orig_bounds[0]))
    except Exception:
        _orig_area = 0.0
        _orig_centroid = np.mean(vertices, axis=0) if len(vertices) > 0 else np.zeros(3)
        _orig_diag = 0.0

    last: Optional[WatertightResult] = None
    cur_v, cur_f = vertices, faces
    for name, fn in attempts:
        try:
            out_v, out_f = fn(cur_v, cur_f)
            ok, out_v, out_f = _validate_watertight(out_v, out_f)
            if ok and len(out_f) > 0 and len(out_v) > 0:
                # Quality gate: reject meshes that are fragmented, shrunken,
                # bloated, or whose centroid has drifted significantly.
                # This catches:
                #  - meshlib_double_offset fragmenting thin structures
                #  - repair/meshlib_offset flooding open LOD2 meshes,
                #    producing bloated blobs whose centroids drift toward
                #    the bounding-box centre (phantom-building bug).
                reject = False
                if _orig_area > 0:
                    try:
                        _wt_tm = _tm.Trimesh(
                            vertices=out_v, faces=out_f, process=False,
                        )
                        _wt_area = _wt_tm.area
                        # Check 1: area should not collapse below 10% of original
                        if _wt_area < 0.1 * _orig_area:
                            reject = True
                            logging.getLogger(__name__).info(
                                "Watertight '%s' rejected: area %.0f < 10%% of "
                                "original %.0f",
                                name, _wt_area, _orig_area,
                            )
                        # Check 2: reject if mesh splits into many components
                        if not reject:
                            _bodies = _wt_tm.split(only_watertight=False)
                            if len(_bodies) > 3:
                                reject = True
                                logging.getLogger(__name__).info(
                                    "Watertight '%s' rejected: %d connected "
                                    "components (max 3 allowed)",
                                    name, len(_bodies),
                                )
                        # Check 3: reject if surface area inflated > 5×
                        # (catches meshlib_offset flooding open LOD2 meshes,
                        #  e.g. 128 verts → 6008 verts bounding-box blob)
                        if not reject and _wt_area > 5.0 * _orig_area:
                            reject = True
                            logging.getLogger(__name__).info(
                                "Watertight '%s' rejected: area %.0f is >5× "
                                "original %.0f (probable interior flood)",
                                name, _wt_area, _orig_area,
                            )
                        # Check 4: reject if centroid drifted too far.
                        # Uses max(5 % of bbox diagonal, 15 m) so that:
                        #  • small buildings: 15 m absolute floor prevents
                        #    over-rejection of proportionally large but
                        #    harmless shifts;
                        #  • large / elongated buildings: 5 % of diagonal
                        #    scales with size, catching bloated blobs whose
                        #    centroids drift toward the bounding-box centre
                        #    (phantom-building bug in LOD2 open meshes).
                        if not reject and _orig_diag > 0:
                            _wt_centroid = np.mean(out_v, axis=0)
                            _drift = float(np.linalg.norm(_wt_centroid - _orig_centroid))
                            _drift_limit = max(0.05 * _orig_diag, 15.0)
                            if _drift > _drift_limit:
                                reject = True
                                logging.getLogger(__name__).info(
                                    "Watertight '%s' rejected: centroid drifted "
                                    "%.1f m (limit %.1f m = max(5%% of %.1f, 15))",
                                    name, _drift, _drift_limit, _orig_diag,
                                )
                    except Exception:
                        pass  # if quality check fails, accept the result
                if reject:
                    # Do NOT pass the fragmented mesh to the next step —
                    # keep using the mesh from *before* this step.
                    last = WatertightResult(out_v, out_f, False, method=name)
                    continue
                return WatertightResult(out_v, out_f, True, method=name)
            # Keep the improved mesh for the next step (progressive)
            if len(out_f) > 0 and len(out_v) > 0:
                cur_v, cur_f = out_v, out_f
            last = WatertightResult(out_v, out_f, False, method=name)
        except Exception as e:
            last = WatertightResult(cur_v, cur_f, False, method=name, error=str(e))

    return last if last is not None else WatertightResult(vertices, faces, False, method="none", error="all methods failed")
