"""
Terrain mesh preprocessing and extrusion for watertight solid generation.

Creates watertight solid volumes from CityGML terrain (TINRelief) meshes by:
1. Merging multiple terrain tiles
2. Removing degenerate triangles
3. Vertex welding (merging duplicate vertices)
4. Fixing non-manifold edges
5. Splitting pinch-point vertices
6. Extruding boundary edges vertically downward (curtain walls)
7. Adding bottom cap at specified Z level
8. Creating a base box that covers the target rectangle
9. Boolean union of extruded terrain + base box to fill gaps (rivers, missing tiles)

The resulting watertight solid can be voxelized with interior-fill algorithms
(MeshLib level-set, winding number, or Z-scanline), producing much more
accurate terrain volumes than the simple per-column DEM fill.

Reference: citygml_mesher.solid.extrusion / citygml_mesher.solid.preprocess /
           citygml_mesher.solid.boolean_ops / citygml_mesher.solid.base_geometry
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .models import Mesh3D

_log = logging.getLogger(__name__)


# =====================================================================
# Statistics
# =====================================================================

@dataclass
class TerrainSolidStats:
    """Bookkeeping for the merge → preprocess → extrude → union pipeline."""
    input_meshes: int = 0
    input_triangles: int = 0
    input_vertices: int = 0
    merged_triangles: int = 0
    merged_vertices: int = 0
    degenerate_removed: int = 0
    vertices_welded: int = 0
    non_manifold_fixed: int = 0
    pinch_points_split: int = 0
    boundary_edges: int = 0
    wall_faces: int = 0
    bottom_faces: int = 0
    final_vertices: int = 0
    final_faces: int = 0
    extrusion_depth: float = 0.0
    bottom_z: float = 0.0
    is_watertight: bool = False
    # Boolean union gap-fill
    boolean_union_attempted: bool = False
    boolean_union_success: bool = False
    boolean_union_engine: str = ""


# =====================================================================
# Low-level helpers (pure-numpy, no external deps)
# =====================================================================

def _compute_triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of each triangle via half cross-product magnitude."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _compute_aspect_ratios(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Aspect ratio (longest / shortest edge) per triangle."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    edges = np.stack([e0, e1, e2], axis=1)
    edges_sorted = np.sort(edges, axis=1)
    return edges_sorted[:, 2] / np.maximum(edges_sorted[:, 0], 1e-15)


# ── Degenerate removal ────────────────────────────────────────────────

def remove_degenerate_triangles(
    mesh: Mesh3D,
    min_area: float = 1e-12,
    max_aspect_ratio: float = 1e6,
) -> Tuple[Mesh3D, int]:
    """Remove near-zero-area or extremely narrow triangles."""
    if len(mesh.faces) == 0:
        return mesh, 0

    areas = _compute_triangle_areas(mesh.vertices, mesh.faces)
    aspect = _compute_aspect_ratios(mesh.vertices, mesh.faces)
    valid = (areas >= min_area) & (aspect <= max_aspect_ratio)

    n_removed = int(np.sum(~valid))
    if n_removed == 0:
        return mesh, 0

    valid_faces = mesh.faces[valid]
    unique_idx, inverse = np.unique(valid_faces.flatten(), return_inverse=True)
    new_verts = mesh.vertices[unique_idx]
    new_faces = inverse.reshape(-1, 3).astype(np.int32)

    return Mesh3D(
        vertices=new_verts,
        faces=new_faces,
        feature_type=mesh.feature_type,
        feature_id=mesh.feature_id,
        attributes=mesh.attributes.copy() if mesh.attributes else {},
    ), n_removed


# ── Vertex welding ────────────────────────────────────────────────────

def weld_vertices(
    mesh: Mesh3D,
    tolerance: float = 1e-10,
) -> Tuple[Mesh3D, int]:
    """Merge duplicate vertices within *tolerance* (triangle-soup → shared)."""
    if len(mesh.vertices) == 0:
        return mesh, 0

    n_orig = len(mesh.vertices)

    # MeshLib fast weld if available
    try:
        import meshlib.mrmeshpy as mrmesh
        import meshlib.mrmeshnumpy as mrmeshnumpy
        verts_f32 = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces_i32 = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        ml = mrmeshnumpy.meshFromFacesVerts(faces_i32, verts_f32)
        settings = mrmesh.VertexMergeSettings()
        settings.maxDistance = float(tolerance)
        mrmesh.uniteCloseVertices(ml, settings)
        new_verts = mrmeshnumpy.getNumpyVerts(ml).astype(np.float64)
        new_faces = mrmeshnumpy.getNumpyFaces(ml.topology).astype(np.int32)
        n_removed = n_orig - len(new_verts)
        return Mesh3D(
            vertices=new_verts,
            faces=new_faces,
            feature_type=mesh.feature_type,
            feature_id=mesh.feature_id,
            attributes=mesh.attributes.copy() if mesh.attributes else {},
        ), n_removed
    except Exception:
        pass

    # Pure-numpy fallback
    if tolerance > 0:
        factor = 1.0 / tolerance
        rounded = np.round(mesh.vertices * factor) / factor
    else:
        rounded = mesh.vertices

    vertex_keys = [tuple(v) for v in rounded]
    unique_map: Dict[tuple, int] = {}
    old_to_new = np.empty(n_orig, dtype=np.int32)
    new_verts_list: list = []
    for i, key in enumerate(vertex_keys):
        if key not in unique_map:
            unique_map[key] = len(new_verts_list)
            new_verts_list.append(mesh.vertices[i])
        old_to_new[i] = unique_map[key]

    new_verts = np.array(new_verts_list, dtype=mesh.vertices.dtype)
    new_faces = old_to_new[mesh.faces]

    # Remove degenerate faces produced by welding
    ok = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[ok]
    n_removed = n_orig - len(new_verts)

    return Mesh3D(
        vertices=new_verts,
        faces=new_faces.astype(np.int32),
        feature_type=mesh.feature_type,
        feature_id=mesh.feature_id,
        attributes=mesh.attributes.copy() if mesh.attributes else {},
    ), n_removed


# ── Overlapping / duplicate face removal ───────────────────────────────

def remove_overlapping_faces(
    mesh: Mesh3D,
    tolerance: float = 1e-6,
) -> Tuple[Mesh3D, int]:
    """Remove duplicate or near-duplicate triangles caused by overlapping tiles.

    When multiple terrain tiles cover the same area (e.g. data from different
    municipalities), the merged mesh contains overlapping triangle pairs with
    nearly identical centroids.  These cause self-intersections that break
    Boolean operations (MeshLib, manifold3d, etc.).

    For each cluster of faces whose sorted vertex-index triple matches
    exactly, only the first face is kept.  Additionally, faces whose
    centroids are within *tolerance* of another face's centroid are
    deduplicated.
    """
    if len(mesh.faces) == 0:
        return mesh, 0

    n_orig = len(mesh.faces)

    # ── Pass 1: exact duplicate removal (same sorted vertex indices) ──
    sorted_faces = np.sort(mesh.faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original order
    n_exact = n_orig - len(unique_idx)

    if n_exact > 0:
        faces = mesh.faces[unique_idx]
    else:
        faces = mesh.faces

    # ── Pass 2: centroid-based near-duplicate removal ─────────────────
    if tolerance > 0 and len(faces) > 1:
        centroids = (
            mesh.vertices[faces[:, 0]]
            + mesh.vertices[faces[:, 1]]
            + mesh.vertices[faces[:, 2]]
        ) / 3.0

        # Quantise centroids and group
        keys = np.round(centroids / tolerance).astype(np.int64)
        # Use a hash of the quantised centroid to find clusters
        seen: set = set()
        keep_mask = np.ones(len(faces), dtype=bool)
        for i in range(len(faces)):
            k = (keys[i, 0], keys[i, 1], keys[i, 2])
            if k in seen:
                keep_mask[i] = False
            else:
                seen.add(k)

        n_near = int((~keep_mask).sum())
        if n_near > 0:
            faces = faces[keep_mask]
    else:
        n_near = 0

    total_removed = n_exact + n_near
    if total_removed == 0:
        return mesh, 0

    # Re-index vertices to keep only those referenced by remaining faces
    unique_verts, inverse = np.unique(faces.flatten(), return_inverse=True)
    new_verts = mesh.vertices[unique_verts]
    new_faces = inverse.reshape(-1, 3).astype(np.int32)

    return Mesh3D(
        vertices=new_verts,
        faces=new_faces,
        feature_type=mesh.feature_type,
        feature_id=mesh.feature_id,
        attributes=mesh.attributes.copy() if mesh.attributes else {},
    ), total_removed


# ── Boundary edge computation ─────────────────────────────────────────

def compute_boundary_edges(faces: np.ndarray) -> Set[Tuple[int, int]]:
    """Edges appearing in exactly one triangle (gaps/holes in mesh)."""
    if len(faces) == 0:
        return set()

    edges = []
    for i in range(3):
        j = (i + 1) % 3
        edges.append(np.stack([faces[:, i], faces[:, j]], axis=1))
    all_edges = np.vstack(edges)
    sorted_edges = np.sort(all_edges, axis=1)

    counts = Counter(tuple(e) for e in sorted_edges)
    return {e for e, c in counts.items() if c == 1}


# ── Non-manifold fix ──────────────────────────────────────────────────

def fix_non_manifold_edges(mesh: Mesh3D) -> Tuple[Mesh3D, int]:
    """Remove excess faces so no edge is shared by >2 triangles."""
    edge_faces: Dict[tuple, list] = {}
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            edge_faces.setdefault(key, []).append(fi)

    to_remove: set = set()
    for _edge, fl in edge_faces.items():
        if len(fl) > 2:
            for fi in fl[2:]:
                to_remove.add(fi)

    if not to_remove:
        return mesh, 0

    mask = np.ones(len(mesh.faces), dtype=bool)
    for fi in to_remove:
        mask[fi] = False

    return Mesh3D(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[mask],
        feature_type=mesh.feature_type,
        feature_id=mesh.feature_id,
    ), len(to_remove)


# ── Pinch-point handling ──────────────────────────────────────────────

def find_pinch_point_vertices(faces: np.ndarray) -> List[Tuple[int, int]]:
    """Find vertices with >2 boundary edges (pinch points)."""
    boundary = compute_boundary_edges(faces)
    vc = Counter()
    for v1, v2 in boundary:
        vc[v1] += 1
        vc[v2] += 1
    return [(v, c) for v, c in vc.items() if c > 2]


def split_pinch_point_vertices(mesh: Mesh3D) -> Tuple[Mesh3D, int]:
    """Duplicate pinch-point vertices so each boundary loop is independent."""
    from collections import defaultdict

    pinch_points = find_pinch_point_vertices(mesh.faces)
    if not pinch_points:
        return mesh, 0

    new_verts = list(mesh.vertices)
    new_faces = mesh.faces.copy().tolist()
    total_added = 0

    for pv, _ec in pinch_points:
        faces_with = [fi for fi, f in enumerate(new_faces) if pv in f]
        if len(faces_with) <= 1:
            continue

        adj: Dict[int, set] = defaultdict(set)
        for i, fi in enumerate(faces_with):
            non_pv_i = [v for v in new_faces[fi] if v != pv]
            for j, fj in enumerate(faces_with):
                if i >= j:
                    continue
                non_pv_j = [v for v in new_faces[fj] if v != pv]
                if set(non_pv_i) & set(non_pv_j):
                    adj[fi].add(fj)
                    adj[fj].add(fi)

        visited: set = set()
        components: List[list] = []
        for sf in faces_with:
            if sf in visited:
                continue
            comp: list = []
            queue = [sf]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                for nb in adj.get(cur, []):
                    if nb not in visited:
                        queue.append(nb)
            if comp:
                components.append(comp)

        for comp in components[1:]:
            new_idx = len(new_verts)
            new_verts.append(mesh.vertices[pv].copy())
            total_added += 1
            for fi in comp:
                new_faces[fi] = [new_idx if v == pv else v for v in new_faces[fi]]

    return Mesh3D(
        vertices=np.array(new_verts, dtype=mesh.vertices.dtype),
        faces=np.array(new_faces, dtype=mesh.faces.dtype),
        feature_type=mesh.feature_type,
        feature_id=mesh.feature_id,
    ), total_added


# =====================================================================
# Merge terrain tiles
# =====================================================================

def merge_terrain_meshes(meshes: List[Mesh3D]) -> Optional[Mesh3D]:
    """Concatenate multiple terrain Mesh3D into one (triangle soup)."""
    non_empty = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]

    all_v: list = []
    all_f: list = []
    offset = 0
    for m in non_empty:
        all_v.append(m.vertices)
        all_f.append(m.faces + offset)
        offset += len(m.vertices)

    return Mesh3D(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f).astype(np.int32),
        feature_type="terrain",
        feature_id="merged_terrain",
    )


# =====================================================================
# Wall + bottom cap creation
# =====================================================================

def _create_wall_faces(
    boundary_edges: Set[Tuple[int, int]],
    n_verts: int,
    faces: np.ndarray,
) -> np.ndarray:
    """Build curtain-wall triangles connecting top boundary to extruded bottom.

    Uses the original face winding to determine correct outward normal
    direction for each wall quad.
    """
    # Map sorted-edge → directed edge as it appears in some face
    edge_dir: Dict[tuple, Tuple[int, int]] = {}
    for face in faces:
        for i in range(3):
            vf, vt = int(face[i]), int(face[(i + 1) % 3])
            key = (min(vf, vt), max(vf, vt))
            if key in boundary_edges:
                edge_dir[key] = (vf, vt)

    offset = n_verts
    wall: list = []
    for ek in boundary_edges:
        v1, v2 = edge_dir.get(ek, ek)
        # Two triangles forming quad: v1–v1'–v2' and v1–v2'–v2
        wall.append([v1, v1 + offset, v2 + offset])
        wall.append([v1, v2 + offset, v2])

    return np.array(wall, dtype=np.int32) if wall else np.empty((0, 3), dtype=np.int32)


def _create_bottom_cap(faces: np.ndarray, vertex_offset: int) -> np.ndarray:
    """Mirror top faces at bottom with reversed winding."""
    b = faces.copy() + vertex_offset
    return b[:, ::-1].astype(np.int32)


# =====================================================================
# Main extrusion function
# =====================================================================

def extrude_terrain_solid(
    mesh: Mesh3D,
    extrusion_depth: float = 10.0,
    bottom_z: Optional[float] = None,
) -> Tuple[Mesh3D, TerrainSolidStats]:
    """Extrude a preprocessed terrain surface into a watertight solid.

    Steps:
        1. Fix non-manifold edges
        2. Split pinch-point vertices
        3. Compute boundary edges
        4. Create extruded vertices at ``bottom_z``
        5. Build curtain walls from boundary edges
        6. Add bottom cap (reversed winding)

    Returns (solid_mesh, stats).
    """
    stats = TerrainSolidStats()
    stats.input_vertices = len(mesh.vertices)
    stats.input_triangles = len(mesh.faces)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh, stats

    # 1. Fix non-manifold
    mesh, n_nm = fix_non_manifold_edges(mesh)
    stats.non_manifold_fixed = n_nm

    # 2. Split pinch points
    pp = find_pinch_point_vertices(mesh.faces)
    if pp:
        mesh, n_added = split_pinch_point_vertices(mesh)
        stats.pinch_points_split = n_added

    # 3. Boundary edges
    boundary = compute_boundary_edges(mesh.faces)
    stats.boundary_edges = len(boundary)

    # 4. Bottom Z
    min_z = float(mesh.vertices[:, 2].min())
    if bottom_z is None:
        bottom_z = min_z - extrusion_depth
    stats.bottom_z = bottom_z
    stats.extrusion_depth = min_z - bottom_z

    # 5. Extruded vertices
    ext_verts = mesh.vertices.copy()
    ext_verts[:, 2] = bottom_z
    all_verts = np.vstack([mesh.vertices, ext_verts])

    # 6. Top surface (unchanged)
    top_faces = mesh.faces.copy()

    # 7. Walls
    wall_faces = _create_wall_faces(boundary, len(mesh.vertices), mesh.faces)
    stats.wall_faces = len(wall_faces)

    # 8. Bottom cap
    bottom_faces = _create_bottom_cap(mesh.faces, len(mesh.vertices))
    stats.bottom_faces = len(bottom_faces)

    # Combine
    parts = [top_faces, wall_faces, bottom_faces]
    all_faces = np.vstack([p for p in parts if len(p) > 0]).astype(np.int32)

    stats.final_vertices = len(all_verts)
    stats.final_faces = len(all_faces)

    solid = Mesh3D(
        vertices=all_verts,
        faces=all_faces,
        feature_type="terrain_solid",
        feature_id=mesh.feature_id + "_solid" if mesh.feature_id else "terrain_solid",
        attributes={"is_solid": True, "bottom_z": bottom_z},
    )
    return solid, stats


# =====================================================================
# Validate watertightness
# =====================================================================

def validate_solid(mesh: Mesh3D) -> dict:
    """Check every edge is shared by exactly 2 faces."""
    if len(mesh.faces) == 0:
        return {"is_watertight": False, "boundary_edges": 0, "non_manifold_edges": 0, "total_edges": 0}

    ec: Dict[tuple, int] = {}
    for face in mesh.faces:
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            ec[key] = ec.get(key, 0) + 1

    bnd = sum(1 for c in ec.values() if c == 1)
    nm = sum(1 for c in ec.values() if c > 2)
    return {
        "is_watertight": bnd == 0 and nm == 0,
        "boundary_edges": bnd,
        "non_manifold_edges": nm,
        "total_edges": len(ec),
    }


# =====================================================================
# Base box for gap filling
# =====================================================================

def create_base_box(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
) -> Mesh3D:
    """Create a watertight axis-aligned box (8 verts, 12 tris, outward CCW normals).

    This box covers the target rectangle and is used as the Boolean union
    operand to fill internal gaps (rivers, missing tiles) in the terrain.

    Reference: citygml_mesher.solid.base_geometry.create_box_mesh
    """
    #   6-------7
    #  /|      /|
    # 4-------5 |
    # | 2-----|-3
    # |/      |/
    # 0-------1
    vertices = np.array([
        [min_x, min_y, min_z],  # 0
        [max_x, min_y, min_z],  # 1
        [min_x, max_y, min_z],  # 2
        [max_x, max_y, min_z],  # 3
        [min_x, min_y, max_z],  # 4
        [max_x, min_y, max_z],  # 5
        [min_x, max_y, max_z],  # 6
        [max_x, max_y, max_z],  # 7
    ], dtype=np.float64)

    faces = np.array([
        # Bottom (-Z)
        [0, 2, 1], [1, 2, 3],
        # Top (+Z)
        [4, 5, 6], [5, 7, 6],
        # Front (-Y)
        [0, 1, 4], [1, 5, 4],
        # Back (+Y)
        [2, 6, 3], [3, 6, 7],
        # Left (-X)
        [0, 4, 2], [2, 4, 6],
        # Right (+X)
        [1, 3, 5], [3, 7, 5],
    ], dtype=np.int32)

    return Mesh3D(
        vertices=vertices,
        faces=faces,
        feature_type="base_box",
        feature_id="base_rectangle_solid",
        attributes={"is_solid": True},
    )


# =====================================================================
# Boolean union (terrain ∪ base box) for gap filling
# =====================================================================

# MeshLib availability (cached at module level)
_HAS_MESHLIB = False
try:
    import meshlib.mrmeshpy as _mrmesh
    import meshlib.mrmeshnumpy as _mrmeshnumpy
    _HAS_MESHLIB = True
except ImportError:
    _mrmesh = None  # type: ignore[assignment]
    _mrmeshnumpy = None  # type: ignore[assignment]


def _fill_holes_meshlib(ml_mesh, max_iterations: int = 5) -> bool:
    """Fill all holes in a MeshLib mesh. Returns True if no holes remain."""
    if not _HAS_MESHLIB:
        return False
    try:
        for _ in range(max_iterations):
            holes = ml_mesh.topology.findHoleRepresentiveEdges()
            if holes.size() == 0:
                return True
            filled_any = False
            for i in range(holes.size()):
                try:
                    params = _mrmesh.FillHoleParams()
                    params.metric = _mrmesh.getUniversalMetric(ml_mesh)
                    _mrmesh.fillHole(ml_mesh, holes[i], params)
                    filled_any = True
                except Exception:
                    pass
            if not filled_any:
                break
            ml_mesh.pack()
        return ml_mesh.topology.findHoleRepresentiveEdges().size() == 0
    except Exception:
        return False


def boolean_union_terrain(
    terrain_solid: Mesh3D,
    base_box: Mesh3D,
    voxel_size: float = 1.0,
    verbose: bool = True,
) -> Tuple[Optional[Mesh3D], str, str]:
    """Boolean union of terrain solid and base box to fill gaps.

    Cascade:
        1. MeshLib geometric boolean (fastest, requires clean meshes)
        2. MeshLib voxel boolean (robust against self-intersections)
        3. trimesh (manifold3d / blender)

    Args:
        terrain_solid: Extruded terrain solid (walls around gaps).
        base_box: Axis-aligned box covering the target rectangle.
        voxel_size: Grid voxel size in metres (for voxel-based fallback).
        verbose: Log progress.

    Returns:
        (result_mesh_or_None, engine_used, error_message)

    Reference: citygml_mesher.solid.boolean_ops.create_watertight_terrain
    """
    ml_a = None
    ml_b = None

    # ── MeshLib geometric boolean ─────────────────────────────────────
    if _HAS_MESHLIB:
        try:
            verts_a = np.ascontiguousarray(terrain_solid.vertices, dtype=np.float32)
            faces_a = np.ascontiguousarray(terrain_solid.faces, dtype=np.int32)
            ml_a = _mrmeshnumpy.meshFromFacesVerts(faces_a, verts_a)

            verts_b = np.ascontiguousarray(base_box.vertices, dtype=np.float32)
            faces_b = np.ascontiguousarray(base_box.faces, dtype=np.int32)
            ml_b = _mrmeshnumpy.meshFromFacesVerts(faces_b, verts_b)

            # Fill holes to help the Boolean engine
            _fill_holes_meshlib(ml_a)
            _fill_holes_meshlib(ml_b)

            result = _mrmesh.boolean(
                ml_a, ml_b, _mrmesh.BooleanOperation.Union,
            )

            if result.valid() and result.mesh is not None:
                rv = _mrmeshnumpy.getNumpyVerts(result.mesh).astype(np.float64)
                rf = _mrmeshnumpy.getNumpyFaces(result.mesh.topology).astype(np.int32)
                mesh = Mesh3D(
                    vertices=rv, faces=rf,
                    feature_type="terrain_watertight",
                    feature_id="boolean_union_result",
                    attributes={"is_solid": True},
                )
                if verbose:
                    _log.info(
                        "  Boolean union (MeshLib): %d verts, %d faces",
                        len(rv), len(rf),
                    )
                return mesh, "meshlib", ""
            else:
                err = getattr(result, "errorString", "unknown")
                if verbose:
                    _log.warning("  MeshLib geometric boolean invalid: %s", err)
        except Exception as exc:
            if verbose:
                _log.warning("  MeshLib geometric boolean failed: %s", exc)

    # ── MeshLib voxel boolean (robust against self-intersections) ─────
    if _HAS_MESHLIB:
        try:
            if ml_a is None:
                verts_a = np.ascontiguousarray(terrain_solid.vertices, dtype=np.float32)
                faces_a = np.ascontiguousarray(terrain_solid.faces, dtype=np.int32)
                ml_a = _mrmeshnumpy.meshFromFacesVerts(faces_a, verts_a)
            if ml_b is None:
                verts_b = np.ascontiguousarray(base_box.vertices, dtype=np.float32)
                faces_b = np.ascontiguousarray(base_box.faces, dtype=np.int32)
                ml_b = _mrmeshnumpy.meshFromFacesVerts(faces_b, verts_b)

            voxel_result = _mrmesh.voxelBooleanUnite(ml_a, ml_b, voxel_size)
            if voxel_result is not None:
                rv = _mrmeshnumpy.getNumpyVerts(voxel_result).astype(np.float64)
                rf = _mrmeshnumpy.getNumpyFaces(voxel_result.topology).astype(np.int32)
                if len(rf) > 0:
                    mesh = Mesh3D(
                        vertices=rv, faces=rf,
                        feature_type="terrain_watertight",
                        feature_id="boolean_union_result",
                        attributes={"is_solid": True},
                    )
                    if verbose:
                        _log.info(
                            "  Boolean union (MeshLib voxel, vs=%.2f): %d verts, %d faces",
                            voxel_size, len(rv), len(rf),
                        )
                    return mesh, "meshlib_voxel", ""
        except Exception as exc:
            if verbose:
                _log.warning("  MeshLib voxel boolean failed: %s", exc)

    # ── Trimesh fallback (manifold3d → blender) ───────────────────────
    try:
        import trimesh

        tm_a = trimesh.Trimesh(
            vertices=terrain_solid.vertices,
            faces=terrain_solid.faces,
            process=False,
        )
        tm_b = trimesh.Trimesh(
            vertices=base_box.vertices,
            faces=base_box.faces,
            process=False,
        )

        engine_used = ""
        r = None
        for eng in ("manifold", "blender"):
            try:
                r = trimesh.boolean.union([tm_a, tm_b], engine=eng)
                engine_used = eng
                break
            except Exception:
                continue

        if r is not None and len(r.faces) > 0:
            mesh = Mesh3D(
                vertices=np.array(r.vertices, dtype=np.float64),
                faces=np.array(r.faces, dtype=np.int32),
                feature_type="terrain_watertight",
                feature_id="boolean_union_result",
                attributes={"is_solid": True},
            )
            if verbose:
                _log.info(
                    "  Boolean union (trimesh/%s): %d verts, %d faces",
                    engine_used, len(mesh.vertices), len(mesh.faces),
                )
            return mesh, engine_used, ""
    except Exception as exc:
        return None, "", str(exc)

    return None, "", "All Boolean engines failed"


# =====================================================================
# High-level: terrain meshes → watertight solid (ready for voxelization)
# =====================================================================

def build_terrain_solid(
    terrain_meshes: List[Mesh3D],
    bottom_z: Optional[float] = None,
    extrusion_depth: float = 10.0,
    weld_tolerance: float = 1e-10,
    min_triangle_area: float = 1e-12,
    max_aspect_ratio: float = 1e6,
    grid_bounds: Optional[Tuple[float, float, float, float]] = None,
    voxel_size: float = 1.0,
    verbose: bool = True,
) -> Tuple[Optional[Mesh3D], TerrainSolidStats]:
    """End-to-end: merge → clean → weld → extrude → Boolean union → validate.

    Args:
        terrain_meshes: Raw terrain Mesh3D objects (already in local metres).
        bottom_z: Explicit bottom Z; if *None*, computed from
                  ``min(z) - extrusion_depth``.
        extrusion_depth: Fallback depth when *bottom_z* is not given.
        weld_tolerance: Distance for merging duplicate vertices.
        min_triangle_area: Degenerate-removal threshold.
        max_aspect_ratio: Degenerate-removal threshold.
        grid_bounds: (min_x, max_x, min_y, max_y) of the target rectangle
                     in local metres.  When provided, a base box is created
                     and Boolean-unioned with the extruded terrain to fill
                     internal gaps (rivers, missing tiles).
        voxel_size: Grid voxel size in metres (used by the voxel-based
                    Boolean fallback when geometric Boolean fails).
        verbose: Log progress.

    Returns:
        (solid_mesh_or_None, stats)
    """
    stats = TerrainSolidStats()
    stats.input_meshes = len(terrain_meshes)
    stats.input_triangles = sum(len(m.faces) for m in terrain_meshes)
    stats.input_vertices = sum(len(m.vertices) for m in terrain_meshes)

    if verbose:
        _log.info(
            "Terrain solid: %d meshes, %d triangles, %d vertices",
            stats.input_meshes, stats.input_triangles, stats.input_vertices,
        )

    # ── 1. Merge ──────────────────────────────────────────────────────
    merged = merge_terrain_meshes(terrain_meshes)
    if merged is None:
        _log.warning("No valid terrain meshes to extrude.")
        return None, stats
    stats.merged_triangles = len(merged.faces)
    stats.merged_vertices = len(merged.vertices)

    # ── 2. Remove degenerate triangles ────────────────────────────────
    merged, n_deg = remove_degenerate_triangles(
        merged, min_area=min_triangle_area, max_aspect_ratio=max_aspect_ratio,
    )
    stats.degenerate_removed = n_deg
    if verbose and n_deg:
        _log.info("  Removed %d degenerate triangles", n_deg)

    # ── 3. Weld vertices ──────────────────────────────────────────────
    merged, n_welded = weld_vertices(merged, tolerance=weld_tolerance)
    stats.vertices_welded = n_welded
    if verbose:
        _log.info(
            "  After weld: %d vertices (-%d), %d faces",
            len(merged.vertices), n_welded, len(merged.faces),
        )

    # ── 3b. Remove overlapping / duplicate faces ──────────────────────
    # When multiple terrain tiles from different municipalities overlap,
    # the merged mesh has duplicate triangles that cause self-intersections.
    # These self-intersections make the Boolean union fail on MeshLib.
    merged, n_overlap = remove_overlapping_faces(
        merged, tolerance=weld_tolerance * 10,
    )
    if verbose and n_overlap:
        _log.info(
            "  Removed %d overlapping faces -> %d faces remain",
            n_overlap, len(merged.faces),
        )

    # ── 4. Extrude to solid ───────────────────────────────────────────
    solid, ext_stats = extrude_terrain_solid(
        merged, extrusion_depth=extrusion_depth, bottom_z=bottom_z,
    )
    # Copy relevant fields
    stats.non_manifold_fixed = ext_stats.non_manifold_fixed
    stats.pinch_points_split = ext_stats.pinch_points_split
    stats.boundary_edges = ext_stats.boundary_edges
    stats.wall_faces = ext_stats.wall_faces
    stats.bottom_faces = ext_stats.bottom_faces
    stats.final_vertices = ext_stats.final_vertices
    stats.final_faces = ext_stats.final_faces
    stats.extrusion_depth = ext_stats.extrusion_depth
    stats.bottom_z = ext_stats.bottom_z

    # ── 5. Boolean union with base box to fill gaps ───────────────────
    if grid_bounds is not None:
        gmin_x, gmax_x, gmin_y, gmax_y = grid_bounds
        # Compute the base box top Z from the boundary-edge vertices.
        # Classify boundary edges into *outer perimeter* vs *internal gap*.
        # The outer perimeter is the longest boundary loop; every other
        # loop is a gap (river, missing tile).  The base box top Z is
        # set to the mean Z of gap-loop vertices (river-bank level).
        # The base box XY is clamped to the terrain mesh footprint so it
        # doesn't protrude beyond the outer boundary and create edge steps.
        boundary = compute_boundary_edges(merged.faces)
        if boundary:
            # ── Trace closed loops from boundary edges ────────────
            adj: Dict[int, List[int]] = {}
            for v1, v2 in boundary:
                adj.setdefault(v1, []).append(v2)
                adj.setdefault(v2, []).append(v1)
            visited_edges: Set[Tuple[int, int]] = set()
            loops: List[List[int]] = []
            for start in adj:
                if all(
                    (min(start, nb), max(start, nb)) in visited_edges
                    for nb in adj[start]
                ):
                    continue
                loop: List[int] = [start]
                cur = start
                while True:
                    found_next = False
                    for nb in adj[cur]:
                        ekey = (min(cur, nb), max(cur, nb))
                        if ekey not in visited_edges:
                            visited_edges.add(ekey)
                            loop.append(nb)
                            cur = nb
                            found_next = True
                            break
                    if not found_next or cur == start:
                        break
                if len(loop) > 2:
                    loops.append(loop)

            if len(loops) > 1:
                # Largest loop = outer perimeter; rest = internal gaps
                loops.sort(key=len, reverse=True)
                gap_loops = loops[1:]
                gap_verts = set()
                for lp in gap_loops:
                    gap_verts.update(lp)
                if gap_verts:
                    gap_z = merged.vertices[list(gap_verts), 2]
                    base_top_z = float(np.min(gap_z))
                else:
                    base_top_z = float(merged.vertices[:, 2].min())
            else:
                # Only one loop (outer boundary, no internal gaps).
                bnd_verts = set()
                for v1, v2 in boundary:
                    bnd_verts.add(v1)
                    bnd_verts.add(v2)
                base_top_z = float(
                    np.min(merged.vertices[list(bnd_verts), 2])
                )
        else:
            base_top_z = float(merged.vertices[:, 2].min())
            loops = []
        base_bottom_z = stats.bottom_z

        # Use the full target rectangle for the base box so it covers
        # the entire area including regions where the terrain mesh has
        # no coverage (river channels, missing tiles at the edges).
        pad = 2.0 * voxel_size
        box_min_x, box_max_x = gmin_x - pad, gmax_x + pad
        box_min_y, box_max_y = gmin_y - pad, gmax_y + pad

        if verbose:
            _log.info(
                "  Creating base box for gap fill: "
                "X[%.1f, %.1f] Y[%.1f, %.1f] Z[%.1f, %.1f]",
                box_min_x, box_max_x, box_min_y, box_max_y,
                base_bottom_z, base_top_z,
            )

        base_box = create_base_box(
            box_min_x, box_max_x, box_min_y, box_max_y,
            base_bottom_z, base_top_z,
        )
        stats.boolean_union_attempted = True

        union_result, engine, err = boolean_union_terrain(
            solid, base_box, voxel_size=voxel_size, verbose=verbose,
        )
        if union_result is not None and len(union_result.faces) > 0:
            solid = union_result
            stats.boolean_union_success = True
            stats.boolean_union_engine = engine
            stats.final_vertices = len(solid.vertices)
            stats.final_faces = len(solid.faces)
        else:
            if verbose:
                _log.warning(
                    "  Boolean union failed (%s) — using extruded solid without gap fill.",
                    err,
                )

    # ── 6. Validate ───────────────────────────────────────────────────
    val = validate_solid(solid)
    stats.is_watertight = val["is_watertight"]
    if verbose:
        _log.info(
            "  Terrain solid: %d verts, %d faces, watertight=%s "
            "(boundary=%d, non-manifold=%d)",
            stats.final_vertices,
            stats.final_faces,
            stats.is_watertight,
            val["boundary_edges"],
            val["non_manifold_edges"],
        )

    return solid, stats
