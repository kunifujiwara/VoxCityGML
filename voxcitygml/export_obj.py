"""
OBJ + MTL export utilities for VoxCityGML.

Exports two files that share the same coordinate system:
  1. **Mesh OBJ** – watertight-processed triangle meshes (building, bridge,
     terrain, vegetation) *before* voxelization.
  2. **Voxel OBJ** – the voxelized model as greedy-meshed boundary faces.

Coordinate convention (matching voxcity reference exporter)
-----------------------------------------------------------
After the reference's ``transpose(2,1,0)`` + axis swap the OBJ axes are:

    OBJ X  =  row_index  * voxel_size   (south from origin)
    OBJ Y  =  col_index  * voxel_size   (east  from origin)
    OBJ Z  =  z_index    * voxel_size   (up    from origin)

Both mesh and voxel OBJs are placed in this space so they overlay exactly.
The mapping from local-metre coordinates (x_east, y_north, z_up) is:

    OBJ_x = max_y - y_local          (row direction)
    OBJ_y = x_local - min_x          (col direction)
    OBJ_z = z_local - min_z          (elevation)

This is a rotation (det = +1) so face winding is preserved.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.prepared import prep as shapely_prep

from .models import Mesh3D, CityGMLMeshCollection, INCLUSIVE_SHELL_THRESHOLD
from .citygml.coordinates import (
    swap_coordinates_3d,
    create_rectangle_frame_transformer,
)
from .voxelizer3d import (
    Grid3DParams,
    _compute_grid_params_3d,
    _allocate_voxel_grid,
    _voxelize_mesh_group,
    _voxelize_single_mesh,
    _voxelize_meshlib_levelset,
    _voxelize_building_solid,
    _MESHLIB_VOXEL_AVAILABLE,
    GROUND_CODE,
    TREE_CODE,
    BUILDING_CODE,
)
from .watertight import make_watertight_mesh
from .terrain_solid import build_terrain_solid, create_base_box

import logging as _logging
_log = _logging.getLogger(__name__)

# MeshLib availability (cached once)
_HAS_MESHLIB = False
try:
    import meshlib.mrmeshpy as _mrmesh
    import meshlib.mrmeshnumpy as _mrmeshnumpy
    _HAS_MESHLIB = True
except ImportError:
    _mrmesh = None  # type: ignore[assignment]
    _mrmeshnumpy = None  # type: ignore[assignment]


def _clip_mesh_to_box(
    verts: np.ndarray,
    faces: np.ndarray,
    clip_box: Mesh3D,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Boolean intersection of a triangle mesh with a clipping box.

    Returns (clipped_verts, clipped_faces), or *None* on failure.
    Tries MeshLib first, then trimesh, then face-centroid filter.
    """
    min_pt = clip_box.vertices.min(axis=0)
    max_pt = clip_box.vertices.max(axis=0)

    # ── MeshLib path ──────────────────────────────────────────────────
    if _HAS_MESHLIB:
        try:
            va = np.ascontiguousarray(verts, dtype=np.float32)
            fa = np.ascontiguousarray(faces, dtype=np.int32)
            ml_a = _mrmeshnumpy.meshFromFacesVerts(fa, va)

            vb = np.ascontiguousarray(clip_box.vertices, dtype=np.float32)
            fb = np.ascontiguousarray(clip_box.faces, dtype=np.int32)
            ml_b = _mrmeshnumpy.meshFromFacesVerts(fb, vb)

            result = _mrmesh.boolean(
                ml_a, ml_b, _mrmesh.BooleanOperation.Intersection,
            )
            if result.valid() and result.mesh is not None:
                rv = _mrmeshnumpy.getNumpyVerts(result.mesh).astype(np.float64)
                rf = _mrmeshnumpy.getNumpyFaces(result.mesh.topology).astype(np.int32)
                if len(rf) > 0:
                    return rv, rf
        except Exception as exc:
            _log.debug("MeshLib intersection failed: %s", exc)

    # ── trimesh fallback ──────────────────────────────────────────────
    try:
        import trimesh
        tm_a = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tm_b = trimesh.Trimesh(
            vertices=clip_box.vertices, faces=clip_box.faces, process=False,
        )
        for eng in ("manifold", "blender"):
            try:
                r = trimesh.boolean.intersection([tm_a, tm_b], engine=eng)
                if r is not None and len(r.faces) > 0:
                    return np.array(r.vertices, dtype=np.float64), np.array(r.faces, dtype=np.int32)
            except Exception:
                continue
    except Exception as exc:
        _log.debug("trimesh intersection failed: %s", exc)

    # ── Last resort: centroid-based face filter ───────────────────────
    # Keep faces whose centroid (XY only) lies inside the clip box.
    return _clip_faces_by_centroid(verts, faces, min_pt, max_pt)


def _clip_faces_by_centroid(
    verts: np.ndarray,
    faces: np.ndarray,
    min_pt: np.ndarray,
    max_pt: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip mesh triangles against 4 XY half-planes of the bounding box.

    Uses Sutherland-Hodgman polygon clipping on each triangle, then
    fan-triangulates the resulting convex polygon.  This produces clean
    trim edges aligned exactly with the rectangle boundary.
    """
    out_verts: List[List[float]] = []
    out_faces: List[List[int]] = []
    v_offset = 0

    # Four clip planes: (axis, sign, threshold)
    #   axis=0 → X,  axis=1 → Y
    #   sign=+1 → keep pts where coord >= threshold (min side)
    #   sign=-1 → keep pts where coord <= threshold (max side)
    planes = [
        (0, +1, min_pt[0]),   # x >= min_x
        (0, -1, max_pt[0]),   # x <= max_x
        (1, +1, min_pt[1]),   # y >= min_y
        (1, -1, max_pt[1]),   # y <= max_y
    ]

    for face in faces:
        # Start with triangle as polygon (list of 3D points)
        poly = [verts[face[0]].tolist(),
                verts[face[1]].tolist(),
                verts[face[2]].tolist()]

        for axis, sign, thresh in planes:
            if not poly:
                break
            poly = _clip_polygon_by_plane(poly, axis, sign, thresh)

        if len(poly) < 3:
            continue

        # Fan-triangulate the convex polygon
        base = v_offset
        for p in poly:
            out_verts.append(p)
        for i in range(1, len(poly) - 1):
            out_faces.append([base, base + i, base + i + 1])
        v_offset += len(poly)

    if not out_faces:
        return None

    return np.array(out_verts, dtype=np.float64), np.array(out_faces, dtype=np.int32)


def _clip_polygon_by_plane(
    poly: List[List[float]],
    axis: int,
    sign: int,
    thresh: float,
) -> List[List[float]]:
    """Sutherland-Hodgman clip of a polygon against one half-plane.

    sign=+1 keeps coord[axis] >= thresh;  sign=-1 keeps coord[axis] <= thresh.
    """
    out: List[List[float]] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        c_in = (cur[axis] - thresh) * sign >= 0
        n_in = (nxt[axis] - thresh) * sign >= 0

        if c_in:
            out.append(cur)
            if not n_in:
                out.append(_intersect_edge(cur, nxt, axis, thresh))
        elif n_in:
            out.append(_intersect_edge(cur, nxt, axis, thresh))
    return out


def _intersect_edge(
    p0: List[float], p1: List[float], axis: int, thresh: float,
) -> List[float]:
    """Linearly interpolate the intersection of edge p0→p1 with plane."""
    d0 = p0[axis] - thresh
    d1 = p1[axis] - thresh
    denom = d0 - d1
    if abs(denom) < 1e-15:
        t = 0.5
    else:
        t = d0 / denom
    return [
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1]),
        p0[2] + t * (p1[2] - p0[2]),
    ]

# ── Default colour palette (RGB 0-1 float) ───────────────────────────
_CATEGORY_COLOURS: Dict[str, Tuple[float, float, float]] = {
    "building":    (0.706, 0.733, 0.847),
    "bridge":      (0.620, 0.620, 0.620),
    "terrain":     (0.737, 0.561, 0.561),
    "vegetation":  (0.306, 0.388, 0.247),
}

_VOXEL_COLOURS: Dict[int, Tuple[str, Tuple[float, float, float]]] = {
    GROUND_CODE:   ("ground",       (0.737, 0.561, 0.561)),
    TREE_CODE:     ("tree_canopy",  (0.306, 0.388, 0.247)),
    BUILDING_CODE: ("building",     (0.706, 0.733, 0.847)),
    1:  ("bareland",      (0.937, 0.894, 0.690)),
    2:  ("rangeland",     (0.482, 0.510, 0.231)),
    3:  ("shrub",         (0.380, 0.549, 0.337)),
    4:  ("agriculture",   (0.439, 0.471, 0.220)),
    5:  ("tree_lc",       (0.455, 0.588, 0.259)),
    6:  ("moss_lichen",   (0.733, 0.800, 0.157)),
    7:  ("wetland",       (0.302, 0.463, 0.388)),
    8:  ("mangrove",      (0.086, 0.239, 0.200)),
    9:  ("water",         (0.173, 0.259, 0.522)),
    10: ("snow_ice",      (0.804, 0.843, 0.878)),
    11: ("developed",     (0.424, 0.467, 0.506)),
    12: ("road",          (0.231, 0.243, 0.341)),
    13: ("building_lc",   (0.588, 0.651, 0.745)),
    14: ("nodata",        (0.937, 0.894, 0.690)),
    -11: ("brick",     (0.318, 0.231, 0.220)),
    -12: ("wood",      (0.973, 0.651, 0.008)),
    -13: ("concrete",  (0.729, 0.733, 0.710)),
    -14: ("metal",     (0.545, 0.584, 0.624)),
    -15: ("stone",     (0.576, 0.549, 0.447)),
    -16: ("glass",     (0.220, 0.306, 0.329)),
    -17: ("plaster",   (0.933, 0.949, 0.918)),
}


# =====================================================================
# Face winding (matching voxcity reference)
# =====================================================================

def _create_face_vertices(coords, positive_direction, axis):
    """Return vertices in correct winding order for a quad face."""
    if axis == 'y':
        if positive_direction:
            return [coords[3], coords[2], coords[1], coords[0]]
        else:
            return [coords[0], coords[1], coords[2], coords[3]]
    else:
        if positive_direction:
            return [coords[0], coords[3], coords[2], coords[1]]
        else:
            return [coords[0], coords[1], coords[2], coords[3]]


# =====================================================================
# Greedy meshing (adapted from voxcity reference exporter)
# =====================================================================

def _greedy_mesh_layer(
    mask, layer_index, axis, positive_direction, normal_idx, voxel_size,
    vertex_dict, vertex_list, faces_per_material, voxel_value_to_material,
):
    """Greedy-mesh one 2D boundary layer into merged quads."""
    mask = mask.copy()
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    vs = voxel_size

    for u in range(h):
        v = 0
        while v < w:
            if visited[u, v] or mask[u, v] == 0:
                v += 1
                continue

            voxel_value = int(mask[u, v])
            mat_name = voxel_value_to_material.get(voxel_value, f"class_{voxel_value}")

            # Find max width of consecutive same-value voxels
            width = 1
            while (v + width < w
                   and mask[u, v + width] == voxel_value
                   and not visited[u, v + width]):
                width += 1

            # Find max height
            height = 1
            done = False
            while u + height < h and not done:
                for k in range(width):
                    if (mask[u + height, v + k] != voxel_value
                            or visited[u + height, v + k]):
                        done = True
                        break
                if not done:
                    height += 1

            visited[u:u + height, v:v + width] = True

            # Generate vertex coordinates (pre-swap space)
            if axis == 'x':
                i = float(layer_index) * vs
                y0 = float(u) * vs
                y1 = float(u + height) * vs
                z0 = float(v) * vs
                z1 = float(v + width) * vs
                coords = [(i, y0, z0), (i, y1, z0), (i, y1, z1), (i, y0, z1)]
            elif axis == 'y':
                i = float(layer_index) * vs
                x0 = float(u) * vs
                x1 = float(u + height) * vs
                z0 = float(v) * vs
                z1 = float(v + width) * vs
                coords = [(x0, i, z0), (x1, i, z0), (x1, i, z1), (x0, i, z1)]
            elif axis == 'z':
                i = float(layer_index) * vs
                x0 = float(u) * vs
                x1 = float(u + height) * vs
                y0 = float(v) * vs
                y1 = float(v + width) * vs
                coords = [(x0, y0, i), (x1, y0, i), (x1, y1, i), (x0, y1, i)]
            else:
                v += width
                continue

            # Convert to right-handed coordinate system (matching reference)
            coords = [(c[2], c[1], c[0]) for c in coords]
            face_verts = _create_face_vertices(coords, positive_direction, axis)

            # Get / create vertex indices (1-based for OBJ)
            indices = []
            for coord in face_verts:
                if coord not in vertex_dict:
                    vertex_list.append(coord)
                    vertex_dict[coord] = len(vertex_list)
                indices.append(vertex_dict[coord])

            # Triangulate with correct winding
            if axis == 'y':
                tris = [
                    {'vertices': [indices[2], indices[1], indices[0]],
                     'normal_idx': normal_idx},
                    {'vertices': [indices[3], indices[2], indices[0]],
                     'normal_idx': normal_idx},
                ]
            else:
                tris = [
                    {'vertices': [indices[0], indices[1], indices[2]],
                     'normal_idx': normal_idx},
                    {'vertices': [indices[0], indices[2], indices[3]],
                     'normal_idx': normal_idx},
                ]

            if mat_name not in faces_per_material:
                faces_per_material[mat_name] = []
            faces_per_material[mat_name].extend(tris)

            v += width


# =====================================================================
# Low-level OBJ / MTL writers
# =====================================================================

def _write_mtl(path: str, materials: Dict[str, Tuple[float, float, float]]) -> None:
    """Write a Wavefront MTL file (matching reference format)."""
    with open(path, "w") as f:
        f.write("# Material file\n\n")
        for name, (r, g, b) in sorted(materials.items()):
            f.write(f"newmtl {name}\n")
            f.write(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"Ke {r:.6f} {g:.6f} {b:.6f}\n")
            f.write("Ks 0.500000 0.500000 0.500000\n")
            f.write("Ns 50.000000\n")
            f.write("illum 2\n\n")


def _fix_face_winding(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Ensure face normals point *outward* from the mesh centroid.

    For each face, compute the face normal (via cross product) and check
    whether it points away from the overall mesh centroid.  If the majority
    of faces point inward, flip **all** face winding (reverse vertex order).

    Returns the (possibly flipped) faces array.
    """
    if len(faces) == 0:
        return faces

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)          # face normals (area-weighted)
    fc = (v0 + v1 + v2) / 3.0                # face centres
    centroid = verts.mean(axis=0)             # mesh centroid
    dots = np.einsum("ij,ij->i", fn, fc - centroid)

    if np.sum(dots < 0) > len(faces) / 2:
        # Majority of normals point inward → flip winding
        faces = faces[:, ::-1].copy()
    return faces


def _write_mesh_obj(
    path: str,
    mtl_filename: str,
    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    materials: Dict[str, Tuple[float, float, float]],
    gp: Grid3DParams,
) -> None:
    """Write triangle-mesh groups, transformed to voxel-OBJ coordinate space.

    Transform: local (x_east, y_north, z_up) -> OBJ (row, col, z_layer)
        OBJ_x = max_y - y_local          (row direction, south from origin)
        OBJ_y = x_local - min_x          (col direction, east from origin)
        OBJ_z = z_local - min_z          (elevation from origin)

    Each face gets its own per-face normal (flat shading) to avoid smooth
    shading artefacts at sharp building edges.  Face winding is also
    validated per-mesh.
    """
    with open(path, "w") as f:
        f.write("# VoxCityGML mesh export\n\n")
        f.write("o \n\n")
        f.write(f"mtllib {mtl_filename}\n\n")
        v_offset = 0
        vn_offset = 0
        for group_name in sorted(groups.keys()):
            meshes = groups[group_name]
            if not meshes:
                continue
            f.write(f"g {group_name}\n")
            f.write("s off\n")
            if group_name in materials:
                f.write(f"usemtl {group_name}\n")
            for verts, faces in meshes:
                # Transform vertices to OBJ coordinate space
                verts_obj = np.empty_like(verts)
                verts_obj[:, 0] = gp.max_y - verts[:, 1]   # row direction
                verts_obj[:, 1] = verts[:, 0] - gp.min_x   # col direction
                verts_obj[:, 2] = verts[:, 2] - gp.min_z   # elevation

                # Fix face winding so normals point outward
                faces = _fix_face_winding(verts_obj, faces)

                # Compute per-face normals (flat shading)
                v0 = verts_obj[faces[:, 0]]
                v1 = verts_obj[faces[:, 1]]
                v2 = verts_obj[faces[:, 2]]
                fn = np.cross(v1 - v0, v2 - v0)
                fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
                fn_len = np.where(fn_len < 1e-12, 1.0, fn_len)
                fn = fn / fn_len

                # Write vertices
                for v in verts_obj:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                # Write per-face normals
                for n in fn:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                # Write faces: each face's 3 vertices share the same face-normal
                for fi, face in enumerate(faces):
                    i0, i1, i2 = face + 1 + v_offset
                    ni = fi + 1 + vn_offset
                    f.write(f"f {i0}//{ni} {i1}//{ni} {i2}//{ni}\n")
                v_offset += len(verts)
                vn_offset += len(faces)
            f.write("\n")


def _greedy_mesh_single_code(
    grid: np.ndarray,
    code: int,
    voxel_size: float,
) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[int, int, int, int]],
]:
    """Greedy-mesh voxels of exactly one class *code*.

    Returns
    -------
    vertex_list : list[(x, y, z)]
        Deduplicated vertex positions.
    faces : list[(v0, v1, v2, normal_idx)]
        Triangle faces as 0-based vertex indices + 1-based normal index.
    """
    mat_name = "mat"
    voxel_value_to_material = {code: mat_name}

    arr = grid.transpose(2, 1, 0)  # (z_orig, col, row) — same as reference
    sx, sy, sz = arr.shape

    vertex_list: list = []
    vertex_dict: dict = {}
    faces_per_material: dict = {}

    normal_indices = {
        'px': 1, 'nx': 2, 'py': 3, 'ny': 4, 'pz': 5, 'nz': 6,
    }
    directions = [
        ('nx', (-1, 0, 0)), ('px', (1, 0, 0)),
        ('ny', (0, -1, 0)), ('py', (0, 1, 0)),
        ('nz', (0, 0, -1)), ('pz', (0, 0, 1)),
    ]

    for direction, _ in directions:
        ni = normal_indices[direction]

        if direction in ('nx', 'px'):
            for x in range(sx):
                voxel_slice = arr[x, :, :]
                if direction == 'nx':
                    neighbor = arr[x - 1, :, :] if x > 0 else np.zeros_like(voxel_slice)
                    layer = x
                else:
                    neighbor = arr[x + 1, :, :] if x + 1 < sx else np.zeros_like(voxel_slice)
                    layer = x + 1
                mask = np.where(
                    (voxel_slice != neighbor) & (voxel_slice != 0),
                    voxel_slice, 0,
                )
                _greedy_mesh_layer(
                    mask, layer, 'x', direction == 'px', ni,
                    voxel_size, vertex_dict, vertex_list,
                    faces_per_material, voxel_value_to_material,
                )

        elif direction in ('ny', 'py'):
            for y in range(sy):
                voxel_slice = arr[:, y, :]
                if direction == 'ny':
                    neighbor = arr[:, y - 1, :] if y > 0 else np.zeros_like(voxel_slice)
                    layer = y
                else:
                    neighbor = arr[:, y + 1, :] if y + 1 < sy else np.zeros_like(voxel_slice)
                    layer = y + 1
                mask = np.where(
                    (voxel_slice != neighbor) & (voxel_slice != 0),
                    voxel_slice, 0,
                )
                _greedy_mesh_layer(
                    mask, layer, 'y', direction == 'py', ni,
                    voxel_size, vertex_dict, vertex_list,
                    faces_per_material, voxel_value_to_material,
                )

        elif direction in ('nz', 'pz'):
            for z in range(sz):
                voxel_slice = arr[:, :, z]
                if direction == 'nz':
                    neighbor = arr[:, :, z - 1] if z > 0 else np.zeros_like(voxel_slice)
                    layer = z
                else:
                    neighbor = arr[:, :, z + 1] if z + 1 < sz else np.zeros_like(voxel_slice)
                    layer = z + 1
                mask = np.where(
                    (voxel_slice != neighbor) & (voxel_slice != 0),
                    voxel_slice, 0,
                )
                _greedy_mesh_layer(
                    mask, layer, 'z', direction == 'pz', ni,
                    voxel_size, vertex_dict, vertex_list,
                    faces_per_material, voxel_value_to_material,
                )

    # The greedy mesher swaps coordinates via (c[2],c[1],c[0])
    # which exchanges X↔Z in output space.  The normal indices
    # still refer to the pre-swap axes, so remap X↔Z normals:
    #   +X(1)↔+Z(5),  -X(2)↔-Z(6),  ±Y unchanged.
    _ni_remap = {1: 5, 2: 6, 3: 3, 4: 4, 5: 1, 6: 2}

    # Flatten triangles with corrected normal indices
    faces: List[Tuple[int, int, int, int]] = []
    for face_list in faces_per_material.values():
        for fd in face_list:
            v0, v1, v2 = fd['vertices']      # 1-based from _greedy_mesh_layer
            ni = _ni_remap[fd['normal_idx']]
            faces.append((v0 - 1, v1 - 1, v2 - 1, ni))

    return vertex_list, faces


def _write_voxel_obj_greedy(
    path: str,
    mtl_filename: str,
    voxel_grid: np.ndarray,
    voxel_size: float,
    materials: Dict[str, Tuple[float, float, float]],
    code_to_name: Dict[int, str],
) -> int:
    """Write voxel OBJ using greedy meshing (matching voxcity reference)."""
    voxel_value_to_material = {code: name for code, name in code_to_name.items()}

    # Transpose to (z_orig, col, row) = (x, y, z) — same as reference
    arr = voxel_grid.transpose(2, 1, 0)
    sx, sy, sz = arr.shape

    vertex_list: list = []
    vertex_dict: dict = {}
    faces_per_material: dict = {}

    # 6 pre-defined normals
    normals = [
        (1.0, 0.0, 0.0),    # 1: +X
        (-1.0, 0.0, 0.0),   # 2: -X
        (0.0, 1.0, 0.0),    # 3: +Y
        (0.0, -1.0, 0.0),   # 4: -Y
        (0.0, 0.0, 1.0),    # 5: +Z
        (0.0, 0.0, -1.0),   # 6: -Z
    ]
    normal_indices = {
        'px': 1, 'nx': 2, 'py': 3, 'ny': 4, 'pz': 5, 'nz': 6,
    }

    directions = [
        ('nx', (-1, 0, 0)), ('px', (1, 0, 0)),
        ('ny', (0, -1, 0)), ('py', (0, 1, 0)),
        ('nz', (0, 0, -1)), ('pz', (0, 0, 1)),
    ]

    for direction, _ in directions:
        ni = normal_indices[direction]

        if direction in ('nx', 'px'):
            for x in range(sx):
                voxel_slice = arr[x, :, :]
                if direction == 'nx':
                    neighbor = arr[x - 1, :, :] if x > 0 else np.zeros_like(voxel_slice)
                    layer = x
                else:
                    neighbor = arr[x + 1, :, :] if x + 1 < sx else np.zeros_like(voxel_slice)
                    layer = x + 1
                mask = np.where((voxel_slice != neighbor) & (voxel_slice != 0), voxel_slice, 0)
                _greedy_mesh_layer(mask, layer, 'x', direction == 'px', ni,
                                   voxel_size, vertex_dict, vertex_list,
                                   faces_per_material, voxel_value_to_material)

        elif direction in ('ny', 'py'):
            for y in range(sy):
                voxel_slice = arr[:, y, :]
                if direction == 'ny':
                    neighbor = arr[:, y - 1, :] if y > 0 else np.zeros_like(voxel_slice)
                    layer = y
                else:
                    neighbor = arr[:, y + 1, :] if y + 1 < sy else np.zeros_like(voxel_slice)
                    layer = y + 1
                mask = np.where((voxel_slice != neighbor) & (voxel_slice != 0), voxel_slice, 0)
                _greedy_mesh_layer(mask, layer, 'y', direction == 'py', ni,
                                   voxel_size, vertex_dict, vertex_list,
                                   faces_per_material, voxel_value_to_material)

        elif direction in ('nz', 'pz'):
            for z in range(sz):
                voxel_slice = arr[:, :, z]
                if direction == 'nz':
                    neighbor = arr[:, :, z - 1] if z > 0 else np.zeros_like(voxel_slice)
                    layer = z
                else:
                    neighbor = arr[:, :, z + 1] if z + 1 < sz else np.zeros_like(voxel_slice)
                    layer = z + 1
                mask = np.where((voxel_slice != neighbor) & (voxel_slice != 0), voxel_slice, 0)
                _greedy_mesh_layer(mask, layer, 'z', direction == 'pz', ni,
                                   voxel_size, vertex_dict, vertex_list,
                                   faces_per_material, voxel_value_to_material)

    # Count total faces
    n_faces = sum(len(fl) for fl in faces_per_material.values())
    print(f"  Greedy meshing: {len(vertex_list):,} vertices, {n_faces:,} faces")

    # Write OBJ
    with open(path, 'w') as f:
        f.write('# VoxCityGML voxel export (greedy meshed)\n\n')
        f.write('o \n\n')
        f.write(f'mtllib {mtl_filename}\n\n')

        f.write('# normals\n')
        for nx, ny, nz in normals:
            f.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
        f.write('\n')

        f.write('# verts\n')
        for vx, vy, vz in vertex_list:
            f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        f.write('\n')

        f.write('# faces\n')
        # Remap normal indices: coord swap (c[2],c[1],c[0]) exchanges X↔Z
        _ni_remap = {1: 5, 2: 6, 3: 3, 4: 4, 5: 1, 6: 2}
        for mat_name, faces in faces_per_material.items():
            f.write(f'usemtl {mat_name}\n')
            for face in faces:
                ni = _ni_remap[face["normal_idx"]]
                face_str = ' '.join(
                    f'{vi}//{ni}' for vi in face['vertices']
                )
                f.write(f'f {face_str}\n')
            f.write('\n')

    return n_faces


# =====================================================================
# Public API
# =====================================================================

def export_meshes_obj(
    collection: CityGMLMeshCollection,
    center_lon: float,
    center_lat: float,
    output_dir: str,
    gp: Grid3DParams,
    basename: str = "meshes",
    watertight: bool = True,
    voxel_size: float = 1.0,
    *,
    rectangle_vertices,
) -> str:
    """Export CityGML meshes as OBJ + MTL.

    Coordinates are transformed to the same index-based space used by
    the voxel OBJ so both files overlay perfectly.

    Parameters
    ----------
    gp : Grid3DParams
        Grid params (needed for the coordinate transform).
    rectangle_vertices : sequence of 4 (lon, lat), keyword-only, required
        Target rectangle, in VoxCity order ``[SW, NW, NE, SE]``.  Meshes
        are placed in the same rectangle-aligned frame as ``gp``.

        This is **required**, not optional.  ``gp`` can only come from
        ``_compute_grid_params_3d``, which always works in the rectangle
        frame, and ``gp.min_x/max_x/min_y/max_y`` set both the OBJ origin
        and the clip box below.  Placing meshes in any other frame would
        rotate them relative to the voxel OBJ and clip them against the
        wrong region — with no exception to reveal it.  There is no
        run-time way to detect the mismatch, so the frame is pinned at the
        call site instead.  It is keyword-only so that an old positional
        call cannot silently bind something else to it.
    """
    os.makedirs(output_dir, exist_ok=True)

    transformer = create_rectangle_frame_transformer(
        center_lon, center_lat, rectangle_vertices)

    def _to_local(mesh: Mesh3D) -> Tuple[np.ndarray, np.ndarray]:
        verts_ll = swap_coordinates_3d(mesh.vertices)
        x_m, y_m = transformer.transform(verts_ll[:, 0], verts_ll[:, 1])
        return np.column_stack([x_m, y_m, verts_ll[:, 2]]), mesh.faces

    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        "building": [],
        "bridge": [],
        "terrain": [],
        "vegetation": [],
    }

    # Buildings – optionally watertight
    print("  [mesh-export] Processing buildings ...")
    for i, mesh in enumerate(collection.buildings):
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        verts, faces = _to_local(mesh)
        if watertight:
            wt = make_watertight_mesh(verts, faces, voxel_size=voxel_size)
            if wt.is_watertight and len(wt.faces) > 0:
                verts, faces = wt.vertices, wt.faces
        groups["building"].append((verts, faces))
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(collection.buildings)} buildings")
    print(f"    Total: {len(groups['building'])} buildings")

    # Bridges
    print("  [mesh-export] Processing bridges ...")
    for mesh in collection.bridges:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        groups["bridge"].append(_to_local(mesh))
    print(f"    Total: {len(groups['bridge'])} bridges")

    # Terrain – build watertight solid with boolean-union gap fill
    print("  [mesh-export] Processing terrain (watertight solid) ...")
    local_terrain: List[Mesh3D] = []
    for mesh in collection.terrain:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        verts, faces = _to_local(mesh)
        local_terrain.append(Mesh3D(
            vertices=verts, faces=faces,
            feature_type=mesh.feature_type,
            feature_id=mesh.feature_id,
        ))
    if local_terrain:
        solid, _stats = build_terrain_solid(
            local_terrain,
            bottom_z=gp.min_z,
            weld_tolerance=voxel_size * 1e-3,
            grid_bounds=(gp.min_x, gp.max_x, gp.min_y, gp.max_y),
            verbose=True,
        )
        if solid is not None and len(solid.faces) > 0:
            groups["terrain"].append((solid.vertices, solid.faces))
            print(f"    Terrain solid: {len(solid.vertices):,} verts, "
                  f"{len(solid.faces):,} faces, "
                  f"watertight={_stats.is_watertight}")
        else:
            # Fallback: raw meshes
            for tm in local_terrain:
                groups["terrain"].append((tm.vertices, tm.faces))
            print(f"    Terrain solid failed – exported {len(local_terrain)} raw meshes")
    print(f"    Total: {len(groups['terrain'])} terrain group(s)")

    # Vegetation
    print("  [mesh-export] Processing vegetation ...")
    for mesh in collection.vegetation:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        groups["vegetation"].append(_to_local(mesh))
    print(f"    Total: {len(groups['vegetation'])} vegetation meshes")

    # ── Clip all meshes to target rectangle ──────────────────────────
    # Build a tall clipping box from the grid bounds so meshes that
    # extend outside the target area are trimmed cleanly.
    clip_box = create_base_box(
        gp.min_x, gp.max_x,
        gp.min_y, gp.max_y,
        gp.min_z, gp.max_z,
    )
    print("  [mesh-export] Clipping meshes to target rectangle ...")
    for group_name in list(groups.keys()):
        clipped: List[Tuple[np.ndarray, np.ndarray]] = []
        n_before = len(groups[group_name])
        for verts, faces in groups[group_name]:
            result = _clip_mesh_to_box(verts, faces, clip_box)
            if result is not None:
                clipped.append(result)
            # else: mesh is entirely outside – drop it
        groups[group_name] = clipped
        n_dropped = n_before - len(clipped)
        if n_dropped:
            print(f"    {group_name}: kept {len(clipped)}/{n_before} "
                  f"(dropped {n_dropped} outside)")
    total_clipped = sum(len(f) for meshes in groups.values() for _, f in meshes)
    print(f"    Clipped total: {total_clipped:,} faces")

    # Write files
    mtl_name = basename + ".mtl"
    obj_path = os.path.join(output_dir, basename + ".obj")
    mtl_path = os.path.join(output_dir, mtl_name)

    _write_mtl(mtl_path, _CATEGORY_COLOURS)
    _write_mesh_obj(obj_path, mtl_name, groups, _CATEGORY_COLOURS, gp)

    total_v = sum(len(v) for meshes in groups.values() for v, _ in meshes)
    total_f = sum(len(f) for meshes in groups.values() for _, f in meshes)
    print(f"  [mesh-export] Wrote {obj_path}  ({total_v:,} verts, {total_f:,} faces)")
    return obj_path, groups


def export_voxels_obj(
    voxel_grid: np.ndarray,
    collection: CityGMLMeshCollection,
    rectangle_vertices,
    center_lon: float,
    center_lat: float,
    meshsize: float,
    output_dir: str,
    basename: str = "voxels",
    underground_depth: float = 0.0,
) -> Tuple[str, Grid3DParams]:
    """Export a voxel grid as OBJ + MTL using greedy meshing.

    Returns
    -------
    (str, Grid3DParams) : path to the written OBJ file + grid params.
    """
    os.makedirs(output_dir, exist_ok=True)

    gp, _ = _compute_grid_params_3d(
        rectangle_vertices, center_lon, center_lat, meshsize, collection,
        underground_depth=underground_depth,
    )

    # Build material dict and code->name mapping
    materials: Dict[str, Tuple[float, float, float]] = {}
    code_to_name: Dict[int, str] = {}
    codes_present = np.unique(voxel_grid)
    for code in codes_present:
        code = int(code)
        if code == 0:
            continue
        if code in _VOXEL_COLOURS:
            name, colour = _VOXEL_COLOURS[code]
        else:
            name = f"class_{code}"
            colour = (0.5, 0.5, 0.5)
        materials[name] = colour
        code_to_name[code] = name

    mtl_name = basename + ".mtl"
    obj_path = os.path.join(output_dir, basename + ".obj")
    mtl_path = os.path.join(output_dir, mtl_name)

    _write_mtl(mtl_path, materials)
    n = _write_voxel_obj_greedy(
        obj_path, mtl_name, voxel_grid, meshsize,
        materials, code_to_name,
    )
    print(f"  [voxel-export] Wrote {obj_path}  ({n:,} faces)")
    return obj_path, gp


# Semantic codes for the per-category voxel grid
_PER_CAT_CODES: Dict[str, int] = {
    "building":   BUILDING_CODE,    # -3
    "bridge":     -4,               # dedicated bridge code
    "terrain":    GROUND_CODE,      # -1
    "vegetation": TREE_CODE,        # -2
}

_PER_CAT_COLOURS: Dict[int, Tuple[str, Tuple[float, float, float]]] = {
    BUILDING_CODE: ("building",   (0.706, 0.733, 0.847)),
    -4:            ("bridge",     (0.620, 0.620, 0.620)),
    GROUND_CODE:   ("terrain",    (0.737, 0.561, 0.561)),
    TREE_CODE:     ("vegetation", (0.306, 0.388, 0.247)),
}


def export_per_category_voxels_obj(
    collection: CityGMLMeshCollection,
    rectangle_vertices,
    center_lon: float,
    center_lat: float,
    meshsize: float,
    output_dir: str,
    # Keyword-only tail: this is an exported 14-parameter function whose
    # optional block has already been inserted into once (2026-08-17, the
    # two shell knobs below), silently shifting mesh_groups and
    # underground_depth right.  The `*` makes the next such insertion
    # structurally incapable of breaking a caller.
    *,
    basename: str = "mesh_voxels",
    max_voxel_ram_mb: Optional[float] = None,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
    building_shell_threshold: float = INCLUSIVE_SHELL_THRESHOLD,
    shell_anchor: str = "connected",
    mesh_groups: Optional[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = None,
    underground_depth: float = 0.0,
) -> Tuple[str, Grid3DParams]:
    """Voxelize each mesh category independently and export as OBJ + MTL.

    Each category is voxelized into its **own** grid so that overlapping
    geometry (e.g. buildings penetrating terrain) is preserved in full.
    The per-category grids are greedy-meshed separately and written as
    distinct material groups in a single OBJ file.

    ``building_shell_threshold`` / ``shell_anchor`` default to the
    inclusive mode (``INCLUSIVE_SHELL_THRESHOLD`` = 0.0 / "connected").
    Callers driving this from a ``VoxelizerConfig`` should pass their
    ``resolved_voxel_params()`` values instead of relying on the defaults,
    so exported building voxels match the main grid produced by
    ``voxelize_citygml_meshes`` (the 2026-08-11 alignment invariant).

    If *mesh_groups* is provided (from ``export_meshes_obj``), those
    already-clipped local-metre meshes are voxelized directly.
    Otherwise falls back to voxelizing raw collection meshes.

    Returns
    -------
    (str, Grid3DParams) : path to the written OBJ file + grid params.
    """
    os.makedirs(output_dir, exist_ok=True)

    gp, transformer = _compute_grid_params_3d(
        rectangle_vertices, center_lon, center_lat, meshsize, collection,
        underground_depth=underground_depth,
    )

    cat_code_map: Dict[str, int] = {
        "building":   BUILDING_CODE,
        "bridge":     -4,
        "terrain":    GROUND_CODE,
        "vegetation": TREE_CODE,
    }

    # Voxelize each category into its own independent grid
    per_cat_grids: Dict[str, np.ndarray] = {}

    if mesh_groups is not None:
        for cat_name, code in cat_code_map.items():
            meshes = mesh_groups.get(cat_name, [])
            if not meshes:
                continue
            cat_grid = _allocate_voxel_grid(gp, max_voxel_ram_mb=max_voxel_ram_mb)
            # Buildings use the same seam as the main voxel grid --
            # grid-aligned winding on the raw mesh + occupancy shell -- so
            # exported building voxels match voxelize_citygml_meshes exactly
            # (alignment fix 2026-08-11), PROVIDED the caller passes the same
            # building_shell_threshold / shell_anchor the main grid used
            # (2026-08-17: both are now parameters, not silent defaults).
            #
            # Bridges and terrain keep the levelset path: they still carry
            # its +half-voxel stamp displacement, consistent with their
            # history but now DIFFERENT from buildings in the same OBJ.
            # Fixing them means fixing
            # _stamp_meshlib_mask's convention, which must happen together
            # with removing the terrain path's -0.5-voxel compensation
            # (voxelizer3d.py ~:446-450); see
            # docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md.
            use_levelset = (
                cat_name in ("bridge", "terrain")
                and _MESHLIB_VOXEL_AVAILABLE
            )
            for verts, faces in meshes:
                if len(verts) == 0 or len(faces) == 0:
                    continue
                if cat_name == "building":
                    _voxelize_building_solid(
                        verts, faces, gp, cat_grid,
                        class_code=code, overwrite=False,
                        occupancy_threshold=occupancy_threshold,
                        occupancy_subdivisions=occupancy_subdivisions,
                        shell_threshold=building_shell_threshold,
                        shell_anchor=shell_anchor,
                    )
                    continue
                # Watertight meshes → prefer MeshLib level-set (no holes)
                if use_levelset:
                    ok = _voxelize_meshlib_levelset(
                        verts, faces, gp, cat_grid,
                        class_code=code, overwrite=False,
                    )
                    if ok:
                        continue
                _voxelize_single_mesh(
                    verts, faces, gp, cat_grid,
                    class_code=code,
                    overwrite=False,
                    seal_surface=(cat_name in ("building", "bridge", "terrain")),
                    occupancy_threshold=occupancy_threshold,
                    occupancy_subdivisions=occupancy_subdivisions,
                )
            n_filled = np.count_nonzero(cat_grid == code)
            if n_filled > 0:
                per_cat_grids[cat_name] = cat_grid
                print(f"  [per-cat] {cat_name}: {n_filled:,} voxels")
    else:
        # Fallback: voxelize raw collection meshes
        categories = [
            ("building",   collection.buildings,  BUILDING_CODE, False, False),
            ("bridge",     collection.bridges,    -4,            False, True),
            ("vegetation", collection.vegetation, TREE_CODE,     False, False),
            ("terrain",    collection.terrain,    GROUND_CODE,   False, False),
        ]
        for cat_name, meshes, code, overwrite, force_surface in categories:
            if not meshes:
                continue
            cat_grid = _allocate_voxel_grid(gp, max_voxel_ram_mb=max_voxel_ram_mb)
            _voxelize_mesh_group(
                meshes, transformer, gp, cat_grid,
                class_code=code,
                overwrite=False,
                occupancy_threshold=occupancy_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
                shell_threshold=building_shell_threshold,
                shell_anchor=shell_anchor,
                force_surface=force_surface,
            )
            n_filled = np.count_nonzero(cat_grid == code)
            if n_filled > 0:
                per_cat_grids[cat_name] = cat_grid
                print(f"  [per-cat] {cat_name}: {n_filled:,} voxels")

    # Build materials
    materials: Dict[str, Tuple[float, float, float]] = {}
    code_to_name: Dict[int, str] = {}
    for cat_name in per_cat_grids:
        code = cat_code_map[cat_name]
        if code in _PER_CAT_COLOURS:
            name, colour = _PER_CAT_COLOURS[code]
        else:
            name = cat_name
            colour = (0.5, 0.5, 0.5)
        materials[name] = colour
        code_to_name[code] = name

    mtl_name = basename + ".mtl"
    obj_path = os.path.join(output_dir, basename + ".obj")
    mtl_path = os.path.join(output_dir, mtl_name)

    _write_mtl(mtl_path, materials)

    # Greedy-mesh each category separately and write to one OBJ
    normals = [
        (1.0, 0.0, 0.0),    # 1: +X
        (-1.0, 0.0, 0.0),   # 2: -X
        (0.0, 1.0, 0.0),    # 3: +Y
        (0.0, -1.0, 0.0),   # 4: -Y
        (0.0, 0.0, 1.0),    # 5: +Z
        (0.0, 0.0, -1.0),   # 6: -Z
    ]
    total_faces = 0
    total_verts = 0
    with open(obj_path, "w") as f:
        f.write("# VoxCityGML per-category voxel export\n\n")
        f.write(f"mtllib {mtl_name}\n\n")
        # Write shared normals
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("\n")
        v_offset = 0
        for cat_name, cat_grid in per_cat_grids.items():
            code = cat_code_map[cat_name]
            mat_name = code_to_name[code]
            verts_list, faces_list = _greedy_mesh_single_code(
                cat_grid, code, meshsize,
            )
            if not verts_list:
                continue
            f.write(f"g {cat_name}\n")
            f.write(f"usemtl {mat_name}\n")
            f.write("s off\n")
            for v in verts_list:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for v0, v1, v2, ni in faces_list:
                i0 = v0 + 1 + v_offset
                i1 = v1 + 1 + v_offset
                i2 = v2 + 1 + v_offset
                f.write(f"f {i0}//{ni} {i1}//{ni} {i2}//{ni}\n")
            v_offset += len(verts_list)
            total_verts += len(verts_list)
            total_faces += len(faces_list)

    print(f"  Greedy meshing: {total_verts:,} vertices, {total_faces:,} faces")
    print(f"  [per-cat-export] Wrote {obj_path}  ({total_faces:,} faces)")
    return obj_path, gp


# =====================================================================
# Land-cover 2-D flat mesh export
# =====================================================================

# Colours keyed by 1-based VoxCity land-cover code
_LC_COLOURS: Dict[int, Tuple[str, Tuple[float, float, float]]] = {
    1:  ("bareland",     (0.937, 0.894, 0.690)),
    2:  ("rangeland",    (0.482, 0.510, 0.231)),
    3:  ("shrub",        (0.380, 0.549, 0.337)),
    4:  ("agriculture",  (0.439, 0.471, 0.220)),
    5:  ("tree",         (0.455, 0.588, 0.259)),
    6:  ("moss_lichen",  (0.733, 0.800, 0.157)),
    7:  ("wetland",      (0.302, 0.463, 0.388)),
    8:  ("mangrove",     (0.086, 0.239, 0.200)),
    9:  ("water",        (0.173, 0.259, 0.522)),
    10: ("snow_ice",     (0.804, 0.843, 0.878)),
    11: ("developed",    (0.424, 0.467, 0.506)),
    12: ("road",         (0.231, 0.243, 0.341)),
    13: ("building_lc",  (0.588, 0.651, 0.745)),
    14: ("nodata",       (0.937, 0.894, 0.690)),
}


def _triangulate_polygon_2d(
    polygon: ShapelyPolygon,
    all_coords: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Triangulate a Shapely polygon (may have holes) via mapbox_earcut.

    Parameters
    ----------
    polygon : ShapelyPolygon
        The polygon to triangulate, **already in OBJ 2-D coordinates**.
        May contain interior rings (holes).
    all_coords : (N, 2) array
        Concatenated vertices of exterior + all interior rings (no
        closing duplicates), in order.  Triangle indices refer to this
        array.

    Returns
    -------
    list[(i0, i1, i2)]
        Triangle vertex indices into *all_coords*.
    """
    import mapbox_earcut

    n = len(all_coords)
    if n < 3:
        return []

    # Build ring-end-indices array for earcut.
    # all_coords is: [exterior_verts..., hole1_verts..., hole2_verts..., ...]
    # mapbox_earcut wants cumulative end-indices: [ext_n, ext_n+h1_n, ...]
    ext_n = len(polygon.exterior.coords) - 1  # drop closing dup
    ring_ends = [ext_n]
    for interior in polygon.interiors:
        ring_ends.append(ring_ends[-1] + len(interior.coords) - 1)

    rings = np.array(ring_ends, dtype=np.uint32)
    coords = np.ascontiguousarray(all_coords, dtype=np.float64)

    tri_indices = mapbox_earcut.triangulate_float64(coords, rings)

    # tri_indices is a flat array of vertex indices, groups of 3
    result = []
    for i in range(0, len(tri_indices), 3):
        result.append((int(tri_indices[i]),
                        int(tri_indices[i + 1]),
                        int(tri_indices[i + 2])))
    return result


def export_landcover_obj(
    land_cover_grid: np.ndarray,
    land_cover_source: str,
    dem_grid: np.ndarray,
    gp: Grid3DParams,
    output_dir: str,
    basename: str = "landcover",
    *,
    citygml_path: str | list[str] | None = None,
    rectangle_vertices=None,
    center_lon: float | None = None,
    center_lat: float | None = None,
) -> str:
    """Export a flat 2-D land-cover mesh as OBJ + MTL.

    When *land_cover_source* is ``'CityGML'`` **and** the CityGML path /
    rectangle / centre are supplied, the actual CityGML polygon geometry
    is exported (true vector polygons, not grid quads).  Otherwise a
    grid-based quad mesh is written as fallback.

    All coordinates use the same X / Y system as the mesh and voxel OBJs
    (z = 0 everywhere).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Decide: polygon mode vs grid mode ────────────────────────────
    use_polygons = (
        land_cover_source == "CityGML"
        and citygml_path is not None
        and rectangle_vertices is not None
        and center_lon is not None
        and center_lat is not None
    )

    if use_polygons:
        return _export_landcover_polygon_obj(
            citygml_path, rectangle_vertices, center_lon, center_lat,
            gp, output_dir, basename,
        )
    else:
        return _export_landcover_grid_obj(
            land_cover_grid, land_cover_source, dem_grid,
            gp, output_dir, basename,
        )


# ------------------------------------------------------------------
# Polygon-based export  (CityGML vector geometry)
# ------------------------------------------------------------------

def _export_landcover_polygon_obj(
    citygml_path: str | list[str],
    rectangle_vertices,
    center_lon: float,
    center_lat: float,
    gp: Grid3DParams,
    output_dir: str,
    basename: str,
) -> str:
    from .landcover.citygml_landcover import get_citygml_land_cover_polygons

    print("  [lc-export] Extracting CityGML land use polygons...")
    polys = get_citygml_land_cover_polygons(citygml_path, rectangle_vertices)

    if not polys:
        obj_path = os.path.join(output_dir, basename + ".obj")
        with open(obj_path, "w") as f:
            f.write("# VoxCityGML land-cover mesh (no polygons found)\n")
        print("  [lc-export] WARNING: no land use polygons found")
        return obj_path

    # Coordinate transform: WGS 84 (lon, lat) → local metres → OBJ.
    # Same rectangle-aligned frame as the voxel / mesh exports, so the
    # land-cover layer overlays them for rotated rectangles too.  The
    # frame alone is not enough: ``get_citygml_land_cover_polygons``
    # clips to the rectangle polygon, not to its bounding box, so the
    # layer also has the same *extent* as the other exports (clipping to
    # the bbox left it overhanging the tile by ~90% extra area at 30 deg).
    transformer = create_rectangle_frame_transformer(
        center_lon, center_lat, rectangle_vertices)

    # ── Collect materials that actually appear ────────────────────────
    codes_present = set(code for code, _ in polys)
    materials: Dict[str, Tuple[float, float, float]] = {}
    code_to_mat: Dict[int, str] = {}
    for code in sorted(codes_present):
        if code in _LC_COLOURS:
            name, colour = _LC_COLOURS[code]
        else:
            name = f"lc_{code}"
            colour = (0.5, 0.5, 0.5)
        materials[name] = colour
        code_to_mat[code] = name

    mtl_name = basename + ".mtl"
    obj_path = os.path.join(output_dir, basename + ".obj")
    mtl_path = os.path.join(output_dir, mtl_name)
    _write_mtl(mtl_path, materials)

    # ── Transform and triangulate each polygon ───────────────────────
    # Group triangulated faces by material
    mat_tris: Dict[str, List[np.ndarray]] = {m: [] for m in materials}

    for code, shapely_poly in polys:
        mat = code_to_mat.get(code)
        if mat is None:
            continue

        # ── Collect all rings (exterior + holes) ─────────────────────
        ext_lonlat = np.array(shapely_poly.exterior.coords[:-1])
        if len(ext_lonlat) < 3:
            continue

        # Transform exterior ring
        ex, ey = transformer.transform(ext_lonlat[:, 0], ext_lonlat[:, 1])
        ext_obj = np.column_stack([gp.max_y - ey, ex - gp.min_x])

        # Transform interior rings (holes)
        hole_objs = []
        for interior in shapely_poly.interiors:
            h_lonlat = np.array(interior.coords[:-1])
            if len(h_lonlat) < 3:
                continue
            hx, hy = transformer.transform(h_lonlat[:, 0], h_lonlat[:, 1])
            hole_objs.append(np.column_stack([gp.max_y - hy, hx - gp.min_x]))

        # Build OBJ-space polygon (with holes) for triangulation filter
        obj_polygon = ShapelyPolygon(ext_obj, holes=hole_objs)

        # Concatenate all ring coords into one array; indices reference this
        all_coords = ext_obj
        for h in hole_objs:
            all_coords = np.vstack([all_coords, h])

        tri_indices = _triangulate_polygon_2d(obj_polygon, all_coords)
        if not tri_indices:
            continue
            continue

        # Build (N_tri, 3, 3) vertex array: each triangle as 3 (x, y, z=0) verts
        for i0, i1, i2 in tri_indices:
            tri_verts = np.array([
                [all_coords[i0, 0], all_coords[i0, 1], 0.0],
                [all_coords[i1, 0], all_coords[i1, 1], 0.0],
                [all_coords[i2, 0], all_coords[i2, 1], 0.0],
            ])
            mat_tris[mat].append(tri_verts)

    # ── Write OBJ ────────────────────────────────────────────────────
    total_verts = 0
    total_faces = 0

    with open(obj_path, "w") as f:
        f.write("# VoxCityGML land-cover polygon mesh export\n\n")
        f.write("o landcover\n\n")
        f.write(f"mtllib {mtl_name}\n\n")
        # Single upward normal for all flat polygons
        f.write("vn 0.000000 0.000000 1.000000\n\n")

        v_offset = 0
        for mat_name in sorted(mat_tris.keys()):
            tris = mat_tris[mat_name]
            if not tris:
                continue
            f.write(f"g {mat_name}\n")
            f.write(f"usemtl {mat_name}\n")
            f.write("s off\n")
            for tri in tris:
                for v in tri:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                i0 = v_offset + 1
                f.write(f"f {i0}//1 {i0+1}//1 {i0+2}//1\n")
                v_offset += 3
                total_verts += 3
                total_faces += 1
            f.write("\n")

    print(f"  [lc-export] Wrote {obj_path}  "
          f"({total_verts:,} verts, {total_faces:,} faces, polygon mode)")
    return obj_path


# ------------------------------------------------------------------
# Grid-based export  (fallback for non-CityGML sources)
# ------------------------------------------------------------------

def _export_landcover_grid_obj(
    land_cover_grid: np.ndarray,
    land_cover_source: str,
    dem_grid: np.ndarray,
    gp: Grid3DParams,
    output_dir: str,
    basename: str,
) -> str:
    from .voxelizer3d import _convert_land_cover

    vs = gp.voxel_size

    # ── Convert land-cover grid to 1-based codes ─────────────────────
    lc_1based = _convert_land_cover(land_cover_grid, land_cover_source)
    lc_north = np.flipud(lc_1based)

    if lc_north.shape != dem_grid.shape:
        from scipy.ndimage import zoom
        factor = (dem_grid.shape[0] / lc_north.shape[0],
                  dem_grid.shape[1] / lc_north.shape[1])
        lc_north = zoom(lc_north, factor, order=0).astype(lc_north.dtype)

    n_rows, n_cols = dem_grid.shape

    codes_present = set(int(c) for c in np.unique(lc_north) if int(c) != 0)
    materials: Dict[str, Tuple[float, float, float]] = {}
    code_to_mat: Dict[int, str] = {}
    for code in sorted(codes_present):
        if code in _LC_COLOURS:
            name, colour = _LC_COLOURS[code]
        else:
            name = f"lc_{code}"
            colour = (0.5, 0.5, 0.5)
        materials[name] = colour
        code_to_mat[code] = name

    mtl_name = basename + ".mtl"
    obj_path = os.path.join(output_dir, basename + ".obj")
    mtl_path = os.path.join(output_dir, mtl_name)
    _write_mtl(mtl_path, materials)

    mat_quads: Dict[str, List[Tuple[tuple, ...]]] = {m: [] for m in materials}

    for r in range(n_rows):
        for c in range(n_cols):
            code = int(lc_north[r, c])
            if code == 0:
                continue
            mat = code_to_mat.get(code)
            if mat is None:
                continue
            v0 = (r * vs, c * vs, 0.0)
            v1 = (r * vs, (c + 1) * vs, 0.0)
            v2 = ((r + 1) * vs, (c + 1) * vs, 0.0)
            v3 = ((r + 1) * vs, c * vs, 0.0)
            mat_quads[mat].append((v0, v1, v2, v3))

    # ── Write OBJ ────────────────────────────────────────────────────
    total_verts = 0
    total_faces = 0

    with open(obj_path, "w") as f:
        f.write("# VoxCityGML land-cover mesh export\n\n")
        f.write("o landcover\n\n")
        f.write(f"mtllib {mtl_name}\n\n")
        # Single upward normal for all flat quads
        f.write("vn 0.000000 0.000000 1.000000\n\n")

        v_offset = 0
        for mat_name in sorted(mat_quads.keys()):
            quads = mat_quads[mat_name]
            if not quads:
                continue
            f.write(f"g {mat_name}\n")
            f.write(f"usemtl {mat_name}\n")
            f.write("s off\n")
            for v0, v1, v2, v3 in quads:
                f.write(f"v {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                f.write(f"v {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write(f"v {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                i0 = v_offset + 1
                # Two triangles: (v0,v1,v2) and (v0,v2,v3)
                f.write(f"f {i0}//1 {i0+1}//1 {i0+2}//1\n")
                f.write(f"f {i0}//1 {i0+2}//1 {i0+3}//1\n")
                v_offset += 4
                total_verts += 4
                total_faces += 2
            f.write("\n")

    print(f"  [lc-export] Wrote {obj_path}  ({total_verts:,} verts, {total_faces:,} faces)")
    return obj_path
