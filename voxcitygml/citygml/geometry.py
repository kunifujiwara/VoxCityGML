"""
Geometry parsing utilities for CityGML.
Handles Polygon, MultiSurface, Solid, and triangulation.
Transplanted from citygml_mesher.geometry.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
    logging.getLogger(__name__).warning(
        "lxml not installed – falling back to stdlib xml.etree.ElementTree. "
        "Install lxml for faster CityGML parsing: pip install lxml"
    )

try:
    import mapbox_earcut as earcut
except ImportError:
    raise ImportError(
        "mapbox-earcut is required for correct polygon triangulation but is not installed.\n"
        "Install it with: pip install mapbox-earcut"
    ) from None


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

def parse_pos_list(pos_text: str, dims: int = 3) -> np.ndarray:
    """Parse gml:posList text into numpy array of coordinates."""
    if not pos_text:
        return np.array([], dtype=np.float64).reshape(0, dims)
    try:
        values = np.fromstring(pos_text, dtype=np.float64, sep=' ')
        n_coords = len(values) // dims
        if n_coords == 0:
            return np.array([], dtype=np.float64).reshape(0, dims)
        return values[:n_coords * dims].reshape(n_coords, dims)
    except (ValueError, TypeError):
        return np.array([], dtype=np.float64).reshape(0, dims)


def parse_pos_list_fast(pos_text: str) -> np.ndarray:
    """Ultra-fast parse for 3D coordinates (no validation)."""
    values = np.fromstring(pos_text, dtype=np.float64, sep=' ')
    return values.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Triangulation helpers
# ---------------------------------------------------------------------------

_TRIANGLE_FACES = np.array([[0, 1, 2]], dtype=np.int32)
_QUAD_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def triangulate_polygon_2d(coords_2d: np.ndarray) -> np.ndarray:
    """Triangulate a 2D polygon using earcut."""
    n = len(coords_2d)
    if n < 3:
        return np.array([], dtype=np.int32).reshape(0, 3)
    if n == 3:
        return _TRIANGLE_FACES
    if n == 4:
        return _QUAD_FACES

    min_c = coords_2d.min(axis=0)
    max_c = coords_2d.max(axis=0)
    extent = max_c - min_c
    extent = np.where(extent < 1e-10, 1.0, extent)
    normalised = (coords_2d - min_c) / extent
    normalised_2d = np.ascontiguousarray(normalised, dtype=np.float64)
    rings = np.array([n], dtype=np.uint32)
    indices = earcut.triangulate_float64(normalised_2d, rings)
    if len(indices) >= 3:
        return indices.reshape(-1, 3).astype(np.int32)

    # earcut returned fewer than 3 indices – degenerate polygon
    logging.getLogger(__name__).warning(
        "earcut returned %d indices for %d-vertex polygon; returning empty",
        len(indices), n,
    )
    return np.array([], dtype=np.int32).reshape(0, 3)


def triangulate_polygon_with_holes_2d(
    exterior_2d: np.ndarray,
    holes_2d: List[np.ndarray],
) -> np.ndarray:
    """Triangulate a 2D polygon with holes using earcut.

    Args:
        exterior_2d: (N, 2) exterior ring.
        holes_2d: list of (M_i, 2) interior rings.

    Returns:
        (T, 3) triangle indices into the concatenated vertex array
        (exterior followed by all holes in order).
    """
    n_ext = len(exterior_2d)
    if n_ext < 3:
        return np.array([], dtype=np.int32).reshape(0, 3)

    all_coords = [exterior_2d]
    ring_sizes = [n_ext]
    for hole in holes_2d:
        if len(hole) >= 3:
            all_coords.append(hole)
            ring_sizes.append(len(hole))

    coords = np.vstack(all_coords)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    extent = max_c - min_c
    extent = np.where(extent < 1e-10, 1.0, extent)
    normalised = np.ascontiguousarray((coords - min_c) / extent, dtype=np.float64)
    # mapbox_earcut expects cumulative ring end indices, not individual sizes
    rings = np.cumsum(ring_sizes, dtype=np.uint32)

    indices = earcut.triangulate_float64(normalised, rings)
    if len(indices) >= 3:
        return indices.reshape(-1, 3).astype(np.int32)

    # earcut returned fewer than 3 indices – degenerate polygon with holes
    logging.getLogger(__name__).warning(
        "earcut returned %d indices for polygon with %d holes; "
        "falling back to exterior-only triangulation",
        len(indices), len(holes_2d),
    )
    return triangulate_polygon_2d(exterior_2d)


def project_to_2d(vertices: np.ndarray) -> np.ndarray:
    """Project 3D polygon vertices to 2D using Newell's-method normal."""
    n = len(vertices)
    if n < 3:
        return vertices[:, :2]

    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    nxt = np.roll(centered, -1, axis=0)

    nx = np.sum((centered[:, 1] - nxt[:, 1]) * (centered[:, 2] + nxt[:, 2]))
    ny = np.sum((centered[:, 2] - nxt[:, 2]) * (centered[:, 0] + nxt[:, 0]))
    nz = np.sum((centered[:, 0] - nxt[:, 0]) * (centered[:, 1] + nxt[:, 1]))

    normal = np.array([nx, ny, nz])
    normal_len = np.linalg.norm(normal)

    if normal_len < 1e-10:
        return vertices[:, :2]

    normal /= normal_len
    up = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    return np.column_stack([np.dot(centered, u), np.dot(centered, v)])


# ---------------------------------------------------------------------------
# Polygon / MultiSurface / Solid parsing
# ---------------------------------------------------------------------------

def _extract_ring_coords(ring_parent, ns: Dict[str, str]) -> np.ndarray:
    """Extract 3D coordinates from gml:exterior or gml:interior."""
    poslist = ring_parent.find('.//gml:LinearRing//gml:posList', ns)
    if poslist is not None and poslist.text:
        coords = parse_pos_list(poslist.text)
    else:
        pos_elems = ring_parent.findall('.//gml:LinearRing//gml:pos', ns)
        if pos_elems:
            pts = []
            for pos in pos_elems:
                try:
                    parts = pos.text.strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except Exception:
                    continue
            coords = np.array(pts, dtype=np.float64) if pts else np.array([], dtype=np.float64).reshape(0, 3)
        else:
            return np.array([], dtype=np.float64).reshape(0, 3)

    if len(coords) < 3:
        return coords
    # Remove closing duplicate
    if len(coords) > 3:
        d = coords[0] - coords[-1]
        if d[0]*d[0] + d[1]*d[1] + d[2]*d[2] < 1e-10:
            coords = coords[:-1]
    return coords


def parse_polygon_to_triangles(polygon_elem, ns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a gml:Polygon (with optional interior rings) and triangulate."""
    empty = np.array([], dtype=np.float64).reshape(0, 3)

    # Exterior ring
    ext_elem = polygon_elem.find('gml:exterior', ns)
    if ext_elem is None:
        ext_elem = polygon_elem.find('.//gml:exterior', ns)
    if ext_elem is None:
        return empty, empty.astype(np.int32)
    ext_coords = _extract_ring_coords(ext_elem, ns)
    if len(ext_coords) < 3:
        return empty, empty.astype(np.int32)

    # Interior rings (holes)
    hole_coords_list: List[np.ndarray] = []
    for interior in polygon_elem.findall('gml:interior', ns):
        hc = _extract_ring_coords(interior, ns)
        if len(hc) >= 3:
            hole_coords_list.append(hc)
    # Also try with descendant axis (some CityGML variants)
    if not hole_coords_list:
        for interior in polygon_elem.findall('.//gml:interior', ns):
            hc = _extract_ring_coords(interior, ns)
            if len(hc) >= 3:
                hole_coords_list.append(hc)

    # Simple triangle — return immediately (no holes possible on 3 verts)
    if len(ext_coords) == 3 and not hole_coords_list:
        return ext_coords, np.array([[0, 1, 2]], dtype=np.int32)

    # Triangulate with or without holes
    if hole_coords_list:
        # Concatenate all ring vertices for a common 2D projection
        all_3d = np.vstack([ext_coords] + hole_coords_list)
        all_2d = project_to_2d(all_3d)

        off = 0
        ext_2d = all_2d[off:off + len(ext_coords)]; off += len(ext_coords)
        holes_2d: List[np.ndarray] = []
        for hc in hole_coords_list:
            holes_2d.append(all_2d[off:off + len(hc)]); off += len(hc)

        faces = triangulate_polygon_with_holes_2d(ext_2d, holes_2d)
        return all_3d, faces
    else:
        if len(ext_coords) == 4:
            return ext_coords, _QUAD_FACES.copy()
        ext_2d = project_to_2d(ext_coords)
        faces = triangulate_polygon_2d(ext_2d)
        return ext_coords, faces


def parse_multisurface(elem, ns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a gml:MultiSurface element into vertices and faces."""
    all_vertices: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    offset = 0

    for polygon in elem.findall('.//gml:Polygon', ns):
        verts, faces = parse_polygon_to_triangles(polygon, ns)
        if len(verts) > 0 and len(faces) > 0:
            all_vertices.append(verts)
            all_faces.append(faces + offset)
            offset += len(verts)

    if not all_vertices:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    return np.vstack(all_vertices), np.vstack(all_faces)


def parse_solid(elem, ns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a gml:Solid element (LOD1 / LOD2 building geometry)."""
    composite = elem.find('.//gml:exterior//gml:CompositeSurface', ns)
    if composite is not None:
        return parse_multisurface(composite, ns)

    surfaces = elem.findall('.//gml:surfaceMember', ns)
    if surfaces:
        all_vertices: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        offset = 0
        for surface in surfaces:
            polygon = surface.find('.//gml:Polygon', ns)
            if polygon is not None:
                verts, faces = parse_polygon_to_triangles(polygon, ns)
                if len(verts) > 0 and len(faces) > 0:
                    all_vertices.append(verts)
                    all_faces.append(faces + offset)
                    offset += len(verts)
        if all_vertices:
            return np.vstack(all_vertices), np.vstack(all_faces)

    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def find_geometry_in_element(elem, ns: Dict[str, str],
                             geometry_tags: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Search for geometry in an element using a list of potential tags."""
    for tag in geometry_tags:
        geom_elem = elem.find(tag, ns)
        if geom_elem is None:
            continue

        solid = geom_elem.find('.//gml:Solid', ns)
        if solid is not None:
            verts, faces = parse_solid(solid, ns)
            if len(verts) > 0:
                return verts, faces

        ms = geom_elem.find('.//gml:MultiSurface', ns)
        if ms is not None:
            verts, faces = parse_multisurface(ms, ns)
            if len(verts) > 0:
                return verts, faces

        polygon = geom_elem.find('.//gml:Polygon', ns)
        if polygon is not None:
            verts, faces = parse_polygon_to_triangles(polygon, ns)
            if len(verts) > 0:
                return verts, faces

    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


# ---------------------------------------------------------------------------
# Implicit Geometry support (CityGML lod*ImplicitRepresentation)
# ---------------------------------------------------------------------------

def parse_transformation_matrix(text: str) -> np.ndarray:
    """Parse a CityGML ``core:transformationMatrix`` text into a 4x4 numpy array.

    The matrix is stored row-major as 16 space-separated floats.
    """
    values = np.fromstring(text.strip(), dtype=np.float64, sep=' ')
    if len(values) != 16:
        raise ValueError(f"Expected 16 values in transformation matrix, got {len(values)}")
    return values.reshape(4, 4)


def parse_implicit_geometry(implicit_elem, ns: Dict[str, str],
                            prototype_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a ``core:ImplicitGeometry`` element into world-coordinate vertices and faces.

    CityGML ImplicitGeometry consists of:
      - ``core:transformationMatrix``: 4x4 affine transform (scales/rotates the template).
      - ``core:relativeGMLGeometry``: Template mesh in local coordinates.
        May be inline or an ``xlink:href`` reference to a shared prototype.
      - ``core:referencePoint``: World-coordinate origin (translation).

    World coordinates = M @ [x, y, z, 1]^T  +  referencePoint

    Parameters
    ----------
    implicit_elem : XML element
        The ``core:ImplicitGeometry`` element.
    ns : dict
        XML namespace dictionary.
    prototype_cache : dict, optional
        Mapping from ``gml:id`` to ``(vertices, faces)`` tuples for shared
        prototype geometries.  Used to resolve ``xlink:href`` references.

    Returns
    -------
    vertices : (N, 3) float64 - world-coordinate vertices.
    faces    : (M, 3) int32   - triangle face indices.
    """
    empty_v = np.array([], dtype=np.float64).reshape(0, 3)
    empty_f = np.array([], dtype=np.int32).reshape(0, 3)

    # --- Transformation matrix ---
    mat_elem = implicit_elem.find('core:transformationMatrix', ns)
    if mat_elem is None or not mat_elem.text:
        mat = np.eye(4, dtype=np.float64)
    else:
        try:
            mat = parse_transformation_matrix(mat_elem.text)
        except ValueError:
            mat = np.eye(4, dtype=np.float64)

    # --- Reference point ---
    ref_point = np.zeros(3, dtype=np.float64)
    ref_elem = implicit_elem.find('core:referencePoint', ns)
    if ref_elem is not None:
        pos = ref_elem.find('.//gml:Point/gml:pos', ns)
        if pos is not None and pos.text:
            parts = pos.text.strip().split()
            if len(parts) >= 3:
                ref_point = np.array([float(parts[0]), float(parts[1]), float(parts[2])],
                                     dtype=np.float64)

    # --- Relative geometry (template mesh) ---
    verts = None
    faces = None

    rel_geom = implicit_elem.find('core:relativeGMLGeometry', ns)
    if rel_geom is None:
        rel_geom = implicit_elem.find('core:relativeGeometry', ns)

    if rel_geom is not None:
        # Check for xlink:href reference to a shared prototype
        xlink_ns = 'http://www.w3.org/1999/xlink'
        href = rel_geom.get(f'{{{xlink_ns}}}href', '')
        if href and prototype_cache is not None:
            proto_id = href.lstrip('#')
            if proto_id in prototype_cache:
                proto_verts, proto_faces = prototype_cache[proto_id]
                verts = proto_verts.copy()
                faces = proto_faces.copy()
        elif not href:
            # Inline geometry
            verts, faces = _parse_geometry_from_element(rel_geom, ns)

    if verts is None or len(verts) == 0:
        return empty_v, empty_f

    # --- Apply transformation: world = M @ [x, y, z, 1]^T, then + ref_point ---
    n = len(verts)
    ones = np.ones((n, 1), dtype=np.float64)
    homo = np.hstack([verts, ones])  # (N, 4)
    transformed = (mat @ homo.T).T  # (N, 4)
    world_verts = transformed[:, :3] + ref_point

    return world_verts, faces.astype(np.int32)


def _parse_geometry_from_element(elem, ns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mesh geometry from a GML element (MultiSurface, Solid, or bare Polygons)."""
    empty_v = np.array([], dtype=np.float64).reshape(0, 3)
    empty_f = np.array([], dtype=np.int32).reshape(0, 3)

    ms = elem.find('.//gml:MultiSurface', ns)
    if ms is not None:
        return parse_multisurface(ms, ns)

    solid = elem.find('.//gml:Solid', ns)
    if solid is not None:
        return parse_solid(solid, ns)

    # Bare polygons
    all_v, all_f, offset = [], [], 0
    for poly in elem.iterfind('.//gml:Polygon', ns):
        v, f = parse_polygon_to_triangles(poly, ns)
        if len(v) > 0 and len(f) > 0:
            all_v.append(v)
            all_f.append(f + offset)
            offset += len(v)
    if all_v:
        return np.vstack(all_v), np.vstack(all_f).astype(np.int32)
    return empty_v, empty_f


def build_prototype_cache(root, ns: Dict[str, str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build a cache of shared prototype geometries from the GML document.

    Scans for all ``gml:MultiSurface``, ``gml:Solid``, and similar geometry
    elements that have a ``gml:id`` attribute and appear inside
    ``core:relativeGMLGeometry``.  These serve as reusable templates
    referenced by ``xlink:href`` in CityGML ImplicitGeometry.

    Returns
    -------
    dict
        Mapping from ``gml:id`` string to ``(vertices, faces)`` tuple.
    """
    _GML_ID = '{http://www.opengis.net/gml}id'
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Find all relativeGMLGeometry elements with inline geometry
    for rel in root.iter(f"{{{ns.get('core', '')}}}relativeGMLGeometry"):
        # Check for inline geometry (not xlink:href reference)
        xlink_ns = 'http://www.w3.org/1999/xlink'
        if rel.get(f'{{{xlink_ns}}}href'):
            continue  # This is a reference, not a definition

        # Find geometry elements with gml:id
        for ms in rel.findall('.//gml:MultiSurface', ns):
            gml_id = ms.get(_GML_ID)
            if gml_id and gml_id not in cache:
                verts, faces = parse_multisurface(ms, ns)
                if len(verts) > 0:
                    cache[gml_id] = (verts, faces)

        for solid in rel.findall('.//gml:Solid', ns):
            gml_id = solid.get(_GML_ID)
            if gml_id and gml_id not in cache:
                verts, faces = parse_solid(solid, ns)
                if len(verts) > 0:
                    cache[gml_id] = (verts, faces)

    return cache
