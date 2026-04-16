"""
Feature extractors for CityGML objects.
Transplanted from citygml_mesher.extractors (terrain, buildings, bridges, vegetation).
"""

import logging
import re
from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
    logging.getLogger(__name__).warning(
        "lxml not installed – falling back to stdlib xml.etree.ElementTree. "
        "Install lxml for faster CityGML parsing: pip install lxml"
    )

from ..models import Mesh3D
from .geometry import (
    parse_pos_list,
    parse_polygon_to_triangles,
    parse_multisurface,
    parse_solid,
    find_geometry_in_element,
    parse_implicit_geometry,
    build_prototype_cache,
)
from .coordinates import mesh_intersects_rectangle


# ========================================================================
# GML helpers
# ========================================================================
_GML_ID = '{http://www.opengis.net/gml}id'


# ========================================================================
# Terrain extractor
# ========================================================================

_POSLIST_PATTERN = re.compile(r'<gml:posList[^>]*>([^<]+)</gml:posList>')
_TIN_ID_PATTERN = re.compile(r'<dem:TINRelief[^>]*gml:id="([^"]+)"')


def _parse_triangles_batch(coord_strings: List[str], entry_len: int = 12) -> np.ndarray:
    """Batch-parse triangle coordinates from posList strings."""
    if not coord_strings:
        return np.array([], dtype=np.float64).reshape(0, 3, 3)
    try:
        all_text = ' '.join(coord_strings)
        all_values = np.array(all_text.split(), dtype=np.float64)
        n_entries = len(all_values) // entry_len
        if n_entries == 0:
            return np.array([], dtype=np.float64).reshape(0, 3, 3)
        reshaped = all_values[:n_entries * entry_len].reshape(n_entries, entry_len)
        return reshaped[:, :9].reshape(n_entries, 3, 3)
    except (ValueError, IndexError):
        triangles = np.empty((len(coord_strings), 3, 3), dtype=np.float64)
        valid = 0
        for text in coord_strings:
            try:
                v = np.array(text.split(), dtype=np.float64)
                if len(v) >= 9:
                    triangles[valid] = v[:9].reshape(3, 3)
                    valid += 1
            except Exception:
                continue
        return triangles[:valid] if valid else np.array([], dtype=np.float64).reshape(0, 3, 3)


def extract_terrain_from_root(root, ns: Dict[str, str],
                              gml_content: str = None) -> List[Mesh3D]:
    """Extract terrain/DEM meshes (TINRelief) from CityGML.

    Supports a fast regex path when *gml_content* is provided,
    and a slower XML-tree fallback otherwise.
    """
    meshes: List[Mesh3D] = []

    # ---- Fast regex path ----
    if gml_content is not None:
        matches = _POSLIST_PATTERN.findall(gml_content)
        if matches:
            first_len = len(matches[0].split())
            entry_len = first_len if first_len >= 9 else 12
            tri_arr = _parse_triangles_batch(matches, entry_len)
            if len(tri_arr) > 0:
                n = len(tri_arr)
                tin_id_match = _TIN_ID_PATTERN.search(gml_content)
                tin_id = tin_id_match.group(1) if tin_id_match else 'terrain'
                meshes.append(Mesh3D(
                    vertices=tri_arr.reshape(-1, 3),
                    faces=np.arange(n * 3, dtype=np.int32).reshape(-1, 3),
                    feature_type='terrain', feature_id=tin_id,
                    attributes={'triangle_coords': tri_arr},
                ))
        return meshes

    # ---- XML fallback ----
    if root is None:
        return meshes
    for relief in root.findall('.//dem:ReliefFeature', ns):
        relief_id = relief.get(_GML_ID, 'unknown')
        for tin in relief.findall('.//dem:TINRelief', ns):
            tin_id = tin.get(_GML_ID, relief_id)
            all_tri = []
            for triangle in tin.findall('.//gml:Triangle', ns):
                for pl in triangle.findall('.//gml:posList', ns):
                    coords = parse_pos_list(pl.text)
                    if len(coords) >= 4 and np.allclose(coords[0], coords[-1]):
                        coords = coords[:-1]
                    if len(coords) >= 3:
                        all_tri.append(coords[:3])
            if all_tri:
                tri_arr = np.array(all_tri)
                n = len(tri_arr)
                meshes.append(Mesh3D(
                    vertices=tri_arr.reshape(-1, 3),
                    faces=np.arange(n * 3).reshape(-1, 3).astype(np.int32),
                    feature_type='terrain', feature_id=tin_id,
                    attributes={'triangle_coords': tri_arr},
                ))
    return meshes


def filter_terrain_by_rectangle_vectorized(terrain_meshes: List[Mesh3D],
                                           rect_polygon: ShapelyPolygon,
                                           prepared_rect) -> List[Mesh3D]:
    """Filter terrain triangles by bbox intersection (vectorised)."""
    if not terrain_meshes:
        return []
    minx, miny, maxx, maxy = rect_polygon.bounds
    result: List[Mesh3D] = []
    for mesh in terrain_meshes:
        if 'triangle_coords' not in mesh.attributes:
            if mesh_intersects_rectangle(mesh, rect_polygon, prepared_rect):
                result.append(mesh)
            continue
        triangles = mesh.attributes['triangle_coords']
        tri_min_lat = triangles[:, :, 0].min(axis=1)
        tri_max_lat = triangles[:, :, 0].max(axis=1)
        tri_min_lon = triangles[:, :, 1].min(axis=1)
        tri_max_lon = triangles[:, :, 1].max(axis=1)
        mask = ((tri_max_lon >= minx) & (tri_min_lon <= maxx) &
                (tri_max_lat >= miny) & (tri_min_lat <= maxy))
        filtered = triangles[mask]
        if len(filtered) > 0:
            n = len(filtered)
            result.append(Mesh3D(
                vertices=filtered.reshape(-1, 3),
                faces=np.arange(n * 3).reshape(-1, 3).astype(np.int32),
                feature_type='terrain', feature_id=mesh.feature_id,
            ))
    return result


# ========================================================================
# Building extractor
# ========================================================================

def _extract_building_polygons(building, ns: Dict[str, str],
                               lod: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract & triangulate polygons (with interior rings) from a building."""
    from .geometry import parse_polygon_to_triangles

    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    offset = 0

    def _collect_polygons(container):
        nonlocal offset
        for polygon in container.iterfind('.//gml:Polygon', ns):
            verts, faces = parse_polygon_to_triangles(polygon, ns)
            if len(verts) > 0 and len(faces) > 0:
                all_verts.append(verts)
                all_faces.append(faces + offset)
                offset += len(verts)

    if lod is not None:
        containers = []
        if lod >= 2:
            bounded = [
                './/bldg:WallSurface', './/bldg:RoofSurface',
                './/bldg:GroundSurface', './/bldg:ClosureSurface',
                './/bldg:OuterCeilingSurface', './/bldg:OuterFloorSurface',
            ]
            for surf_type in bounded:
                for surface in building.findall(surf_type, ns):
                    lod_geom = surface.find(f'.//bldg:lod{lod}MultiSurface', ns)
                    if lod_geom is not None:
                        containers.append(lod_geom)
            if not containers:
                for tag in [f'.//bldg:lod{lod}MultiSurface', f'.//bldg:lod{lod}Solid']:
                    containers.extend(building.findall(tag, ns))
        else:
            for tag in [f'.//bldg:lod{lod}Solid', f'.//bldg:lod{lod}MultiSurface']:
                containers.extend(building.findall(tag, ns))

        for container in containers:
            _collect_polygons(container)
    else:
        _collect_polygons(building)

    if not all_verts:
        return (np.array([], dtype=np.float64).reshape(0, 3),
                np.array([], dtype=np.int32).reshape(0, 3))
    return np.vstack(all_verts), np.vstack(all_faces).astype(np.int32)


def _get_building_best_lod(building, ns: Dict[str, str], max_lod: int = 4) -> int:
    """Determine the best (highest) LOD available for a building."""
    for lod in [4, 3, 2]:
        if lod > max_lod:
            continue
        if building.find(f'.//bldg:lod{lod}Solid', ns) is not None:
            return lod
        if building.find(f'.//bldg:lod{lod}MultiSurface', ns) is not None:
            return lod
        for surf in ['WallSurface', 'RoofSurface', 'GroundSurface']:
            s = building.find(f'.//bldg:{surf}', ns)
            if s is not None and s.find(f'.//bldg:lod{lod}MultiSurface', ns) is not None:
                return lod
    if max_lod >= 1:
        if building.find('.//bldg:lod1Solid', ns) is not None:
            return 1
        if building.find('.//bldg:lod1MultiSurface', ns) is not None:
            return 1
    return 0


def extract_buildings_from_root(root, ns: Dict[str, str],
                                prefer_lod: int = None,
                                max_lod: int = 4) -> List[Mesh3D]:
    """Extract building meshes from parsed CityGML root.  One LOD per building."""
    meshes: List[Mesh3D] = []
    for building in root.iterfind('.//bldg:Building', ns):
        bid = building.get(_GML_ID, 'unknown')
        h_elem = building.find('bldg:measuredHeight', ns)
        height = float(h_elem.text) if h_elem is not None and h_elem.text else None

        verts = np.array([]).reshape(0, 3)
        faces = np.array([]).reshape(0, 3)
        extracted_lod = None

        if prefer_lod is not None and prefer_lod <= max_lod:
            verts, faces = _extract_building_polygons(building, ns, lod=prefer_lod)
            if len(verts) > 0:
                extracted_lod = prefer_lod
        if len(verts) == 0:
            best = _get_building_best_lod(building, ns, max_lod=max_lod)
            if best > 0:
                verts, faces = _extract_building_polygons(building, ns, lod=best)
                extracted_lod = best
            else:
                verts, faces = _extract_building_polygons(building, ns, lod=None)

        if len(verts) > 0 and len(faces) > 0:
            meshes.append(Mesh3D(
                vertices=verts, faces=faces.astype(np.int32),
                feature_type='building', feature_id=bid,
                attributes={'height': height, 'lod': extracted_lod},
            ))
    return meshes


# ========================================================================
# Bridge extractor
# ========================================================================

def extract_bridges_from_root(root, ns: Dict[str, str]) -> List[Mesh3D]:
    """Extract bridge meshes from parsed CityGML root.
    
    Strategy:
        1. Prefer Solid representations (lod2Solid, lod3Solid, etc.) - these define closed volumes
        2. Search construction elements (BridgeConstructionElement, BridgePart, BridgeInstallation)
        3. Fall back to boundary surfaces (WallSurface, RoofSurface, etc.) - these are open
        
    This prioritization ensures we get watertight geometry when available, rather than
    mixing solids with open boundary surfaces which creates non-manifold meshes.
    """
    meshes: List[Mesh3D] = []
    solid_tags = [
        './/brid:lod4Solid', './/brid:lod3Solid', './/brid:lod2Solid', './/brid:lod1Solid',
    ]
    fallback_geom_tags = [
        './/brid:lod4MultiSurface', './/brid:lod3MultiSurface',
        './/brid:lod2MultiSurface', './/brid:lod1MultiSurface',
        './/brid:lod4Geometry', './/brid:lod3Geometry',
        './/brid:lod2Geometry', './/brid:lod1Geometry',
    ]

    for bridge in root.findall('.//brid:Bridge', ns):
        bridge_id = bridge.get(_GML_ID, 'unknown')
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        offset = 0

        # Step 1: Try to get Solid directly from bridge
        verts, faces = find_geometry_in_element(bridge, ns, solid_tags)
        if len(verts) > 0 and len(faces) > 0:
            meshes.append(Mesh3D(
                vertices=verts, faces=faces.astype(np.int32),
                feature_type='bridge', feature_id=bridge_id,
            ))
            continue

        # Step 2: Construction elements, parts, installations (often contain Solids)
        for tag in ['.//brid:BridgeConstructionElement',
                    './/brid:BridgePart',
                    './/brid:BridgeInstallation']:
            for elem in bridge.findall(tag, ns):
                ev, ef = find_geometry_in_element(elem, ns, solid_tags)
                if len(ev) > 0:
                    all_verts.append(ev)
                    all_faces.append(ef + offset)
                    offset += len(ev)

        if all_verts:
            verts = np.vstack(all_verts)
            faces = np.vstack(all_faces)
            meshes.append(Mesh3D(
                vertices=verts, faces=faces.astype(np.int32),
                feature_type='bridge', feature_id=bridge_id,
            ))
            continue

        # Step 3: Fall back to boundary surfaces + construction elements with MultiSurface/Geometry
        all_verts = []
        all_faces = []
        offset = 0

        # Boundary surfaces
        for surf_type in ['.//brid:WallSurface', './/brid:RoofSurface',
                          './/brid:GroundSurface', './/brid:ClosureSurface',
                          './/brid:OuterCeilingSurface', './/brid:OuterFloorSurface']:
            for surface in bridge.findall(surf_type, ns):
                for ms in surface.findall('.//gml:MultiSurface', ns):
                    sv, sf = parse_multisurface(ms, ns)
                    if len(sv) > 0:
                        all_verts.append(sv)
                        all_faces.append(sf + offset)
                        offset += len(sv)

        # Construction elements, parts, installations (with fallback geom tags)
        for tag in ['.//brid:BridgeConstructionElement',
                    './/brid:BridgePart',
                    './/brid:BridgeInstallation']:
            for elem in bridge.findall(tag, ns):
                ev, ef = find_geometry_in_element(elem, ns, fallback_geom_tags)
                if len(ev) > 0:
                    all_verts.append(ev)
                    all_faces.append(ef + offset)
                    offset += len(ev)

        if all_verts:
            verts = np.vstack(all_verts)
            faces = np.vstack(all_faces)
        else:
            verts, faces = find_geometry_in_element(bridge, ns, fallback_geom_tags)

        if len(verts) > 0 and len(faces) > 0:
            meshes.append(Mesh3D(
                vertices=verts, faces=faces.astype(np.int32),
                feature_type='bridge', feature_id=bridge_id,
            ))
    return meshes


# ========================================================================
# Vegetation extractor
# ========================================================================

def _extract_all_veg_polygons(element, ns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract all polygons from a vegetation element."""
    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    offset = 0
    for polygon in element.iterfind('.//gml:Polygon', ns):
        verts, faces = parse_polygon_to_triangles(polygon, ns)
        if len(verts) > 0 and len(faces) > 0:
            all_verts.append(verts)
            all_faces.append(faces + offset)
            offset += len(verts)
    if not all_verts:
        return (np.array([], dtype=np.float64).reshape(0, 3),
                np.array([], dtype=np.int32).reshape(0, 3))
    return np.vstack(all_verts), np.vstack(all_faces).astype(np.int32)


def _extract_veg_implicit_geometry(element, ns: Dict[str, str],
                                    prototype_cache: dict = None,
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract geometry from a vegetation element with ImplicitGeometry (LOD 1–4).

    Searches for ``lod{n}ImplicitRepresentation`` elements and applies the
    transformation matrix + reference point to produce world-coordinate meshes.
    Falls back to direct polygon extraction if no implicit geometry is found.
    """
    # Try implicit representations LOD4 → LOD1
    for lod in [4, 3, 2, 1]:
        implicit_tag = f'veg:lod{lod}ImplicitRepresentation'
        impl_repr = element.find(implicit_tag, ns)
        if impl_repr is None:
            continue
        impl_geom = impl_repr.find('core:ImplicitGeometry', ns)
        if impl_geom is not None:
            verts, faces = parse_implicit_geometry(impl_geom, ns,
                                                    prototype_cache=prototype_cache)
            if len(verts) > 0 and len(faces) > 0:
                return verts, faces

    # Also try direct LOD geometry (MultiSurface / Solid)
    for lod in [4, 3, 2, 1]:
        for geom_type in ['MultiSurface', 'Geometry']:
            geom_tag = f'veg:lod{lod}{geom_type}'
            geom_elem = element.find(geom_tag, ns)
            if geom_elem is not None:
                ms = geom_elem.find('.//gml:MultiSurface', ns)
                if ms is not None:
                    verts, faces = parse_multisurface(ms, ns)
                    if len(verts) > 0:
                        return verts, faces

    # Fall back to extracting all polygons directly
    return _extract_all_veg_polygons(element, ns)


def _get_generic_double_attr(element, ns: Dict[str, str], name: str):
    """Extract a gen:doubleAttribute value by name from a CityGML element."""
    for attr in element.iterfind('gen:doubleAttribute', ns):
        attr_name = attr.get('name', '')
        if attr_name == name:
            val_elem = attr.find('gen:value', ns)
            if val_elem is not None and val_elem.text:
                try:
                    return float(val_elem.text)
                except ValueError:
                    pass
    return None


def extract_vegetation_from_root(root, ns: Dict[str, str]) -> List[Mesh3D]:
    """Extract vegetation meshes (PlantCover + SolitaryVegetationObject).

    Supports both direct polygon geometry and CityGML ImplicitGeometry
    (used e.g. in Munich tree models where each tree is an instance of
    a parametric template, transformed by a 4×4 matrix).
    """
    meshes: List[Mesh3D] = []

    # Build prototype cache for shared implicit geometry templates
    proto_cache = build_prototype_cache(root, ns)
    if proto_cache:
        logging.getLogger(__name__).debug(
            "Built prototype cache with %d templates: %s",
            len(proto_cache), list(proto_cache.keys()),
        )

    for cover in root.iterfind('.//veg:PlantCover', ns):
        cid = cover.get(_GML_ID, 'unknown')
        h_elem = cover.find('veg:averageHeight', ns)
        h = float(h_elem.text) if h_elem is not None and h_elem.text else None
        v, f = _extract_all_veg_polygons(cover, ns)
        if len(v) > 0 and len(f) > 0:
            meshes.append(Mesh3D(vertices=v, faces=f,
                                 feature_type='vegetation', feature_id=cid,
                                 attributes={'height': h, 'type': 'PlantCover'}))

    for sol in root.iterfind('.//veg:SolitaryVegetationObject', ns):
        vid = sol.get(_GML_ID, 'unknown')
        h_elem = sol.find('veg:height', ns)
        h = float(h_elem.text) if h_elem is not None and h_elem.text else None
        cd_elem = sol.find('veg:crownDiameter', ns)
        crown_diam = float(cd_elem.text) if cd_elem is not None and cd_elem.text else None

        # Extract generic attributes (Munich tree models)
        crown_height = _get_generic_double_attr(sol, ns, 'crown_height')
        trunk_height = _get_generic_double_attr(sol, ns, 'trunk_height')
        crown_volume = _get_generic_double_attr(sol, ns, 'crown_volume')

        v, f = _extract_veg_implicit_geometry(sol, ns, prototype_cache=proto_cache)
        if len(v) > 0 and len(f) > 0:
            attrs = {
                'height': h,
                'type': 'SolitaryVegetationObject',
                'crown_diameter': crown_diam,
                'crown_height': crown_height,
                'trunk_height': trunk_height,
                'crown_volume': crown_volume,
            }
            meshes.append(Mesh3D(vertices=v, faces=f,
                                 feature_type='vegetation', feature_id=vid,
                                 attributes=attrs))
    return meshes
