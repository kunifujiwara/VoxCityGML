"""
CityGML parsing layer – transplanted from citygml_mesher.

Provides namespace handling, geometry parsing, coordinate transforms,
and feature extraction (terrain, buildings, bridges, vegetation).
"""

from .namespaces import (
    build_namespaces,
    DEFAULT_NAMESPACES,
    detect_crs_from_root,
    detect_crs_from_file_header,
)
from .geometry import (
    parse_pos_list,
    parse_polygon_to_triangles,
    parse_multisurface,
    parse_solid,
    find_geometry_in_element,
    triangulate_polygon_2d,
    project_to_2d,
    parse_implicit_geometry,
    build_prototype_cache,
)
from .coordinates import (
    create_rectangle,
    rectangle_to_shapely,
    swap_coordinates_3d,
    reproject_vertices,
    mesh_intersects_rectangle,
    file_intersects_rectangle,
    transform_to_local_meters,
    create_local_transformer,
)
from .extractors import (
    extract_terrain_from_root,
    filter_terrain_by_rectangle_vectorized,
    extract_buildings_from_root,
    extract_bridges_from_root,
    extract_vegetation_from_root,
)
from .parser import parse_citygml_directory, merge_terrain_meshes

__all__ = [
    "build_namespaces",
    "DEFAULT_NAMESPACES",
    "detect_crs_from_root",
    "detect_crs_from_file_header",
    "parse_pos_list",
    "parse_polygon_to_triangles",
    "parse_multisurface",
    "parse_solid",
    "find_geometry_in_element",
    "triangulate_polygon_2d",
    "project_to_2d",
    "parse_implicit_geometry",
    "build_prototype_cache",
    "create_rectangle",
    "rectangle_to_shapely",
    "swap_coordinates_3d",
    "reproject_vertices",
    "mesh_intersects_rectangle",
    "file_intersects_rectangle",
    "transform_to_local_meters",
    "create_local_transformer",
    "extract_terrain_from_root",
    "filter_terrain_by_rectangle_vectorized",
    "extract_buildings_from_root",
    "extract_bridges_from_root",
    "extract_vegetation_from_root",
    "parse_citygml_directory",
    "merge_terrain_meshes",
]
