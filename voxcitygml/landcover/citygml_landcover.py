"""
CityGML-based land cover grid generation.

Parses ``luse:LandUse`` features from CityGML (PLATEAU) datasets,
maps their class codes to the internal VoxCity 1-based land cover
indices, and rasterizes the polygons onto a regular grid matching the
target rectangle and resolution.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box, Point
from shapely.prepared import prep as shapely_prep

try:
    import lxml.etree as ET
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
    HAS_LXML = False

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# PLATEAU Common_landUseType → internal 1-based VoxCity land-cover codes
# -----------------------------------------------------------------------
# Internal codes (from voxelizer3d / voxcity):
#   1=Bareland, 2=Rangeland, 3=Shrub, 4=Agriculture, 5=Tree,
#   6=Moss/lichen, 7=Wetland, 8=Mangrove, 9=Water, 10=Snow/ice,
#   11=Developed space, 12=Road, 13=Building, 14=No Data
#
# Priority (higher number = higher priority, wins when polygons overlap).
# Road (12) is the lowest so every other class overrides it.
_VOXCITY_PRIORITY: Dict[int, int] = {
    14: 0,   # No Data – lowest
    12: 1,   # Road – only above No Data
    1:  2,   # Bareland
    2:  3,   # Rangeland
    3:  4,   # Shrub
    4:  5,   # Agriculture
    5:  6,   # Tree
    6:  7,   # Moss / lichen
    7:  8,   # Wetland
    8:  9,   # Mangrove
    9:  10,  # Water
    10: 11,  # Snow / ice
    11: 12,  # Developed space
    13: 13,  # Building
}
_PLATEAU_LUSE_TO_VOXCITY = {
    201: 4,   # 田 (Paddy field) → Agriculture
    202: 4,   # 畑 (Farm/orchard) → Agriculture
    203: 5,   # 山林 (Forest) → Tree
    204: 9,   # 水面 (Water) → Water
    205: 2,   # その他自然地 (Other natural) → Rangeland
    211: 13,  # 住宅用地 (Residential) → Building
    212: 13,  # 商業用地 (Commercial) → Building
    213: 13,  # 工業用地 (Industrial) → Building
    214: 11,  # 公益施設用地 (Public facility) → Developed space
    215: 12,  # 道路用地 (Road) → Road
    216: 12,  # 交通施設用地 (Transport facility) → Road
    217: 2,   # 公共空地 (Public open space/park) → Rangeland
    218: 11,  # その他公的施設用地 (Other public) → Developed space
    219: 4,   # 農林漁業施設用地 (Agri/forestry facility) → Agriculture
    220: 2,   # その他① ゴルフ場 (Golf course) → Rangeland
    221: 11,  # その他② 太陽光発電 (Solar power) → Developed space
    222: 11,  # その他③ 平面駐車場 (Parking lot) → Developed space
    223: 11,  # その他④ (Other urban) → Developed space
    224: 1,   # 低未利用土地 (Vacant land) → Bareland
    231: 14,  # 不明 (Unknown) → No Data
    251: 11,  # 可住地 (Habitable land) → Developed space
    252: 1,   # 非可住地 (Non-habitable) → Bareland
    260: 4,   # 農地 (Farmland, no subdivision) → Agriculture
    261: 13,  # 宅地 (Residential, no subdivision) → Building
    262: 12,  # 道路・鉄軌道敷 (Road/rail) → Road
    263: 1,   # 空地 (Vacant land) → Bareland
}

# Regex patterns for fast extraction without full XML parse
_LANDUSE_BLOCK_RE = re.compile(
    r'<luse:LandUse[^>]*>(.*?)</luse:LandUse>',
    re.DOTALL,
)
_CLASS_RE = re.compile(r'<luse:class[^>]*>(\d+)</luse:class>')
_POSLIST_RE = re.compile(r'<gml:posList[^>]*>([^<]+)</gml:posList>')
_ENVELOPE_RE = re.compile(
    r'<gml:lowerCorner>([^<]+)</gml:lowerCorner>\s*'
    r'<gml:upperCorner>([^<]+)</gml:upperCorner>',
)


def _file_envelope_intersects(filepath: Path,
                              min_lon: float, min_lat: float,
                              max_lon: float, max_lat: float) -> bool:
    """Quick check: does the file's GML envelope intersect the target bbox?"""
    try:
        with open(str(filepath), 'r', encoding='utf-8', errors='replace') as f:
            header = f.read(4096)
        m = _ENVELOPE_RE.search(header)
        if m is None:
            return True  # Can't determine → include
        lower = m.group(1).split()
        upper = m.group(2).split()
        # PLATEAU luse coords: lat lon z
        f_min_lat, f_min_lon = float(lower[0]), float(lower[1])
        f_max_lat, f_max_lon = float(upper[0]), float(upper[1])
        # AABB intersection test
        if f_max_lon < min_lon or f_min_lon > max_lon:
            return False
        if f_max_lat < min_lat or f_min_lat > max_lat:
            return False
        return True
    except Exception:
        return True


def _parse_landuse_polygons_regex(
    gml_content: str,
    min_lon: float = -180, min_lat: float = -90,
    max_lon: float = 180, max_lat: float = 90,
) -> List[Tuple[int, List[np.ndarray]]]:
    """Fast regex extraction of LandUse features with bbox pre-filter.

    Returns list of (plateau_class_code, [ring_coords, ...]) tuples.
    Each ring_coords is an (N, 2) array of (lat, lon) pairs.
    Only features whose exterior ring intersects the given bbox are returned.
    """
    features = []
    for block_match in _LANDUSE_BLOCK_RE.finditer(gml_content):
        block = block_match.group(1)
        cls_match = _CLASS_RE.search(block)
        if cls_match is None:
            continue
        cls_code = int(cls_match.group(1))

        rings = []
        for pos_match in _POSLIST_RE.finditer(block):
            vals = np.fromstring(pos_match.group(1), sep=' ')
            # Coordinates are (lat, lon, z) triples
            if len(vals) >= 6:
                coords = vals.reshape(-1, 3)[:, :2]  # (lat, lon)
                if len(coords) >= 3:
                    rings.append(coords)
        if not rings:
            continue

        # Quick bbox check on exterior ring (lat, lon columns)
        ext = rings[0]
        ext_lats, ext_lons = ext[:, 0], ext[:, 1]
        if ext_lons.max() < min_lon or ext_lons.min() > max_lon:
            continue
        if ext_lats.max() < min_lat or ext_lats.min() > max_lat:
            continue

        features.append((cls_code, rings))
    return features


def _find_luse_files(citygml_path: str) -> List[Path]:
    """Locate luse GML files in a CityGML dataset directory."""
    root = Path(citygml_path)
    # PLATEAU layout
    luse_dir = root / 'udx' / 'luse'
    if luse_dir.exists():
        files = sorted(luse_dir.glob('*.gml'))
        if files:
            return files

    # Flat layout – look for files with 'luse' in name
    flat_files = sorted(root.glob('*luse*.gml'))
    if flat_files:
        return flat_files

    return []


def get_citygml_land_cover_grid(
    citygml_path: str,
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
) -> np.ndarray:
    """Generate a land-cover grid from CityGML ``luse:LandUse`` features.

    Parameters
    ----------
    citygml_path : str
        Root directory of the CityGML dataset (must contain
        ``udx/luse/*.gml`` for PLATEAU datasets).
    rectangle_vertices : list[(lon, lat)]
        [SW, NW, NE, SE] target rectangle in WGS 84.
    meshsize : float
        Grid resolution in metres.

    Returns
    -------
    np.ndarray
        2-D int32 grid of 1-based VoxCity land-cover codes.
        Values are already in the final internal format
        (no further conversion needed by ``_convert_land_cover``).
    """
    luse_files = _find_luse_files(citygml_path)
    if not luse_files:
        raise FileNotFoundError(
            f"No CityGML land use (luse) files found in {citygml_path}. "
            "Ensure the dataset contains udx/luse/*.gml files."
        )
    print(f"  Found {len(luse_files)} CityGML land use file(s)")

    # ── Build the target grid ────────────────────────────────────────
    # rectangle_vertices is [(lon,lat), ...] in [SW, NW, NE, SE] order
    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Estimate grid dimensions from metre size via simple degree approx
    lat_mid = (min_lat + max_lat) / 2
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(lat_mid)))

    n_rows = max(1, int(round((max_lat - min_lat) / (meshsize * deg_per_m_lat))))
    n_cols = max(1, int(round((max_lon - min_lon) / (meshsize * deg_per_m_lon))))

    # Cell edges in lat/lon  (south-up: row 0 = min_lat, to match
    # upstream voxcity convention where _apply_land_cover does flipud)
    lat_edges = np.linspace(min_lat, max_lat, n_rows + 1)  # south-up
    lon_edges = np.linspace(min_lon, max_lon, n_cols + 1)

    # Cell centre coordinates (precomputed for rasterisation)
    row_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    col_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Initialise grid with No Data (14)
    grid = np.full((n_rows, n_cols), 14, dtype=np.int32)

    # ── Parse and rasterize each file ────────────────────────────────
    total_features = 0
    total_rasterized = 0

    for gml_file in luse_files:
        # Quick envelope check to skip irrelevant files
        if not _file_envelope_intersects(gml_file, min_lon, min_lat,
                                         max_lon, max_lat):
            print(f"  Skipping {gml_file.name} (outside target area)")
            continue

        print(f"  Parsing {gml_file.name}...")
        try:
            with open(str(gml_file), 'r', encoding='utf-8',
                       errors='replace') as f:
                content = f.read()
        except Exception as exc:
            log.warning("Failed to read %s: %s", gml_file, exc)
            continue

        features = _parse_landuse_polygons_regex(
            content, min_lon, min_lat, max_lon, max_lat,
        )
        total_features += len(features)

        for cls_code, rings in features:
            voxcity_code = _PLATEAU_LUSE_TO_VOXCITY.get(cls_code, 14)

            # Use the exterior ring (first ring)
            exterior = rings[0]
            # CityGML coords are (lat, lon); Shapely needs (lon, lat)
            try:
                poly_lonlat = ShapelyPolygon(exterior[:, ::-1])
                if not poly_lonlat.is_valid:
                    poly_lonlat = poly_lonlat.buffer(0)
            except Exception:
                continue

            # Rasterize: find grid cells whose centres fall inside polygon
            pmin_lon, pmin_lat, pmax_lon, pmax_lat = poly_lonlat.bounds

            # Row/col range (south-up: row 0 = min_lat)
            r_start = max(0, int(np.floor(
                (pmin_lat - min_lat) / (meshsize * deg_per_m_lat))))
            r_end = min(n_rows, int(np.ceil(
                (pmax_lat - min_lat) / (meshsize * deg_per_m_lat))))
            c_start = max(0, int(np.floor(
                (pmin_lon - min_lon) / (meshsize * deg_per_m_lon))))
            c_end = min(n_cols, int(np.ceil(
                (pmax_lon - min_lon) / (meshsize * deg_per_m_lon))))

            if r_start >= r_end or c_start >= c_end:
                continue

            sub_lats = row_centres[r_start:r_end]
            sub_lons = col_centres[c_start:c_end]

            prep_poly = shapely_prep(poly_lonlat)
            for ri, lat_c in enumerate(sub_lats):
                for ci, lon_c in enumerate(sub_lons):
                    if prep_poly.contains(Point(lon_c, lat_c)):
                        existing = grid[r_start + ri, c_start + ci]
                        # Only overwrite if the new code has higher
                        # priority (road = lowest).
                        if _VOXCITY_PRIORITY.get(voxcity_code, 2) >= \
                                _VOXCITY_PRIORITY.get(int(existing), 2):
                            grid[r_start + ri, c_start + ci] = voxcity_code

            total_rasterized += 1

    print(f"  CityGML land use: {total_features} features parsed, "
          f"{total_rasterized} rasterized onto {n_rows}×{n_cols} grid")

    return grid


# -----------------------------------------------------------------------
# Vector polygon extraction (for OBJ export – no rasterization)
# -----------------------------------------------------------------------

def get_citygml_land_cover_polygons(
    citygml_path: str,
    rectangle_vertices: List[Tuple[float, float]],
) -> List[Tuple[int, ShapelyPolygon]]:
    """Return CityGML land use polygons clipped to the target rectangle.

    Unlike :func:`get_citygml_land_cover_grid`, this function returns the
    **raw vector geometry** (as Shapely polygons in *lon, lat* WGS 84) so
    that exporters can write true polygon meshes rather than grid quads.

    Parameters
    ----------
    citygml_path : str
        Root directory of the CityGML dataset.
    rectangle_vertices : list[(lon, lat)]
        [SW, NW, NE, SE] target rectangle in WGS 84.

    Returns
    -------
    list[(int, ShapelyPolygon)]
        Each entry is ``(voxcity_code, polygon_lonlat)``.
    """
    luse_files = _find_luse_files(citygml_path)
    if not luse_files:
        raise FileNotFoundError(
            f"No CityGML land use (luse) files found in {citygml_path}."
        )

    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    clip_box = box(min_lon, min_lat, max_lon, max_lat)

    result: List[Tuple[int, ShapelyPolygon]] = []

    for gml_file in luse_files:
        if not _file_envelope_intersects(gml_file, min_lon, min_lat,
                                         max_lon, max_lat):
            continue
        try:
            with open(str(gml_file), 'r', encoding='utf-8',
                       errors='replace') as f:
                content = f.read()
        except Exception as exc:
            log.warning("Failed to read %s: %s", gml_file, exc)
            continue

        features = _parse_landuse_polygons_regex(
            content, min_lon, min_lat, max_lon, max_lat,
        )

        for cls_code, rings in features:
            voxcity_code = _PLATEAU_LUSE_TO_VOXCITY.get(cls_code, 14)

            exterior = rings[0]
            # CityGML coords are (lat, lon); Shapely needs (lon, lat)
            try:
                poly_lonlat = ShapelyPolygon(exterior[:, ::-1])
                if not poly_lonlat.is_valid:
                    poly_lonlat = poly_lonlat.buffer(0)
            except Exception:
                continue

            # Clip to target rectangle
            clipped = poly_lonlat.intersection(clip_box)
            if clipped.is_empty:
                continue

            # intersection() may return MultiPolygon, GeometryCollection, etc.
            if clipped.geom_type == 'Polygon':
                result.append((voxcity_code, clipped))
            elif clipped.geom_type == 'MultiPolygon':
                for poly in clipped.geoms:
                    result.append((voxcity_code, poly))
            elif hasattr(clipped, 'geoms'):
                for geom in clipped.geoms:
                    if geom.geom_type == 'Polygon':
                        result.append((voxcity_code, geom))

    # ── Priority handling: subtract non-road areas from road polygons ──
    result = _apply_polygon_priority(result)

    print(f"  CityGML land use polygons: {len(result)} polygons extracted")
    return result


def _apply_polygon_priority(
    polys: List[Tuple[int, ShapelyPolygon]],
) -> List[Tuple[int, ShapelyPolygon]]:
    """Remove overlapping areas from lower-priority polygons.

    Polygons are grouped by their VoxCity code and processed from
    highest priority to lowest (see ``_VOXCITY_PRIORITY``).  For each
    priority level, the union of all *higher*-priority polygons is
    subtracted so that no two classes overlap.  This ensures, for
    example, that road polygons only cover areas not claimed by any
    other land-cover class.
    """
    from shapely.ops import unary_union

    if not polys:
        return polys

    # Group polygons by priority level (higher number = higher priority)
    from collections import defaultdict
    prio_groups: Dict[int, List[Tuple[int, ShapelyPolygon]]] = defaultdict(list)
    for code, poly in polys:
        prio = _VOXCITY_PRIORITY.get(code, 2)
        prio_groups[prio].append((code, poly))

    # If everything is the same priority, nothing to subtract
    if len(prio_groups) <= 1:
        return polys

    # Process from highest priority to lowest.
    # Accumulate a running union of all geometries already placed;
    # subtract that union from each subsequent (lower) priority level.
    sorted_prios = sorted(prio_groups.keys(), reverse=True)

    out: List[Tuple[int, ShapelyPolygon]] = []
    placed_union = None  # cumulative union of higher-priority geometry

    for prio in sorted_prios:
        group = prio_groups[prio]

        if placed_union is None:
            # Highest priority – emit as-is
            out.extend(group)
            placed_union = unary_union([p for _, p in group])
            continue

        # Subtract everything already placed from this priority level
        new_entries: List[Tuple[int, ShapelyPolygon]] = []
        for code, poly in group:
            diff = poly.difference(placed_union)
            if diff.is_empty:
                continue
            if diff.geom_type == 'Polygon':
                new_entries.append((code, diff))
            elif diff.geom_type == 'MultiPolygon':
                for g in diff.geoms:
                    new_entries.append((code, g))
            elif hasattr(diff, 'geoms'):
                for g in diff.geoms:
                    if g.geom_type == 'Polygon':
                        new_entries.append((code, g))

        if new_entries:
            out.extend(new_entries)
            # Expand the placed union with this level's geometry
            level_union = unary_union([p for _, p in new_entries])
            placed_union = placed_union.union(level_union)

    return out
