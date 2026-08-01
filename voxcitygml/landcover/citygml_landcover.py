"""
CityGML-based land cover grid generation.

Parses ``luse:LandUse`` features from CityGML (PLATEAU) datasets,
maps their class codes to the internal VoxCity 1-based land cover
indices, and rasterizes the polygons onto a regular grid matching the
target rectangle and resolution.

Rasterisation runs in the shared affine :class:`~voxcitygml.grid_utils.GridParams`
frame (NW origin, basis vectors along the rectangle's own sides), so the
grid follows a **rotated** rectangle rather than the axis-aligned bounding
box of its corners.  See :func:`get_citygml_land_cover_grid` for the row
ordering of the returned array.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box, Point
from shapely.prepared import prep as shapely_prep

from ..grid_utils import compute_grid_params

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


def _find_luse_files(citygml_path: Union[str, List[str]]) -> List[Path]:
    """Locate luse GML files in one or more CityGML dataset directories.

    *citygml_path* may be a single dataset root or a list of roots (as
    returned by :func:`resolve_citygml_paths`).  When a list is given,
    luse files from every dataset are collected and de-duplicated so the
    target area is found regardless of which dataset contains it.
    """
    # Multiple dataset roots – collect luse files from each and dedupe
    if isinstance(citygml_path, (list, tuple)):
        collected: List[Path] = []
        seen = set()
        for p in citygml_path:
            for f in _find_luse_files(p):
                key = str(f.resolve())
                if key not in seen:
                    seen.add(key)
                    collected.append(f)
        return collected

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


def _cell_window(gp, bounds) -> Optional[Tuple[int, int, int, int]]:
    """Candidate ``(r_start, r_end, c_start, c_end)`` for a lon/lat bbox.

    Maps the bbox's four corners through the affine frame and takes the
    min/max of the resulting (row, col) pairs.  Because the frame is
    affine, the image of the bbox is a parallelogram whose row/col extent
    is exactly the extent of its corner images — so the returned window is
    a guaranteed superset of the cells whose centres lie in the bbox
    (and therefore of those inside any polygon it bounds), for a rotated
    rectangle as well as an axis-aligned one.

    Returns ``None`` when the bbox misses the grid entirely, or is empty
    (``buffer(0)`` on a self-intersecting ring can collapse a polygon to
    nothing, and an empty geometry has no ``bounds``).
    """
    if len(bounds) != 4:
        return None
    bmin_lon, bmin_lat, bmax_lon, bmax_lat = bounds
    corner_lons = np.array([bmin_lon, bmin_lon, bmax_lon, bmax_lon])
    corner_lats = np.array([bmin_lat, bmax_lat, bmin_lat, bmax_lat])
    rows, cols = gp.lonlat_to_rowcol(corner_lons, corner_lats)

    # +/- 1 cell of slack absorbs floating-point noise at the boundary.
    r_start = max(0, int(np.floor(rows.min())) - 1)
    r_end = min(gp.n_rows, int(np.ceil(rows.max())) + 2)
    c_start = max(0, int(np.floor(cols.min())) - 1)
    c_end = min(gp.n_cols, int(np.ceil(cols.max())) + 2)

    if r_start >= r_end or c_start >= c_end:
        return None
    return r_start, r_end, c_start, c_end


def get_citygml_land_cover_grid(
    citygml_path: Union[str, List[str]],
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
        2-D int32 grid of 1-based VoxCity land-cover codes, shape
        ``(gp.n_rows, gp.n_cols)`` for ``gp = compute_grid_params(...)``.
        Values are already in the final internal format
        (no further conversion needed by ``_convert_land_cover``).

        **Row order is reversed relative to the** :class:`GridParams`
        **frame** (row 0 = the *last* grid row, i.e. the southern edge for
        an unrotated rectangle).  That is the voxcity land-cover
        convention: every consumer — ``voxelizer3d._apply_land_cover`` and
        ``export_obj`` — applies ``np.flipud`` before use, exactly as it
        does for the OpenStreetMap / ESA WorldCover grids.  Rasterisation
        happens in the canonical north-up frame and the array is flipped
        once, at the end, so this function stays interchangeable with the
        other land-cover sources.
    """
    luse_files = _find_luse_files(citygml_path)
    if not luse_files:
        raise FileNotFoundError(
            f"No CityGML land use (luse) files found in {citygml_path}. "
            "Ensure the dataset contains udx/luse/*.gml files."
        )
    print(f"  Found {len(luse_files)} CityGML land use file(s)")

    # ── Build the target grid ────────────────────────────────────────
    # The affine frame is shared with the DEM / building / canopy
    # rasterizers, so the land-cover grid follows the rectangle's own
    # sides (correct under rotation) and its shape is guaranteed to match
    # the DEM grid — no lossy nearest-neighbour resize downstream.
    gp = compute_grid_params(rectangle_vertices, meshsize)
    n_rows, n_cols = gp.n_rows, gp.n_cols

    # The bounding box is still what the file/feature pre-filters use: it
    # contains the rectangle under any rotation, so it never discards a
    # feature that could contribute a cell.
    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Initialise grid with No Data (14).  Rasterised north-up (row 0 =
    # the NW-anchored first row of the affine frame); flipped to the
    # voxcity land-cover row order just before returning.
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
            window = _cell_window(gp, poly_lonlat.bounds)
            if window is None:
                continue
            r_start, r_end, c_start, c_end = window

            rows = np.arange(r_start, r_end)
            cols = np.arange(c_start, c_end)
            sub_lons, sub_lats = gp.cell_centres(rows=rows, cols=cols)

            prep_poly = shapely_prep(poly_lonlat)
            for ri in range(r_end - r_start):
                for ci in range(c_end - c_start):
                    if prep_poly.contains(Point(sub_lons[ri, ci],
                                                sub_lats[ri, ci])):
                        existing = grid[r_start + ri, c_start + ci]
                        # Only overwrite if the new code has higher
                        # priority (road = lowest).
                        if _VOXCITY_PRIORITY.get(voxcity_code, 2) >= \
                                _VOXCITY_PRIORITY.get(int(existing), 2):
                            grid[r_start + ri, c_start + ci] = voxcity_code

            total_rasterized += 1

    print(f"  CityGML land use: {total_features} features parsed, "
          f"{total_rasterized} rasterized onto {n_rows}×{n_cols} grid")

    # Flip to the voxcity land-cover row order (see the docstring): every
    # consumer applies np.flipud, so this hands back the same orientation
    # the OpenStreetMap / ESA WorldCover sources do.
    return np.flipud(grid).copy()


# -----------------------------------------------------------------------
# Vector polygon extraction (for OBJ export – no rasterization)
# -----------------------------------------------------------------------

def get_citygml_land_cover_polygons(
    citygml_path: Union[str, List[str]],
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
