"""
CityGML directory parser with parallel processing.

Supports both Japanese PLATEAU datasets (``udx/`` sub-directory structure,
WGS84-geographic coordinates) and European CityGML datasets (flat directory,
projected CRS like UTM).  Non-WGS84 coordinates are automatically
reprojected to (lat, lon, z) at parse time.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.prepared import prep

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]

from ..models import Mesh3D, CityGMLMeshCollection
from .namespaces import build_namespaces, detect_crs_from_root, detect_crs_from_file_header
from .coordinates import (
    rectangle_to_shapely, file_intersects_rectangle,
    mesh_intersects_rectangle, reproject_vertices,
)
from .extractors import (
    extract_buildings_from_root,
    extract_bridges_from_root,
    extract_terrain_from_root,
    filter_terrain_by_rectangle_vectorized,
    extract_vegetation_from_root,
)
from .parse_cache import (
    load_cached_meshes, store_cached_meshes, reset_store_failures,
)

log = logging.getLogger(__name__)

# Feature types the extractor/filter pair understands. Anything else is
# rejected before parsing so it can never be written to the cache: an entry
# saying "this file contains nothing" is structurally valid and correctly
# versioned, so no CACHE_VERSION bump would ever retire it.
_KNOWN_TYPES = ('building', 'bridge', 'terrain', 'vegetation')



def _filter_meshes_by_rectangle(meshes: List[Mesh3D],
                                rect_polygon: ShapelyPolygon,
                                prepared_rect) -> List[Mesh3D]:
    """Filter meshes by partial intersection with rectangle.
    
    Returns all meshes whose 2D bounding box has ANY overlap with the rectangle.
    """
    return [m for m in meshes if mesh_intersects_rectangle(m, rect_polygon, prepared_rect)]


def _filter_building_and_bridge_meshes(meshes: List[Mesh3D],
                                       rect_polygon: ShapelyPolygon,
                                       prepared_rect) -> List[Mesh3D]:
    """Filter buildings/bridges by partial intersection with target rectangle.
    
    IMPORTANT: Buildings and bridges are included if they even PARTIALLY
    intersect the target rectangle. The voxelizer will include the full
    building/bridge mesh even if parts extend outside the rectangle.
    
    Args:
        meshes: Building or bridge mesh objects.
        rect_polygon: Target rectangle (Shapely polygon).
        prepared_rect: Prepared rectangle for faster intersection tests.
    
    Returns:
        List of meshes that have any overlap with the rectangle.
    """
    return [m for m in meshes if mesh_intersects_rectangle(m, rect_polygon, prepared_rect)]


def merge_terrain_meshes(terrain_meshes: List[Mesh3D]) -> List[Mesh3D]:
    """Merge multiple terrain meshes into a single mesh."""
    if not terrain_meshes:
        return []
    all_verts, all_faces = [], []
    offset = 0
    for mesh in terrain_meshes:
        if len(mesh.vertices) > 0:
            all_verts.append(mesh.vertices)
            all_faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
    if not all_verts:
        return []
    return [Mesh3D(
        vertices=np.vstack(all_verts),
        faces=np.vstack(all_faces).astype(np.int32),
        feature_type='terrain', feature_id='merged_terrain',
    )]


def _parse_single_file(gml_file: Path, feature_type: str,
                        rect_polygon, prepared_rect,
                        building_lod: Optional[int] = None,
                        source_epsg: Optional[str] = None,
                        *,
                        failures: Optional[List[str]] = None,
                        use_cache: bool = True) -> List[Mesh3D]:
    """Parse a single GML file.

    If *source_epsg* is given (e.g. ``'EPSG:25832'``), all extracted
    vertex coordinates are reprojected to WGS84 (lat, lon, z) before
    the meshes are returned.

    When *use_cache* is True (default), the unfiltered, reprojected
    extraction result is snapshotted to ``.voxcitygml_cache`` beside the
    dataset and reused on later calls; the rectangle filter is always
    applied afresh, so cached results are rectangle-independent.
    """
    if feature_type not in _KNOWN_TYPES:
        return []
    try:
        # Reconstruct prepared_rect if only rect_polygon is given
        # (PreparedGeometry is not picklable for multiprocessing)
        if rect_polygon is not None and prepared_rect is None:
            prepared_rect = prep(rect_polygon)

        if rect_polygon is not None and source_epsg is None:
            # Only apply mesh-code filtering for WGS84 / PLATEAU data
            if not file_intersects_rectangle(gml_file.name, rect_polygon):
                return []

        meshes = None
        if use_cache:
            meshes = load_cached_meshes(
                gml_file, feature_type, building_lod, source_epsg)
        # ``None`` is a miss; ``[]`` is a legitimate cached result. Testing
        # truthiness here would re-parse every empty tile on every request.
        if meshes is None:
            # Stat before extraction, never after: a source replaced mid-parse
            # must invalidate the entry rather than stamp the new size/mtime
            # onto arrays holding the old content.
            src_stat = os.stat(gml_file) if use_cache else None
            meshes = _extract_file_meshes(
                gml_file, feature_type, building_lod, source_epsg)
            if use_cache:
                store_cached_meshes(gml_file, feature_type, building_lod,
                                    meshes, source_epsg, src_stat=src_stat)

        return _filter_for_type(meshes, feature_type, rect_polygon, prepared_rect)
    except Exception as exc:
        log.warning("Failed to parse %s: %s", gml_file, exc)
        if failures is not None:
            failures.append(f"{gml_file}: {exc}")
        return []


def _extract_file_meshes(gml_file: Path, feature_type: str,
                         building_lod: Optional[int],
                         source_epsg: Optional[str]) -> List[Mesh3D]:
    """Extract + reproject one GML file. Rectangle-independent (cacheable)."""
    tree = ET.parse(str(gml_file))
    root = tree.getroot()
    ns = build_namespaces(root)

    # Auto-detect CRS from file if not already known
    file_epsg = source_epsg
    if file_epsg is None:
        file_epsg = detect_crs_from_root(root)

    if feature_type == 'building':
        meshes = extract_buildings_from_root(
            root, ns,
            prefer_lod=building_lod,
            max_lod=4,
        )
    elif feature_type == 'bridge':
        meshes = extract_bridges_from_root(root, ns)
    elif feature_type == 'terrain':
        try:
            with open(str(gml_file), 'r', encoding='utf-8') as f:
                content = f.read()
            meshes = extract_terrain_from_root(None, None, gml_content=content)
        except Exception:
            meshes = extract_terrain_from_root(root, ns)
    elif feature_type == 'vegetation':
        meshes = extract_vegetation_from_root(root, ns)
    else:
        # Unreachable via _parse_single_file, which gates on _KNOWN_TYPES;
        # kept only so this function is total if called directly.
        return []

    if file_epsg:
        meshes = _reproject_meshes(meshes, file_epsg)
    return meshes


def _filter_for_type(meshes: List[Mesh3D], feature_type: str,
                     rect_polygon, prepared_rect) -> List[Mesh3D]:
    """Apply the per-feature-type rectangle filter (post-parse, post-cache)."""
    if rect_polygon is None or not meshes:
        return meshes
    if feature_type in ('building', 'bridge'):
        return _filter_building_and_bridge_meshes(meshes, rect_polygon, prepared_rect)
    if feature_type == 'terrain':
        return filter_terrain_by_rectangle_vectorized(meshes, rect_polygon, prepared_rect)
    if feature_type == 'vegetation':
        return _filter_meshes_by_rectangle(meshes, rect_polygon, prepared_rect)
    # Unreachable: _parse_single_file gates on _KNOWN_TYPES. Passing the input
    # through matches the ``rect_polygon is None`` early return above, so an
    # unhandled type behaves the same way whether or not a rectangle is given.
    return meshes


def _reproject_meshes(meshes: List[Mesh3D], source_epsg: str) -> List[Mesh3D]:
    """Reproject mesh vertices from *source_epsg* → WGS84 (lat, lon, z)."""
    for mesh in meshes:
        if len(mesh.vertices) > 0:
            mesh.vertices = reproject_vertices(mesh.vertices, source_epsg)
            # Also reproject triangle_coords cached in terrain meshes
            if 'triangle_coords' in mesh.attributes:
                tri = mesh.attributes['triangle_coords']  # (N, 3, 3)
                shape = tri.shape
                flat = tri.reshape(-1, 3)
                flat_reproj = reproject_vertices(flat, source_epsg)
                mesh.attributes['triangle_coords'] = flat_reproj.reshape(shape)
    return meshes


def parse_citygml_directory(
    citygml_path: str,
    max_files: Optional[int] = None,
    rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
    n_workers: int = 4,
    feature_types: Optional[List[str]] = None,
    building_lod: Optional[int] = None,
    dem_path: Optional[str] = None,
    tree_citygml_path: Optional[str] = None,
    use_parse_cache: bool = True,
) -> CityGMLMeshCollection:
    """Parse all CityGML files in a dataset directory.

    Supports two directory layouts:

    * **PLATEAU** (Japan): ``udx/bldg/``, ``udx/dem/``, ``udx/brid/``,
      ``udx/veg/`` sub-directories with mesh-code filenames.
    * **Flat / European**: all ``.gml`` files in a single directory;
      feature types are detected from file content.

    Non-WGS84 CRS (e.g. UTM) is auto-detected from the GML envelope
    and reprojected to (lat, lon, z) at parse time.

    Args:
        citygml_path: Root directory of the CityGML dataset.
        max_files: Limit number of files per category (for testing).
        rectangle_vertices: [(lon,lat), ...] filter rectangle [SW, NW, NE, SE].
        n_workers: Parallel worker count.
        feature_types: List of types to parse ('building', 'bridge', 'terrain', 'vegetation').
        building_lod: Preferred CityGML building LOD (1-4). If None, use highest available.
        dem_path: Optional path to a GeoTIFF DEM file (e.g. ``*.tif``).  When
            supplied and no terrain is found in the GML files, the raster is
            loaded, reprojected to WGS84 and converted to terrain triangles.
        tree_citygml_path: Optional path to a separate CityGML directory
            containing vegetation/tree data (e.g. Munich semantic tree models).
            If supplied, vegetation GML files are loaded from this directory
            *in addition to* any vegetation found in the main citygml_path.
        use_parse_cache: If True (default), cache parsed meshes as binary
            snapshots in ``.voxcitygml_cache`` beside the dataset and reuse
            them on later calls. Set False to always parse the XML.

    Returns:
        CityGMLMeshCollection with parsed meshes.
    """
    # Re-arm cache writes per call, not per process. The failure latch exists
    # to bound warning noise within one run, but it cannot tell a permanently
    # read-only dataset from a transient hiccup (on Windows, os.replace fails
    # with "access is denied" while a concurrent request holds an entry open
    # for reading). Without this, a long-lived server would silently lose
    # caching for good after a few unrelated collisions.
    if use_parse_cache:
        reset_store_failures()

    collection = CityGMLMeshCollection()
    citygml_root = Path(citygml_path)

    # Determine directory layout
    udx_path = citygml_root / 'udx'
    is_plateau = udx_path.exists()

    if feature_types is None:
        feature_types = ['building', 'bridge', 'terrain', 'vegetation']

    parse_failures: List[str] = []

    rect_polygon = None
    prepared_rect = None
    if rectangle_vertices is not None:
        rect_polygon = rectangle_to_shapely(rectangle_vertices)
        prepared_rect = prep(rect_polygon)
        print(f"Filtering to rectangle: {rectangle_vertices[0]} to {rectangle_vertices[2]}")

    # ------------------------------------------------------------------
    # Detect CRS from the first available GML file
    # ------------------------------------------------------------------
    source_epsg = _detect_dataset_crs(citygml_root, is_plateau)
    if source_epsg:
        print(f"Detected projected CRS: {source_epsg} – coordinates will be reprojected to WGS84")

    # ------------------------------------------------------------------
    # PLATEAU layout: sub-directories per feature type
    # ------------------------------------------------------------------
    if is_plateau:
        base = udx_path

        def process_folder(folder: Path, ftype: str, desc: str) -> List[Mesh3D]:
            if not folder.exists():
                return []
            gml_files = list(folder.glob('*.gml'))
            if rect_polygon is not None and source_epsg is None:
                gml_files = [f for f in gml_files
                             if file_intersects_rectangle(f.name, rect_polygon)]
            if max_files:
                gml_files = gml_files[:max_files]
            if not gml_files:
                return []
            print(f"Parsing {len(gml_files)} {desc} files...")
            all_meshes: List[Mesh3D] = []
            for gml_file in tqdm(gml_files, desc=desc):
                meshes = _parse_single_file(
                    gml_file, ftype, rect_polygon, prepared_rect,
                    building_lod=building_lod, source_epsg=source_epsg,
                    failures=parse_failures,
                    use_cache=use_parse_cache,
                )
                if meshes:
                    all_meshes.extend(meshes)
            return all_meshes

        if 'building' in feature_types:
            collection.buildings = process_folder(base / 'bldg', 'building', 'Buildings')
        if 'bridge' in feature_types:
            collection.bridges = process_folder(base / 'brid', 'bridge', 'Bridges')
        if 'terrain' in feature_types:
            collection.terrain = process_folder(base / 'dem', 'terrain', 'Terrain')
        if 'vegetation' in feature_types:
            collection.vegetation = process_folder(base / 'veg', 'vegetation', 'Vegetation')

    # ------------------------------------------------------------------
    # Flat directory: probe each GML file for its feature type
    # ------------------------------------------------------------------
    else:
        gml_files = sorted(citygml_root.glob('*.gml'))
        if max_files:
            gml_files = gml_files[:max_files]

        if gml_files:
            print(f"Flat directory layout – scanning {len(gml_files)} GML files...")
            _parse_flat_directory(
                gml_files, collection, feature_types,
                rect_polygon, prepared_rect,
                building_lod, source_epsg,
                failures=parse_failures,
                use_cache=use_parse_cache,
            )

    # Keep this summary after all failure-accumulating parse paths.
    if parse_failures:
        print(f"WARNING: {len(parse_failures)} file(s) failed to parse:")
        for f in parse_failures[:10]:
            print(f"  - {f}")
        if len(parse_failures) > 10:
            print(f"  ... and {len(parse_failures) - 10} more")

    # ------------------------------------------------------------------
    # GeoTIFF DEM fallback (explicit path only)
    # ------------------------------------------------------------------
    if 'terrain' in feature_types and not collection.terrain and dem_path:
        tif_meshes = _load_geotiff_dem(Path(dem_path), rect_polygon)
        if tif_meshes:
            collection.terrain = tif_meshes

    # ------------------------------------------------------------------
    # External tree / vegetation CityGML directory
    # ------------------------------------------------------------------
    if 'vegetation' in feature_types and tree_citygml_path:
        tree_root = Path(tree_citygml_path)
        if tree_root.exists():
            tree_gml_files = sorted(tree_root.glob('*.gml'))
            if max_files:
                tree_gml_files = tree_gml_files[:max_files]
            if tree_gml_files:
                tree_epsg = _detect_dataset_crs(tree_root, is_plateau=False)
                if tree_epsg:
                    print(f"Tree CityGML CRS: {tree_epsg}")
                # Spatial pre-filter: only parse tiles overlapping the target rect
                n_before = len(tree_gml_files)
                if rect_polygon is not None and tree_epsg:
                    tree_gml_files = _filter_tiles_by_rectangle(
                        tree_gml_files, rect_polygon, tree_epsg,
                    )
                    print(f"  Tile filter: {n_before} -> {len(tree_gml_files)} tiles")
                print(f"Parsing {len(tree_gml_files)} tree/vegetation files"
                      f" from {tree_root} ({n_workers} workers)...")
                tree_meshes: List[Mesh3D] = []
                effective_workers = min(n_workers, len(tree_gml_files))
                if effective_workers > 1:
                    ctx = multiprocessing.get_context('spawn')
                    with ProcessPoolExecutor(
                        max_workers=effective_workers, mp_context=ctx,
                    ) as executor:
                        futures = {
                            executor.submit(
                                _parse_single_file, gml_file, 'vegetation',
                                rect_polygon, None,  # prepared_rect rebuilt in child
                                None, tree_epsg,
                                use_cache=use_parse_cache,
                            ): gml_file
                            for gml_file in tree_gml_files
                        }
                        for future in tqdm(
                            as_completed(futures), total=len(futures), desc="Trees",
                        ):
                            try:
                                meshes = future.result()
                                if meshes:
                                    tree_meshes.extend(meshes)
                            except Exception as exc:
                                log.debug("Tree file error: %s", exc)
                else:
                    for gml_file in tqdm(tree_gml_files, desc="Trees"):
                        meshes = _parse_single_file(
                            gml_file, 'vegetation', rect_polygon, prepared_rect,
                            building_lod=None, source_epsg=tree_epsg,
                            use_cache=use_parse_cache,
                        )
                        if meshes:
                            tree_meshes.extend(meshes)
                if tree_meshes:
                    print(f"  Extracted {len(tree_meshes)} tree meshes from external directory")
                    collection.vegetation.extend(tree_meshes)
        else:
            log.warning("tree_citygml_path does not exist: %s", tree_citygml_path)

    print(f"\nParsing complete:")
    print(f"  Buildings:  {len(collection.buildings)}")
    print(f"  Bridges:    {len(collection.bridges)}")
    print(f"  Terrain:    {len(collection.terrain)}")
    print(f"  Vegetation: {len(collection.vegetation)}")
    print(f"  Vertices:   {collection.total_vertices():,}")
    print(f"  Faces:      {collection.total_faces():,}")
    return collection


# ------------------------------------------------------------------
# Tile spatial filtering
# ------------------------------------------------------------------

def _filter_tiles_by_rectangle(
    gml_files: List[Path],
    rect_polygon: ShapelyPolygon,
    source_epsg: str,
    tile_size_m: float = 1000.0,
) -> List[Path]:
    """Filter GML tile files whose filenames encode grid coordinates.

    Supports filenames like ``..._tile_680_5338.gml`` where the two
    trailing integers are the SW corner of the tile in *source_epsg*
    kilometres.  Each tile is assumed to cover *tile_size_m* x *tile_size_m*
    metres.

    The WGS84 *rect_polygon* is reprojected to *source_epsg* and only
    the tiles that intersect the reprojected bounding box (with a small
    margin) are kept.
    """
    import re
    from pyproj import Transformer

    # Parse EPSG code
    epsg_code = source_epsg.split(':')[-1]

    # Reproject rectangle bounds to the tile CRS
    t = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_code}', always_xy=True)
    minlon, minlat, maxlon, maxlat = rect_polygon.bounds
    corners = [
        t.transform(minlon, minlat),
        t.transform(minlon, maxlat),
        t.transform(maxlon, minlat),
        t.transform(maxlon, maxlat),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    margin = tile_size_m  # one-tile margin
    rect_min_x = min(xs) - margin
    rect_max_x = max(xs) + margin
    rect_min_y = min(ys) - margin
    rect_max_y = max(ys) + margin

    tile_re = re.compile(r'tile_(\d+)_(\d+)')
    kept: List[Path] = []
    for f in gml_files:
        m = tile_re.search(f.stem)
        if not m:
            kept.append(f)  # no tile coords -> keep (safe fallback)
            continue
        tx = int(m.group(1)) * 1000  # SW corner easting  (m)
        ty = int(m.group(2)) * 1000  # SW corner northing (m)
        # Tile covers [tx, tx+tile_size_m] x [ty, ty+tile_size_m]
        if (tx + tile_size_m >= rect_min_x and tx <= rect_max_x and
                ty + tile_size_m >= rect_min_y and ty <= rect_max_y):
            kept.append(f)
    return kept


# ------------------------------------------------------------------
# Flat-directory helpers
# ------------------------------------------------------------------

def _detect_feature_type_from_header(filepath: Path) -> Optional[str]:
    """Read first ~4 KB of a GML file and guess the dominant feature type."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            header = f.read(8192)
        # Simple heuristics based on namespace/element presence
        if 'building' in header.lower() or 'bldg:' in header:
            return 'building'
        if 'bridge' in header.lower() or 'brid:' in header:
            return 'bridge'
        if 'TINRelief' in header or 'dem:' in header:
            return 'terrain'
        if 'vegetation' in header.lower() or 'veg:' in header:
            return 'vegetation'
    except Exception:
        pass
    return None


def _detect_dataset_crs(root_dir: Path, is_plateau: bool) -> Optional[str]:
    """Detect CRS from the first GML file found in the dataset."""
    if is_plateau:
        search_dirs = [root_dir / 'udx' / sub for sub in ('bldg', 'brid', 'dem', 'veg')]
    else:
        search_dirs = [root_dir]
    for d in search_dirs:
        if not d.exists():
            continue
        for gml in sorted(d.glob('*.gml'))[:3]:
            epsg = detect_crs_from_file_header(str(gml))
            if epsg:
                return epsg
    return None


def _parse_flat_directory(
    gml_files: List[Path],
    collection: CityGMLMeshCollection,
    feature_types: List[str],
    rect_polygon, prepared_rect,
    building_lod: Optional[int],
    source_epsg: Optional[str],
    *,
    failures: Optional[List[str]] = None,
    use_cache: bool = True,
) -> None:
    """Parse a flat directory of GML files, auto-detecting feature types."""
    # Group files by detected feature type
    groups: Dict[str, List[Path]] = {ft: [] for ft in feature_types}

    for gml_file in gml_files:
        ftype = _detect_feature_type_from_header(gml_file)
        if ftype and ftype in groups:
            groups[ftype].append(gml_file)
        elif ftype is None:
            # Could not detect – try as building (most common)
            if 'building' in groups:
                groups['building'].append(gml_file)

    for ftype, files in groups.items():
        if not files:
            continue
        print(f"Parsing {len(files)} {ftype} files...")
        all_meshes: List[Mesh3D] = []
        for gml_file in tqdm(files, desc=ftype.capitalize()):
            meshes = _parse_single_file(
                gml_file, ftype, rect_polygon, prepared_rect,
                building_lod=building_lod, source_epsg=source_epsg,
                failures=failures,
                use_cache=use_cache,
            )
            if meshes:
                all_meshes.extend(meshes)

        if ftype == 'building':
            collection.buildings.extend(all_meshes)
        elif ftype == 'bridge':
            collection.bridges.extend(all_meshes)
        elif ftype == 'terrain':
            collection.terrain.extend(all_meshes)
        elif ftype == 'vegetation':
            collection.vegetation.extend(all_meshes)


# ------------------------------------------------------------------
# GeoTIFF DEM loader
# ------------------------------------------------------------------

def _load_geotiff_dem(dem_path: Path,
                      rect_polygon: Optional[ShapelyPolygon] = None,
                      ) -> List[Mesh3D]:
    """Load terrain from a GeoTIFF DEM file.

    The raster is reprojected to WGS84. Each pixel becomes a triangle pair
    in the returned Mesh3D so that the rest of the pipeline (TIN interpolation)
    works unchanged.

    Parameters
    ----------
    dem_path : Path
        Path to a single GeoTIFF file (``*.tif`` / ``*.tiff``).
    rect_polygon : ShapelyPolygon, optional
        Spatial filter rectangle in WGS84 (lon, lat).
    """
    if not dem_path.exists():
        log.warning("DEM file not found: %s", dem_path)
        return []

    try:
        import rasterio
    except ImportError:
        log.warning("rasterio not installed – cannot read GeoTIFF DEM")
        return []

    try:
        return _read_single_geotiff_dem(dem_path, rect_polygon)
    except Exception as exc:
        log.warning("Failed to read GeoTIFF %s: %s", dem_path, exc)
        return []


def _read_single_geotiff_dem(tif_path: Path,
                             rect_polygon: Optional[ShapelyPolygon],
                             max_points: int = 500_000,
                             ) -> List[Mesh3D]:
    """Read one GeoTIFF DEM and return triangle meshes in (lat, lon, z)."""
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box as shapely_box

    with rasterio.open(str(tif_path)) as src:
        src_crs = src.crs
        transform = src.transform
        data = src.read(1)
        nodata = src.nodata

    nrows, ncols = data.shape

    # Build coordinate arrays in the source CRS
    col_idx = np.arange(ncols)
    row_idx = np.arange(nrows)
    xs = transform.c + (col_idx + 0.5) * transform.a
    ys = transform.f + (row_idx + 0.5) * transform.e
    XX, YY = np.meshgrid(xs, ys)

    # Reproject to WGS84
    if src_crs and not src_crs.is_geographic:
        t = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
        LON, LAT = t.transform(XX, YY)
    else:
        LON, LAT = XX, YY

    # Spatial filter: only keep data within the rectangle (with some margin)
    if rect_polygon is not None:
        rlon = LON.ravel()
        rlat = LAT.ravel()
        minx, miny, maxx, maxy = rect_polygon.bounds
        margin = max(maxx - minx, maxy - miny) * 0.1
        mask_2d = ((LON >= minx - margin) & (LON <= maxx + margin) &
                   (LAT >= miny - margin) & (LAT <= maxy + margin))
        if not mask_2d.any():
            return []
        # Crop to bounding rows/cols for efficiency
        rows_any = mask_2d.any(axis=1)
        cols_any = mask_2d.any(axis=0)
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        LAT = LAT[r0:r1+1, c0:c1+1]
        LON = LON[r0:r1+1, c0:c1+1]
        data = data[r0:r1+1, c0:c1+1]
        nrows, ncols = data.shape

    # Subsample if too large
    step = 1
    while nrows * ncols > max_points:
        step += 1
        nrows_s = len(range(0, nrows, step))
        ncols_s = len(range(0, ncols, step))
        if nrows_s * ncols_s <= max_points:
            break
    if step > 1:
        LAT = LAT[::step, ::step]
        LON = LON[::step, ::step]
        data = data[::step, ::step]
        nrows, ncols = data.shape

    # Build triangles (2 per quad cell)
    valid = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        valid = data != nodata

    triangles = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            if not (valid[r, c] and valid[r, c+1] and valid[r+1, c] and valid[r+1, c+1]):
                continue
            # Triangle 1: top-left, top-right, bottom-left
            t1 = np.array([
                [LAT[r, c],   LON[r, c],   data[r, c]],
                [LAT[r, c+1], LON[r, c+1], data[r, c+1]],
                [LAT[r+1, c], LON[r+1, c], data[r+1, c]],
            ])
            # Triangle 2: top-right, bottom-right, bottom-left
            t2 = np.array([
                [LAT[r, c+1],   LON[r, c+1],   data[r, c+1]],
                [LAT[r+1, c+1], LON[r+1, c+1], data[r+1, c+1]],
                [LAT[r+1, c],   LON[r+1, c],   data[r+1, c]],
            ])
            triangles.append(t1)
            triangles.append(t2)

    if not triangles:
        return []

    tri_arr = np.array(triangles)  # (N, 3, 3) in (lat, lon, z)
    n = len(tri_arr)
    print(f"  Loaded GeoTIFF DEM: {tif_path.name} → {n} triangles")

    return [Mesh3D(
        vertices=tri_arr.reshape(-1, 3),
        faces=np.arange(n * 3, dtype=np.int32).reshape(-1, 3),
        feature_type='terrain',
        feature_id=f'geotiff_{tif_path.stem}',
        attributes={'triangle_coords': tri_arr},
    )]
