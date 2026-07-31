"""
Data models for the VoxCityGML pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np


def resolve_citygml_paths(raw_path: Union[str, List[str]]) -> List[str]:
    """Resolve one or more CityGML dataset paths, with auto-discovery.

    Accepts a single path string, or a list of path strings.  Each path
    is checked:

    * If it contains a ``udx`` sub-directory (PLATEAU layout) or ``.gml``
      files directly, it is treated as a dataset root and kept as-is.
    * Otherwise, child and grandchild directories are scanned for
      ``udx`` sub-directories or ``.gml`` files, and every matching
      directory is included.

    This allows users to pass a high-level parent folder such as
    ``/data/citygml/plateau`` and have all CityGML datasets
    beneath it discovered automatically.

    Returns
    -------
    list[str]
        Sorted list of resolved dataset directory paths.
        Raises ``FileNotFoundError`` if no datasets are found.
    """
    if isinstance(raw_path, list):
        paths_to_check = [Path(p) for p in raw_path]
    else:
        paths_to_check = [Path(raw_path)]

    resolved: List[str] = []

    for p in paths_to_check:
        if not p.exists():
            raise FileNotFoundError(f"CityGML path does not exist: {p}")

        if _is_citygml_dataset(p):
            resolved.append(str(p))
        else:
            # Scan children and grandchildren (depth 1-2)
            found = False
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    if _is_citygml_dataset(child):
                        resolved.append(str(child))
                        found = True
                    else:
                        for grandchild in sorted(child.iterdir()):
                            if grandchild.is_dir() and _is_citygml_dataset(grandchild):
                                resolved.append(str(grandchild))
                                found = True
            if not found:
                raise FileNotFoundError(
                    f"No CityGML datasets found under: {p}\n"
                    f"Expected directories containing a 'udx' folder "
                    f"or .gml files."
                )

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in resolved:
        norm = str(Path(r).resolve())
        if norm not in seen:
            seen.add(norm)
            unique.append(r)
    return unique


def _is_citygml_dataset(directory: Path) -> bool:
    """Return True if *directory* looks like a CityGML dataset root."""
    if (directory / 'udx').is_dir():
        return True
    # Flat layout: directory contains .gml files
    try:
        return any(directory.glob('*.gml'))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Mesh data structures (transplanted from citygml_mesher.models)
# ---------------------------------------------------------------------------

@dataclass
class Mesh3D:
    """Represents a 3D triangle mesh with vertices, faces, and metadata."""
    vertices: np.ndarray   # (N, 3) float64 – vertex positions
    faces: np.ndarray      # (M, 3) int32   – triangle indices
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    feature_type: str = "unknown"
    feature_id: str = ""
    attributes: Dict = field(default_factory=dict)


@dataclass
class CityGMLMeshCollection:
    """Collection of 3D meshes organised by CityGML feature type."""
    buildings: List[Mesh3D] = field(default_factory=list)
    bridges: List[Mesh3D] = field(default_factory=list)
    terrain: List[Mesh3D] = field(default_factory=list)
    vegetation: List[Mesh3D] = field(default_factory=list)

    def total_vertices(self) -> int:
        return sum(len(m.vertices) for grp in
                   [self.buildings, self.bridges, self.terrain, self.vegetation]
                   for m in grp)

    def total_faces(self) -> int:
        return sum(len(m.faces) for grp in
                   [self.buildings, self.bridges, self.terrain, self.vegetation]
                   for m in grp)

    def merge(self, other: 'CityGMLMeshCollection') -> None:
        """Merge another collection into this one (in-place).

        Buildings and bridges are deduplicated by ``feature_id`` (gml:id).
        When two CityGML directories cover adjacent areas (e.g. different
        wards), the same building can appear in both datasets.  Keeping
        both copies leads to duplicate building-ID entries that corrupt
        downstream per-building analyses (GVI, solar, etc.).
        """
        existing_building_ids = {m.feature_id for m in self.buildings}
        new_buildings = [m for m in other.buildings
                        if m.feature_id not in existing_building_ids]
        n_dup_bldg = len(other.buildings) - len(new_buildings)

        existing_bridge_ids = {m.feature_id for m in self.bridges}
        new_bridges = [m for m in other.bridges
                       if m.feature_id not in existing_bridge_ids]
        n_dup_brid = len(other.bridges) - len(new_bridges)

        if n_dup_bldg or n_dup_brid:
            print(f"  Merge: skipped {n_dup_bldg} duplicate buildings, "
                  f"{n_dup_brid} duplicate bridges (same gml:id)")

        self.buildings.extend(new_buildings)
        self.bridges.extend(new_bridges)
        self.terrain.extend(other.terrain)
        self.vegetation.extend(other.vegetation)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class VoxelizerConfig:
    """Configuration for the CityGML → VoxCity pipeline.

    Attributes:
        citygml_path: Path to a CityGML dataset directory (or a list of
                      paths).  Accepts both PLATEAU-style layouts
                      (``udx/bldg/``, ``udx/dem/``, …) and flat directories
                      with ``.gml`` files.  When a list is given, each
                      directory is parsed and the results are merged.
        center_lon:  Centre longitude of the target area (WGS 84).
        center_lat:  Centre latitude of the target area (WGS 84).
        size_meters: Side length of the square target area in metres.
        meshsize:    Voxel resolution in metres (default 1 m).
        land_cover_source:      Land-cover data source name
                                (forwarded to ``voxcity.generator.grids``).
                                If ``None``, automatically selected based on
                                location using ``voxcity``.
        canopy_height_source:   Canopy-height data source name
                                (forwarded to ``voxcity.generator.grids``).
                                If ``None``, automatically selected based on
                                location using ``voxcity``.
        output_dir:  Directory for saved artefacts.
        n_workers:   Number of parallel workers for CityGML parsing.
        trunk_height_ratio: Ratio of trunk height to total tree height.
        static_tree_height: Default tree height when canopy source is ``Static``.
        gridvis:     Show matplotlib grid visualisations during processing.
        save_output: Persist the final VoxCity object to disk.
        buffer_meters: Extra buffer around the target rectangle when parsing
                       CityGML (avoids edge artefacts).
        use_3d_voxelizer: If True, voxelize all CityGML meshes in a shared
                  3-D grid. If False, use the legacy 2.5-D grids.
        max_voxel_ram_mb: Optional hard limit for 3-D voxel grid allocation.
        occupancy_threshold: Minimum volume overlap fraction (0.0–1.0) a
            boundary voxel must reach to be kept during surface voxelization.
            0.0 (default) keeps every voxel with any geometric contact.
        occupancy_subdivisions: Sub-divisions per axis when estimating
            volume fraction (default 3 → 27 sub-samples per voxel).
        building_lod: Preferred CityGML building LOD (1–4) to voxelize.
                      If ``None``, the highest available LOD for each
                      building is selected automatically.
        dem_path: Optional path to a GeoTIFF DEM/DTM file.  Used as terrain
                  source when the CityGML dataset does not include TINRelief
                  geometry (common for German / European datasets).
        gee_project: Optional Google Earth Engine cloud project ID
                     (e.g. ``'my-gee-project'``).  Passed to
                     ``ee.Initialize(project=...)``.  Required for GEE-based
                     data sources like canopy height maps.
        tree_citygml_path: Optional path to a separate CityGML directory
                     containing vegetation / tree models (e.g. Munich
                     semantic 3-D tree models).  These are parsed as
                     vegetation and merged with any vegetation found in
                     ``citygml_path``.
        include_bridges: If True (default), CityGML bridge features are
                     voxelized along with buildings.  Set to False to
                     exclude bridges from the voxel model entirely.
        use_parse_cache: If True (default), cache parsed CityGML meshes as
            binary snapshots in ``.voxcitygml_cache`` beside the dataset and
            reuse them on later runs. Set False to always parse the XML.
        rectangle_vertices: Optional explicit target rectangle
                      ``[(lon, lat), ...]`` in VoxCity order
                      [SW, NW, NE, SE].  When given, ``center_lon`` /
                      ``center_lat`` / ``size_meters`` are ignored and the
                      centre is derived from the vertex centroid.
    """
    citygml_path: Union[str, List[str]] = ""
    rectangle_vertices: Optional[List[Tuple[float, float]]] = None
    center_lon: float = 0.0
    center_lat: float = 0.0
    size_meters: float = 500.0
    meshsize: float = 1.0
    land_cover_source: Optional[str] = None
    canopy_height_source: Optional[str] = None
    gee_project: Optional[str] = None
    output_dir: str = "output"
    n_workers: int = 4
    trunk_height_ratio: Optional[float] = None
    static_tree_height: float = 10.0
    gridvis: bool = False
    save_output: bool = True
    buffer_meters: float = 50.0
    use_3d_voxelizer: bool = True
    max_voxel_ram_mb: Optional[float] = None
    occupancy_threshold: float = 0.0
    occupancy_subdivisions: int = 3
    building_lod: Optional[int] = None
    dem_path: Optional[str] = None
    tree_citygml_path: Optional[str] = None
    terrain_underground_depth: float = 0.0
    include_bridges: bool = True
    use_parse_cache: bool = True


# ---------------------------------------------------------------------------
# Rectangle resolution
# ---------------------------------------------------------------------------

def _buffered_vertices(vertices: List[Tuple[float, float]],
                       buffer_meters: float) -> List[Tuple[float, float]]:
    """Expand a rectangle outward by ~buffer_meters on every side.

    Scales each vertex away from the centroid by a factor derived from the
    shortest side, so the result always contains the original rectangle
    (over-buffering the longer side is acceptable: the buffered rectangle is
    only used as a parse filter).
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    sw, nw, ne, se = vertices
    _, _, d_ns = geod.inv(sw[0], sw[1], nw[0], nw[1])
    _, _, d_ew = geod.inv(sw[0], sw[1], se[0], se[1])
    min_side = max(1.0, min(d_ns, d_ew))
    factor = 1.0 + 2.0 * buffer_meters / min_side
    c_lon = sum(v[0] for v in vertices) / 4.0
    c_lat = sum(v[1] for v in vertices) / 4.0
    return [(c_lon + (v[0] - c_lon) * factor,
             c_lat + (v[1] - c_lat) * factor) for v in vertices]


def resolve_rectangles(
    cfg: VoxelizerConfig,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float, float]:
    """Resolve the target and buffered rectangles from a config.

    Returns
    -------
    (rectangle, buffered_rectangle, center_lon, center_lat)
        ``rectangle`` and ``buffered_rectangle`` are ``[(lon, lat), ...]``
        in VoxCity order [SW, NW, NE, SE].
    """
    from .citygml.coordinates import create_rectangle

    if cfg.buffer_meters < 0:
        raise ValueError(
            f"buffer_meters must be >= 0, got {cfg.buffer_meters}")

    if cfg.rectangle_vertices is not None:
        rect = []
        for v in cfg.rectangle_vertices:
            if len(v) < 2:
                raise ValueError(
                    f"each rectangle vertex must have at least 2 "
                    f"components (lon, lat), got {v!r}")
            rect.append(tuple(v[:2]))
        if len(rect) != 4:
            raise ValueError(
                f"rectangle_vertices must have 4 vertices, got {len(rect)}")
        buffered = _buffered_vertices(rect, cfg.buffer_meters)
        center_lon = sum(v[0] for v in rect) / 4.0
        center_lat = sum(v[1] for v in rect) / 4.0
        return rect, buffered, center_lon, center_lat

    if not cfg.size_meters or cfg.size_meters <= 0:
        raise ValueError(
            "rectangle_vertices not set and size_meters is not positive; "
            "provide rectangle_vertices or center_lon/center_lat/size_meters")
    rect = create_rectangle(cfg.center_lon, cfg.center_lat, cfg.size_meters)
    buffered = create_rectangle(
        cfg.center_lon, cfg.center_lat,
        cfg.size_meters + 2 * cfg.buffer_meters)
    return rect, buffered, cfg.center_lon, cfg.center_lat
