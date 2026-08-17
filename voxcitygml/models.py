"""
Data models for the VoxCityGML pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, NamedTuple
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

#: Surface-contact occupancy threshold for the inclusive-mode surface
#: shell.  Zero: the shell rasterizer already tests PENETRATION of the
#: voxel interior (see the shrunk SAT box in ``_overlay_surface_shell``),
#: so every cell it reports genuinely contains mesh volume and no further
#: filtering is wanted.  Raising this above ~0.33 starts dropping
#: single-face thin walls -- the comb bug this mode exists to fix.
#: History: a 0.25 "calibrated plateau" was tried on 2026-08-17 and
#: rejected -- it only looked exact on round grid origins; on a real
#: pyproj origin every solid face still leaked one layer.  The fix
#: belonged in the metric, not the threshold.
INCLUSIVE_SHELL_THRESHOLD = 0.0


class ResolvedVoxelParams(NamedTuple):
    """Concrete voxelizer knobs after applying ``voxelization_mode``.

    ``VoxelizerConfig.resolved_voxel_params`` is the only producer; the
    pipeline and the OBJ exporters consume the same instance so the main
    grid and exported voxels always agree (the 2026-08-11 invariant).
    """
    building_shell_threshold: float
    occupancy_threshold: float
    shell_anchor: str


_MODE_PARAMS: Dict[str, ResolvedVoxelParams] = {
    # Every voxel CONTAINING mesh volume becomes solid; thin features
    # survive via the connectivity-flood anchor.  Right semantics for
    # obstruction (sunlight / wind): nothing leaks through a voxel the
    # geometry occupies, and no empty voxel is marked solid.
    "inclusive": ResolvedVoxelParams(
        building_shell_threshold=INCLUSIVE_SHELL_THRESHOLD,
        occupancy_threshold=0.0,
        shell_anchor="connected",
    ),
    # The 2026-08-11 tight envelope: shell kept only at >= 0.5
    # surface-contact occupancy, 1-step adjacency anchor.
    "tight": ResolvedVoxelParams(
        building_shell_threshold=0.5,
        occupancy_threshold=0.0,
        shell_anchor="adjacent",
    ),
}

#: Derived from ``_MODE_PARAMS`` rather than restated: a mode the table
#: cannot resolve must never be accepted by validation.
VOXELIZATION_MODES = tuple(_MODE_PARAMS)


def _invalid_mode_error(mode) -> ValueError:
    """The single message both validation seams report for a bad mode."""
    return ValueError(
        f"voxelization_mode must be one of {VOXELIZATION_MODES}, "
        f"got {mode!r}")


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
        voxelization_mode: "inclusive" (default) or "tight".  Inclusive
            marks every voxel containing any part of a mesh and floods
            thin features from their anchors, so nothing sub-voxel-thin is
            lost for obstruction (sunlight / wind) analyses.  Tight
            restores the 2026-08-11 envelope: building shell kept only at
            >= 0.5 surface-contact occupancy with a 1-step adjacency
            anchor.  The mode resolves to concrete knobs via
            ``resolved_voxel_params()``; explicitly-set threshold fields
            override it.
        occupancy_threshold: Minimum SURFACE-CONTACT occupancy (0.0–1.0) a
            boundary voxel must reach to be kept during surface voxelization
            — the fraction of a voxel's subdivided sub-cells that touch
            mesh geometry, not the fraction of its volume enclosed.
            ``None`` (default) defers to ``voxelization_mode`` (both modes
            resolve it to 0.0: keep every voxel with any geometric
            contact).  Does not govern the building surface shell; see
            ``building_shell_threshold``.
        occupancy_subdivisions: Sub-divisions per axis when estimating
            surface-contact occupancy (default 3 → 27 sub-samples per voxel).
        building_shell_threshold: Minimum SURFACE-CONTACT occupancy for the
            building surface-shell overlay.  ``None`` (default) defers to
            ``voxelization_mode``: 0.0 (``INCLUSIVE_SHELL_THRESHOLD``) in
            inclusive mode, 0.5 in tight mode.  The metric applies to a
            candidate set that is already PENETRATION-tested (the shell
            rasterizer's SAT box tests whether the mesh enters a voxel's
            interior, not whether it merely touches the boundary), so this
            is NOT volume overlap on top of that: a lone flat face crossing
            a voxel scores ~0.33 and is dropped at 0.5; two crossing faces
            or a slab spanning two sub-slabs score >= 0.5 and are kept.
            The interior fill independently keeps every centre-inside cell,
            so the shell only decides thin-feature and edge cells.
            Inclusive mode uses 0.0 because the penetration-tested
            candidate set already answers the volume question -- no further
            occupancy filtering is wanted; see ``INCLUSIVE_SHELL_THRESHOLD``.
        building_lod: Preferred CityGML building LOD (1–4) to voxelize.
                      If ``None``, the highest available LOD for each
                      building is selected automatically.
        dem_path: Optional path to a GeoTIFF DEM/DTM file.  Used as terrain
                  source when the CityGML dataset does not include TINRelief
                  geometry (common for German / European datasets).
        dem_source: DEM source selection.  ``None`` or ``"CityGML Terrain"``
                  keeps the existing behaviour (TINRelief terrain, with the
                  ``dem_path`` GeoTIFF fallback).  ``"Flat"`` uses an
                  all-zeros DEM and re-seats buildings on it.  Any other
                  name (e.g. ``"FABDEM"``, ``"GSI DEM Japan"``) is fetched
                  via ``voxcity.generator.grids.get_dem_grid`` and buildings
                  are re-anchored onto it.
        dem_interpolation: Forwarded to ``get_dem_grid`` for named
                  ``dem_source`` values.  Ignored otherwise.
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
    occupancy_threshold: Optional[float] = None
    occupancy_subdivisions: int = 3
    building_shell_threshold: Optional[float] = None
    voxelization_mode: str = "inclusive"
    building_lod: Optional[int] = None
    dem_path: Optional[str] = None
    dem_source: Optional[str] = None
    dem_interpolation: Optional[bool] = None
    tree_citygml_path: Optional[str] = None
    terrain_underground_depth: float = 0.0
    include_bridges: bool = True
    use_parse_cache: bool = True

    def __post_init__(self):
        if self.voxelization_mode not in VOXELIZATION_MODES:
            raise _invalid_mode_error(self.voxelization_mode)
        if self.buffer_meters < 0:
            raise ValueError(
                f"buffer_meters must be >= 0, got {self.buffer_meters}")

    def resolved_voxel_params(self) -> ResolvedVoxelParams:
        """Mode defaults with explicit threshold overrides applied.

        ``None`` in ``building_shell_threshold`` / ``occupancy_threshold``
        means "the mode decides"; an explicit value always wins over the
        mode.  ``shell_anchor`` has no per-field override — it follows the
        mode.
        """
        # Re-validated, not just trusted from __post_init__: this is a
        # plain mutable dataclass, so `cfg.voxelization_mode = "loose"`
        # after construction reaches here unchecked.  This method is on
        # the hot path at three production call sites; a bare
        # KeyError('loose') mid-pipeline is a much worse diagnostic than
        # the construction-time message.
        base = _MODE_PARAMS.get(self.voxelization_mode)
        if base is None:
            raise _invalid_mode_error(self.voxelization_mode)
        overrides = {
            key: value
            for key, value in (
                ("building_shell_threshold", self.building_shell_threshold),
                ("occupancy_threshold", self.occupancy_threshold),
            )
            if value is not None
        }
        return base._replace(**overrides)


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

    # Primary validation is now in VoxelizerConfig.__post_init__, which
    # fails at the user's construction site instead of here, deep in the
    # pipeline.  This copy stays as the mutation guard, for the same
    # reason resolved_voxel_params() re-checks its mode: the dataclass is
    # mutable, so `cfg.buffer_meters = -1` after construction reaches here
    # unvalidated and would otherwise shrink the buffered rectangle
    # INSIDE the target one.
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
