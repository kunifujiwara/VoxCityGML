"""Per-file parse cache: binary snapshots of extracted CityGML meshes.

Parsing PLATEAU CityGML XML dominates LOD2 generation time (~57% of a warm
request). Extraction is rectangle-independent (the request rectangle is a
post-parse filter), so the unfiltered, already-reprojected meshes of each GML
file can be snapshotted once and reloaded on every later request.

Layout: ``<dataset>/.voxcitygml_cache/<relative-path>.{ftype}[.lodK].npz``
where the dataset root is the directory containing ``udx`` (PLATEAU) or the
GML file's own directory (flat layouts). ``building`` files are keyed by the
requested LOD (``auto`` when None -- auto-selection may pick different LODs per
building, so it must not share a key with an explicit LOD).

Validity: recorded source size + mtime_ns must match the GML file and the
format version must match ``CACHE_VERSION``. Writes are atomic
(temp file + ``os.replace``), so parallel workers never observe a torn file.

The cache is strictly an accelerator: every failure in load or store logs one
warning and the caller falls back to normal parsing.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..models import Mesh3D

log = logging.getLogger(__name__)

CACHE_VERSION = 1
CACHE_DIR_NAME = ".voxcitygml_cache"


def _dataset_root_for(gml_file: Path) -> Path:
    """Dataset root: the directory containing ``udx``, else the file's dir."""
    resolved = gml_file.resolve()
    parts = resolved.parts
    if "udx" in parts:
        # Nearest enclosing ``udx`` -- a path may contain several (e.g. a
        # parent folder that happens to be named ``udx``), and the dataset
        # root is the innermost one.
        idx = len(parts) - 1 - parts[::-1].index("udx")
        return Path(*parts[:idx])
    return resolved.parent


def _cache_path(gml_file: Path, feature_type: str,
                building_lod: Optional[int]) -> Path:
    """Cache file path for one (GML file, feature type, LOD) combination."""
    gml_file = Path(gml_file)
    root = _dataset_root_for(gml_file)
    rel = gml_file.resolve().relative_to(root)
    suffix = f".{feature_type}"
    if feature_type == "building":
        suffix += f".lod{building_lod if building_lod is not None else 'auto'}"
    return root / CACHE_DIR_NAME / rel.parent / (rel.name + suffix + ".npz")


def load_cached_meshes(gml_file: Path, feature_type: str,
                       building_lod: Optional[int]) -> Optional[List[Mesh3D]]:
    """Return cached meshes for *gml_file*, or None on miss/stale/error."""
    try:
        cache_file = _cache_path(gml_file, feature_type, building_lod)
        if not cache_file.exists():
            return None
        st = os.stat(gml_file)
        with np.load(cache_file, allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
            if meta.get("version") != CACHE_VERSION:
                return None
            if (meta.get("src_size") != st.st_size
                    or meta.get("src_mtime_ns") != st.st_mtime_ns):
                return None
            meshes: List[Mesh3D] = []
            for i, mm in enumerate(meta["meshes"]):
                attributes = dict(mm["attrs"])
                for name in mm["array_attrs"]:
                    attributes[name] = data[f"a{i}_{name}"]
                meshes.append(Mesh3D(
                    vertices=data[f"v{i}"],
                    faces=data[f"f{i}"],
                    normals=data[f"n{i}"] if mm["has_normals"] else None,
                    colors=data[f"c{i}"] if mm["has_colors"] else None,
                    feature_type=mm["feature_type"],
                    feature_id=mm["feature_id"],
                    attributes=attributes,
                ))
        return meshes
    except Exception as exc:
        log.warning("Parse cache load failed for %s (re-parsing): %s",
                    gml_file, exc)
        return None


def store_cached_meshes(gml_file: Path, feature_type: str,
                        building_lod: Optional[int],
                        meshes: List[Mesh3D]) -> None:
    """Snapshot *meshes* (unfiltered, reprojected). Never raises."""
    tmp_name = None
    try:
        cache_file = _cache_path(gml_file, feature_type, building_lod)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        st = os.stat(gml_file)

        # Array keys are ``v{i}``/``f{i}``/``n{i}``/``c{i}``/``a{i}_{name}``.
        # The mesh index is a pure digit run and the separator is not a digit,
        # so ``a{i}_{name}`` is injective: no attribute name can make two
        # different (index, name) pairs collide.
        arrays = {}
        meta_meshes = []
        for i, m in enumerate(meshes):
            arrays[f"v{i}"] = m.vertices
            arrays[f"f{i}"] = m.faces
            if m.normals is not None:
                arrays[f"n{i}"] = m.normals
            if m.colors is not None:
                arrays[f"c{i}"] = m.colors
            attrs, array_attrs = {}, []
            for name, value in m.attributes.items():
                if isinstance(value, np.ndarray):
                    arrays[f"a{i}_{name}"] = value
                    array_attrs.append(name)
                else:
                    attrs[name] = value
            meta_meshes.append({
                "feature_type": m.feature_type,
                "feature_id": m.feature_id,
                "attrs": attrs,
                "array_attrs": array_attrs,
                "has_normals": m.normals is not None,
                "has_colors": m.colors is not None,
            })
        meta = {
            "version": CACHE_VERSION,
            "src_size": st.st_size,
            "src_mtime_ns": st.st_mtime_ns,
            "meshes": meta_meshes,
        }
        # default=str stringifies anything json can't encode (spec behavior)
        arrays["meta"] = np.array(json.dumps(meta, default=str))

        fd, tmp_name = tempfile.mkstemp(
            dir=str(cache_file.parent), suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_name, cache_file)
        tmp_name = None
    except Exception as exc:
        log.warning("Parse cache store failed for %s (continuing): %s",
                    gml_file, exc)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
