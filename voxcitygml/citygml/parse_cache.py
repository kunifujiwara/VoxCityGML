"""Per-file parse cache: binary snapshots of extracted CityGML meshes.

Parsing PLATEAU CityGML XML dominates LOD2 generation time (~57% of a warm
request). Extraction is rectangle-independent (the request rectangle is a
post-parse filter), so the unfiltered, already-reprojected meshes of each GML
file can be snapshotted once and reloaded on every later request.

Layout: ``<dataset>/.voxcitygml_cache/<relative-path>.{ftype}[.lodK].npz``
where the dataset root is the directory containing ``udx`` (PLATEAU) or the
GML file's own directory (flat layouts). ``building`` files are keyed by the
requested LOD (``auto`` when None -- auto-selection may pick different LODs per
building, so it must not share a key with an explicit LOD). Note that each LOD
key stores a **full independent snapshot**: a dataset queried at LOD1, LOD2 and
``auto`` holds three complete copies of its building meshes, so disk use for
the largest feature type can reach 3x.

Validity: recorded source size + mtime_ns must match the GML file and the
format version must match ``CACHE_VERSION``. Bump ``CACHE_VERSION`` whenever
extraction output changes (see ``extractors.py``) -- it is the only thing that
invalidates caches already sitting in users' dataset directories. Writes are
atomic (temp file + ``os.replace``), so parallel workers never observe a torn
file.

Mesh attributes are stored as strict JSON (ndarray values are split out into
the npz alongside the geometry). Values must therefore be JSON-round-trip
identical -- ``str``/``int``/``float``/``bool``/``None``/``list``/``dict`` with
string keys. Anything else (numpy scalars such as ``np.int64``/``np.bool_``,
tuples, non-string dict keys) either fails to serialize or comes back as a
different type, so a cache hit would disagree with a cache miss; when strict
serialization fails the file is simply not cached.

The cache is strictly an accelerator: every failure in load or store logs a
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

# A read-only or network-mounted dataset makes every store fail. Warning once
# per (file x feature type) would emit hundreds of identical lines per request,
# so failures are counted and writes latch off. State is per-process, so each
# multiprocessing worker bounds its own noise.
_MAX_STORE_FAILURE_WARNINGS = 3
_store_failures = 0
_stores_disabled = False


def reset_store_failures() -> None:
    """Re-enable cache writes after the failure latch tripped (for tests)."""
    global _store_failures, _stores_disabled
    _store_failures = 0
    _stores_disabled = False


def _note_store_failure(gml_file, cache_file, reason: str) -> None:
    """Warn about a store that did not happen; latch writes off after a few."""
    global _store_failures, _stores_disabled
    _store_failures += 1
    if _store_failures <= _MAX_STORE_FAILURE_WARNINGS:
        log.warning("Parse cache store failed for %s (cache file %s); parsing "
                    "continues uncached: %s", gml_file, cache_file, reason)
    if _store_failures >= _MAX_STORE_FAILURE_WARNINGS and not _stores_disabled:
        _stores_disabled = True
        log.warning("Parse cache: %d store failures; disabling cache writes for "
                    "the rest of this process (is the dataset directory "
                    "read-only?). Parsing is unaffected.", _store_failures)


def _dataset_root_for(resolved: Path) -> Path:
    """Dataset root for an already-resolved GML path.

    The directory containing ``udx`` (PLATEAU), else the file's own directory.
    """
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
    resolved = Path(gml_file).resolve()
    root = _dataset_root_for(resolved)
    rel = resolved.relative_to(root)
    suffix = f".{feature_type}"
    if feature_type == "building":
        suffix += f".lod{building_lod if building_lod is not None else 'auto'}"
    return root / CACHE_DIR_NAME / rel.parent / (rel.name + suffix + ".npz")


def load_cached_meshes(gml_file: Path, feature_type: str,
                       building_lod: Optional[int]) -> Optional[List[Mesh3D]]:
    """Return cached meshes for *gml_file*, or None on miss/stale/error."""
    cache_file = None
    try:
        cache_file = _cache_path(gml_file, feature_type, building_lod)
        if not cache_file.exists():
            return None
        st = os.stat(gml_file)
        with np.load(cache_file, allow_pickle=False) as data:
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))
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
        log.warning("Parse cache unusable for %s (cache file %s); falling back "
                    "to normal parsing -- delete the %s directory if this "
                    "persists: %s",
                    gml_file, cache_file, CACHE_DIR_NAME, exc)
        return None


def store_cached_meshes(gml_file: Path, feature_type: str,
                        building_lod: Optional[int],
                        meshes: List[Mesh3D]) -> None:
    """Snapshot *meshes* (unfiltered, reprojected). Never raises."""
    if _stores_disabled:
        return
    tmp_name = None
    cache_file = None
    try:
        cache_file = _cache_path(gml_file, feature_type, building_lod)
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
        # Strict: a lenient encoder (``default=str``) would let a cache hit
        # differ from a cache miss -- np.bool_(False) would come back as the
        # truthy string 'False'. Skipping keeps hit == miss by construction.
        try:
            meta_json = json.dumps(meta)
        except TypeError as exc:
            _note_store_failure(
                gml_file, cache_file,
                f"attribute is not JSON-serializable ({exc}); extend the "
                f"serializer if this type is now expected")
            return
        # utf-8 bytes, not a unicode array: np.array(str) is UTF-32, 4x larger
        # to write and read back. json.dumps escapes non-ASCII by default and
        # the utf-8 round-trip is exact either way.
        arrays["meta"] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(cache_file.parent), suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_name, cache_file)
        tmp_name = None
    except Exception as exc:
        _note_store_failure(gml_file, cache_file, str(exc))
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
