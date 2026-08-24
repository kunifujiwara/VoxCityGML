# Building–Terrain Contact Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate unintentional air gaps between PLATEAU LOD2 building bottoms and the voxelized terrain, while never filling intentional pilotis voids.

**Architecture:** Three layers, per `docs/superpowers/specs/2026-08-24-building-terrain-contact-design.md`: (D1) scope the terrain solid's −0.5·voxel pre-shift to the levelset call only and grid-align the winding fallback; (D2) replace the two DEM fill helpers with one `_fill_air_to_dem_surface` that conforms terrain to the *surface voxel* (`ceil(t)−1`, penetration convention, raise-only); (D3) a mesh-accurate, tolerance-bounded ground-contact closure `close_building_ground_gaps` driven by `building_min_height_grid` (pilotis have bottom heights ≥ ~2.2 m and are excluded by the metres-based tolerance regardless of voxel size).

**Tech Stack:** numpy, scipy.ndimage, meshlib (optional), numba, pytest. Run tests with the project env: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest ...` from the repo root.

**Background you need:** `voxcitygml/voxelizer3d.py` builds one int16 grid `(n_rows, n_cols, n_z)`, north-up, `GROUND_CODE=-1`, `BUILDING_CODE=-3`, air `0`. Terrain is voxelized first (`_voxelize_terrain_solid`: levelset when watertight, else winding, else Numba scanline), then DEM fills, then buildings (`_voxelize_building_solid`: grid-aligned winding fill + penetration shell). `t = (z − gp.min_z)/voxel_size` is a z coordinate in voxel units; the voxel *containing* elevation `z` under the half-open-upward convention has index `ceil(t)−1` (equals `floor(t)` for fractional `t`, and `t−1` when `z` lies exactly on a lattice plane).

---

### Task 1: Failing tests — terrain surface exactness and building contact

**Files:**
- Create: `tests/test_terrain_building_contact.py`

- [ ] **Step 1: Write the test file**

```python
"""Terrain surface placement and building ground contact.

Design: docs/superpowers/specs/2026-08-24-building-terrain-contact-design.md

A flat terrain TIN at elevation z_t must produce a terrain whose topmost
GROUND voxel is exactly the *surface voxel* ceil(t)-1 (t = (z_t-min_z)/vs)
at every fractional phase, on all three terrain paths (levelset, winding
fallback, Numba scanline fallback).  A box building whose base lies exactly
on that terrain must then touch it (zero air voxels below its lowest
building voxel in every footprint column).
"""
import numpy as np
import pytest
import trimesh

from voxcitygml import voxelizer3d as v3
from voxcitygml.models import Mesh3D

VS = 1.0
NXY = 40.0
PHASES = [0.0, 0.1, 0.25, 0.5, 0.6, 0.7, 0.9]


class IdentityTransformer:
    def transform(self, a, b):
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def make_grid(z_t):
    # min_z is an integer, so frac((z_t - min_z)/VS) equals frac(z_t).
    z_min = float(np.floor(z_t)) - 5.0
    z_max = float(np.floor(z_t)) + 14.0
    n = int(round(NXY / VS))
    nz = int(round((z_max - z_min) / VS))
    gp = v3.Grid3DParams(
        n_rows=n, n_cols=n, n_z=nz,
        min_x=0.0, max_x=NXY, min_y=0.0, max_y=NXY,
        min_z=z_min, max_z=z_max, voxel_size=VS,
    )
    return gp, np.zeros((n, n, nz), dtype=np.int16)


def flat_terrain_mesh(z_t):
    # Mesh3D vertices are (lat, lon, z); swap_coordinates_3d converts to
    # (lon, lat, z) and IdentityTransformer maps lon->x, lat->y.
    verts = np.array([
        [0.0, 0.0, z_t], [0.0, NXY, z_t],
        [NXY, NXY, z_t], [NXY, 0.0, z_t],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh3D(vertices=verts, faces=faces,
                  feature_type="terrain", feature_id="t")


def box_building(z_base, x0=15.0, y0=15.0, w=8.0, h=9.0):
    b = trimesh.creation.box(extents=[w, w, h])
    b.apply_translation([x0 + w / 2, y0 + w / 2, z_base + h / 2])
    return np.asarray(b.vertices, float), np.asarray(b.faces)


def voxelize_terrain(gp, grid, z_t, path, monkeypatch):
    """Run the requested terrain path, then the DEM surface conform."""
    tmesh = flat_terrain_mesh(z_t)
    if path == "scanline":
        monkeypatch.setattr(v3, "_MESHLIB_VOXEL_AVAILABLE", False)
    elif path == "winding":
        real = v3.build_terrain_solid

        def not_watertight(*a, **k):
            solid, stats = real(*a, **k)
            if stats is not None:
                stats.is_watertight = False
            return solid, stats
        monkeypatch.setattr(v3, "build_terrain_solid", not_watertight)
    ok = v3._voxelize_terrain_solid([tmesh], IdentityTransformer(), gp, grid)
    assert ok
    dem = np.full((gp.n_rows, gp.n_cols), z_t, dtype=np.float64)
    v3._fill_air_to_dem_surface(grid, gp, dem)


def surface_voxel_index(gp, z_t):
    t = (z_t - gp.min_z) / gp.voxel_size
    return int(np.ceil(np.round(t, 9))) - 1


def max_air_gap_below_buildings(grid):
    is_b = grid == v3.BUILDING_CODE
    gaps = []
    for r, c in zip(*np.nonzero(is_b.any(axis=2))):
        bm = int(np.argmax(is_b[r, c]))
        below = np.nonzero(grid[r, c, :bm] != 0)[0]
        gaps.append(bm - (int(below[-1]) if len(below) else -1) - 1)
    assert gaps, "building produced no voxels"
    return max(gaps)


needs_meshlib = pytest.mark.skipif(
    not v3._MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")

TERRAIN_PATHS = [
    pytest.param("levelset", marks=needs_meshlib),
    pytest.param("winding", marks=needs_meshlib),
    "scanline",
]


@pytest.mark.parametrize("path", TERRAIN_PATHS)
@pytest.mark.parametrize("phase", PHASES)
def test_terrain_top_is_surface_voxel(path, phase, monkeypatch):
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, path, monkeypatch)
    is_g = grid == v3.GROUND_CODE
    tops = gp.n_z - 1 - np.argmax(np.flip(is_g, axis=2), axis=2)
    interior = tops[5:-5, 5:-5]  # avoid boundary-column artifacts
    expected = surface_voxel_index(gp, z_t)
    # The conform is raise-only, so no path may sit LOW.  The levelset
    # stamp's accepted corner-sampling overfill may sit one voxel HIGH at
    # some phases (2026-08-24 design, "Consequences accepted"); the
    # centre-sampled winding/scanline paths must be exact.
    high = expected + 1 if path == "levelset" else expected
    assert interior.min() >= expected and interior.max() <= high, (
        f"phase {phase}: terrain top {interior.min()}..{interior.max()}, "
        f"expected [{expected}, {high}]")


@pytest.mark.parametrize("path", TERRAIN_PATHS)
@pytest.mark.parametrize("phase", PHASES)
def test_building_on_terrain_touches(path, phase, monkeypatch):
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, path, monkeypatch)
    bverts, bfaces = box_building(z_base=z_t)
    v3._voxelize_building_solid(bverts, bfaces, gp, grid,
                                v3.BUILDING_CODE, True)
    assert max_air_gap_below_buildings(grid) == 0, f"phase {phase}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_terrain_building_contact.py -x -q`
Expected: FAIL/ERROR with `AttributeError: ... has no attribute '_fill_air_to_dem_surface'` (the function does not exist yet). This is the correct failure.

- [ ] **Step 3: Commit**

```bash
git add tests/test_terrain_building_contact.py
git commit -m "test: terrain surface-voxel exactness and building contact (failing)"
```

---

### Task 2: D2 — `_fill_air_to_dem_surface` replaces the two DEM fill helpers

**Files:**
- Modify: `voxcitygml/voxelizer3d.py:269-284` (call site), `voxcitygml/voxelizer3d.py:550-589` (helpers)
- Modify: `tests/test_optimized.py:6-45`, `tests/test_profile.py`, `tests/test_profile_detail.py` (import/caller updates)

- [ ] **Step 1: Replace `_fill_terrain_from_dem` and `_fill_terrain_gaps_from_dem`**

Delete both functions (voxelizer3d.py:550-589) and add in their place:

```python
def _fill_air_to_dem_surface(
    voxel_grid: np.ndarray, gp: Grid3DParams, dem_grid: np.ndarray,
) -> None:
    """Fill AIR cells up to the DEM *surface voxel* with GROUND_CODE.

    The surface voxel of elevation ``z`` is the voxel CONTAINING ``z``
    under the half-open-upward convention: index ``ceil(t) - 1`` with
    ``t = (z - min_z)/vs``.  An elevation exactly on a lattice plane
    belongs to the voxel below (its top face IS the surface), matching
    ``_penetration_half`` semantics — so a building base coincident with
    the terrain surface lands in, or directly above, the terrain's top
    voxel at every fractional phase (2026-08-24 contact design).

    Raise-only and air-only: cells already claimed by the terrain solid
    (or anything else) are never lowered or overwritten.  Runs after the
    terrain solid, so it simultaneously (a) builds the whole ground when
    no terrain meshes exist, (b) fills river / failed-union gap columns,
    and (c) tops up columns the solid voxelization left below the
    surface — replacing the pre-2026-08-24 pair of rint-based helpers
    (_fill_terrain_from_dem / _fill_terrain_gaps_from_dem), whose
    round-half-up level sat up to half a voxel off the surface either
    way with grid phase.
    """
    t = (np.asarray(dem_grid, dtype=np.float64) - gp.min_z) / gp.voxel_size
    surface = (np.ceil(np.round(t, 9)) - 1).astype(np.intp)
    surface = np.clip(surface, -1, gp.n_z - 1)  # -1: DEM below grid, no fill
    z_indices = np.arange(gp.n_z, dtype=np.intp)
    fill = (z_indices[np.newaxis, np.newaxis, :] <= surface[:, :, np.newaxis])
    fill &= (voxel_grid == 0)
    voxel_grid[fill] = GROUND_CODE
```

- [ ] **Step 2: Rewrite the call site in `voxelize_citygml_meshes`**

Replace the current block (voxelizer3d.py:269-284):

```python
    terrain_filled = False
    if collection.terrain:
        terrain_filled = _voxelize_terrain_solid(
            collection.terrain,
            transformer,
            gp,
            voxel_grid,
        )

    if not terrain_filled and dem_grid is not None:
        _fill_terrain_from_dem(voxel_grid, gp, dem_grid)
    elif terrain_filled and dem_grid is not None:
        # The terrain solid may leave gaps (rivers, missing tiles, failed
        # Boolean union).  Fill columns that are still empty using the DEM
        # as a safety-net — only writes to cells that are currently AIR (0).
        _fill_terrain_gaps_from_dem(voxel_grid, gp, dem_grid)
```

with:

```python
    if collection.terrain:
        _voxelize_terrain_solid(
            collection.terrain,
            transformer,
            gp,
            voxel_grid,
        )

    if dem_grid is not None:
        # Conform the ground to the DEM surface voxel: builds the whole
        # ground when there is no terrain solid, fills river / failed-union
        # gap columns, and raises columns the solid left below the surface.
        # Raise-only and air-only — never carves the solid down.
        _fill_air_to_dem_surface(voxel_grid, gp, dem_grid)
```

- [ ] **Step 3: Update the old helpers' other callers**

In `tests/test_optimized.py`: change the import (line 8) from `_fill_terrain_from_dem` to `_fill_air_to_dem_surface`, and in `test_fill_terrain` (lines 32-45) change the call and the expectation — dem 20.0 at vs 2.0 and min_z 0.0 lies exactly on a lattice plane, so the surface voxel is index 9, not rint's 10:

```python
    _fill_air_to_dem_surface(grid, gp, dem)
    ...
    expected_level = 9  # 20 m on-lattice: surface voxel is ceil(10)-1
```

Also fix the other two calls in that file (lines 78 and 197) to the new name. In `tests/test_profile.py` (lines 11, 47) and `tests/test_profile_detail.py` (line 13) rename import and call the same way; those scripts only time the call, no expectation change.

- [ ] **Step 4: Run the Task 1 tests**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_terrain_building_contact.py -q`
Expected: `test_terrain_top_is_surface_voxel` PASSES for all paths/phases (the conform raises every path to the surface voxel; it cannot lower the winding fallback's misplaced solid, but a flat plane's fallback error is always LOW, which the raise corrects). `test_building_on_terrain_touches` PASSES. If any `winding` case still fails, continue to Task 3 — D1 removes the fallback's downward displacement — then re-run.

- [ ] **Step 5: Run the optimized/profile suites**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_optimized.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add voxcitygml/voxelizer3d.py tests/test_optimized.py tests/test_profile.py tests/test_profile_detail.py
git commit -m "fix: conform terrain to the DEM surface voxel (penetration convention, raise-only)"
```

---

### Task 3: D1 — scope the −0.5·voxel pre-shift to the levelset call; align the fallback

**Files:**
- Modify: `voxcitygml/voxelizer3d.py:513-547` (`_voxelize_terrain_solid` tail)

- [ ] **Step 1: Rewrite the path dispatch**

Replace (voxelizer3d.py:513-547):

```python
    verts = solid.vertices.copy()
    faces = solid.faces

    # Shift the solid down by half a voxel to compensate for the MeshLib
    # level-set SDF centre-sampling bias: ...
    verts[:, 2] -= 0.5 * gp.voxel_size

    # ── MeshLib path ──────────────────────────────────────────────────
    if _MESHLIB_VOXEL_AVAILABLE:
        if stats.is_watertight:
            ok = _voxelize_meshlib_levelset(
                verts, faces, gp, voxel_grid, GROUND_CODE, overwrite=False,
            )
            ...
        # Fallback: winding number (works even if not perfectly watertight)
        ok = _voxelize_meshlib_winding(
            verts, faces, gp, voxel_grid, GROUND_CODE, overwrite=False,
        )
        ...
    # ── Legacy Numba path ─────────────────────────────────────────────
    _voxelize_single_mesh(
        verts, faces, gp, voxel_grid, ...
```

with:

```python
    verts = solid.vertices
    faces = solid.faces

    # ── MeshLib path ──────────────────────────────────────────────────
    if _MESHLIB_VOXEL_AVAILABLE:
        if stats.is_watertight:
            # The levelset stamp is corner-sampled but stamped as centre
            # samples (+0.5 voxel displacement, 2026-08-11 diagnosis); the
            # -0.5-voxel pre-shift compensates THAT PATH ONLY.  The winding
            # and scanline paths below are centre-sampled and must receive
            # the raw solid — feeding them the shifted vertices sank the
            # terrain a full voxel at ~40% of grid phases and left air
            # gaps under buildings (2026-08-24 contact design).
            shifted = verts.copy()
            shifted[:, 2] -= 0.5 * gp.voxel_size
            ok = _voxelize_meshlib_levelset(
                shifted, faces, gp, voxel_grid, GROUND_CODE, overwrite=False,
            )
            if ok:
                _log.info("Terrain voxelized via MeshLib level-set.")
                return True
        # Fallback: winding number (works even if not perfectly watertight).
        # align_origin=True snaps the SDF lattice to the grid lattice, the
        # same phase-exactness the 2026-08-11 fix gave buildings.
        ok = _voxelize_meshlib_winding(
            verts, faces, gp, voxel_grid, GROUND_CODE, overwrite=False,
            align_origin=True,
        )
        if ok:
            _log.info("Terrain voxelized via MeshLib winding number.")
            return True

    # ── Legacy Numba path ─────────────────────────────────────────────
    _voxelize_single_mesh(
        verts, faces, gp, voxel_grid,
        class_code=GROUND_CODE,
        overwrite=False,
        seal_surface=False,  # solid is already closed, no dilation needed
    )
    _log.info("Terrain voxelized via Numba Z-scanline.")
    return True
```

(Keep the `if ok:` bodies and log lines exactly as shown — the only changes are: shift applied to a copy inside the watertight branch, `align_origin=True` on the fallback, and raw `verts` everywhere else. `verts` no longer needs the module-level `.copy()` since the levelset branch copies.)

- [ ] **Step 2: Run the contact tests**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_terrain_building_contact.py -q`
Expected: all PASS, including every `winding` and `scanline` phase.

- [ ] **Step 3: Run the full unit suite (offline subset)**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q -m "not slow"`
Expected: PASS. If a test pinned the old rint fill level or fallback placement, update it to the surface-voxel convention (`ceil(t)-1`) with a comment referencing the 2026-08-24 design.

- [ ] **Step 4: Commit**

```bash
git add voxcitygml/voxelizer3d.py
git commit -m "fix: scope terrain -0.5-voxel pre-shift to the levelset path, align fallback winding"
```

---

### Task 4: D3 — pilotis-safe ground-contact closure

**Files:**
- Modify: `voxcitygml/voxelizer3d.py` (new function, after `_fill_air_to_dem_surface`)
- Modify: `voxcitygml/models.py:341` (config field), `voxcitygml/pipeline.py:430` area (wiring)
- Test: `tests/test_terrain_building_contact.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_terrain_building_contact.py`:

```python
def _min_height_grid(gp, cells):
    """Object grid of [h_min, h_max] segment lists; cells maps (r, c) -> h_min."""
    g = np.empty((gp.n_rows, gp.n_cols), dtype=object)
    for i in range(gp.n_rows):
        for j in range(gp.n_cols):
            g[i, j] = []
    for (r, c), h in cells.items():
        g[r, c] = [[h, h + 9.0]]
    return g


@needs_meshlib
def test_small_base_offset_closed(monkeypatch):
    # Base 1.45 m above the terrain: with exact terrain a gap needs the
    # offset to cross a second lattice plane, so at VS=1 an offset in
    # (1.0, 1.5] is the smallest that floats AND must still be closed
    # (1.45 <= tolerance 1.5).  z_t=10.6, z_b=12.05: terrain top voxel 5,
    # building bottom voxel 7 -> a real 1-voxel air gap before closure.
    z_t = 10.6
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, "levelset", monkeypatch)
    bverts, bfaces = box_building(z_base=z_t + 1.45)
    v3._voxelize_building_solid(bverts, bfaces, gp, grid,
                                v3.BUILDING_CODE, True)
    assert max_air_gap_below_buildings(grid) >= 1  # reproduces the float
    is_b = grid == v3.BUILDING_CODE
    cells = {(r, c): 1.45 for r, c in zip(*np.nonzero(is_b.any(axis=2)))}
    v3.close_building_ground_gaps(grid, _min_height_grid(gp, cells), 1.5)
    assert max_air_gap_below_buildings(grid) == 0


@needs_meshlib
def test_pilotis_void_preserved(monkeypatch):
    # Slab 3 m above the terrain (pilotis clearance): must NOT be closed.
    z_t = 10.6
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, "levelset", monkeypatch)
    bverts, bfaces = box_building(z_base=z_t + 3.0, h=6.0)
    v3._voxelize_building_solid(bverts, bfaces, gp, grid,
                                v3.BUILDING_CODE, True)
    before_ground = int((grid == v3.GROUND_CODE).sum())
    is_b = grid == v3.BUILDING_CODE
    cells = {(r, c): 3.0 for r, c in zip(*np.nonzero(is_b.any(axis=2)))}
    v3.close_building_ground_gaps(grid, _min_height_grid(gp, cells), 1.5)
    assert int((grid == v3.GROUND_CODE).sum()) == before_ground
    assert max_air_gap_below_buildings(grid) >= 2  # the void survives


@needs_meshlib
def test_closure_fringe_column_uses_nearest_height(monkeypatch):
    # A column with building voxels but NO min-height segments (the
    # centre-inside rasterization fringe) inherits the nearest claimed
    # column's h_min instead of being skipped.
    z_t = 10.6
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, "levelset", monkeypatch)
    bverts, bfaces = box_building(z_base=z_t + 1.45)
    v3._voxelize_building_solid(bverts, bfaces, gp, grid,
                                v3.BUILDING_CODE, True)
    is_b = grid == v3.BUILDING_CODE
    cols = list(zip(*np.nonzero(is_b.any(axis=2))))
    cells = {(r, c): 1.45 for r, c in cols[:-1]}  # drop one column's segments
    v3.close_building_ground_gaps(grid, _min_height_grid(gp, cells), 1.5)
    assert max_air_gap_below_buildings(grid) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_terrain_building_contact.py -q -k "closed or pilotis or fringe"`
Expected: FAIL with `AttributeError: ... 'close_building_ground_gaps'`.

- [ ] **Step 3: Implement the closure in voxelizer3d.py**

Add directly after `_fill_air_to_dem_surface`:

```python
def close_building_ground_gaps(
    voxel_grid: np.ndarray,
    building_min_height_grid: np.ndarray,
    tolerance_m: float,
) -> int:
    """Close sub-tolerance air gaps between building bottoms and the ground.

    PLATEAU LOD2 building bases sit up to ~1 m above the local terrain in
    the source data; voxelized at absolute elevation this becomes a visible
    air layer under the building.  For every column containing BUILDING
    voxels, the building's mesh-accurate bottom height above ground is
    taken from *building_min_height_grid* (the object grid of
    ``[h_min, h_max]`` segments built by ``meshes_to_building_grids``;
    ``h_min`` is metres above the DEM).  Columns whose lowest segment
    starts at ``h_min <= tolerance_m`` get any air between their lowest
    building voxel and the support below filled with GROUND_CODE.

    Pilotis are preserved by construction: an intentional open ground
    floor puts the column's lowest building geometry (the slab underside)
    at ``h_min`` >= ~2.2 m, above any sane tolerance — the discrimination
    is in metres of real geometry, so it holds at every voxel size,
    where a voxel-count rule could not tell a 2.5 m pilotis from a 0.3 m
    misalignment at 2 m voxels (2026-08-24 contact design).

    Columns the voxelizer claimed but the centre-inside rasterization did
    not (the ``fill_building_id_gaps`` fringe, up to one cell wide) have
    no segments; they inherit the nearest claimed column's ``h_min`` via
    the same distance-transform device that repairs the id grid.

    Returns the number of columns closed.  ``tolerance_m <= 0`` disables
    closure.  Raises ValueError on a frame mismatch, mirroring
    ``fill_building_id_gaps`` — these grids come from separately-computed
    frames, and a silent mismatch would misplace every fill.
    """
    if tolerance_m <= 0:
        return 0
    if building_min_height_grid.shape != voxel_grid.shape[:2]:
        raise ValueError(
            "building_min_height_grid and voxel_grid must share their first "
            f"two axes; got {building_min_height_grid.shape} and "
            f"{voxel_grid.shape[:2]}.")

    is_building = voxel_grid == BUILDING_CODE
    has_building = is_building.any(axis=2)
    if not has_building.any():
        return 0

    h_min = np.full(voxel_grid.shape[:2], np.nan)
    for r, c in zip(*np.nonzero(has_building)):
        segments = building_min_height_grid[r, c]
        if segments:
            h_min[r, c] = min(seg[0] for seg in segments)

    known = ~np.isnan(h_min)
    if not known.any():
        return 0
    from scipy.ndimage import distance_transform_edt
    _dist, idx = distance_transform_edt(~known, return_indices=True)
    h_min = h_min[idx[0], idx[1]]

    lowest_building = np.argmax(is_building, axis=2)
    n_closed = 0
    for r, c in zip(*np.nonzero(has_building & (h_min <= tolerance_m))):
        bottom = lowest_building[r, c]
        support = np.nonzero(voxel_grid[r, c, :bottom] != 0)[0]
        top = int(support[-1]) if len(support) else -1
        if top < bottom - 1:
            voxel_grid[r, c, top + 1:bottom] = GROUND_CODE
            n_closed += 1
    if n_closed:
        _log.info("  Ground-contact closure: closed %d building columns "
                  "(tolerance %.2f m)", n_closed, tolerance_m)
    return n_closed
```

- [ ] **Step 4: Run the closure tests**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_terrain_building_contact.py -q`
Expected: all PASS.

- [ ] **Step 5: Add the config field**

In `voxcitygml/models.py`, after `terrain_underground_depth: float = 0.0` (line 341):

```python
    # Maximum building-bottom height above ground (m) that still counts as
    # unintentional base/terrain misalignment: columns whose lowest building
    # geometry starts at or below this height get the air under them filled
    # with ground (voxelizer3d.close_building_ground_gaps).  Pilotis and
    # other intentional voids sit higher (>= ~2.2 m) and are never filled.
    # 0 disables closure.  3-D voxelizer path only.
    ground_contact_tolerance: float = 1.5
```

- [ ] **Step 6: Wire into run_core**

In `voxcitygml/pipeline.py`, import `close_building_ground_gaps` alongside `BUILDING_CODE` (line 51):

```python
from .voxelizer3d import (
    BUILDING_CODE, close_building_ground_gaps, voxelize_citygml_meshes,
)
```

and in `run_core`, immediately after the `voxelize_citygml_meshes(...)` call returns (before the `voxel_min_z = float(...)` line):

```python
        # Close sub-tolerance air gaps between building bottoms and the
        # ground (PLATEAU base elevations sit up to ~1 m above the local
        # TIN).  Mesh-accurate: driven by building_min_height_grid, so
        # pilotis (bottom >= ~2.2 m above ground) are never filled.
        close_building_ground_gaps(
            voxel_grid, building_min_height_grid,
            cfg.ground_contact_tolerance,
        )
```

(`building_min_height_grid` and `voxel_grid` are both north-up here and share the 2-D frame; the ValueError guard catches drift.)

- [ ] **Step 7: Run the offline suite**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q -m "not slow"`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add voxcitygml/voxelizer3d.py voxcitygml/models.py voxcitygml/pipeline.py tests/test_terrain_building_contact.py
git commit -m "feat: pilotis-safe ground-contact closure with ground_contact_tolerance config"
```

---

### Task 5: CLI flag and stale-doc cleanup

**Files:**
- Modify: `voxcitygml/cli.py` (argument + config pass-through)
- Modify: `voxcitygml/voxelizer3d.py:633-649` (levelset warning note)
- Modify: `docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md` (terrain-note addendum)

- [ ] **Step 1: CLI argument**

In `voxcitygml/cli.py`, after the `--voxelization-mode` argument (line 82 area):

```python
    parser.add_argument('--ground-contact-tolerance', type=float, default=1.5,
                        help='Max building-bottom height above ground (m) '
                             'closed as base/terrain misalignment; pilotis '
                             'sit higher and are preserved. 0 disables.')
```

and pass it where the `VoxelizerConfig` is constructed in the same file:

```python
        ground_contact_tolerance=args.ground_contact_tolerance,
```

- [ ] **Step 2: Fix the stale levelset warning**

In `_voxelize_meshlib_levelset`'s docstring (voxelizer3d.py:633-649), replace the sentence "Note the terrain path pre-shifts its solid by -0.5 voxel (see ``_voxelize_terrain_solid`` around voxelizer3d.py:446-450) to compensate for this same bias — a future fix of the stamp convention must remove that compensation in the same change, or terrain will double-correct." with:

```
   Note the terrain path pre-shifts a COPY of its solid by -0.5 voxel
   inside its watertight/levelset branch only (see
   ``_voxelize_terrain_solid``) to compensate for this same bias — a
   future fix of the stamp convention must remove that scoped
   compensation in the same change, or terrain will double-correct.
   The winding and scanline terrain paths consume the raw solid
   (2026-08-24 contact fix).
```

- [ ] **Step 3: Addendum in the 2026-08-11 design doc**

Append to the "Consequences accepted" bullet about the terrain pre-shift in `docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md`:

```markdown
  **2026-08-24 update:** the compensation is now scoped to the levelset
  branch only; the winding/scanline fallbacks take the raw solid and the
  fallback winding is grid-aligned (`align_origin=True`).  See
  `2026-08-24-building-terrain-contact-design.md`.
```

- [ ] **Step 4: Run the offline suite and commit**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q -m "not slow"`
Expected: PASS.

```bash
git add voxcitygml/cli.py voxcitygml/voxelizer3d.py docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md
git commit -m "feat: --ground-contact-tolerance CLI flag; scope-shift doc updates"
```

---

### Task 6: Dataset-gated integration test + acceptance

**Files:**
- Modify: `tests/test_integration_plateau.py` (new test)

- [ ] **Step 1: Add the integration test**

Append to `tests/test_integration_plateau.py` (it already defines `DATASET`, `requires_dataset`, and imports; add `run_core` usage in the style of the file's other cfg-building tests):

```python
@requires_dataset
@pytest.mark.slow
def test_lod2_buildings_touch_terrain(tmp_path):
    """No unintentional air gap between building bottoms and the ground.

    Columns whose mesh-accurate building bottom is within the
    ground-contact tolerance must have zero AIR voxels between the lowest
    building voxel and the support below (2026-08-24 contact design).
    Pilotis columns (bottom above the tolerance) are exempt.
    """
    from voxcitygml import VoxelizerConfig
    from voxcitygml.pipeline import run_core
    from voxcitygml.citygml.coordinates import create_rectangle

    rect = create_rectangle(139.7725, 35.6481, 200)
    cfg = VoxelizerConfig(
        citygml_path=DATASET,
        rectangle_vertices=rect,
        meshsize=2.0,
        building_lod=2,
        land_cover_source="CityGML",
        canopy_height_source="Static",
        output_dir=str(tmp_path),
        save_output=False,
        use_parse_cache=False,
    )
    art = run_core(cfg)
    grid = art.voxel_grid
    is_b = grid == BUILDING_CODE
    bad = []
    for r, c in zip(*np.nonzero(is_b.any(axis=2))):
        bottom = int(np.argmax(is_b[r, c]))
        below = np.nonzero(grid[r, c, :bottom] != 0)[0]
        gap = bottom - (int(below[-1]) if len(below) else -1) - 1
        if gap > 0:
            segments = art.building_min_height_grid[r, c]
            h_min = min((s[0] for s in segments), default=None)
            if h_min is not None and h_min <= cfg.ground_contact_tolerance:
                bad.append((r, c, gap, h_min))
    assert not bad, f"{len(bad)} sub-tolerance floating columns: {bad[:10]}"
```

- [ ] **Step 2: Run it**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_integration_plateau.py::test_lod2_buildings_touch_terrain -q`
Expected: PASS (skips automatically when the dataset directory is absent).

- [ ] **Step 3: Full suite**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q`
Expected: PASS (slow dataset tests included; the roof-slope numbers in `test_lod2_generate_voxcity_end_to_end` measure building geometry only and should not move — if one moves, investigate before recalibrating).

- [ ] **Step 4: Acceptance rerun of the diagnostics**

Run the session diagnostic script (scratchpad `diagnose_floating_buildings.py`) on the four investigated configurations — Chuo 2 m (139.7725, 35.6481, 200 m), Kudanzaka 1 m and 2 m (139.7467, 35.6952, 200 m, Chiyoda dataset), Ochanomizu 2 m 500 m (139.7592, 35.6989, Chiyoda dataset) — and confirm: `terrain_top_face − dem` median within `[0, vs)` everywhere, and zero gap columns except those with `h_min > 1.5` (report any as pilotis/podium columns, expected to remain).

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_plateau.py
git commit -m "test: PLATEAU integration guard — buildings touch terrain within tolerance"
```
