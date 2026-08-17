# Inclusive Voxelization Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `voxelization_mode: "inclusive" | "tight"` option (default `"inclusive"`) so thin geometry (< 1 voxel) voxelizes with no gaps for sunlight/wind obstruction analyses.

**Architecture:** Mode is policy resolved at the `VoxelizerConfig` boundary into three mechanism knobs (`building_shell_threshold`, `occupancy_threshold`, `shell_anchor`). The voxelizer keeps plain numeric/flag parameters, with defaults flipped to the inclusive values (shell threshold `INCLUSIVE_SHELL_THRESHOLD` = 0.25, connectivity-flood anchor). `"tight"` reproduces the 2026-08-11 behavior exactly.

**Shell metric corrected 2026-08-17 — see the design spec's "Shell metric calibration".** The inclusive threshold is **0.0**, as originally specced. What changed is the metric: `_overlay_surface_shell` now builds its SAT box *shrunk* (`voxel_size/2 - voxel_size*1e-6`) instead of expanded, so it tests whether the mesh **penetrates** a cell rather than whether it **touches the cell's boundary**. With the expanded box a face lying on a cell boundary marked both neighbours, inflating solid buildings 2–2.5× with empty voxels (aligned box 180 → 448). An intermediate "calibrated 0.25 threshold" was tried and rejected — it only looked exact on a round grid origin and leaked a full layer on a real pyproj origin. Tight mode is measurably unaffected by the shrink. Tasks 2 and 3 below still describe the constant `INCLUSIVE_SHELL_THRESHOLD` in `models.py`; its value is 0.0.

**Tech Stack:** Python, numpy, scipy.ndimage (`binary_propagation`), numba, meshlib, trimesh (tests), pytest.

**Spec:** `docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md`

**Test command prefix (PowerShell — conda is NOT on PATH):**
`& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest`

**File map:**
- Modify: `voxcitygml/voxelizer3d.py` (Tasks 1–2: anchor mechanism, inclusive defaults)
- Modify: `voxcitygml/models.py` (Task 3: mode enum + resolver)
- Modify: `voxcitygml/pipeline.py`, `voxcitygml/pipeline_export.py`, `voxcitygml/export_obj.py`, `voxcitygml/cli.py` (Task 4: plumbing)
- Create: `tests/test_inclusive_voxelization.py` (Tasks 1–3)
- Modify: `tests/test_voxelizer_alignment.py` (Task 2: pin tight settings explicitly)
- Modify: `tests/test_integration_plateau.py` (Task 5: calibration comment)

---

### Task 1: `anchor` parameter on `_overlay_surface_shell`

The shell overlay currently keeps only voxels 6-adjacent (1 step) to an already-filled voxel (`surface &= _dilate6(existing)` in `voxcitygml/voxelizer3d.py:826-831`). Add an `anchor` parameter: `"adjacent"` (current rule) or `"connected"` (flood through the shell from anchored seeds; keep whole shell if no seeds exist anywhere).

**Files:**
- Modify: `voxcitygml/voxelizer3d.py:778-837` (`_overlay_surface_shell`), `voxcitygml/voxelizer3d.py:16` (scipy import)
- Create: `tests/test_inclusive_voxelization.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inclusive_voxelization.py`:

```python
"""Inclusive voxelization: shell anchor rules and gap-free thin volumes.

Pins the 2026-08-17 inclusive-voxelization design
(docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md):

- ``_overlay_surface_shell`` ``anchor="connected"`` keeps thin features
  connected *through the shell* to any filled voxel, still drops
  disconnected fragments, and keeps the whole shell when no anchor exists
  at all (per-category export grids contain no terrain to anchor on).
- Building defaults are inclusive: shell threshold 0.0 + connected anchor
  produce gap-free thin walls (the Plateau LOD2 "comb" bug).
- ``VoxelizerConfig.voxelization_mode`` resolves mode -> mechanism knobs,
  with explicit threshold values overriding the mode.
"""
import numpy as np
import pytest
import trimesh

from voxcitygml.voxelizer3d import (
    _MESHLIB_VOXEL_AVAILABLE,
    Grid3DParams,
    _overlay_surface_shell,
)

MS = 2.0


def make_gp():
    # Same deliberately y-incongruent grid as tests/test_voxelizer_alignment.py:
    # (max_y - min_y) is not a whole number of voxels, matching production.
    return Grid3DParams(n_rows=12, n_cols=12, n_z=10,
                        min_x=-6.0, max_x=18.0, min_y=-6.9, max_y=18.0,
                        min_z=-6.0, max_z=14.0, voxel_size=MS)


def box(min_corner, extents):
    b = trimesh.creation.box(extents=list(extents))
    b.apply_translation([min_corner[i] + extents[i] / 2 for i in range(3)])
    return np.asarray(b.vertices, float), np.asarray(b.faces)


def filled(grid, code=-3):
    return set(zip(*np.nonzero(grid == code)))


# Thin wall used throughout: 0.5 m thick on a 2 m grid, crossing the cell
# boundary at x=4 so each of columns 4 and 5 sees a single face
# (~9/27 = 0.33 surface contact).  Extents chosen so no face lies exactly on
# a cell boundary: x in [3.9, 4.4], y in [0.3, 11.7], z in [0.3, 9.7].
# Cells the wall crosses: rows 3..8, cols {4, 5}, zi 3..7.
# No voxel-column centre (x = ..., 3, 5, ...) lies inside the wall, so the
# winding fill contributes nothing — the shell must supply every voxel.
WALL = ((3.9, 0.3, 0.3), (0.5, 11.4, 9.4))


def wall_cells():
    return [(row, col, zi)
            for row in range(3, 9) for col in (4, 5) for zi in range(3, 8)]


def test_adjacent_anchor_drops_upper_thin_wall():
    """Documents the pre-fix rule: only shell voxels 6-adjacent to a filled
    voxel survive, so a tall thin wall keeps just its bottom slice."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1                     # ground layer under the wall
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="adjacent")
    got = filled(grid)
    assert (5, 4, 3) in got                # bottom slice: adjacent to ground
    assert (5, 4, 6) not in got            # upper wall: dropped by adjacency


def test_connected_anchor_keeps_full_thin_wall():
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="connected")
    got = filled(grid)
    for cell in wall_cells():
        assert cell in got, f"gap at {cell}"


def test_connected_anchor_drops_disconnected_fragment():
    """One mesh containing an anchored wall AND a floating cube far away:
    the wall survives the flood, the fragment does not."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1
    v1, f1 = box(*WALL)
    v2, f2 = box((-3.7, 0.3, 10.3), (1.4, 1.4, 1.4))   # one cell: (8, 1, 8)
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="connected")
    got = filled(grid)
    assert (5, 4, 5) in got                # wall kept
    assert (8, 1, 8) not in got            # floating fragment dropped


def test_connected_anchor_without_any_seed_keeps_whole_shell():
    """No filled voxel anywhere (per-category export grids have no terrain):
    dropping the whole feature would be worse than keeping it unanchored."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)   # completely empty: no anchors
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="connected")
    got = filled(grid)
    for cell in wall_cells():
        assert cell in got, f"gap at {cell}"


def test_adjacent_anchor_without_any_seed_keeps_nothing():
    """Current behavior, unchanged in tight mode."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    v, f = box(*WALL)
    _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="adjacent")
    assert filled(grid) == set()


def test_unknown_anchor_raises():
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    v, f = box(*WALL)
    with pytest.raises(ValueError):
        _overlay_surface_shell(v, f, gp, grid, -3, True, anchor="loose")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_inclusive_voxelization.py -q`

Expected: FAIL — `TypeError: _overlay_surface_shell() got an unexpected keyword argument 'anchor'` on every test.

- [ ] **Step 3: Implement the anchor parameter**

In `voxcitygml/voxelizer3d.py`, extend the scipy import (line 16):

```python
from scipy.ndimage import binary_fill_holes, binary_propagation, zoom
```

Replace `_overlay_surface_shell` (currently lines 778–837) with:

```python
def _overlay_surface_shell(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: "Grid3DParams",
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
    anchor: str = "adjacent",
) -> None:
    """Stamp the triangle surface shell onto the voxel grid via SAT.

    This guarantees that every voxel touching an original mesh triangle
    is marked, even when the SDF discretisation misses thin walls whose
    thickness is smaller than the voxel size.

    When *occupancy_threshold* > 0, boundary voxels whose SURFACE-CONTACT
    occupancy is below the threshold are discarded.  This controls how
    "inclusive" the surface shell is.  The metric is the fraction of the
    voxel's sub-cells that a triangle passes through, not how much of the
    voxel lies inside the mesh: a lone flat face crossing the voxel scores
    ~1/3 and is dropped at 0.5, while an edge or corner where two faces
    cross scores higher and survives.  See ``_compute_occupancy_fraction``.

    *anchor* controls which shell voxels survive relative to already-filled
    voxels (2026-08-17 inclusive-voxelization design):

    ``"adjacent"``
        Keep only shell voxels 6-adjacent (1 step) to a filled voxel.
        Historic rule; a tall thin feature whose interior fill is empty
        keeps just the slice next to its anchor.
    ``"connected"``
        Flood (``scipy.ndimage.binary_propagation``, 6-connectivity) from
        the adjacent seeds through the shell itself: every shell voxel
        connected to an anchored voxel survives, disconnected floating
        fragments are still discarded.  If NO seed exists anywhere — a
        fully thin mesh whose winding fill produced nothing, e.g. buildings
        in per-category export grids with no terrain — the whole shell is
        kept: dropping an entire real feature is worse for obstruction than
        keeping an unanchored one.
    """
    if anchor not in ("adjacent", "connected"):
        raise ValueError(
            f"anchor must be 'adjacent' or 'connected', got {anchor!r}")
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    r0, r1, c0, c1, z0, z1 = _bbox_to_index_range(gp, vmin, vmax)
    if r0 > r1 or c0 > c1 or z0 > z1:
        return
    nr, nc, nz = r1 - r0 + 1, c1 - c0 + 1, z1 - z0 + 1
    half_eps = gp.voxel_size * 1e-6
    half = np.array([gp.voxel_size / 2.0 + half_eps] * 3, dtype=np.float64)
    verts_f64 = np.ascontiguousarray(verts, dtype=np.float64)
    faces_ip = np.ascontiguousarray(faces, dtype=np.intp)
    surface = _surface_voxelize_numba(
        verts_f64, faces_ip,
        gp.min_x, gp.max_y, gp.min_z, gp.voxel_size,
        r0, r1, c0, c1, z0, z1,
        nr, nc, nz, half,
    )
    if occupancy_threshold > 0.0:
        surface = _filter_surface_by_occupancy(
            surface, verts_f64, faces_ip, gp,
            r0, c0, z0, nr, nc, nz,
            occupancy_threshold, occupancy_subdivisions,
        )

    # Anchor filter: prevents stray / disconnected mesh fragments from
    # creating floating artifacts.  See the docstring for the two rules.
    subgrid = voxel_grid[r0:r1 + 1, c0:c1 + 1, z0:z1 + 1]
    existing = subgrid != 0  # any non-empty voxel counts as anchor
    seeds = surface & _dilate6(existing)
    if anchor == "adjacent":
        surface = seeds
    elif seeds.any():
        surface = binary_propagation(seeds, mask=surface)
    # else: connected with no seed anywhere -> keep the whole shell

    if overwrite:
        subgrid[surface] = class_code
    else:
        mask = surface & (subgrid == 0)
        subgrid[mask] = class_code
```

(Only the docstring, the `anchor` validation, and the anchor-filter block change; the rasterization body is identical to the current code.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_inclusive_voxelization.py -q`
Expected: 6 passed.

Also run the alignment suite (its behavior must be untouched — every call still uses the `anchor="adjacent"` default):

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_voxelizer_alignment.py -q`
Expected: all passed (or skipped if meshlib missing).

- [ ] **Step 5: Commit**

```powershell
git add tests/test_inclusive_voxelization.py voxcitygml/voxelizer3d.py
git commit -m "feat: connectivity-flood anchor option for the surface shell"
```

---

### Task 2: Inclusive defaults through the building/vegetation voxelization paths

Flip the mechanism defaults to inclusive (`shell_threshold` 0.5 → 0.0, `shell_anchor="connected"`) and thread `shell_anchor` through `_voxelize_building_solid`, `_voxelize_mesh_group`, and `voxelize_citygml_meshes`. Pin the old tight settings explicitly in the alignment tests.

**Files:**
- Modify: `voxcitygml/voxelizer3d.py` (`voxelize_citygml_meshes` ~120–310, `_voxelize_building_solid` ~840–896, `_voxelize_mesh_group` ~901–1012)
- Modify: `tests/test_voxelizer_alignment.py`
- Test: `tests/test_inclusive_voxelization.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inclusive_voxelization.py`:

```python
# ── Inclusive defaults through the building path ──────────────────────

@pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")
def test_thin_wall_voxelizes_gap_free_by_default():
    """The Plateau LOD2 comb bug: a wall thinner than a voxel, crossing a
    cell boundary so each cell sees a single face (~0.33 surface contact),
    must voxelize with NO gaps under the default (inclusive) settings."""
    from voxcitygml.voxelizer3d import _voxelize_building_solid
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1                     # terrain under the wall
    v, f = box(*WALL)
    _voxelize_building_solid(v, f, gp, grid, -3, True)
    got = filled(grid)
    for cell in wall_cells():
        assert cell in got, f"gap at {cell}"


@pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")
def test_tight_settings_reproduce_2026_08_11_behavior():
    """Explicit tight knobs (shell 0.5, adjacent anchor) keep the same wall
    sparse — the behavior 'voxelization_mode=\"tight\"' resolves to."""
    from voxcitygml.voxelizer3d import _voxelize_building_solid
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 2] = -1
    v, f = box(*WALL)
    _voxelize_building_solid(v, f, gp, grid, -3, True,
                             shell_threshold=0.5, shell_anchor="adjacent")
    got = filled(grid)
    assert (5, 4, 6) not in got            # single-face cell dropped at 0.5


@pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")
def test_shell_anchor_reaches_shell(monkeypatch):
    """_voxelize_building_solid must forward shell_anchor (and the new 0.0
    shell default) to _overlay_surface_shell.

    Covers seam -> shell only.  The upper plumbing (VoxelizerConfig ->
    pipeline / export) is single-line pass-throughs left to review, matching
    test_voxelizer_alignment.py::test_building_shell_threshold_reaches_shell.
    """
    import voxcitygml.voxelizer3d as vx

    seen = {}
    real = vx._overlay_surface_shell

    def spy(verts, faces, gp, grid, code, overwrite, **kw):
        seen.update(kw)
        return real(verts, faces, gp, grid, code, overwrite, **kw)

    monkeypatch.setattr(vx, "_overlay_surface_shell", spy)
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    v, f = box((0.0, 0.3, 0.3), (12.0, 11.4, 9.4))
    vx._voxelize_building_solid(v, f, gp, grid, -3, True,
                                shell_anchor="adjacent")
    assert seen["anchor"] == "adjacent"
    assert seen["occupancy_threshold"] == INCLUSIVE_SHELL_THRESHOLD


# The calibration guard.  Inclusive mode must produce EXACTLY the set of
# voxels containing mesh volume: no gaps (the comb bug) and no empty
# voxels (the threshold-0.0 inflation).  Both failure directions are
# pinned here, so retuning INCLUSIVE_SHELL_THRESHOLD outside the
# [0.15, 0.33] plateau fails loudly instead of silently changing every
# building envelope.  See "Threshold calibration" in the design spec.
INCLUSIVE_CASES = [
    ("box exactly aligned",       (0.0,   0.0, 0.0, 12.0, 12.0, 10.0)),
    ("box +0.05 off aligned",     (0.05,  0.0, 0.0, 12.0, 12.0, 10.0)),
    ("box offset 0.7",            (0.7,   0.7, 0.0, 12.0, 12.0, 10.0)),
    ("thin wall 0.5m mid-cell",   (3.9,   0.3, 0.3, 0.5, 11.4, 9.4)),
    ("thin wall 0.5m on bound.",  (3.75,  0.3, 0.3, 0.5, 11.4, 9.4)),
    ("very thin wall 0.1m",       (3.95,  0.3, 0.3, 0.1, 11.4, 9.4)),
    ("thin slab 0.3m horizontal", (0.3,   0.3, 4.85, 11.4, 11.4, 0.3)),
]


def cells_containing_volume(x0, y0, z0, ex, ey, ez):
    """Voxels whose volume meets the box in a set of positive measure."""
    gp = make_gp()
    out = set()
    for row in range(gp.n_rows):
        for col in range(gp.n_cols):
            for zi in range(gp.n_z):
                cx0 = gp.min_x + col * MS
                cy1 = gp.max_y - row * MS
                cz0 = gp.min_z + zi * MS
                ox = min(cx0 + MS, x0 + ex) - max(cx0, x0)
                oy = min(cy1, y0 + ey) - max(cy1 - MS, y0)
                oz = min(cz0 + MS, z0 + ez) - max(cz0, z0)
                if ox > 1e-9 and oy > 1e-9 and oz > 1e-9:
                    out.add((row, col, zi))
    return out


@pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")
@pytest.mark.parametrize("label,args", INCLUSIVE_CASES,
                         ids=[c[0] for c in INCLUSIVE_CASES])
def test_inclusive_defaults_match_volume_exactly(label, args):
    from voxcitygml.voxelizer3d import _voxelize_building_solid
    x0, y0, z0, ex, ey, ez = args
    b = trimesh.creation.box(extents=[ex, ey, ez])
    b.apply_translation([x0 + ex / 2, y0 + ey / 2, z0 + ez / 2])
    v = np.asarray(b.vertices, float)
    f = np.asarray(b.faces)

    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int16)
    grid[:, :, 0] = -1                     # terrain floor, well below
    _voxelize_building_solid(v, f, gp, grid, -3, True)

    got = filled(grid)
    want = cells_containing_volume(*args)
    assert got == want, (
        f"{label}: {len(got)} cells vs ideal {len(want)} "
        f"({len(got - want)} empty voxels added, "
        f"{len(want - got)} solid voxels missed)")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_inclusive_voxelization.py -q`

Expected: the 6 Task-1 tests pass; the 3 new tests fail —
`test_thin_wall_voxelizes_gap_free_by_default` with gap assertions (old 0.5/adjacent defaults),
`test_tight_settings_reproduce_2026_08_11_behavior` and `test_shell_anchor_reaches_shell` with `TypeError: ... unexpected keyword argument 'shell_anchor'`.

- [ ] **Step 3: Thread `shell_anchor` and flip defaults in `voxelizer3d.py`**

Five edits. **(0) first**, because the other four reference the constant:

**(0) The shared threshold constant.** Add to `voxcitygml/models.py`, just above the "Pipeline configuration" banner comment:

```python
#: Surface-contact occupancy threshold for the inclusive-mode surface
#: shell.  A voxel is kept when this fraction of its 3x3x3 sub-cells
#: touch mesh geometry.  Calibrated 2026-08-17: any value in
#: [0.15, 0.33] reproduces the analytic ideal — exactly the voxels
#: containing mesh volume — on all ten calibration geometries, so 0.25
#: is the plateau midpoint.  Below ~0.15 the shell also marks the empty
#: voxel outside every boundary face (solid buildings inflate 2-2.5x);
#: above ~0.33 a lone flat face scores too low and thin walls vanish
#: again.  See "Threshold calibration" in
#: docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md.
#: Retuning this single value retunes inclusive mode everywhere.
INCLUSIVE_SHELL_THRESHOLD = 0.25
```

This is the ONLY change to `models.py` in this task (Task 3 adds the mode enum and resolver, and consumes this constant). Import it in `voxcitygml/voxelizer3d.py` by extending the existing models import (no import cycle — `models.py` does not import `voxelizer3d`):

```python
from .models import Mesh3D, CityGMLMeshCollection, INCLUSIVE_SHELL_THRESHOLD
```

Then the four threading edits:

**(a) `_voxelize_building_solid`** — signature (currently `shell_threshold: float = 0.5` is at line ~849):

```python
def _voxelize_building_solid(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
    shell_threshold: float = INCLUSIVE_SHELL_THRESHOLD,
    shell_anchor: str = "connected",
) -> None:
```

and its shell call becomes:

```python
        if ok:
            _overlay_surface_shell(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                occupancy_threshold=shell_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
                anchor=shell_anchor,
            )
            return
```

Append to its docstring (after the `occupancy_threshold` paragraph):

```
    ``shell_threshold`` / ``shell_anchor`` default to the INCLUSIVE mode
    (``INCLUSIVE_SHELL_THRESHOLD`` / "connected", 2026-08-17 design):
    every voxel that CONTAINS part of the raw mesh becomes solid, and
    thin features survive via the connectivity flood.  Note the shell
    threshold is deliberately not 0: at 0 the surface-contact metric also
    marks the empty voxel on the far side of every boundary face, which
    inflates solid buildings 2-2.5x without adding any obstruction.  Pass
    0.5 / "adjacent" for the historic tight envelope
    (``voxelization_mode="tight"``).
```

**(b) `_voxelize_mesh_group`** — signature (currently `shell_threshold: float = 0.5` at line ~910):

```python
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
    shell_threshold: float = INCLUSIVE_SHELL_THRESHOLD,
    shell_anchor: str = "connected",
    force_surface: bool = False,
```

Buildings branch adds the pass-through:

```python
        if class_code == BUILDING_CODE and not force_surface:
            _voxelize_building_solid(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                occupancy_threshold=occupancy_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
                shell_threshold=shell_threshold,
                shell_anchor=shell_anchor,
            )
```

Vegetation branch's shell call adds the anchor:

```python
                if ok:
                    # Union with surface shell so leaves / branches thinner
                    # than one voxel are not lost by the narrow-band SDF.
                    _overlay_surface_shell(
                        verts, faces, gp, voxel_grid,
                        class_code, overwrite,
                        occupancy_threshold=occupancy_threshold,
                        occupancy_subdivisions=occupancy_subdivisions,
                        anchor=shell_anchor,
                    )
                    continue
```

Add to the `Args:` section of its docstring:

```
        shell_anchor: Anchor rule for every ``_overlay_surface_shell``
            call this group makes (building shell and vegetation shell):
            "connected" (default, inclusive) or "adjacent" (tight).  See
            ``_overlay_surface_shell``.
```

**(c) `voxelize_citygml_meshes`** — signature (currently `building_shell_threshold: float = 0.5` at line ~135):

```python
    occupancy_threshold: float = 0.0,
    occupancy_subdivisions: int = 3,
    building_shell_threshold: float = INCLUSIVE_SHELL_THRESHOLD,
    shell_anchor: str = "connected",
    underground_depth: float = 0.0,
```

Buildings group call adds `shell_anchor=shell_anchor,` after `shell_threshold=building_shell_threshold,`; vegetation group call adds `shell_anchor=shell_anchor,` after its `occupancy_subdivisions=occupancy_subdivisions,`. (The bridges call is untouched — its `force_surface` path never reaches the shell.)

In the docstring, replace `(default 0.5)` in the `building_shell_threshold` entry with `(default 0.0 — inclusive)` and add after that entry:

```
        shell_anchor: Anchor rule for surface-shell overlays ("connected"
            default — inclusive; "adjacent" — tight).  See
            ``_overlay_surface_shell``.
```

**(d)** No change to the bridges/`force_surface` or fallback cascades — `occupancy_threshold` keeps governing them, default 0.0 as before.

- [ ] **Step 4: Pin tight settings explicitly in the alignment tests**

`tests/test_voxelizer_alignment.py` pins the 2026-08-11 tight behavior; with the anchor default now `"connected"` it must say so explicitly. Add `shell_anchor="adjacent"` to every `building_solid(...)` call that passes `shell_threshold`:

- `test_building_path_box_exact` (line ~89): `building_solid(v, f, gp, grid, -3, True, occupancy_threshold=0.0, occupancy_subdivisions=3, shell_threshold=0.5, shell_anchor="adjacent")`
- `test_aligned_box_no_dilation` (line ~126): same four kwargs.
- `test_shell_threshold_discriminates_surface_contact` (lines ~148 and ~155): `building_solid(v, f, gp, grid, -3, True, shell_threshold=thr, shell_anchor="adjacent")` and `building_solid(v2, f2, gp, grid, -3, True, shell_threshold=0.5, shell_anchor="adjacent")`.

(`test_building_shell_threshold_reaches_shell` passes `shell_threshold=0.31` only — leave it; it asserts the threshold seam, not shell contents.)

Also update the module docstring's last sentence to note the tight settings are now explicit:

```python
"""Regression tests for building voxelization alignment.

An axis-aligned box whose faces lie exactly on the 2 m grid must voxelize to
exactly its analytic cell set -- no displacement, no dilation.  These tests
pin the 2026-08-11 diagnosis: the old levelset path filled 294 cells where
the answer is 180 (corner-sampled SDF stamped as centre-sampled).

Since the 2026-08-17 inclusive-voxelization change the production DEFAULTS
are shell_threshold=0.0 / shell_anchor="connected"
(voxelization_mode="inclusive"); these tests pass the tight settings
explicitly because they pin the tight envelope contract that
voxelization_mode="tight" resolves to.
"""
```

- [ ] **Step 5: Run both test files to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_inclusive_voxelization.py tests/test_voxelizer_alignment.py -q`
Expected: all passed (alignment tests skipped if meshlib missing).

- [ ] **Step 6: Commit**

```powershell
git add voxcitygml/voxelizer3d.py tests/test_inclusive_voxelization.py tests/test_voxelizer_alignment.py
git commit -m "feat: inclusive voxelization defaults - gap-free thin walls"
```

---

### Task 3: `voxelization_mode` enum + resolver on `VoxelizerConfig`

**Files:**
- Modify: `voxcitygml/models.py` (`VoxelizerConfig`, lines ~155–271)
- Test: `tests/test_inclusive_voxelization.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inclusive_voxelization.py`:

```python
# ── VoxelizerConfig mode resolution ───────────────────────────────────

# Imported inside each test, not at module level: until Step 3 lands,
# `ResolvedVoxelParams` does not exist, and a module-level import would
# turn the red step into a collection ERROR for the whole file — taking
# the passing Task 1 / Task 2 tests down with it.  Same lazy-import
# rationale as tests/test_voxelizer_alignment.py's `building_solid`.


def test_default_mode_is_inclusive():
    from voxcitygml.models import (
        INCLUSIVE_SHELL_THRESHOLD, ResolvedVoxelParams, VoxelizerConfig)
    p = VoxelizerConfig().resolved_voxel_params()
    assert p == ResolvedVoxelParams(
        building_shell_threshold=INCLUSIVE_SHELL_THRESHOLD,
        occupancy_threshold=0.0,
        shell_anchor="connected",
    )


def test_inclusive_threshold_is_zero():
    """Inclusive mode does no occupancy filtering: the shell rasterizer's
    shrunk SAT box already answers the volume question (see the design
    spec's "Shell metric calibration").  Raising this above ~0.33 would
    reintroduce the comb bug this mode exists to fix, so the value is
    pinned rather than left as a tuning knob."""
    from voxcitygml.models import INCLUSIVE_SHELL_THRESHOLD
    assert INCLUSIVE_SHELL_THRESHOLD == 0.0


def test_tight_mode_reproduces_2026_08_11_defaults():
    from voxcitygml.models import ResolvedVoxelParams, VoxelizerConfig
    p = VoxelizerConfig(voxelization_mode="tight").resolved_voxel_params()
    assert p == ResolvedVoxelParams(
        building_shell_threshold=0.5,
        occupancy_threshold=0.0,
        shell_anchor="adjacent",
    )


def test_explicit_threshold_overrides_mode():
    from voxcitygml.models import VoxelizerConfig
    p = VoxelizerConfig(
        building_shell_threshold=0.5).resolved_voxel_params()
    assert p.building_shell_threshold == 0.5
    assert p.shell_anchor == "connected"   # anchor still from the mode


def test_unknown_mode_raises_at_construction():
    from voxcitygml.models import VoxelizerConfig
    with pytest.raises(ValueError):
        VoxelizerConfig(voxelization_mode="loose")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_inclusive_voxelization.py -q`
Expected: the 4 new tests fail — `ImportError: cannot import name 'ResolvedVoxelParams'` in the first two, `TypeError: __init__() got an unexpected keyword argument 'voxelization_mode'` in the last; `test_explicit_threshold_overrides_mode` fails with `AttributeError: 'VoxelizerConfig' object has no attribute 'resolved_voxel_params'`. All Task 1 / Task 2 tests still pass (the lazy imports keep collection working).

- [ ] **Step 3: Implement mode field, validation, and resolver**

In `voxcitygml/models.py`:

**(a)** Extend the typing import to include `NamedTuple` (the module already imports from `typing`).

**(b)** Immediately above the `VoxelizerConfig` dataclass (after the "Pipeline configuration" banner comment at lines ~151–153), add:

```python
VOXELIZATION_MODES = ("inclusive", "tight")


class ResolvedVoxelParams(NamedTuple):
    """Concrete voxelizer knobs after applying ``voxelization_mode``.

    ``VoxelizerConfig.resolved_voxel_params`` is the only producer; the
    pipeline and the OBJ exporters consume the same instance so the main
    grid and exported voxels always agree (the 2026-08-11 invariant).
    """
    building_shell_threshold: float
    occupancy_threshold: float
    shell_anchor: str


_MODE_PARAMS = {
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
```

**(c)** Change the two threshold fields (lines ~261, ~263) to mode-deferring optionals:

```python
    occupancy_threshold: Optional[float] = None
    occupancy_subdivisions: int = 3
    building_shell_threshold: Optional[float] = None
```

and add the mode field next to them:

```python
    voxelization_mode: str = "inclusive"
```

**(d)** Add validation and the resolver at the end of the dataclass body:

```python
    def __post_init__(self):
        if self.voxelization_mode not in VOXELIZATION_MODES:
            raise ValueError(
                f"voxelization_mode must be one of {VOXELIZATION_MODES}, "
                f"got {self.voxelization_mode!r}")

    def resolved_voxel_params(self) -> ResolvedVoxelParams:
        """Mode defaults with explicit threshold overrides applied.

        ``None`` in ``building_shell_threshold`` / ``occupancy_threshold``
        means "the mode decides"; an explicit value always wins over the
        mode.  ``shell_anchor`` has no per-field override — it follows the
        mode.
        """
        base = _MODE_PARAMS[self.voxelization_mode]
        overrides = {
            key: value
            for key, value in (
                ("building_shell_threshold", self.building_shell_threshold),
                ("occupancy_threshold", self.occupancy_threshold),
            )
            if value is not None
        }
        return base._replace(**overrides)
```

**(e)** Update the docstring attributes (lines ~188–206). Replace the `occupancy_threshold` and `building_shell_threshold` entries with:

```
        voxelization_mode: "inclusive" (default) or "tight".  Inclusive
            keeps every voxel any geometry touches and floods thin features
            from their anchors — no gaps for obstruction (sunlight / wind)
            analyses, at the cost of up to ~1 voxel of envelope inflation
            on corner-grazed edges.  Tight restores the 2026-08-11
            behavior: building shell kept only at >= 0.5 surface-contact
            occupancy with a 1-step adjacency anchor.  The mode resolves to
            concrete knobs via ``resolved_voxel_params()``.
        occupancy_threshold: Minimum SURFACE-CONTACT occupancy (0.0–1.0) a
            boundary voxel must reach to be kept during surface
            voxelization — the fraction of a voxel's subdivided sub-cells
            that touch mesh geometry, not the fraction of its volume
            enclosed.  ``None`` (default) defers to ``voxelization_mode``
            (both modes use 0.0: keep every voxel with any geometric
            contact).  Does not govern the building surface shell; see
            ``building_shell_threshold``.
        occupancy_subdivisions: Sub-divisions per axis when estimating
            surface-contact occupancy (default 3 → 27 sub-samples per voxel).
        building_shell_threshold: Minimum SURFACE-CONTACT occupancy for the
            building surface-shell overlay.  ``None`` (default) defers to
            ``voxelization_mode``: ``INCLUSIVE_SHELL_THRESHOLD`` (0.25) in
            inclusive mode, 0.5 in tight mode.  This is NOT volume overlap
            — a lone flat face crossing a voxel scores ~0.33 and is dropped
            at 0.5; two crossing faces or a slab spanning two sub-slabs
            score >= 0.5 and are kept.  The interior fill independently
            keeps every centre-inside cell, so the shell only decides
            thin-feature and edge cells.  Setting it to 0 is a trap: the
            metric then also keeps the empty cell on the outside of every
            boundary face, inflating solid buildings 2-2.5x (measured
            2026-08-17) — which is why inclusive mode uses 0.25, not 0.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_inclusive_voxelization.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```powershell
git add voxcitygml/models.py tests/test_inclusive_voxelization.py
git commit -m "feat: voxelization_mode enum with inclusive default on VoxelizerConfig"
```

---

### Task 4: Plumb resolved params through pipeline, exports, and CLI

All single-line pass-throughs of `cfg.resolved_voxel_params()`; the seam below them is covered by `test_shell_anchor_reaches_shell` (Task 2) and `test_building_shell_threshold_reaches_shell` (existing). `cfg.occupancy_threshold` / `cfg.building_shell_threshold` may now be `None`, so **every** direct read of them must go through the resolver — the four call sites below are all of them (verified by grep 2026-08-17).

**Files:**
- Modify: `voxcitygml/pipeline.py:392-407`
- Modify: `voxcitygml/pipeline_export.py:92-105`
- Modify: `voxcitygml/export_obj.py:958-1075` (`export_per_category_voxels_obj`)
- Modify: `voxcitygml/cli.py`

- [ ] **Step 1: `pipeline.py` — resolve once, pass everywhere**

At line 392, before the `if cfg.use_3d_voxelizer:` voxelize call, insert:

```python
    vox_params = cfg.resolved_voxel_params()
```

and change the three threshold kwargs of `voxelize_citygml_meshes` (lines 402–404) to:

```python
            occupancy_threshold=vox_params.occupancy_threshold,
            occupancy_subdivisions=cfg.occupancy_subdivisions,
            building_shell_threshold=vox_params.building_shell_threshold,
            shell_anchor=vox_params.shell_anchor,
```

- [ ] **Step 2: `export_obj.py` — accept and forward shell params**

`export_per_category_voxels_obj` signature (after `occupancy_subdivisions: int = 3,` at line ~968) gains:

```python
    building_shell_threshold: float = 0.0,
    shell_anchor: str = "connected",
```

Docstring: add after the first paragraph:

```
    ``building_shell_threshold`` / ``shell_anchor`` default to the
    inclusive mode; pass the caller's resolved values so exported building
    voxels match the main grid (the 2026-08-11 invariant).
```

The building branch (line ~1029) becomes:

```python
                if cat_name == "building":
                    _voxelize_building_solid(
                        verts, faces, gp, cat_grid,
                        class_code=code, overwrite=False,
                        occupancy_threshold=occupancy_threshold,
                        occupancy_subdivisions=occupancy_subdivisions,
                        shell_threshold=building_shell_threshold,
                        shell_anchor=shell_anchor,
                    )
                    continue
```

(Note this call previously omitted `shell_threshold`, silently using the 0.5 default — passing it explicitly is itself a fix for mode consistency.)

The fallback `_voxelize_mesh_group` call (line ~1068) gains, after `occupancy_subdivisions=occupancy_subdivisions,`:

```python
                shell_threshold=building_shell_threshold,
                shell_anchor=shell_anchor,
```

- [ ] **Step 3: `pipeline_export.py` — pass resolved values**

Before the `export_per_category_voxels_obj(` call (line ~92), insert:

```python
        vox_params = cfg.resolved_voxel_params()
```

and change its threshold kwargs (lines 101–102) to:

```python
            occupancy_threshold=vox_params.occupancy_threshold,
            occupancy_subdivisions=cfg.occupancy_subdivisions,
            building_shell_threshold=vox_params.building_shell_threshold,
            shell_anchor=vox_params.shell_anchor,
```

- [ ] **Step 4: `cli.py` — expose the mode**

After the `--occupancy-subdivisions` argument (line ~79), add:

```python
    parser.add_argument('--voxelization-mode', type=str, default='inclusive',
                        choices=['inclusive', 'tight'],
                        help="Voxelization completeness (default: inclusive). "
                             "'inclusive' keeps every voxel geometry touches - no gaps in "
                             "thin walls, right for sunlight/wind obstruction. 'tight' "
                             "restores the pre-2026-08-17 tight envelope (building shell "
                             "at >= 0.5 surface-contact occupancy, adjacency anchor).")
```

Change `--occupancy-threshold` (line ~75) to defer to the mode:

```python
    parser.add_argument('--occupancy-threshold', type=float, default=None,
                        help="Min surface-contact occupancy (0-1) for boundary voxels: the fraction "
                             "of a voxel's sub-cells a triangle passes through, not volume filled "
                             "(default: decided by --voxelization-mode; both modes use 0 = any "
                             "contact). Does not govern the building shell.")
```

And in the `VoxelizerConfig(` construction (line ~84), add:

```python
        voxelization_mode=args.voxelization_mode,
```

- [ ] **Step 5: Run the full offline suite**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q -m "not slow"`
Expected: all passed, no errors. (The slow PLATEAU integration tests run in Task 5.)

- [ ] **Step 6: Commit**

```powershell
git add voxcitygml/pipeline.py voxcitygml/pipeline_export.py voxcitygml/export_obj.py voxcitygml/cli.py
git commit -m "feat: plumb voxelization_mode through pipeline, exports, and CLI"
```

---

### Task 5: Integration verification, calibration note, spec update

The PLATEAU integration tests measure real-data roof-slope fractions against a lower bound (`MIN_ROOF_SLOPE_FRACTION = 0.03`, `tests/test_integration_plateau.py:79`). Inclusive mode keeps MORE thin roof-skin cells (the ~0.33-contact cells the 0.5 threshold dropped), so measured slope figures rise back toward their pre-2026-08-11 values — the lower bound still holds, but the calibration comment describes the tight default and must be updated.

**Files:**
- Modify: `tests/test_integration_plateau.py:45-78` (comment only)
- Modify: `docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md` (acceptance record)

- [ ] **Step 1: Run the integration suite (needs the local PLATEAU dataset)**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_integration_plateau.py tests/test_frame_contract.py -q`

Expected: all passed (several minutes), with the printed slope figures **at or above** the tabulated tight-mode values (unrotated LOD2 was 0.0741, rotated 0.0594). If the dataset directory (`D:\03_Data\citygml\plateau\13102_chuo-ku_pref_2023_citygml_2_op` or `$env:VOXCITYGML_PLATEAU_TEST_DATA`) is absent, the tests skip — note that in the completion report and still do Step 2.

If any integration test FAILS (not skips), stop and use superpowers:systematic-debugging — do not adjust `MIN_ROOF_SLOPE_FRACTION` to make it pass.

- [ ] **Step 2: Update the calibration comment**

In `tests/test_integration_plateau.py`, append to the calibration comment block (after line 78, before `MIN_ROOF_SLOPE_FRACTION = 0.03`):

```python
# 2026-08-17 inclusive-voxelization change: the production default is now
# voxelization_mode="inclusive" (shell threshold 0.0, connectivity-flood
# anchor), which KEEPS the ~0.33-contact roof-skin cells the 0.5 threshold
# dropped.  Measured slope fractions therefore sit at or above the
# tight-mode figures tabulated above, and the 0.03 lower bound holds with
# more margin, not less.  The tight figures remain valid for
# voxelization_mode="tight".
```

Record the actually-measured inclusive-mode figures from Step 1 in this comment (replace "sit at or above" prose with the numbers) if the dataset was available.

- [ ] **Step 3: Record acceptance in the design spec**

In `docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md`, change `**Status:** Approved` to `**Status:** Implemented (2026-08-17)` and append a short `## Acceptance` section listing: unit/alignment/inclusive suites green, and the integration slope figures (or "integration suite skipped — dataset unavailable").

- [ ] **Step 4: Full suite one last time**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q`
Expected: all passed (slow tests included if the dataset exists).

- [ ] **Step 5: Commit**

```powershell
git add tests/test_integration_plateau.py docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md
git commit -m "test: recalibration note and acceptance record for inclusive voxelization"
```
