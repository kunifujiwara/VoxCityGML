# Voxelizer Alignment Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Building voxelization places every voxel where the geometry actually is: grid-aligned winding SDF (no half-voxel stamp error, no per-building phase jitter), raw mesh in (no watertight resculpting), shell at occupancy 0.5 (no corner-graze dilation).

**Architecture:** New seam `_voxelize_building_solid` in `voxelizer3d.py` replaces the inline building branch; `_voxelize_meshlib_winding` gains `align_origin`; one new config field threads through `models.py` → `pipeline.py` → `voxelize_citygml_meshes`.

**Tech stack:** Python, MeshLib (`mrmeshpy`), numpy, trimesh (tests), pytest.

**Repo:** `C:\Users\kunih\OneDrive\00_Codes\python\VoxCityGML` (editable install in conda env `voxcitygml`).
**Test command prefix (conda not on PATH):**
`& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest`

**Design doc:** `docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md`

**Background for the implementer:** the diagnosis proved `_voxelize_meshlib_levelset` + `_stamp_meshlib_mask` displace every building's voxels +1 m in x/y/z (corner-sampled SDF stamped as centre-sampled), that `make_watertight_mesh`'s `doubleOffsetMesh` at 2 m resculpts surfaces by ~1 m, and that the threshold-0 shell adds 20–42 % corner-grazed cells. The winding path (`meshToDistanceVolume`) is measured exact. An aligned 12×12×10 m box must fill exactly 6×6×5 = 180 cells; today the building path fills 294.

---

### Task 1: Failing regression tests

**Files:**
- Create: `tests/test_voxelizer_alignment.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Regression tests for building voxelization alignment.

An axis-aligned box whose faces lie exactly on the 2 m grid must voxelize to
exactly its analytic cell set -- no displacement, no dilation.  These tests
pin the 2026-08-11 diagnosis: the old levelset path filled 294 cells where
the answer is 180 (corner-sampled SDF stamped as centre-sampled).
"""
import numpy as np
import pytest
import trimesh

from voxcitygml.voxelizer3d import (
    _MESHLIB_VOXEL_AVAILABLE,
    Grid3DParams,
    _voxelize_meshlib_winding,
)

# _voxelize_building_solid does not exist until Task 3; import it lazily so
# earlier tasks get test FAILURES, not a module-collection error that would
# also block the winding tests.


def building_solid(*args, **kw):
    from voxcitygml.voxelizer3d import _voxelize_building_solid
    return _voxelize_building_solid(*args, **kw)

pytestmark = pytest.mark.skipif(
    not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")

MS = 2.0


def make_gp():
    # 24 x 24 x 20 m grid; rows span [max_y - n_rows*vs, max_y] = [-6, 18].
    # min_y is deliberately NOT congruent with max_y modulo voxel_size:
    # every row computation in voxelizer3d.py anchors at max_y
    # (_stamp_meshlib_mask, Grid3DParams.box_center, _bbox_to_index_range),
    # and production grids keep max_y as the raw rectangle coordinate while
    # n_rows is rounded (_compute_grid_params_3d), so (max_y - min_y) is
    # generally not a whole number of voxels.  A snap anchored at min_y
    # passes on a congruent grid but leaves a half-voxel y-phase error in
    # production — this grid makes that bug fail the tests.
    return Grid3DParams(n_rows=12, n_cols=12, n_z=10,
                        min_x=-6.0, max_x=18.0, min_y=-6.9, max_y=18.0,
                        min_z=-6.0, max_z=14.0, voxel_size=MS)


def box_mesh(dx=0.0, dy=0.0, dz=0.0, extents=(12.0, 12.0, 10.0)):
    b = trimesh.creation.box(extents=list(extents))
    b.apply_translation([extents[0] / 2 + dx, extents[1] / 2 + dy,
                         extents[2] / 2 + dz])
    return np.asarray(b.vertices, float), np.asarray(b.faces)


def filled(grid):
    return set(zip(*np.nonzero(grid == -3)))


def expected_box_cells(gp, dx=0.0, dy=0.0, dz=0.0, extents=(12.0, 12.0, 10.0)):
    """Centre-inside cells of the translated box, in (row, col, z) indices."""
    out = set()
    for row in range(gp.n_rows):
        for col in range(gp.n_cols):
            for zi in range(gp.n_z):
                x = gp.min_x + (col + 0.5) * MS
                y = gp.max_y - (row + 0.5) * MS
                z = gp.min_z + (zi + 0.5) * MS
                if (dx <= x <= dx + extents[0] and dy <= y <= dy + extents[1]
                        and dz <= z <= dz + extents[2]):
                    out.add((row, col, zi))
    return out


@pytest.mark.parametrize("off", [0.0, 0.7, 1.3])
def test_winding_aligned_box_exact(off):
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh(dx=off, dy=off)
    ok = _voxelize_meshlib_winding(v, f, gp, grid, -3, True, align_origin=True)
    assert ok
    assert filled(grid) == expected_box_cells(gp, dx=off, dy=off)


@pytest.mark.parametrize("off", [0.0, 0.7, 1.3])
def test_building_path_box_exact(off):
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh(dx=off, dy=off)
    building_solid(v, f, gp, grid, -3, True,
                   occupancy_threshold=0.0,
                   occupancy_subdivisions=3,
                   shell_threshold=0.5)
    got = filled(grid)
    want = expected_box_cells(gp, dx=off, dy=off)
    # The aligned winding fill supplies every centre-inside cell.  (Volume
    # overlap >= 0.5 implies centre-inside, but NOT the converse: a corner
    # cell can be centre-inside at only ~0.42 overlap — 0.65 x 0.65 at
    # off=0.7 — which is why the >= 0.5 bound below is applied to the
    # EXTRA cells only, never demanded of `want`.)  The shell
    # measures SURFACE-CONTACT occupancy — the fraction of 3x3x3 sub-cells
    # touching a triangle — so a lone flat face scores ~1/3 and is dropped
    # at threshold 0.5; only multi-face cells (corners/edges) can survive,
    # and those are centre-inside anyway.  The envelope bound still holds:
    # never lose a centre-inside cell, never add a cell the box covers by
    # less than half its volume.
    assert want <= got
    for cell in got - want:
        row, col, zi = cell
        # every extra cell must overlap the box by >= 0.5 along each axis
        x0 = gp.min_x + col * MS
        y0 = gp.max_y - (row + 1) * MS
        z0 = gp.min_z + zi * MS
        fx = max(0.0, min(x0 + MS, off + 12.0) - max(x0, off)) / MS
        fy = max(0.0, min(y0 + MS, off + 12.0) - max(y0, off)) / MS
        fz = max(0.0, min(z0 + MS, 10.0) - max(z0, 0.0)) / MS
        assert fx * fy * fz >= 0.5 - 1e-9, (
            f"cell {cell} kept with overlap {fx*fy*fz:.2f} < 0.5")


def test_aligned_box_no_dilation():
    """The historic failure: aligned box produced 294 cells, one extra layer
    on every +x/+y/+z face.  Exactly 180, exactly placed."""
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh()
    building_solid(v, f, gp, grid, -3, True,
                   occupancy_threshold=0.0,
                   occupancy_subdivisions=3,
                   shell_threshold=0.5)
    assert filled(grid) == expected_box_cells(gp)
    assert len(filled(grid)) == 180


def test_shell_threshold_discriminates_surface_contact():
    """The shell keeps a cell only when enough sub-cells TOUCH geometry.

    A 0.3 m sliver crosses one sub-slab of its cell (contact 9/27 = 0.33):
    kept at threshold 0.0, dropped at 0.5.  The shell's 6-connected anchor
    guard needs a neighbouring filled voxel (voxelizer3d.py, `_dilate6`),
    so the layer below is pre-filled — without an anchor the sliver
    disappears at ANY threshold and the test would prove nothing.
    """
    v, f = box_mesh(extents=(12.0, 12.0, 0.3))
    for thr, expect_cells in ((0.0, True), (0.5, False)):
        gp = make_gp()
        grid = np.zeros((12, 12, 10), np.int32)
        grid[:, :, 2] = -1                 # anchor layer under the sliver
        building_solid(v, f, gp, grid, -3, True, shell_threshold=thr)
        assert (len(filled(grid)) > 0) == expect_cells, f"threshold {thr}"
    # A >= half-voxel slab (1.2 m) survives threshold 0.5 regardless: its
    # centre-inside cells come from the winding fill, not the shell.
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v2, f2 = box_mesh(extents=(12.0, 12.0, 1.2))
    building_solid(v2, f2, gp, grid, -3, True, shell_threshold=0.5)
    assert len(filled(grid)) > 0
```

- [ ] **Step 2: Run and verify they fail for the right reason**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_voxelizer_alignment.py -q`
Expected: ALL tests fail — the three `winding_aligned` tests with `TypeError: ... unexpected keyword argument 'align_origin'`, the building-path tests with `ImportError: cannot import name '_voxelize_building_solid'` raised inside the lazy `building_solid` helper. No collection error (the lazy import exists precisely so the file still collects).

- [ ] **Step 3: Commit**

```bash
git add tests/test_voxelizer_alignment.py
git commit -m "test: pin exact building voxelization on aligned/offset boxes (red)"
```

---

### Task 2: `align_origin` for the winding path

**Files:**
- Modify: `voxcitygml/voxelizer3d.py` — `_voxelize_meshlib_winding` (line ~601)

- [ ] **Step 1: Add the parameter and the snap**

Replace the function header and the origin/dimension setup (keep the rest of the body unchanged):

```python
def _voxelize_meshlib_winding(
    verts: np.ndarray,
    faces: np.ndarray,
    gp: Grid3DParams,
    voxel_grid: np.ndarray,
    class_code: int,
    overwrite: bool,
    align_origin: bool = False,
) -> bool:
    """Voxelize a mesh (possibly open) via MeshLib's Generalized Winding Number.

    ``meshToDistanceVolume`` with ``SignDetectionMode.HoleWindingRule``
    robustly classifies inside/outside even for meshes with holes, gaps,
    or self-intersections.

    When ``align_origin`` is True the SDF lattice origin is snapped to the
    main voxel-grid lattice, so SDF cell centres coincide exactly with main
    grid cell centres and ``_stamp_meshlib_mask`` becomes a phase-free
    integer copy.  The snap anchors are per-axis — ``(min_x, max_y, min_z)``
    — because those are the anchors the stamp itself uses (rows count down
    from ``max_y``, and ``max_y - min_y`` is generally not a whole number of
    voxels, so ``min_y`` is NOT on the row lattice).  Without the snap, each
    mesh's own bounding box sets the lattice phase and the stamped result
    can land up to half a voxel away (2026-08-11 diagnosis).

    Returns True on success.
    """
    try:
        ml_mesh, shift = _meshlib_mesh_from_numpy(verts, faces)
        vs = float(gp.voxel_size)

        box = ml_mesh.computeBoundingBox()
        expansion = _mr.Vector3f.diagonal(3 * vs)

        origin_local = box.min - expansion
        if align_origin:
            # Snap in world coordinates, then convert back to the shifted
            # MeshLib frame.  floor() moves the origin down by < 1 voxel,
            # covered by the +2 dimension margin below.
            #
            # Anchor per axis to the lattice the stamp actually indexes
            # against: cols from min_x, z from min_z, but ROWS from max_y
            # (_stamp_meshlib_mask: row = (max_y - y)/vs).  min_y must NOT
            # be used for y — production grids keep max_y raw while n_rows
            # is rounded (_compute_grid_params_3d), so max_y - min_y is
            # generally not a multiple of vs and the min_y lattice is out
            # of phase with the row lattice by up to half a voxel.
            anchor = np.array([gp.min_x, gp.max_y, gp.min_z],
                              dtype=np.float64)
            world = np.array(
                [origin_local.x, origin_local.y, origin_local.z],
                dtype=np.float64) + shift
            snapped = anchor + np.floor((world - anchor) / vs) * vs
            local = snapped - shift
            origin_local = _mr.Vector3f(
                float(local[0]), float(local[1]), float(local[2]))

        params = _mr.MeshToDistanceVolumeParams()
        params.vol.origin = origin_local
        params.vol.voxelSize = _mr.Vector3f.diagonal(vs)
        dim_f = (box.max + expansion - origin_local) / vs
        params.vol.dimensions = _mr.Vector3i(
            int(dim_f.x) + 2, int(dim_f.y) + 2, int(dim_f.z) + 2,
        )
        params.dist.signMode = _mr.SignDetectionMode.HoleWindingRule
        params.dist.maxDistSq = (3 * vs) ** 2
```

(The remainder — `meshToDistanceVolume`, origin recovery, `inside_mask`, `binary_fill_holes`, `_stamp_meshlib_mask`, `return True/except` — is unchanged.)

Note: the dimension margin changes from `+1` to `+2` **for all callers**, including the vegetation and terrain-fallback winding calls that keep `align_origin=False`. This is intentional — one extra empty SDF plane per axis, negligible cost — and simpler than making the margin conditional on the snap.

- [ ] **Step 2: Run the winding tests**

Run (Git-Bash; `tail` is not a PowerShell command): `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_voxelizer_alignment.py -q -k winding_aligned 2>&1 | tail -5`
Expected: 3 passed. (The building-path tests still fail on the seam's lazy ImportError until Task 3 — run with `-k`, do not weaken them.)

- [ ] **Step 3: Commit**

```bash
git add voxcitygml/voxelizer3d.py
git commit -m "feat: grid-aligned SDF lattice option for winding voxelization"
```

---

### Task 3: Building seam — winding-aligned, raw mesh, shell threshold

**Files:**
- Modify: `voxcitygml/voxelizer3d.py` — add `_voxelize_building_solid` above `_voxelize_mesh_group` (line ~773); rewrite the building branch inside `_voxelize_mesh_group` (lines ~819–871); add a warning line to `_voxelize_meshlib_levelset`'s docstring (line ~546)

- [ ] **Step 1: Add the seam function**

Insert immediately before `def _voxelize_mesh_group(`:

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
    shell_threshold: float = 0.5,
) -> None:
    """Voxelize one building solid.

    MeshLib path: grid-aligned winding SDF on the RAW mesh
    (``HoleWindingRule`` signs open / self-intersecting shells robustly),
    then the raw-mesh surface shell at ``shell_threshold`` occupancy.
    ``make_watertight_mesh`` is deliberately NOT used here: repairing at grid
    resolution resculpted surfaces by ~1 m and the old levelset stamping
    displaced every building by +half a voxel (2026-08-11 diagnosis,
    docs/superpowers/specs/2026-08-11-voxelizer-alignment-fix-design.md).

    Fallback (meshlib missing or winding failed): the legacy watertight →
    occupancy → sealed-surface path, unchanged.
    """
    if _MESHLIB_VOXEL_AVAILABLE:
        ok = _voxelize_meshlib_winding(
            verts, faces, gp, voxel_grid, class_code, overwrite,
            align_origin=True,
        )
        if ok:
            _overlay_surface_shell(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                occupancy_threshold=shell_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
            )
            return

    wt = make_watertight_mesh(verts, faces, voxel_size=gp.voxel_size)
    if wt.is_watertight and len(wt.faces) > 0 and len(wt.vertices) > 0:
        ok = _voxelize_by_occupancy(
            wt.vertices, wt.faces, gp, voxel_grid, class_code, overwrite,
            occupancy_threshold=occupancy_threshold,
            occupancy_subdivisions=occupancy_subdivisions,
        )
        if ok:
            return
    _voxelize_single_mesh(
        verts, faces, gp, voxel_grid, class_code, overwrite,
        seal_surface=True,
        occupancy_threshold=occupancy_threshold,
        occupancy_subdivisions=occupancy_subdivisions,
    )
```

- [ ] **Step 2: Rewire the building branch**

In `_voxelize_mesh_group`, replace the `if class_code == BUILDING_CODE and not force_surface:` block with the snippet below. **Preserve the bridge-rationale comment block** (`# ── Bridges (force_surface) ──…` lines) that sits between this block and the `elif class_code == BUILDING_CODE and force_surface:` — it documents the elif, not the block being replaced.

```python
        # ── Buildings (solid) ─────────────────────────────────────────
        if class_code == BUILDING_CODE and not force_surface:
            _voxelize_building_solid(
                verts, faces, gp, voxel_grid, class_code, overwrite,
                occupancy_threshold=occupancy_threshold,
                occupancy_subdivisions=occupancy_subdivisions,
                shell_threshold=shell_threshold,
            )
```

Add `shell_threshold: float = 0.5,` to `_voxelize_mesh_group`'s signature, directly after `occupancy_subdivisions: int = 3,`.

- [ ] **Step 3: Mark the levelset path**

Append to `_voxelize_meshlib_levelset`'s docstring (before the closing `"""`):

```
    .. warning:: Not used for buildings since 2026-08-11: ``meshToVolume``'s
       SDF is corner-sampled, but ``_stamp_meshlib_mask`` assumes centre
       samples, displacing the fill by +half a voxel per axis.  Fix the
       convention before reusing this for solid stamping.  Note the terrain
       path pre-shifts its solid by -0.5 voxel (see ``_voxelize_terrain_solid``
       around voxelizer3d.py:446-450) to compensate for this same bias — a
       future fix of the stamp convention must remove that compensation in
       the same change, or terrain will double-correct.

       Adjacent defect, also out of scope here: ``_stamp_meshlib_mask``
       truncates indices with ``astype(np.intp)`` instead of flooring, so
       SDF cells less than one voxel outside the grid on the min_x / min_z /
       max_y sides produce index fractions in (-1, 0) that truncate to 0 and
       pass the ``>= 0`` validity check — folding out-of-grid cells onto the
       edge row/col/layer.  With the grid-aligned lattice those fractions
       become exactly -0.5, so the folding is now deterministic instead of
       phase-dependent.  Eventual fix: ``np.floor`` before the cast.
```

- [ ] **Step 4: Run the alignment tests**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_voxelizer_alignment.py -q 2>&1 | tail -5`
Expected: all tests pass (winding 3, building path 3, no-dilation 1, surface-contact threshold 1 = 8 passed).

- [ ] **Step 5: Commit**

```bash
git add voxcitygml/voxelizer3d.py
git commit -m "fix: buildings voxelize via grid-aligned winding on raw mesh, shell at 0.5"
```

---

### Task 4: Config plumbing

**Files:**
- Modify: `voxcitygml/models.py` — `VoxelizerConfig` (field block ~line 248, docstring ~line 190)
- Modify: `voxcitygml/voxelizer3d.py` — `voxelize_citygml_meshes` signature (~line 133) and its building `_voxelize_mesh_group` call
- Modify: `voxcitygml/pipeline.py` — call site (~line 402)

- [ ] **Step 1: Add the config field**

In `models.py`, directly after the line `occupancy_subdivisions: int = 3`, add:

```python
    building_shell_threshold: float = 0.5
```

and in the class docstring, after the `occupancy_subdivisions:` entry, add:

```
        building_shell_threshold: Minimum SURFACE-CONTACT occupancy for the
            building surface-shell overlay (default 0.5): the fraction of a
            boundary voxel's 3x3x3 sub-cells that touch building geometry.
            This is NOT volume overlap — a lone flat face crossing a voxel
            scores ~0.33 and is dropped at 0.5; two crossing faces or a slab
            spanning two sub-slabs score >= 0.5 and are kept.  The interior
            fill independently keeps every centre-inside cell, so the shell
            only decides thin-feature and edge cells.  At 0 the shell keeps
            every corner-grazed cell and visibly inflates the envelope
            (2026-08-11 diagnosis).
```

Note: the pre-existing `occupancy_threshold` docstrings (models.py:188,
voxelizer3d.py:150) describe the same filter as "volume overlap fraction",
which is inaccurate for the same reason. Correcting them in this commit is
optional but encouraged; the NEW text above must not repeat the error.

- [ ] **Step 2: Thread through voxelize_citygml_meshes**

In `voxelize_citygml_meshes`'s signature, after `occupancy_subdivisions: int = 3,`, add:

```python
    building_shell_threshold: float = 0.5,
```

and pass `shell_threshold=building_shell_threshold,` in the `_voxelize_mesh_group` call that voxelizes buildings (the call passing `BUILDING_CODE` without `force_surface=True`; bridges and vegetation calls are untouched).

- [ ] **Step 3: Pipeline call site**

In `pipeline.py`, after the line `occupancy_subdivisions=cfg.occupancy_subdivisions,`, add:

```python
            building_shell_threshold=cfg.building_shell_threshold,
```

- [ ] **Step 4: Config test**

Append to `tests/test_voxelizer_alignment.py`:

```python
def test_building_shell_threshold_reaches_shell(monkeypatch):
    """The config value must arrive at _overlay_surface_shell for buildings.

    Covers seam -> shell only.  The upper plumbing (VoxelizerConfig ->
    voxelize_citygml_meshes -> _voxelize_mesh_group) is left to review, as
    exercising it needs a full CityGMLMeshCollection fixture; the three
    edits are single-line pass-throughs.
    """
    import voxcitygml.voxelizer3d as vx

    seen = {}
    real = vx._overlay_surface_shell

    def spy(verts, faces, gp, grid, code, overwrite, **kw):
        seen["occupancy_threshold"] = kw.get("occupancy_threshold")
        return real(verts, faces, gp, grid, code, overwrite, **kw)

    monkeypatch.setattr(vx, "_overlay_surface_shell", spy)
    gp = make_gp()
    grid = np.zeros((12, 12, 10), np.int32)
    v, f = box_mesh()
    vx._voxelize_building_solid(v, f, gp, grid, -3, True,
                                shell_threshold=0.31)
    assert seen["occupancy_threshold"] == 0.31
```

- [ ] **Step 5: Run and commit**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests/test_voxelizer_alignment.py -q 2>&1 | tail -3`
Expected: 9 passed.

```bash
git add voxcitygml/models.py voxcitygml/voxelizer3d.py voxcitygml/pipeline.py tests/test_voxelizer_alignment.py
git commit -m "feat: expose building_shell_threshold in VoxelizerConfig"
```

---

### Task 5: Full suite and real-building acceptance

**Files:**
- Create: `scripts/diagnose_voxelization2.py` — copied **verbatim** from the CityMesher
  session scratchpad; no source changes otherwise.

- [ ] **Step 0: Vendor the diagnosis script into the repo**

The script currently lives only in another session's Temp scratchpad, which gets
cleaned; the acceptance below must stay reproducible.  Copy it verbatim (it may
reference absolute data paths from the CityMesher session — leave those alone):

```bash
mkdir -p scripts
cp "/c/Users/kunih/AppData/Local/Temp/claude/c--Users-kunih-OneDrive-00-Codes-python-CityMesher/c28d7136-0bdf-4af2-a57f-cf6510459177/scratchpad/diagnose_voxelization2.py" scripts/
git add scripts/diagnose_voxelization2.py
git commit -m "chore: vendor voxelization diagnosis script for acceptance"
```

- [ ] **Step 1: Full VoxCityGML test suite**

Run (Git-Bash): `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcitygml python -m pytest tests -q 2>&1 | tail -5`
Expected: no new failures relative to the pre-change baseline (record the baseline first with `git stash` if unsure; data-dependent suites like munich/plateau may already skip).

Known blast radius beyond the main pipeline — NOT regressions: `export_obj.py:1048` (fallback per-category OBJ export) also reaches the building branch of `_voxelize_mesh_group`, so exported building voxels switch to the new seam too (desired: exports stay consistent with the main grid); `tests/test_profile.py` only prints timings. Any diff in export-related tests should be read in that light before being treated as a failure.

- [ ] **Step 2: Diagnosis Part C reruns clean**

Run `scripts/diagnose_voxelization2.py` unchanged (it calls the patched internals). The Part C reproduction must now report, for all six buildings:
- `fill shift` = `(+0.00, +0.00, +0.00)`
- levelset→(now winding) IoU vs truth ≥ 0.90 with no improvement from shifting
- `shell +` cells ≤ 10 % of the fill

Note: Part B (stored pickle) will still show the OLD shifts — the pickle predates the fix; only Part C reflects patched code.

- [ ] **Step 3: Commit acceptance note**

```bash
git add docs/superpowers/plans/2026-08-11-voxelizer-alignment-fix.md
git commit -m "docs: record voxelizer alignment acceptance results"
```

(Record the Part C numbers in this file under a `## Acceptance results` heading.)

---

## Acceptance results (2026-08-11)

### Correction to Step 2 as originally written

The plan said to run `diagnose_voxelization2.py` "unchanged (it calls the
patched internals)". **That was wrong.** Part C hardcodes the *old* pipeline —
`make_watertight_mesh` → `_voxelize_meshlib_levelset` → shell at threshold 0 —
and never touches `_voxelize_building_solid` or `align_origin`. Run unchanged,
it reproduces the pre-fix numbers exactly and shows the fix doing nothing.

The script was therefore vendored to `scripts/diagnose_voxelization2.py` and a
**Part E** was added that runs the real new seam on the same buildings, on the
same lattice, against the same truth oracle. Part C is retained unmodified as
the "before" reference, so C and E are directly comparable. Data artefacts stay
outside the repo; point `VOXCITYGML_DIAG_DATA` at the directory holding
`plateau_raw.pkl` and `kanda_city.pkl`.

### Six tall interior Kanda buildings, 2 m voxels

| building | old IoU (C) | **new IoU (E)** | old shell + | **new shell +** |
|---|---|---|---|---|
| dbff04d5 | 0.809 | **0.973** | 1092 / 4376 = 25.0 % | **51 / 4489 = 1.1 %** |
| 03bcf44f | 0.592 | **0.942** | 272 / 1111 = 24.5 % | **49 / 1090 = 4.5 %** |
| 2e623427 | 0.617 | **0.914** | 481 / 2350 = 20.5 % | **5 / 2321 = 0.2 %** |
| a64f70dc | 0.679 | **0.977** | 409 / 970 = 42.2 % | **22 / 997 = 2.2 %** |
| d738e6a5 | 0.664 | **0.958** | 370 / 1829 = 20.2 % | **4 / 1897 = 0.2 %** |
| 518b754b | 0.559 | **0.980** | 221 / 878 = 25.2 % | **6 / 906 = 0.7 %** |

Against the three acceptance criteria:

- **IoU ≥ 0.90 — MET.** 0.914–0.980, up from 0.559–0.809. Fill accuracy is the
  headline result: the worst new building beats the best old one by 0.11.
- **Shell adds ≤ 10 % of fill — MET.** 0.2–4.5 %, down from 20.2–42.2 %. The
  old spread reproduces the diagnosis's "20–42 %" figure exactly, which is a
  good check that Part C really is the old path.
- **Fill shift = (0, 0, 0) — MET IN SUBSTANCE, NOT LITERALLY.** Old best-fit
  shifts were systematically `(+1.00, +1.00, ·)` on every building — a clean
  half-voxel displacement in both x and y, the diagnosed bug. New best-fit
  shifts are scattered with no common direction (x ∈ {0.00, 0.25};
  y ∈ {−0.75, −0.50, 0.00, +0.25}; z ∈ {−0.75, −0.50, 0.00, +0.25}) and the
  IoU they buy is mostly noise (median gain ≈ 0.005; worst case 0.072 on
  03bcf44f). The literal criterion was too strict to be measurable: the truth
  oracle is itself a 0.25 m lattice whose `contains()` quantizes query points,
  so ±0.25 m is the measurement's own resolution floor. **Systematic
  displacement is eliminated; what remains is at the noise floor of the
  instrument.** A future tightening would need a finer oracle (`FINE = 0.05`)
  to resolve further, which was not run.

Also confirmed in passing: **M2 is real.** The old path's watertight step moved
surfaces by p95 = 0.11–0.97 m (`meshlib_double_offset` was selected for three of
the six buildings, and those are exactly the three with the largest deviation).
The new seam skips watertighting entirely on the MeshLib path.

### Test suite

`222 passed, 1 skipped, 1 deselected` — the 213-test pre-change baseline, plus
the 9 new alignment tests. Two PLATEAU roof-slope assertions were recalibrated
in `99113a9`; see that commit for the measured LOD2/LOD1 figures and why the
metric's absolute scale moved.

---

### Task 6 (follow-up, needs user go-ahead — GPU time): regenerate downstream

- [ ] Re-voxelize the Kanda tile with patched voxcitygml → new `kanda_city.pkl`
- [ ] Re-run `plateau_combo.py` (annual cumulative solar) and `finalize.py`
- [ ] Expect: z-residual optimum ≈ 0 beyond the pure per-column datum; roof-top excess mode 0; projection match at 1.5× meshsize radius materially above the old 80 %; east/west facade ratio drops toward the weather-only 1.07 (solver diffuse bias remains)
- [ ] Update the two published artifacts with corrected numbers
