# Voxel Frame Cell-Count Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 3-D voxel grid take its row/column counts from the canonical 2-D grid frame, so `run_core` can never pair two differently-sized grids and fail LOD2 generation.

**Architecture:** `run_core` builds two frames from the same rectangle. Every 2-D rasteriser (`buildings/processor.py:178`, `canopy/processor.py:158`, `landcover/citygml_landcover.py:297`, `terrain/processor.py:54`, `pipeline.py:329`) calls the same pure `compute_grid_params`, which sizes cells from the **geodesic** side length. Only `voxelizer3d._compute_grid_params_3d` is the odd one out: it sizes from the **projected bbox** of the rectangle. Both round with `int(x + 0.5)`, so when a side lands within floating-point reach of a half cell they round to different integers and `fill_building_id_gaps` — which deliberately refuses a mismatch rather than mis-attributing every column — raises. The fix deletes the projected-bbox sizing: `_compute_grid_params_3d` derives its counts from `compute_grid_params` by default, and accepts an explicit `grid_shape` override for callers that work in a synthetic local frame.

**Tech Stack:** Python 3.10, NumPy, pyproj, pytest, Docker.

## Global Constraints

- Only `n_rows` and `n_cols` change. **`min_x`, `max_x`, `min_y`, `max_y`, `min_z`, `max_z` must keep their current values.** Every row/col/z index in `voxelizer3d.py` anchors at `(min_x, max_y, min_z)` — see `Grid3DParams.xyz_to_indices`, `box_center`, `_bbox_to_index_range`, `_stamp_meshlib_mask`. Rebasing the bounds to make the extent a whole number of voxels would re-register the entire grid by up to half a voxel, and `tests/test_voxelizer_alignment.py::make_gp` pins deliberately that `max_y - min_y` is *not* a whole number of voxels.
- **The projected-bbox rounding `max(1, int((max_x - min_x) / meshsize + 0.5))` must not survive anywhere as a fallback.** A conditional that keeps it alive for some inputs re-introduces exactly the drift this task removes. The only permitted escape is an explicit caller-supplied `grid_shape`.
- `n_z` keeps its existing `max(1, int((z_max - z_min) / meshsize + 0.5))` rounding. There is no 2-D counterpart to unify it with.
- `_compute_grid_params_3d` has three call sites besides the pipeline: `export_obj.py:908` and `export_obj.py:1004`, plus `tests/test_terrain_building_contact.py`. The two `export_obj` sites recompute the frame for a voxel grid they are handed, so **they must end up with the same counts as that grid** — they therefore take the new canonical default and must NOT be given a `grid_shape`.
- No new dependency. `voxelizer3d.py` already imports from `.grid_utils` (line 23), and `grid_utils` does not import `voxelizer3d`, so there is no cycle.
- Do not change `voxcitygml/reapply.py`'s synthetic `Grid3DParams`.
- Baseline to preserve: `372 passed, 2 skipped, 12 deselected`.

## Test Runner

The host Python cannot import `voxcitygml` (missing `mapbox-earcut`). Tests run in a Docker image that layers `pytest` onto the app image, **already built** as `voxcitygml-test:latest`. Rebuild only if `docker image inspect voxcitygml-test:latest` fails:

```bash
cd /work-misc/code/VoxCityGML
docker build -t voxcitygml-test:latest -f - . <<'DOCKERFILE'
FROM optree-voxcity-local:latest
RUN python -m pip install --no-cache-dir pytest
ENTRYPOINT ["python", "-m", "pytest"]
DOCKERFILE
```

Mounting the checkout at `/src` with `-w /src` makes `import voxcitygml` resolve to the working tree, not site-packages.

## File Structure

- `voxcitygml/voxelizer3d.py` — the fix: the `.grid_utils` import, a new `grid_shape` parameter on `_compute_grid_params_3d` and on `voxelize_citygml_meshes`, and deletion of the projected-bbox rounding.
- `voxcitygml/pipeline.py` — `run_core` computes the canonical frame once and hands its shape to the voxeliser.
- `tests/test_grid_rotation.py` — regression tests for the frame agreement and the override.
- `tests/test_terrain_building_contact.py` — its identity-transformer fixture works in a synthetic local frame and must now say so explicitly.

---

### Task 1: Derive the 3-D frame's cell counts from the canonical 2-D frame

**Files:**
- Modify: `voxcitygml/voxelizer3d.py` (import line 23; `voxelize_citygml_meshes` signature and its `_compute_grid_params_3d` call at line 257; `_compute_grid_params_3d` signature at line 388 and its cell-count block at lines 444-446)
- Modify: `voxcitygml/pipeline.py` (`run_core`, at the `voxelize_citygml_meshes` call, line 525)
- Modify: `tests/test_terrain_building_contact.py` (the two calls at lines 341 and 354)
- Test: `tests/test_grid_rotation.py` (append at end of file)

**Interfaces:**
- Consumes: `compute_grid_params(rectangle_vertices, meshsize) -> GridParams`, with `.n_rows: int`, `.n_cols: int` and a `.shape -> Tuple[int, int]` property returning `(n_rows, n_cols)`. Already imported in `pipeline.py:39` and in `tests/test_grid_rotation.py:8`.
- Consumes: `geodesic_rect(center_lon, center_lat, width_m, height_m, rotation_deg) -> [SW, NW, NE, SE]`, aliased as `_geodesic_rect` at `tests/test_grid_rotation.py:11`.
- Produces: `_compute_grid_params_3d(rectangle_vertices, center_lon, center_lat, meshsize, collection, underground_depth=0.0, dem_grid=None, *, grid_shape: Optional[Tuple[int, int]] = None)`. When `grid_shape` is `None` the counts come from `compute_grid_params`; when supplied they are used verbatim.
- Produces: `voxelize_citygml_meshes(..., *, info_out=None, grid_shape: Optional[Tuple[int, int]] = None)` — forwards `grid_shape` unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grid_rotation.py`:

```python
# ---------------------------------------------------------------------
# The 3-D voxel frame must size to the same cells as the 2-D frame
# ---------------------------------------------------------------------

class _NoMeshes:
    """Empty collection: `_compute_grid_params_3d` only reads the four mesh
    lists to find the z range, and an empty one takes the documented
    z_min=0.0 / z_max=meshsize fallback."""
    terrain = buildings = bridges = vegetation = ()


def test_voxel_frame_adopts_the_2d_cell_counts_at_a_half_cell_boundary():
    """`run_core` pairs these two grids cell-for-cell, and
    `fill_building_id_gaps` refuses a mismatch rather than mis-attributing
    every column, so a disagreement fails LOD2 generation outright.

    250.52 m x 250.00 m at 1 m is a measured trigger: the geodesic side
    length `compute_grid_params` sizes from has already crossed 250.5 while
    the projected bbox the voxeliser used to measure had not, so the frames
    sized 251 and 250 columns for one rectangle.
    """
    from voxcitygml.voxelizer3d import _compute_grid_params_3d

    clon, clat = 139.765, 35.681
    rect = _geodesic_rect(clon, clat, 250.52, 250.0, 0.0)
    gp2d = compute_grid_params(rect, 1.0)
    gp3d, _ = _compute_grid_params_3d(
        rect, clon, clat, 1.0, _NoMeshes(), 0.0, None)
    assert (gp3d.n_rows, gp3d.n_cols) == (gp2d.n_rows, gp2d.n_cols)


def test_voxel_frame_matches_the_2d_frame_across_a_size_sweep():
    """One boundary is an anecdote. Sweeping a half-cell crossing in 2 cm
    steps catches the whole band where the two measurements straddle the
    rounding threshold; before the fix this band is 2-4% of all sizes."""
    from voxcitygml.voxelizer3d import _compute_grid_params_3d

    clon, clat = 139.765, 35.681
    bad = []
    for i in range(60):
        width = 250.0 + i * 0.02
        rect = _geodesic_rect(clon, clat, width, 250.0, 0.0)
        gp2d = compute_grid_params(rect, 1.0)
        gp3d, _ = _compute_grid_params_3d(
            rect, clon, clat, 1.0, _NoMeshes(), 0.0, None)
        if (gp3d.n_rows, gp3d.n_cols) != (gp2d.n_rows, gp2d.n_cols):
            bad.append((round(width, 2), (gp2d.n_rows, gp2d.n_cols),
                        (gp3d.n_rows, gp3d.n_cols)))
    assert not bad, f"frames disagreed at {bad}"


def test_voxel_frame_keeps_the_raw_projected_bbox_as_its_lattice_anchor():
    """Adopting the 2-D counts must not rebase the bounds.

    Rows index as `(max_y - y) / voxel_size` and columns as
    `(x - min_x) / voxel_size` (`Grid3DParams.xyz_to_indices`), so `min_x`
    and `max_y` are the lattice origin, not free metadata: shifting them to
    make the extent a whole number of voxels would move every voxel by up
    to half a cell.
    """
    from voxcitygml.citygml.coordinates import create_rectangle_frame_transformer
    from voxcitygml.voxelizer3d import _compute_grid_params_3d

    clon, clat = 139.765, 35.681
    rect = _geodesic_rect(clon, clat, 250.52, 250.0, 0.0)
    gp3d, _ = _compute_grid_params_3d(
        rect, clon, clat, 1.0, _NoMeshes(), 0.0, None)

    transformer = create_rectangle_frame_transformer(clon, clat, rect)
    rx, ry = transformer.transform([v[0] for v in rect], [v[1] for v in rect])
    assert gp3d.min_x == min(rx)
    assert gp3d.max_y == max(ry)


def test_voxel_frame_uses_an_explicit_grid_shape_verbatim():
    """The escape hatch for callers working in a synthetic local frame.

    `tests/test_terrain_building_contact.py` drives the voxeliser through an
    IdentityTransformer on `[0, 40]^2` local-metre vertices, which are not
    lon/lat and have no geodesic side length. Such a caller states its own
    cell counts; the default geodesic derivation would read those vertices
    as degrees.
    """
    from voxcitygml.voxelizer3d import _compute_grid_params_3d

    clon, clat = 139.765, 35.681
    rect = _geodesic_rect(clon, clat, 250.52, 250.0, 0.0)
    gp3d, _ = _compute_grid_params_3d(
        rect, clon, clat, 1.0, _NoMeshes(), 0.0, None, grid_shape=(7, 9))
    assert (gp3d.n_rows, gp3d.n_cols) == (7, 9)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /work-misc/code/VoxCityGML
docker run --rm -v /work-misc/code/VoxCityGML:/src -w /src voxcitygml-test:latest \
  tests/test_grid_rotation.py -k "voxel_frame" -q
```

Expected: failures. `..._at_a_half_cell_boundary` and `..._across_a_size_sweep` fail on the count mismatch; `..._uses_an_explicit_grid_shape_verbatim` fails with `TypeError: _compute_grid_params_3d() got an unexpected keyword argument 'grid_shape'`; `..._lattice_anchor` passes already (it guards the fix, it does not drive it). Report the real output; do not edit tests to match a prediction.

- [ ] **Step 3: Add `compute_grid_params` to the existing grid_utils import**

In `voxcitygml/voxelizer3d.py`, line 23:

```python
from .grid_utils import check_non_degenerate, compute_grid_params
```

- [ ] **Step 4: Add the `grid_shape` parameter to `_compute_grid_params_3d`**

Change the signature at `voxcitygml/voxelizer3d.py:388` to:

```python
def _compute_grid_params_3d(
    rectangle_vertices: List[Tuple[float, float]],
    center_lon: float,
    center_lat: float,
    meshsize: float,
    collection: CityGMLMeshCollection,
    underground_depth: float = 0.0,
    dem_grid: Optional[np.ndarray] = None,
    *,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Grid3DParams, object]:
```

- [ ] **Step 5: Replace the projected-bbox rounding**

Replace lines 444-446:

```python
    n_cols = max(1, int((max_x - min_x) / meshsize + 0.5))
    n_rows = max(1, int((max_y - min_y) / meshsize + 0.5))
    n_z = max(1, int((z_max - z_min) / meshsize + 0.5))
```

with:

```python
    # Cell counts come from the canonical 2-D frame, never from the bbox
    # above. Both measure the same side -- geodesic length there, projected
    # bbox here -- so a side within floating-point reach of a half cell
    # rounds up in one frame and down in the other, and run_core then pairs
    # the two grids (fill_building_id_gaps refuses a mismatch). A caller
    # working in a synthetic local frame, whose vertices are not lon/lat,
    # supplies grid_shape instead. The bounds stay raw either way: rows
    # index off max_y, columns off min_x.
    if grid_shape is None:
        n_rows, n_cols = compute_grid_params(rectangle_vertices, meshsize).shape
    else:
        n_rows, n_cols = grid_shape
    n_z = max(1, int((z_max - z_min) / meshsize + 0.5))
```

- [ ] **Step 6: Forward `grid_shape` through `voxelize_citygml_meshes`**

In `voxcitygml/voxelizer3d.py`, add the parameter to the keyword-only block of `voxelize_citygml_meshes` (which currently ends `*, info_out: Optional[dict] = None,`):

```python
    *,
    info_out: Optional[dict] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
```

and forward it in the `_compute_grid_params_3d` call at line 257, keeping the existing arguments unchanged:

```python
    gp, transformer = _compute_grid_params_3d(
        rectangle_vertices,
        center_lon,
        center_lat,
        meshsize,
        collection,
        underground_depth,
        dem_grid,
        grid_shape=grid_shape,
    )
```

Match the existing call's argument style — if it passes `underground_depth`/`dem_grid` by keyword, keep doing that and only add the `grid_shape=grid_shape` line.

- [ ] **Step 7: Hand the canonical shape to the voxeliser from `run_core`**

`run_core` does not currently hold a `GridParams` (the only `compute_grid_params` call in `pipeline.py` is at line 329, inside `_resolve_dem_step`). Immediately before the `voxelize_citygml_meshes` call at line 525, add:

```python
        # The 2-D rasterisers above each sized themselves from this same
        # frame; passing it here is what makes fill_building_id_gaps' pairing
        # below hold by construction rather than by two roundings agreeing.
        gp_2d = compute_grid_params(rectangle, cfg.meshsize)
```

and add to that call's keyword arguments:

```python
            grid_shape=gp_2d.shape,
```

This is deliberately belt-and-braces: the parameter's default already derives the same counts. It is passed explicitly so the pairing that `fill_building_id_gaps` depends on is visible at the seam that depends on it, rather than being an invisible coincidence of two functions calling the same helper.

- [ ] **Step 8: Declare the local frame in the identity-transformer fixture**

`tests/test_terrain_building_contact.py` drives the voxeliser with `water_rect()` — `[(0.0, 0.0), (0.0, NXY), (NXY, NXY), (NXY, 0.0)]`, local metres, not lon/lat — through an `IdentityTransformer`. It must now state its own counts. `n = int(round(NXY / VS))` is already computed at line 336.

Add `grid_shape=(n, n)` to the `v3.voxelize_citygml_meshes(...)` call (line 341) and `grid_shape=(n, n)` to the `v3._compute_grid_params_3d(...)` call (line 354). Change nothing else in that file — the existing assertion `grid.shape == (gp.n_rows, gp.n_cols, gp.n_z)` must still hold.

- [ ] **Step 9: Run the new tests to verify they pass**

```bash
docker run --rm -v /work-misc/code/VoxCityGML:/src -w /src voxcitygml-test:latest \
  tests/test_grid_rotation.py -k "voxel_frame" -q
```

Expected: `4 passed`.

- [ ] **Step 10: Run the full fast suite for regressions**

```bash
docker run --rm -v /work-misc/code/VoxCityGML:/src -w /src voxcitygml-test:latest \
  -m "not slow" -q
```

Expected: `376 passed, 2 skipped, 12 deselected` (the 372 baseline plus the 4 new tests). Any other delta is a regression — stop and report it rather than adjusting the expectation or editing the failing test.

- [ ] **Step 11: Commit**

```bash
cd /work-misc/code/VoxCityGML
git add voxcitygml/voxelizer3d.py voxcitygml/pipeline.py \
        tests/test_grid_rotation.py tests/test_terrain_building_contact.py
git commit -m "fix(voxelizer): size the 3-D frame from the canonical 2-D cell counts

_compute_grid_params_3d rounded the projected bbox of the rectangle while
compute_grid_params rounds its geodesic side length. The two measurements
differ slightly, so a side within floating-point reach of a half cell sized
the frames one column apart and run_core's fill_building_id_gaps refused the
pair, failing LOD2 generation with a 400. Measured on ~2-4% of sizes.

Callers working in a synthetic local frame pass grid_shape instead. Only the
counts change; min_x/max_y/min_z stay raw because every row, column and z
index anchors on them."
```

---

### Task 2: Re-pin optree to the fixed commit and verify end to end

**BLOCKED** until Task 1 is pushed to `origin/main` — the optree Dockerfile installs `voxcitygml` from a git URL, so an unpushed commit cannot be built. Do not start this task without explicit human consent to push.

**Files:**
- Modify: `/work-misc/code/optree_voxcity/Dockerfile:118`

**Interfaces:**
- Consumes: the commit SHA produced by Task 1.

- [ ] **Step 1: Push Task 1**

```bash
cd /work-misc/code/VoxCityGML && git push origin HEAD
```

- [ ] **Step 2: Capture the new SHA**

```bash
cd /work-misc/code/VoxCityGML && git rev-parse HEAD
```

- [ ] **Step 3: Update the pin**

In `/work-misc/code/optree_voxcity/Dockerfile`, line 118, replace `e43c6823b0327bd7f19cb5b32bd548eab4b7e84c` with the SHA from Step 2:

```dockerfile
        "voxcitygml[mesh] @ git+https://github.com/kunifujiwara/VoxCityGML@<NEW_SHA>"
```

- [ ] **Step 4: Rebuild and restart**

```bash
cd /work-misc/code/optree_voxcity
docker compose down
docker compose up -d --build
```

- [ ] **Step 5: Verify the app is healthy**

```bash
curl -fsS --retry 20 --retry-delay 5 --retry-all-errors \
  http://localhost:${HOST_PORT:-8004}/api/health
```

Expected: `"plateau_lod2":{"available":true,"reason":""}`.

- [ ] **Step 6: Verify the fix is in the running image**

```bash
docker compose exec -T optree python -c "
import inspect, voxcitygml.voxelizer3d as v
src = inspect.getsource(v._compute_grid_params_3d)
assert 'grid_shape' in src, 'old pin still installed'
print('fixed voxelizer3d is installed')"
```

Expected: `fixed voxelizer3d is installed`.

- [ ] **Step 7: Re-run the failing generation in the browser**

Generate mode PLATEAU, LOD2, mesh size 1, on the same target rectangle that produced
`building_id_grid and voxel_grid must share their first two axes; got (250, 250) and (250, 251)`.
Expected: generation completes. Confirm no `ValueError` in `docker compose logs optree`.

- [ ] **Step 8: Commit the pin bump**

```bash
cd /work-misc/code/optree_voxcity
git add Dockerfile
git commit -m "chore: bump voxcitygml to the unified voxel frame cell counts"
```

---

## Notes for the implementer

**Why not resize `building_id_grid` to match, like the other 2-D inputs?**
`voxelize_citygml_meshes` already resizes `dem_grid`, `land_cover_grid` and `canopy_top/bottom` into its own frame. Doing the same to `building_id_grid` is the symptom fix and is wrong: those are *ids*, not measurements, so a zoom would shift ownership of every column by up to a cell. `fill_building_id_gaps` raises on purpose for exactly this reason.

**Why an explicit `grid_shape` rather than sniffing the coordinate system?**
A heuristic that guesses whether vertices are lon/lat (magnitude checks, extent thresholds) silently routes unrecognised input back to the buggy rounding, and the guess is untestable at the boundary. A caller in a synthetic frame knows its own counts; making it say so keeps one code path for every real caller.

**Bonus effect worth knowing about.** Because the frames now always agree, the `_resize_float_grid` / `_resize_int_grid` calls at `voxelizer3d.py:271`, `:338` and `:356-357` become no-ops. Before the fix, whenever the counts diverged the DEM and land-cover grids were silently zoom-stretched by one row or column across the whole grid — a sub-cell misregistration that never raised. Those resize calls should stay as defensive no-ops; removing them is out of scope.
