# Inclusive Voxelization Mode — Design

**Date:** 2026-08-17
**Status:** Approved

## Problem

Plateau LOD2 buildings with walls, parapets, or slabs thinner than one voxel
voxelize sparsely: alternating filled/empty columns ("combs") that sunlight
and wind pass straight through. Two mechanisms cause this:

1. **Shell threshold.** The building surface shell drops boundary voxels
   whose surface-contact occupancy is below `building_shell_threshold`
   (default 0.5). A lone flat face crossing a voxel scores ~0.33
   (`_compute_occupancy_fraction`), so any wall thinner than a voxel is
   dropped wherever only one face crosses the cell. The winding-number
   interior fill only keeps centre-inside cells, so a sub-voxel wall fills
   only the columns its volume happens to straddle — the comb pattern.
2. **Anchor rule.** `_overlay_surface_shell` keeps only shell voxels
   6-adjacent (1 step) to an already-filled voxel. A thin feature whose
   interior fill is empty and which sits ≥2 voxels from any filled voxel
   (high parapet, awning; any building in the per-category OBJ grids, which
   contain no terrain to anchor on) is silently dropped even at threshold 0.

For obstruction analyses (solar, wind) the grid must represent obstructing
geometry *completely*: no voxel a surface passes through may be empty.

## Decision summary (user-approved)

| Question | Decision |
|---|---|
| Scope | All obstructing classes (buildings, bridges, vegetation) — in practice mostly changes buildings; bridges/vegetation already default to threshold 0. |
| Semantics | **Any material → solid**: shell threshold **0.25**. A voxel is solid iff it contains any part of the mesh volume. See "Threshold calibration" below — this was revised from the originally-specced 0.0 after measurement. |
| API shape | Named mode enum `voxelization_mode: "inclusive" \| "tight"`, default `"inclusive"`. Explicit threshold values override the mode. |
| Anchor rule | **Connectivity flood** in inclusive mode: keep every shell voxel connected *through the shell* to an anchored voxel; floating disconnected fragments still discarded. |

`"tight"` reproduces today's behavior exactly (shell 0.5, 1-step adjacency
anchor) and stays available for tight-envelope visualisation.

## Threshold calibration (2026-08-17, measured)

The inclusive shell threshold was originally specced as **0.0** ("every
voxel any triangle touches"). Measurement during implementation showed
that is strictly worse than a small positive threshold, and the spec was
revised to **0.25** with the user's agreement.

Why 0.0 fails: surface-contact occupancy marks every voxel a triangle
*passes through*, including the voxel on the **outside** of a face. For a
solid building every boundary face therefore marks one extra layer of
empty voxels on all six sides. Those voxels contain no material, so they
add no obstruction — only error (streets narrow, buildings over-shade).

Measured on ten geometries (voxel size 2 m), against the analytic ideal —
the set of voxels whose volume intersects the mesh in a set of positive
measure:

| geometry | ideal | thr 0.0 | thr 0.15 / 0.25 / 0.33 | thr 0.5 (tight) |
|---|---|---|---|---|
| box exactly grid-aligned | 180 | 448 (2.5×) | **180** ✓ | 180 |
| box +0.001 m off aligned | 210 | 392 | **210** ✓ | 180 (−30) |
| box −0.001 m off aligned | 210 | 392 | **210** ✓ | 180 (−30) |
| box +0.05 m off aligned | 210 | 392 | **210** ✓ | 180 (−30) |
| box offset 0.7 m | 245 | 343 | **245** ✓ | 180 (−65) |
| box offset 1.0 m | 245 | 343 | **245** ✓ | 245 |
| thin wall 0.5 m mid-cell | 60 | 60 ✓ | **60** ✓ | 0 ✗ |
| thin wall 0.5 m on boundary | 60 | 60 ✓ | **60** ✓ | 0 ✗ |
| very thin wall 0.1 m | 60 | 60 ✓ | **60** ✓ | 0 ✗ |
| thin slab 0.3 m horizontal | 36 | 36 ✓ | **36** ✓ | 36 ✓ |

Any threshold in **[0.15, 0.33]** reproduces the ideal set exactly on all
ten cases — zero extra, zero missing. The plateau is wide and stable
(the ±0.001 m perturbation cases confirm it is not a floating-point
coincidence at exact alignment), so **0.25** is chosen as its midpoint for
maximum margin on both sides.

The upper edge of the plateau is ~0.33: a lone flat face crossing a voxel
touches about 9 of 27 sub-cells, so at 0.34 and above thin single-face
walls start disappearing again. The lower edge is where boundary-coincident
faces begin marking the empty neighbour cell.

The `"connected"` anchor is required regardless of threshold: at 0.25 with
the historic `"adjacent"` anchor, thin walls still vanish entirely (0 of 60
voxels), because their winding fill is empty and nothing anchors them.

**Retuning.** 0.25 is expected to be revisited against real PLATEAU data.
The value lives in exactly one place — the `"inclusive"` entry of
`_MODE_PARAMS` in `models.py` — so retuning is a one-line change. Callers
can also override per-run via `VoxelizerConfig(building_shell_threshold=…)`
without touching the mode. Any new value should be re-checked against the
ten-geometry table above; staying inside [0.15, 0.33] preserves exactness.

## Architecture

Mode is **policy** resolved at the config boundary; the voxelizer keeps
plain numeric/flag **mechanism** parameters.

### 1. Config surface (`models.py`)

- New field `voxelization_mode: str = "inclusive"`; validated against
  `{"inclusive", "tight"}`.
- `building_shell_threshold: Optional[float] = None` and
  `occupancy_threshold: Optional[float] = None` — `None` means "mode
  decides"; an explicit value always wins over the mode.
- Resolver (function or property on `VoxelizerConfig`) maps mode →
  `(building_shell_threshold, occupancy_threshold, shell_anchor)`:
  - `inclusive` → `(0.25, 0.0, "connected")`
  - `tight` → `(0.5, 0.0, "adjacent")`

### 2. Voxelizer mechanism (`voxelizer3d.py`)

- `_overlay_surface_shell` gains `anchor: "adjacent" | "connected"`:
  - `"adjacent"`: current rule — `surface & _dilate6(existing)`.
  - `"connected"`: `seeds = surface & _dilate6(existing)`, then
    `scipy.ndimage.binary_propagation(seeds, mask=surface)` with
    6-connectivity. Every shell voxel connected through the shell to an
    anchored voxel survives; disconnected fragments are discarded.
  - **No-anchor fallback** (`"connected"` only): if `seeds` is empty —
    a fully thin mesh whose winding fill produced nothing, e.g. buildings
    in per-category OBJ grids with no terrain anchor — keep the whole
    shell. Dropping an entire real feature is worse for obstruction than
    keeping an unanchored one.
- Defaults flip to inclusive at the mechanism layer too:
  `voxelize_citygml_meshes`, `_voxelize_mesh_group`,
  `_voxelize_building_solid` default `building_shell_threshold`/
  `shell_threshold` to **0.25** and thread `shell_anchor="connected"`
  through. The anchor parameter applies wherever `_overlay_surface_shell`
  is called: the building shell and the vegetation shell. Bridges already
  use the sealed-surface path at threshold 0 (no anchor filter) and are
  unchanged.
- The winding-number interior fill itself is unchanged; inclusive output is
  `winding fill ∪ complete shell`.

### 3. Callers

- `pipeline.py` (`run()` → `voxelize_citygml_meshes`): pass the resolved
  values from `VoxelizerConfig`.
- `export_obj.py` `export_per_category_voxels_obj` and
  `pipeline_export.py`: accept and forward the same resolved parameters so
  exported building voxels keep matching the main grid (the invariant the
  2026-08-11 alignment fix established).
- `cli.py`: `--voxelization-mode {inclusive,tight}`, default `inclusive`.

## Error handling

- Unknown `voxelization_mode` → `ValueError` at config construction /
  resolution, not deep in the voxelizer.
- Unknown `anchor` value in `_overlay_surface_shell` → `ValueError`.
- Behavior with meshes producing empty shells/fills is defined above
  (no-anchor fallback); no silent drops of whole features in inclusive
  mode.

## Testing

- **Red-first comb test:** a thin slab/wall (thickness < voxel size,
  positioned so faces do not straddle cell centres) voxelizes with **no
  gaps** in inclusive mode.
- **Exactness test (both directions):** on solid boxes — grid-aligned,
  slightly off-aligned, and half-cell offset — inclusive mode produces
  *exactly* the analytic set of voxels containing mesh volume: no gaps and
  no empty voxels. This is the guard against the threshold drifting back
  toward 0.0 (envelope inflation) or up past 0.33 (thin features lost).
- **Anchor tests:** a thin parapet ≥2 voxels above the winding fill
  survives via the connectivity flood; a detached floating fragment is
  still dropped; a mesh whose fill is empty keeps its full shell
  (no-anchor fallback).
- **Tight-mode pinning:** existing tests that pin the 0.5 shell (e.g. the
  roof-slope guard) switch to explicit `voxelization_mode="tight"` /
  `building_shell_threshold=0.5` so they keep pinning that contract.
- **Override precedence:** explicit `building_shell_threshold=0.5` with
  `voxelization_mode="inclusive"` uses 0.5 for the shell while keeping the
  connected anchor rule.

## Known trade-offs

- Inclusive envelopes are *wider than tight ones wherever a building's
  faces do not fall on cell boundaries* — a boundary cell holding 35 % of a
  wall becomes solid instead of being rounded away. That is the intended
  semantics (it holds material, so it obstructs), and it is bounded: the
  envelope never exceeds the set of voxels the mesh actually occupies.
  Measured against tight on the calibration geometries: +30 voxels on a
  slightly off-aligned box (210 vs 180), +65 on a 0.7 m offset box
  (245 vs 180), and no change at all on grid-aligned geometry. The
  2026-08-11 tight envelope remains one word away
  (`voxelization_mode="tight"`).
- The no-anchor fallback can admit a genuinely floating mesh if that mesh
  has *no* filled voxels at all; accepted because completeness of real
  features outweighs suppressing that rare artifact class in inclusive
  mode.
