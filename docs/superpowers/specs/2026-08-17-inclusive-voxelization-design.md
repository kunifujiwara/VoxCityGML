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
| Semantics | **Any material → solid**: a voxel is solid iff it contains any part of the mesh volume. Shell threshold **0.0**, with the shell rasterizer testing voxel *penetration* rather than boundary *contact*. See "Shell metric calibration" below — the threshold is as originally specced; the metric had to be corrected to make it mean what it says. |
| API shape | Named mode enum `voxelization_mode: "inclusive" \| "tight"`, default `"inclusive"`. Explicit threshold values override the mode. |
| Anchor rule | **Connectivity flood** in inclusive mode: keep every shell voxel connected *through the shell* to an anchored voxel; floating disconnected fragments still discarded. |

`"tight"` reproduces today's behavior exactly (shell 0.5, 1-step adjacency
anchor) and stays available for tight-envelope visualisation.

## Shell metric calibration (2026-08-17, measured)

This section records a two-step correction made during implementation.
The conclusion is that the **shell metric** was wrong, not the threshold:
the inclusive threshold is **0.0**, exactly as originally specced, but the
shell rasterizer now tests voxel *penetration* rather than boundary
*contact*.

### Step 1 - the inflation bug

`_overlay_surface_shell` built its SAT test box **expanded** by a
tolerance (`half = voxel_size/2 + voxel_size*1e-6`). That answers "does
the mesh touch this cell's closure?", so a building face lying on a cell
boundary registered in **both** neighbouring cells - marking the empty
cell just outside every solid wall. Measured on a 2 m grid, an
exactly-aligned 12x12x10 m box filled 448 cells where 180 is correct
(2.5x); an offset box filled 343 where 245 is correct. Those extra voxels
contain no material, so they add no obstruction - only error (streets
narrow, buildings over-shade).

### Step 2 - the rejected threshold "fix"

A shell threshold of 0.25 appeared to fix this: on the calibration grid a
sweep showed any value in [0.15, 0.33] reproducing the analytic ideal
exactly on all ten geometries, so 0.25 was adopted as the plateau
midpoint.

**That result was an artifact of the test fixture.** Every case used a
grid with a round origin (`min_x = -6.0`), where the boundary coincidence
cancelled by floating-point luck. Production grids take their origin from
a pyproj transform and are never round. Re-measured on a realistic origin
(`min_x = -100.10367673553799`), threshold 0.25 leaked a full layer on
every case - aligned box 180 -> 336, thin wall 30 -> 60 - while 0.34 and
above still deleted thin walls entirely. No threshold value works, because
on a real origin an outside-face cell and a genuine thin-wall cell both
score ~0.333: the surface-contact metric cannot distinguish them.

This was caught by the `test_export_building_alignment` regression, whose
grid comes from an actual rectangle transform.

### Step 3 - the metric fix

Negating the tolerance (`half = voxel_size/2 - voxel_size*1e-6`) makes the
rasterizer ask "does the mesh penetrate this cell's interior?" - the
volume question inclusive mode actually means. It is robust to origin
rounding: the tolerance is ~2 microns at 2 m voxels, orders of magnitude
above coordinate float noise (~1e-12) and below any real feature.

With the shrunk box at threshold 0.0, output equals the analytic ideal
exactly - zero extra, zero missing - on **both** grid origins across all
seven geometries kept in the test suite (aligned box, off-aligned box,
offset box, thin walls at 0.5 m and 0.1 m both mid-cell and
boundary-coincident, horizontal 0.3 m slab).

### Consequences

- `INCLUSIVE_SHELL_THRESHOLD = 0.0`. It is not a tuning knob: the
  rasterizer already answers the volume question. Raising it above ~0.33
  reintroduces the comb bug.
- **Tight mode is unaffected.** Measured on all seven geometries at
  `shell_threshold=0.5, shell_anchor="adjacent"` on the real-origin grid:
  identical counts with the shrunk and expanded box. `half` only decides
  which cells are offered as candidates; `_compute_occupancy_fraction`
  runs its own sub-voxel subdivision that never used `half`, and it
  already scored boundary-leak cells at ~0.33, below the 0.5 cut.
- The `"connected"` anchor is required regardless: with the historic
  `"adjacent"` anchor, thin walls still vanish entirely, because their
  winding fill is empty and nothing anchors them.

### Testing rule this produced

Every geometric calibration case runs against **two** grid fixtures - a
round origin and a realistic pyproj origin - because a round origin hides
boundary-coincidence bugs. Single-origin geometry tests are how the 0.25
error reached the spec in the first place.

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
  - `inclusive` → `(INCLUSIVE_SHELL_THRESHOLD = 0.0, 0.0, "connected")`
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
- `_overlay_surface_shell` tests **penetration, not contact**: its SAT box
  is `voxel_size/2 - voxel_size*1e-6` (shrunk), not `+` (expanded). See
  "Shell metric calibration" — this is what makes threshold 0.0 mean "the
  cell contains mesh volume" instead of "a face touches the cell's
  boundary". Applies to both modes; measured to leave tight mode's output
  identical.
- Defaults flip to inclusive at the mechanism layer too:
  `voxelize_citygml_meshes`, `_voxelize_mesh_group`,
  `_voxelize_building_solid` default `building_shell_threshold`/
  `shell_threshold` to `INCLUSIVE_SHELL_THRESHOLD` (**0.0**) and thread
  `shell_anchor="connected"` through. The anchor parameter applies
  wherever `_overlay_surface_shell` is called: the building shell and the
  vegetation shell. Bridges already use the sealed-surface path at
  threshold 0 (no anchor filter) and are unchanged.
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
- **Exactness test (both directions, both origins):** across seven
  geometries — solid boxes grid-aligned, slightly off-aligned and offset;
  thin walls at 0.5 m and 0.1 m, mid-cell and boundary-coincident; a
  horizontal 0.3 m slab — inclusive mode produces *exactly* the analytic
  set of voxels containing mesh volume: no gaps and no empty voxels. Every
  case runs against **two grid fixtures**, one with a round origin and one
  with a realistic pyproj origin. The two-origin requirement is not
  optional: a round origin hides boundary-coincidence bugs by
  floating-point luck, which is exactly how the rejected 0.25 threshold
  passed review.
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
  (245 vs 180), and no change at all on grid-aligned geometry — plus the
  thin features tight loses entirely (60 vs 0). The 2026-08-11 tight
  envelope remains one word away (`voxelization_mode="tight"`).
- **Boundary-coincident flat features (2026-08-18).** A mesh that is a
  single flat surface exactly coincident with a cell-boundary plane
  penetrates no voxel interior and would vanish. This is reachable in
  production, not synthetic: the grid's z origin is
  `scene_min_z - meshsize` (geometry-derived, not projected), so with
  `dem_source="Flat"` at the default `meshsize=1.0` every integer
  elevation sits exactly on a cell plane. `_overlay_surface_shell`
  therefore re-rasterizes with the contact box when the penetration
  rasterization is empty for a non-degenerate mesh **and** that mesh's own
  bbox has no prior coverage. The second condition is required: an exactly
  grid-aligned *solid* box also rasterizes to an empty penetration shell
  on all six faces, but its interior is already filled by the winding
  pass, so that emptiness is correct.
  *Residual limitation:* the fallback does not fire for a flat feature
  whose bbox (padded one voxel) already overlaps other filled voxels — a
  deck one voxel above terrain, say. Such a feature is still dropped. It
  was judged acceptable because the obstruction is adjacent to solid
  geometry either way; revisit if flat decks over open space appear in
  real data.
  *Not fixable by tolerance:* a symmetric epsilon cannot separate "solid
  face on a boundary" (outer cell must stay empty) from "zero-thickness
  surface on a boundary" (one cell must own it) — that needs orientation
  or degeneracy information. Half-open and shifted boxes were both tested
  and rejected.
- The no-anchor fallback can admit a genuinely floating mesh if that mesh
  has *no* filled voxels at all; accepted because completeness of real
  features outweighs suppressing that rare artifact class in inclusive
  mode.
