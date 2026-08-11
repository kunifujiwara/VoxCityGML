# Voxelizer Alignment Fix — Design

**Date:** 2026-08-11
**Scope:** `voxcitygml` building voxelization (voxelizer3d.py, models.py, pipeline.py)
**Diagnosis:** claude.ai artifact `3c329c4e-66fc-42f9-ae2c-d89f62b92479`; scripts
`diagnose_voxelization2.py`, `calibrate_stamp.py` (CityMesher session scratchpad).

## Problem

Three mechanisms make LoD2 building voxels uneven:

1. **M1 (primary):** `_voxelize_meshlib_levelset` reads MeshLib `meshToVolume`,
   whose SDF is corner-sampled, but `_stamp_meshlib_mask` maps it into the city
   grid assuming centre samples. Every building's fill is displaced +½ voxel in
   x, y, z and dilated one layer on the +faces; per-building bbox phase
   quantizes the realised shift differently for each building. Proven on an
   aligned 12×12×10 m box: 294 cells filled where the answer is 180. The
   winding path (`meshToDistanceVolume`, centre-sampled) is measured exact.
2. **M2:** `make_watertight_mesh` falls through to `doubleOffsetMesh` at
   `voxelSize = meshsize` for some buildings, resculpting the surface by
   p95 ≈ 0.8–1.0 m before voxelization starts.
3. **M3:** `_overlay_surface_shell` with `occupancy_threshold = 0` marks every
   corner-grazed cell, adding 20–42 % cells, grid-aligned — patching the
   displaced fill asymmetrically.

## Decisions (approved 2026-08-11)

| # | decision |
|---|----------|
| M1 | Buildings switch to the **winding path with a grid-aligned SDF lattice**: `_voxelize_meshlib_winding` gains `align_origin=True`, snapping `vol.origin` to the `Grid3DParams` lattice so SDF cell centres coincide exactly with grid cell centres. The levelset path is retired for buildings (function kept, call removed, warning added). |
| M2 | **`make_watertight_mesh` is skipped for buildings** on the MeshLib path — `HoleWindingRule` signs raw soup robustly, so both the SDF and the shell consume the raw mesh. The cascade remains only in the legacy no-meshlib fallback. |
| M3 | Building shell runs at **threshold 0.5, exposed in config** as a new `VoxelizerConfig.building_shell_threshold` field (default 0.5), threaded through `voxelize_citygml_meshes` → `_voxelize_mesh_group`. The existing shared `occupancy_threshold` (vegetation etc.) is untouched. **Semantics (verified in review):** the filter measures *surface-contact* occupancy — the fraction of a cell's 3×3×3 sub-cells that touch a triangle — not volume overlap. A lone flat face scores ~⅓ (dropped at 0.5); two crossing faces or a ≥2-sub-slab-thick slab score ≥0.5 (kept). The combination still yields the intended envelope because the aligned winding *fill* independently supplies every centre-inside cell; the shell only decides thin-feature and edge cells. |

## Architecture

A new seam `_voxelize_building_solid(verts, faces, gp, grid, code, overwrite,
occupancy_threshold, occupancy_subdivisions, shell_threshold)` replaces the
inline building branch of `_voxelize_mesh_group`:

```
raw mesh ── winding SDF (align_origin=True) ── stamp (exact, phase-free)
   │                                              │
   └────────── surface shell (threshold 0.5) ── union
fallback (meshlib absent / winding failed): watertight → occupancy → single-mesh (unchanged)
```

The `force_surface` (bridges) and vegetation branches are unchanged.

## Consequences accepted

- Sub-half-voxel building features (≈ <1 m slabs at 2 m grid) that only
  survived via the threshold-0 shell will now be dropped. Envelope accuracy is
  preferred over sub-voxel features; tests document the boundary (1.2 m slab
  kept, 0.3 m slab dropped, at 2 m voxels).
- The shell previously used the watertight mesh "to avoid stray-triangle
  artifacts"; it now uses the raw mesh. The existing 6-connected-anchor guard
  in `_overlay_surface_shell` already suppresses floating fragments.
- `meshToDistanceVolume` is somewhat slower than the OpenVDB levelset;
  per-building cost measured acceptable during diagnosis (< 1 s/building).
- **The per-category OBJ export is NOT fixed by this change.** Corrected
  2026-08-11 after the final review; the original wording here claimed exports
  stay consistent with the main grid, which is false for the path real callers
  use. `export_per_category_voxels_obj` has two branches:
  `export_obj.py:1003–1035` runs when `mesh_groups` is supplied and calls
  `_voxelize_meshlib_levelset` directly on watertight-repaired meshes, with no
  grid alignment and no shell union — so it still carries M1 and M2. The `else`
  fallback at `export_obj.py:1036+` does reach `_voxelize_mesh_group` and hence
  the new seam, but it only runs when `mesh_groups is None`, and
  `pipeline_export.py:103` always supplies it. So in practice the fixed path is
  the dead one. Buildings in `mesh_voxels.obj` remain displaced by +½ voxel per
  axis and resculpted by up to ~1 m. Nothing tests this
  (`test_pipeline_core.py:156` monkeypatches the exporter away). Routing that
  branch through `_voxelize_building_solid` is a separate follow-up.
- The terrain path pre-shifts its solid by −0.5 voxel
  (`_voxelize_terrain_solid`, voxelizer3d.py:446–450) to compensate for the
  very stamp bias diagnosed here. Out of scope, but any future fix of the
  stamp convention must remove that compensation in the same change.

## Error handling

`align_origin` snapping happens in world coordinates then converts back to the
MeshLib shifted frame; dimensions gain a +2 margin to cover the ≤1-voxel
origin move (applied unconditionally, so non-aligned callers also gain one
empty SDF plane — benign). The snap anchors **per axis at (min_x, max_y,
min_z)** — the anchors `_stamp_meshlib_mask` actually indexes against. min_y
must not be used for y: production grids keep `max_y` as the raw rectangle
coordinate while `n_rows` is rounded, so `max_y − min_y` is generally not a
whole number of voxels and the min_y lattice is out of phase with the row
lattice; the unit-test grid sets a non-congruent `min_y` to pin this. Winding failure falls through to the legacy path exactly as
before (no behaviour change on failure).

## Testing

`tests/test_voxelizer_alignment.py`:
- aligned 12×12×10 box through the building seam → exactly 180 cells at the
  analytically known indices; repeated at sub-voxel offsets +0.7 m and +1.3 m
  (whole-lattice expectations shift accordingly, count stays 180).
- `align_origin=True` winding at the same three phases → exact.
- shell threshold: 1.2 m slab (0.6 cell fill) kept, 0.3 m slab (0.15) dropped.
- config: `building_shell_threshold` reaches the shell call.

Acceptance beyond unit tests: rerun diagnosis Part C reproduction on the six
Kanda buildings → best-fit fill shift (0, 0, 0) for all, shell adds < 10 %.

## Out of scope

Full Kanda re-voxelization + solar comparison rerun (separate follow-up, GPU
time); fixing `_voxelize_meshlib_levelset` internally; terrain path.
