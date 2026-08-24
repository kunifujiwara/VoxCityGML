# Building–Terrain Contact Fix — Design

**Date:** 2026-08-24
**Scope:** `voxcitygml` terrain voxelization + building ground contact
(voxelizer3d.py, pipeline.py, models.py, cli.py)
**Symptom (VoxCityApp, PLATEAU LOD2):** unintentional air gaps between
building bottoms and the terrain surface below, for some buildings.
**Constraint (user, 2026-08-24):** pilotis — intentional open ground
floors — must NOT be closed by the fix.

## Investigation summary

Reproduction/diagnosis scripts: session scratchpad
`diagnose_floating_buildings.py` (real-data per-column gap measurement)
and `phase_sweep.py` (synthetic flat-TIN + box-building phase sweep).

A gap appears in a column when

```
(terrain placement error) + (building base − terrain surface) ≥ 1 voxel
```

Both terms were measured on real PLATEAU data:

1. **Terrain fallback double-shift (deterministic bug).**
   `_voxelize_terrain_solid` pre-shifts the terrain solid down by
   0.5·voxel to compensate the *levelset* stamp's corner-sampling bias —
   but the same pre-shifted vertices are also fed to the **winding
   fallback** (non-watertight solids) and the **Numba scanline
   fallback**, which are centre-sampled and need no compensation.
   Synthetic sweep result (base exactly on terrain, winding fallback):
   **1-voxel air gap under the whole building at ~40 % of fractional
   phases** (0.1, 0.2, 0.6, 0.7 of a voxel). The fallback winding call
   also omits `align_origin=True`, so its lattice phase is per-mesh
   (bbox-dependent) — the same class of bug fixed for buildings on
   2026-08-11.

2. **Levelset terrain path is phase-dependent even when watertight.**
   Measured `terrain_top_face − DEM` on real TINs (expected `[0, vs)`):
   Chuo 2 m: median **−1.48 m**; Ochanomizu 500 m 2 m: median −0.97 m;
   same site 1 m: +0.49 m; Kudanzaka 2 m: +0.79 m. The sign flips with
   the grid's z-phase (`min_z` is derived from scene geometry, so moving
   the target rectangle changes it). On an adverse phase the terrain
   sits up to a voxel low **district-wide**.

3. **No ground-contact invariant + source-data offsets.** Buildings are
   voxelized at absolute elevation. In PLATEAU LOD2, building bases sit
   slightly above the local TIN for a meaningful minority (Kudanzaka: 8
   of 70 up to +0.7 m; Ochanomizu: 53 of 870, max +3.0 m; Sumida: median
   −0.01 m — i.e. entire districts sit *exactly at* the surface, one bad
   phase away from floating). The legacy 2.5-D voxelizer guaranteed
   contact by construction; the 3-D voxelizer has no equivalent.

4. **Replacement-DEM path.** `_fill_terrain_from_dem` uses
   `np.rint` (over/under-fills by up to half a voxel, phase-dependent),
   and `anchor_meshes_to_dem` seats each mesh rigidly from one centroid
   cell (documented limitation).

Why it reproduces only sometimes: flat districts share one fractional
phase, so a generation either has no gaps at all or floats whole blocks;
the phase changes with the drawn rectangle (via `min_z`), which matches
the "some buildings / some runs" symptom from the app.

Also observed, out of scope here (flagged to the user):
`dem_source="GSI DEM Japan"` silently returns an all-zero DEM when Earth
Engine is not initialized.

## Decisions

| # | decision |
|---|----------|
| D1 | **Scope the −0.5·voxel pre-shift to the levelset call only.** The winding fallback gets raw vertices and `align_origin=True`; the Numba scanline fallback gets raw vertices. This kills the deterministic fallback bias and makes the fallback phase-exact, mirroring the 2026-08-11 building fix. |
| D2 | **Conform terrain to the DEM surface voxel (raise-only).** One new function `_fill_air_to_dem_surface(voxel_grid, gp, dem_grid)` fills every AIR cell at or below the *surface voxel index* `floor((dem − min_z)/vs − ε)` with `GROUND_CODE`. It runs after terrain-solid voxelization whenever a DEM exists and **replaces both** `_fill_terrain_from_dem` (no-terrain case: fills whole columns) and `_fill_terrain_gaps_from_dem` (river/failed-union gaps), and additionally tops up columns the solid left low. `rint` → `floor(−ε)` fixes the half-voxel overfill and establishes the penetration-consistent convention: the terrain surface voxel is the voxel *containing* the surface (half-open upward; a surface exactly on a lattice plane belongs to the voxel below, matching `_penetration_half` semantics). No carving: overfill from the levelset path is left in place (pre-existing, benign, out of scope). |
| D3 | **Pilotis-safe bounded ground-contact closure.** New `close_building_ground_gaps(voxel_grid, building_min_height_grid, tolerance_m)` in voxelizer3d.py, called from `run_core` after `voxelize_citygml_meshes`. For each column containing building voxels, take the **mesh-accurate** bottom height above ground `h_min = min(seg[0] for seg in building_min_height_grid[r, c])` (fringe columns with voxels but no segments get the nearest claimed column's value via `distance_transform_edt`, the `fill_building_id_gaps` precedent). If `0 ≤ h_min ≤ tolerance_m` and air separates the lowest building voxel from the support below, fill that air with `GROUND_CODE`. Pilotis columns have `h_min` ≈ slab clearance (≥ ~2.2 m) and are never touched — the discrimination is in metres of real geometry, so it works at any voxel size (a voxel-count rule cannot distinguish a 2.5 m pilotis from a 0.3 m misalignment at 2 m voxels). |
| D4 | **Config knob** `VoxelizerConfig.ground_contact_tolerance: float = 1.5` (metres; `0` disables closure), exposed in the CLI as `--ground-contact-tolerance`. Default 1.5 sits between observed data offsets (p95 < 1 m) and minimum plausible pilotis clearance (~2.2 m). |

## Contact invariant achieved

With D1+D2, terrain and buildings share penetration semantics: a
building whose base coincides with the terrain surface claims (or is
adjacent above) the same voxel the terrain surface claims — contact at
every phase, proven by the phase-sweep tests across all three terrain
paths. In fact any base offset smaller than one voxel can no longer
gap at all (a gap now requires the offset to span a second lattice
plane), so D3 only ever acts on genuine source-data offsets in the
`(voxel_size, tolerance]` band; anything larger (pilotis, podiums,
genuine overhangs) is preserved.

## Consequences accepted

- Flat/named-DEM ground drops by up to one voxel versus the old `rint`
  fill (the old behaviour over-filled; re-anchored buildings sit exactly
  on the surface voxel under the new convention, and D3 covers sub-
  tolerance offsets). Dataset-gated integration numbers may shift.
- The levelset path keeps its compensated corner-sampling stamp and its
  occasional +1 overfill; fixing `_stamp_meshlib_mask`'s convention
  remains out of scope (as documented on 2026-08-11) and D2 makes the
  fix insensitive to it in the gap direction.
- Closure writes plain `GROUND_CODE` under buildings; it does not
  re-run the land-cover overlay for those hidden cells.
- A pilotis thinner than one voxel (stilts at coarse mesh) may still
  leave its slab visually floating — correct per the constraint: we do
  not fabricate ground there.

## Testing

New `tests/test_terrain_building_contact.py`: synthetic flat-TIN phase
sweeps (10 phases × {levelset, forced-winding-fallback, forced-scanline})
asserting (a) terrain top voxel == surface voxel (exact for winding /
scanline; levelset may over-fill +1, accepted), (b) zero air gap for a
box building with base on the terrain, (c) +1.45 m base offset (the
smallest that floats once terrain is exact at 1 m voxels, still ≤
tolerance) closed by D3, (d) 3 m slab (pilotis) NOT closed. Dataset-
gated addition to `test_integration_plateau.py`:
zero sub-tolerance gap columns on the reference rectangle.

Acceptance: rerun the scratchpad diagnostic on Chuo 2 m / Kudanzaka 1 m &
2 m / Ochanomizu 500 m 2 m — expect `terrain_top_face − dem` median
within `[0, vs)` on every run and zero gap columns with `h_min ≤ 1.5`.
