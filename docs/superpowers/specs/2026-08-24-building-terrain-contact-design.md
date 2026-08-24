# Building–Terrain Contact Fix — Design

**Date:** 2026-08-24 (substantially revised 2026-08-25 after measurement
refuted the original D1 — see "Corrections" below)
**Scope:** `voxcitygml` terrain voxelization + building ground contact
(voxelizer3d.py, pipeline.py)
**Symptom (VoxCityApp, PLATEAU LOD2):** unintentional air gaps between
building bottoms and the terrain surface below, for some buildings.
**Constraint (user, 2026-08-24):** pilotis — intentional open ground
floors — must NOT be closed by the fix.

## Conventions

`t = (z − gp.min_z) / voxel_size` is an elevation in voxel units.

* **containing voxel** — `ceil(t) − 1`: the voxel whose interior holds
  `z`. For a surface exactly on a lattice plane this is the voxel below,
  matching `_penetration_half` semantics.
* **centre-sampled voxel** — `floor(t − 0.5)`: the topmost voxel whose
  *centre* lies at or below `z`. This is what any centre-sampled solid
  voxelizer produces.

The two differ exactly when `frac(t) ∈ (0, 0.5)`, where the centre-sampled
result is one lower. This distinction is load-bearing throughout.

## Measured facts

Tools: session scratchpad `acceptance_contact.py` (per-column gaps,
classified by mesh-accurate bottom height) and `mechanism.py` (per-gap-column
attribution). Site: Ochanomizu 500 m @ 2 m (139.7592, 35.6989, Chiyoda
dataset), 37 807 building columns, plus Chuo 200 m @ 2 m.

1. **The building term is correct.** For gap columns,
   `building_bottom − floor(t + h_min/vs)` has mean −0.20 and range
   [−1, +1]. Buildings land where the model predicts. The building path
   is not implicated.

2. **The terrain term is systematically low.** Across all open terrain
   columns, `terrain_top − floor(t − 0.5)` is **−1 for 47.6%** and 0 for
   50.8%, with a thin tail to −4 and only 0.4% above 0. A ~50/50 split
   between −1 and 0 is the signature of a **half-voxel downward
   displacement**: terrain is landing near `floor(t − 1)` rather than
   `floor(t − 0.5)`. On the 26 gap columns specifically the offset is
   worse — mean −1.38, reaching −3.

3. **The error is one-sided.** 48.6% of columns are placed too low;
   0.4% too high. Whatever the precise cause, a *raise-only* correction
   addresses essentially the entire defect and risks nothing in the
   over-fill direction.

4. **Source-data offsets are real but small.** PLATEAU LOD2 bases sit
   above the local TIN for a minority of buildings (Kudanzaka: 8 of 70,
   max +0.71 m; Ochanomizu: 53 of 870, max +3.0 m). Every one of the 26
   observed gap columns had `h_min ≤ 1.12 m` — i.e. **less than one voxel
   at 2 m**. Data offset alone did not cause a single observed gap.

5. **Pre-fix gap counts** (2 m voxels): Chuo 200 m — 0 gap columns (its
   terrain is low enough that buildings are *buried* rather than
   floating); Ochanomizu 500 m — 15 sub-tolerance gap columns plus 11
   "fringe" columns that have building voxels but no
   `building_min_height_grid` segments. Several gap columns report
   `h_min = 0.00 m`: the base sits exactly at ground in the source data
   and the column still floats. Those are pure voxelizer placement error
   with no data offset component at all.

## Corrections to the original (2026-08-24) design

The original design attributed the bug primarily to **D1**: a
−0.5·voxel pre-shift, intended for the MeshLib levelset stamp, leaking
into the centre-sampled winding and Numba scanline fallbacks. That
premise is **withdrawn**. It rested on a synthetic sweep
(`phase_sweep.py`) whose grid phase never actually varied: it set
`z_min = z_t − 4.0 − VS` while sweeping `z_t`, so `min_z` tracked `z_t`
and `t` stayed pinned at exactly 5.0 at every "phase". The apparent
phase-dependent failures were floating-point tie-breaking around an
integral `t`, not a real phase sweep.

Measurement on a correct sweep (grid phase genuinely varied) shows:

* the −0.5 pre-shift has **no effect at all** on the winding path — that
  path derives its SDF origin from the mesh bounding box, so a uniform
  z-translation of the whole solid moves the bbox with it and cancels;
* the pre-shift **is currently a correct compensation** on the Numba
  scanline path, which over-marks by one voxel without it (it stamps
  with the inclusive `_contact_half` box). Removing it regresses that
  path by +1 at every phase;
* consequently the planned D1 change would have fixed nothing on winding
  and introduced a scanline regression.

**D1 is dropped. The three terrain paths are left exactly as they are.**

Two further consequences of the corrected model, both verified
analytically:

* With centre-sampled terrain (`floor(t−0.5)`) and a penetration-shelled
  building (bottom = `floor(t)` for fractional `t`), the gap is
  `floor(t) − floor(t−0.5) − 1 ∈ {−1, 0}` — **never positive**. A
  synthetic flat case therefore cannot reproduce the bug at all, which
  is why the real-data measurement above was necessary.
* `ceil(t) − 1` is **unachievable** by a centre-sampled path: such a path
  cannot claim a voxel that is only 10% submerged. Any test asserting it
  against the raw terrain solid is asserting an impossible target.

## Decisions

| # | decision |
|---|----------|
| **D2** (the fix) | **Conform terrain to the DEM surface voxel, raise-only.** One new `_fill_air_to_dem_surface(voxel_grid, gp, dem_grid)` fills every AIR cell at or below the containing voxel `ceil(t)−1` with `GROUND_CODE`. It runs after terrain-solid voxelization whenever a DEM exists, and **replaces both** `_fill_terrain_from_dem` (no-terrain case) and `_fill_terrain_gaps_from_dem` (river / failed-union gaps), additionally raising columns the solid left low. Two defects die together: the old helpers' `np.rint` level (round-half-up, up to half a voxel off in *either* direction), and the one-sided terrain displacement measured above. Raise-only and air-only: nothing is ever carved down, so the 0.4% of columns placed high are untouched and the levelset path's occasional overfill is preserved as-is (pre-existing, out of scope). |
| **D2 gives the contact invariant** | After the conform, `terrain_top ≥ ceil(t_dem)−1`, while a building based at the DEM has `bottom = floor(t_dem)`, which equals `ceil(t_dem)−1` for fractional `t` and `t−1+1` at integral `t`. Contact (or overlap) is guaranteed at **every** grid phase, independent of which path placed the solid and independent of its bias. This is why the fix works without needing to diagnose MeshLib's stamp convention. |
| **D3** (deferred, not implemented) | A pilotis-safe, tolerance-bounded ground-contact closure was designed for residual source-data offsets. Measured directly at both resolutions on Ochanomizu 500 m (same site): at 2 m, **zero** gap columns have `h_min ≤ 1.5 m` (the app's default tolerance) — the closure would fire on nothing. At 1 m — where the naive `(voxel_size, tolerance]` band argument does *not* predict an empty band, since `1 < 1.5` — measurement still finds **zero** sub-tolerance gap columns; the only residual gaps are 4 columns forming one contiguous feature (rows 67–68, cols 184–186) with `h_min` of 1.63 m, 1.74 m and 3.16 m, plus one adjacent fringe column with no `building_min_height_grid` segments. That is precisely the elevated-structure class — bottom well above ground — the pilotis constraint requires be left alone, not closed. D3 is therefore **not implemented**: both resolutions agree it would address a case not observed, and it would add a config field, a CLI flag, pipeline wiring and a closure pass for that. Not fabricating ground also honours the pilotis constraint in the strongest possible way — nothing is ever invented under a building. It is revisited only if a future measurement finds a genuine sub-tolerance gap. |

## Testing

`tests/test_terrain_building_contact.py` asserts at two levels:

* **Pre-conform** (`test_terrain_solid_top_is_surface_voxel`) — the
  terrain solid alone must be exactly `floor(t − 0.5)` on the
  centre-sampled winding and scanline paths, at every phase. This is a
  characterization/regression guard, not a target: it goes red if the
  scanline pre-shift is ever removed (the regression the withdrawn D1
  would have caused), and it is the only coverage of the
  `dem_grid is None` configuration production can still reach. The
  levelset path is asserted separately with its measured tolerance.
* **Post-conform** (`test_terrain_top_is_surface_voxel`,
  `test_building_on_terrain_touches`) — solid + conform together, which
  is what the pipeline ships. `ceil(t)−1` is the right expectation here
  because the conform can achieve it.

`test_winding_fallback_exact_on_off_lattice_mesh` was **removed**: the
inset mesh it relied on does not put the solid's bbox off-lattice,
because `build_terrain_solid` unions the TIN with a base box spanning
`grid_bounds` (terrain_solid.py) and the union's bbox spans the whole
grid regardless. It produced byte-identical results to the on-lattice
case at all phases — duplicate coverage at 8× the cost.

Dataset-gated addition to `test_integration_plateau.py`: zero gap
columns on the reference rectangle.

**Acceptance** (pre-fix numbers recorded for comparison): Ochanomizu
500 m @ 2 m must go from 15 sub-tolerance + 11 fringe gap columns to
**0**, and the share of open columns whose `terrain_top_face − dem` lies
in `[0, vs)` must rise from **1.4%** toward 100%. Chuo 200 m @ 2 m must
stay at 0 gap columns and rise from **1.1%**.

## Consequences accepted

* **The ground surface now sits about one voxel higher on real data than
  before.** Median `terrain_top_face − dem` moved from **−0.97 m** to
  **+1.09 m** at 2 m on Ochanomizu. This is a correction, not a
  regression: the old placement was demonstrably too low (see "Measured
  facts" above), and every consumer that locates the surface dynamically
  — `_apply_land_cover`, `_apply_canopy` via `_ground_surface_index`,
  and building placement — follows the corrected surface, not a stale
  index.
* **At on-lattice DEM elevations the fill is one voxel lower than the
  old `np.rint` level.** `t` exactly integral used to round to `t`
  itself; it now resolves to the containing voxel `ceil(t)−1 = t−1`.
  This is why `test_fill_terrain` (`tests/test_optimized.py`) moved its
  expected level from 10 to 9.
* **voxcity's legacy 2.5-D voxelizer is untouched and stays on its own
  `round(t)`-style ground level.** The 3-D path's containing-voxel
  convention now differs from that separate product path by up to a
  voxel. Accepted: nothing cross-reads voxel indices between the two
  paths, so the divergence has no observable effect.

## Open, deliberately not chased

The precise reason the levelset path lands half a voxel low on real TINs
while measuring exact on a synthetic flat plane is unresolved. Candidates:
the documented `_stamp_meshlib_mask` truncation-instead-of-floor defect
combined with a non-grid-aligned SDF lattice (the levelset path passes no
`align_origin`), and possible registration differences between the
lon/lat DEM grid and the rotated metric voxel grid. D2's raise-only
conform corrects the low direction whatever the cause, so this is
recorded rather than pursued. A future fix of `_stamp_meshlib_mask`'s
convention must still remove the levelset branch's −0.5 compensation in
the same change, or terrain will double-correct.

Separately: `dem_source="GSI DEM Japan"` silently returns an all-zero DEM
when Earth Engine is not initialized, and the pipeline proceeds on that
flat plane. Reported, not fixed here.
