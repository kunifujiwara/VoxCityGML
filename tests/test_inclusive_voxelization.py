"""Inclusive voxelization: shell anchor rules and gap-free thin volumes.

Pins the 2026-08-17 inclusive-voxelization design
(docs/superpowers/specs/2026-08-17-inclusive-voxelization-design.md):

- ``_overlay_surface_shell`` ``anchor="connected"`` keeps thin features
  connected *through the shell* to any filled voxel, still drops
  disconnected fragments, and keeps the whole shell when no anchor exists
  at all (per-category export grids contain no terrain to anchor on).
- Building defaults are inclusive: shell threshold
  ``INCLUSIVE_SHELL_THRESHOLD`` (0.0) + connected anchor produce gap-free
  thin walls (the Plateau LOD2 "comb" bug) without inflating solid
  buildings.  The threshold can stay at 0 because
  ``_overlay_surface_shell``'s SAT test tests voxel-interior PENETRATION,
  not boundary contact (2026-08-17 metric fix, after an earlier 0.25
  "calibrated threshold" attempt was found to only mask the boundary-
  contact bug on round grid origins -- see
  ``test_inclusive_defaults_match_volume_exactly``'s real-origin grid).
- ``VoxelizerConfig.voxelization_mode`` resolves mode -> mechanism knobs,
  with explicit threshold values overriding the mode.

IMPORTANT -- a SKIPPED run of this module is a NON-RESULT, not a pass.
Every test below that exercises the building path is ``skipif``'d on
``_MESHLIB_VOXEL_AVAILABLE``, because the non-meshlib fallback in
``_voxelize_building_solid`` never receives ``shell_threshold`` /
``shell_anchor`` at all and its own surface stamp still uses the
boundary-contact (expanded) SAT box -- i.e. without meshlib this module
cannot verify the inclusive-mode fix even exists, let alone holds.  Two
wrong calibrations (2026-08-17: threshold-0.0 inflation, then a 0.25
"calibrated" threshold that only looked exact on round grid origins)
shipped past a green suite before this was caught, so
``test_meshlib_available_or_explicitly_opted_out`` below fails the run
outright when meshlib is missing, unless
``VOXCITYGML_ALLOW_NO_MESHLIB=1`` is set to explicitly accept an
unverified run.  Even then, "0 failed" for this file means "opted out
and skipped", never "the inclusive-mode contract was checked."
"""
import os

import numpy as np
import pytest
import trimesh

from voxcitygml.models import INCLUSIVE_SHELL_THRESHOLD
from voxcitygml.voxelizer3d import (
    _MESHLIB_VOXEL_AVAILABLE,
    Grid3DParams,
    _overlay_surface_shell,
)

MS = 2.0


def test_meshlib_available_or_explicitly_opted_out():
    """The non-skippable guard (see the module docstring).  Every other
    test in this file is skipif'd on meshlib and would happily vanish
    into "skipped" if meshlib were missing -- exactly the configuration
    where the inclusive-mode fix is entirely unverified.  This test has
    no skipif: it fails the run unless meshlib is present or a human
    explicitly opted out via VOXCITYGML_ALLOW_NO_MESHLIB=1."""
    if _MESHLIB_VOXEL_AVAILABLE:
        return
    if os.environ.get("VOXCITYGML_ALLOW_NO_MESHLIB"):
        pytest.skip(
            "meshlib unavailable; VOXCITYGML_ALLOW_NO_MESHLIB=1 explicitly "
            "accepts that every meshlib-dependent test in this module will "
            "now skip too -- this file verifies nothing about inclusive "
            "voxelization in this run.")
    pytest.fail(
        "meshlib is not installed, so every inclusive-mode test in "
        "tests/test_inclusive_voxelization.py is about to SKIP rather "
        "than run -- a green suite in that state does not mean the "
        "inclusive-mode fix (2026-08-17) holds; it means it was never "
        "checked.  Install meshlib, or set VOXCITYGML_ALLOW_NO_MESHLIB=1 "
        "to explicitly accept an unverified run.",
        pytrace=False)


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


def filled(grid):
    return set(zip(*np.nonzero(grid == -3)))


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
    # Exact-set on purpose: a superset assertion would still pass even if
    # the anchor filter were deleted from the function entirely.
    assert filled(grid) == set(wall_cells())


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
    # Exact-set on purpose: a superset assertion would still pass even if
    # the anchor filter were deleted from the function entirely.
    assert filled(grid) == set(wall_cells())


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
    sparse — the behavior 'voxelization_mode="tight"' resolves to."""
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


def make_gp_real():
    """Production grids get their origin from a pyproj transform, so it is
    never a round number.  The 2026-08-17 threshold calibration was wrong
    precisely because it was only ever measured on make_gp()'s round
    origin: boundary-coincident faces cancelled by floating-point luck
    there and leaked a full layer here."""
    return Grid3DParams(n_rows=12, n_cols=12, n_z=10,
                        min_x=-100.10367673553799, max_x=-76.10367673553799,
                        min_y=75.402981512345, max_y=100.302981512345,
                        min_z=-3.7071067811865475, max_z=16.2928932188,
                        voxel_size=MS)


# The calibration guard.  Inclusive mode must produce EXACTLY the set of
# voxels containing mesh volume: no gaps (the comb bug) and no empty
# voxels (the old boundary-contact metric's inflation).  Both failure
# directions are pinned here, on TWO grid origins (see GRID_FIXTURES
# below) -- the 2026-08-17 threshold-calibration bug slipped through
# because every case was originally measured on make_gp()'s round origin
# only, where a face lying exactly on a cell boundary cancelled by
# floating-point luck.  The fix belongs in the SAT rasterizer's tolerance
# (_overlay_surface_shell: shrink, not expand), which is why the threshold
# itself is back to 0.0; these cases now guard that fix on a realistic
# (non-round) origin too.  See "Threshold calibration" in the design spec.
INCLUSIVE_CASES = [
    ("box exactly aligned",       (0.0,   0.0, 0.0, 12.0, 12.0, 10.0)),
    ("box +0.05 off aligned",     (0.05,  0.0, 0.0, 12.0, 12.0, 10.0)),
    ("box offset 0.7",            (0.7,   0.7, 0.0, 12.0, 12.0, 10.0)),
    ("thin wall 0.5m mid-cell",   (3.9,   0.3, 0.3, 0.5, 11.4, 9.4)),
    ("thin wall 0.5m on bound.",  (3.75,  0.3, 0.3, 0.5, 11.4, 9.4)),
    ("very thin wall 0.1m",       (3.95,  0.3, 0.3, 0.1, 11.4, 9.4)),
    ("thin slab 0.3m horizontal", (0.3,   0.3, 4.85, 11.4, 11.4, 0.3)),
]

# Every case above is written relative to make_gp()'s own anchors (min_x
# for x, max_y for y, min_z for z -- see Grid3DParams.xyz_to_indices).
# GRID_FIXTURES lets the same case table run against a second grid with a
# non-round origin; _case_position() re-anchors a case's coordinates onto
# whichever grid is under test, preserving each case's exact/fractional
# alignment (an integer-voxel offset stays an integer-voxel offset; a
# +0.05 m off-alignment stays +0.05 m off) regardless of that grid's own
# origin.
GRID_FIXTURES = [("clean-origin", make_gp), ("real-origin", make_gp_real)]


def _case_position(gp, x0, y0, z0):
    """Re-anchor a case position defined relative to make_gp() onto *gp*.

    This deliberately holds each case's RELATIVE alignment fixed -- its
    offset in voxel units (and fractional metres) from the grid's own
    anchors (min_x / max_y / min_z) -- so that coordinate MAGNITUDE is the
    only thing that changes between fixtures.  That isolation is the
    entire point of testing a second, non-round-origin grid: if a case's
    outcome differs between fixtures, it can only be float-noise at large
    coordinate magnitude, never a different underlying geometry.  See
    ``test_case_position_preserves_relative_alignment``, which checks this
    invariant directly rather than leaving it implicit.
    """
    ref = make_gp()
    return (gp.min_x + (x0 - ref.min_x),
            gp.max_y - (ref.max_y - y0),
            gp.min_z + (z0 - ref.min_z))


def cells_containing_volume(gp, x0, y0, z0, ex, ey, ez):
    """Voxels whose volume meets the box in a set of positive measure.

    The inclusion tolerance matches the rasterizer's own SAT tolerance
    (``_penetration_half`` / ``gp.voxel_size * 1e-6``), not an arbitrary
    epsilon: this is the oracle for a penetration test, and using a
    different tolerance would let the oracle and the contract quietly
    drift apart in the zero-to-tol band.
    """
    tol = gp.voxel_size * 1e-6
    out = set()
    for row in range(gp.n_rows):
        for col in range(gp.n_cols):
            for zi in range(gp.n_z):
                cx0 = gp.min_x + col * gp.voxel_size
                cy1 = gp.max_y - row * gp.voxel_size
                cz0 = gp.min_z + zi * gp.voxel_size
                ox = min(cx0 + gp.voxel_size, x0 + ex) - max(cx0, x0)
                oy = min(cy1, y0 + ey) - max(cy1 - gp.voxel_size, y0)
                oz = min(cz0 + gp.voxel_size, z0 + ez) - max(cz0, z0)
                if ox > tol and oy > tol and oz > tol:
                    out.add((row, col, zi))
    return out


@pytest.mark.parametrize("label,args", INCLUSIVE_CASES,
                         ids=[c[0] for c in INCLUSIVE_CASES])
def test_case_position_preserves_relative_alignment(label, args):
    """Locks down the invariant _case_position's docstring claims: the
    ideal (volume-containment) cell SET for a case must be index-identical
    across grid origins.  Pure geometry, no meshlib needed -- if this
    ever fails, a mismatch in test_inclusive_defaults_match_volume_exactly
    between fixtures would be an artifact of the re-anchoring transform,
    not a real rasterizer bug."""
    ideal_sets = []
    for _, make_grid in GRID_FIXTURES:
        gp = make_grid()
        x0, y0, z0 = _case_position(gp, *args[:3])
        ideal_sets.append(cells_containing_volume(gp, x0, y0, z0, *args[3:]))
    assert ideal_sets[0] == ideal_sets[1], (
        f"{label}: ideal cell set differs between grid origins "
        f"({len(ideal_sets[0])} vs {len(ideal_sets[1])} cells) -- "
        "_case_position should hold relative alignment fixed")


@pytest.mark.skipif(not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")
@pytest.mark.parametrize("grid_id,make_grid", GRID_FIXTURES,
                         ids=[g[0] for g in GRID_FIXTURES])
@pytest.mark.parametrize("label,args", INCLUSIVE_CASES,
                         ids=[c[0] for c in INCLUSIVE_CASES])
def test_inclusive_defaults_match_volume_exactly(label, args, grid_id, make_grid):
    from voxcitygml.voxelizer3d import _voxelize_building_solid
    gp = make_grid()
    x0, y0, z0, ex, ey, ez = args
    x0, y0, z0 = _case_position(gp, x0, y0, z0)

    b = trimesh.creation.box(extents=[ex, ey, ez])
    b.apply_translation([x0 + ex / 2, y0 + ey / 2, z0 + ez / 2])
    v = np.asarray(b.vertices, float)
    f = np.asarray(b.faces)

    grid = np.zeros((gp.n_rows, gp.n_cols, gp.n_z), np.int16)
    grid[:, :, 0] = -1                     # terrain floor, well below
    _voxelize_building_solid(v, f, gp, grid, -3, True)

    got = filled(grid)
    want = cells_containing_volume(gp, x0, y0, z0, ex, ey, ez)
    assert got == want, (
        f"{grid_id}/{label}: {len(got)} cells vs ideal {len(want)} "
        f"({len(got - want)} empty voxels added, "
        f"{len(want - got)} solid voxels missed)")
