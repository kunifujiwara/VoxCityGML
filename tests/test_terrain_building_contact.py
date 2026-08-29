"""Terrain surface placement and building ground contact.

Design: docs/superpowers/specs/2026-08-24-building-terrain-contact-design.md

A flat terrain TIN at elevation z_t must, after the DEM surface conform,
produce a terrain whose topmost GROUND voxel is exactly the *surface
voxel* ceil(t)-1 (t = (z_t-min_z)/vs) at every fractional phase, on all
three terrain paths (levelset, winding fallback, Numba scanline
fallback).  A box building whose base lies exactly on that terrain must
then touch it (zero air voxels below its lowest building voxel in every
footprint column).

The terrain is asserted at two levels, deliberately:

* ``test_terrain_solid_top_is_surface_voxel`` checks the terrain SOLID
  alone, before any DEM conform, against what each path's sampling
  convention can actually deliver (``floor(t-0.5)`` for the
  centre-sampled winding and scanline paths).  The conform is raise-only,
  so without this level any downward drift in the solid would be masked.
  It is also the only coverage of the configuration production reaches
  whenever ``dem_grid`` is None (voxelize_citygml_meshes takes it as
  Optional).
* ``test_terrain_top_is_surface_voxel`` checks solid + conform together,
  which is what the pipeline actually ships.

Note ``path="scanline"`` clears ``_MESHLIB_VOXEL_AVAILABLE`` for the whole
test, so in ``test_building_on_terrain_touches`` it selects the Numba
BUILDING voxelizer as well as the Numba terrain path.  That is deliberate
extra coverage, not an oversight -- for that test the parameter names a
(terrain path, building path) pair.
"""
import logging
import warnings

import numpy as np
import pytest
import trimesh

from voxcitygml import voxelizer3d as v3
from voxcitygml.models import Mesh3D

VS = 1.0
NXY = 40.0
PHASES = [0.0, 0.1, 0.2, 0.25, 0.5, 0.6, 0.7, 0.9]


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


def select_terrain_path(path, monkeypatch):
    """Force the module down one of the three terrain voxelization paths.

    Module-level patching, so it steers ``voxelize_citygml_meshes`` just
    as it steers a direct ``_voxelize_terrain_solid`` call.
    """
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


def build_terrain_only(gp, grid, z_t, path, monkeypatch):
    """Voxelize the terrain solid via *path*, with no DEM conform."""
    tmesh = flat_terrain_mesh(z_t)
    select_terrain_path(path, monkeypatch)
    assert v3._voxelize_terrain_solid([tmesh], IdentityTransformer(), gp, grid)


def conform_to_dem(gp, grid, z_t):
    dem = np.full((gp.n_rows, gp.n_cols), z_t, dtype=np.float64)
    v3._fill_air_to_dem_surface(grid, gp, dem)


def voxelize_terrain(gp, grid, z_t, path, monkeypatch):
    """Terrain solid + DEM surface conform -- the production sequence."""
    build_terrain_only(gp, grid, z_t, path, monkeypatch)
    conform_to_dem(gp, grid, z_t)


def ground_tops(gp, grid):
    """Topmost GROUND voxel index per column; fails if any column is bare.

    Asserted rather than trimmed: argmax on an all-False column silently
    returns 0, so a bare column would otherwise report a bogus top index
    instead of failing.
    """
    is_g = grid == v3.GROUND_CODE
    bare = int((~is_g.any(axis=2)).sum())
    assert bare == 0, f"{bare} columns have no ground voxel"
    return gp.n_z - 1 - np.argmax(np.flip(is_g, axis=2), axis=2)


def allowed_solid_top_range(gp, z_t, path):
    """[low, high] terrain-top indices for the RAW SOLID, pre-conform.

    Characterization, not aspiration.  The winding and scanline paths are
    centre-sampled, so the highest voxel they can claim is
    ``floor(t - 0.5)``; a centre-sampled fill cannot claim a voxel that is
    only 10% submerged, and demanding the containing voxel ``ceil(t)-1``
    here asserts an impossible target (2026-08-25 measurement, see the
    design doc's "Corrections").  Pinning the achievable value EXACTLY is
    what makes this a regression guard: it goes red if the scanline
    path's -0.5 pre-shift is ever removed, which would over-mark that
    path by one voxel at every phase.

    The levelset path measures at the containing voxel, +1 when the
    surface lies exactly on a lattice plane.
    """
    t = (z_t - gp.min_z) / gp.voxel_size
    if path == "levelset":
        expected = int(np.ceil(np.round(t, 9))) - 1
        on_lattice = abs(t - round(t)) < 1e-9
        return expected, expected + (1 if on_lattice else 0)
    centre = int(np.floor(np.round(t - 0.5, 9)))
    return centre, centre


def allowed_top_range(gp, z_t, path):
    """Inclusive [low, high] range of acceptable terrain-top indices.

    The centre-sampled winding / scanline paths must be exact.  The
    levelset stamp is corner-sampled (2026-08-11 diagnosis) and overfills
    by one voxel only when the surface lies exactly on a lattice plane;
    that overfill is accepted and out of scope (2026-08-24 design), so it
    is tolerated at integral t alone rather than at every phase.
    """
    t = (z_t - gp.min_z) / gp.voxel_size
    expected = int(np.ceil(np.round(t, 9))) - 1
    on_lattice = abs(t - round(t)) < 1e-9
    high = expected + 1 if (path == "levelset" and on_lattice) else expected
    return expected, high


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
def test_terrain_solid_top_is_surface_voxel(path, phase, monkeypatch):
    """The terrain solid alone lands where its sampling convention allows.

    Regression guard on the three terrain paths, asserted before any DEM
    conform: the conform is raise-only, so it would otherwise mask any
    downward drift here.  This is also the only coverage of the
    ``dem_grid is None`` configuration production can still reach.
    """
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    build_terrain_only(gp, grid, z_t, path, monkeypatch)
    tops = ground_tops(gp, grid)
    low, high = allowed_solid_top_range(gp, z_t, path)
    assert tops.min() >= low and tops.max() <= high, (
        f"phase {phase} {path}: terrain solid top "
        f"{tops.min()}..{tops.max()}, expected [{low}, {high}]")


@pytest.mark.parametrize("path", TERRAIN_PATHS)
@pytest.mark.parametrize("phase", PHASES)
def test_terrain_top_is_surface_voxel(path, phase, monkeypatch):
    """Terrain solid + DEM conform together -- what the pipeline ships."""
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, path, monkeypatch)
    tops = ground_tops(gp, grid)
    low, high = allowed_top_range(gp, z_t, path)
    assert tops.min() >= low and tops.max() <= high, (
        f"phase {phase} {path}: terrain top {tops.min()}..{tops.max()}, "
        f"expected [{low}, {high}]")


def test_conform_skips_nonfinite_dem_cells(caplog):
    """A non-finite DEM cell must not be fed to the integer cast.

    Asserting the grid alone does NOT pin this guard: ``np.ceil(nan)``
    cast to an integer is undefined behaviour, and on this platform it
    lands on INT64_MIN, which the clip turns into -1 -- so the unguarded
    code also merely leaves the column bare, and a grid-only assertion
    passes against it.  What the guard actually buys is independence from
    that UB: on a platform where the cast lands positive, the same clip
    yields ``n_z - 1`` and buries every building in the column under a
    full stack of ground.

    So this pins the two observables that exist only once the cells are
    excluded explicitly -- no cast warning escapes, and the skip is
    reported -- alongside the grid outcome.
    """
    gp, grid = make_grid(10.5)
    dem = np.full((gp.n_rows, gp.n_cols), 10.5, dtype=np.float64)
    dem[3, 4] = np.nan
    dem[5, 6] = np.inf
    with warnings.catch_warnings():
        # The unguarded cast emits "invalid value encountered in cast";
        # promoting it to an error is what makes this test fail if
        # non-finite cells are ever handed to astype() again.
        warnings.simplefilter("error", RuntimeWarning)
        with caplog.at_level(logging.WARNING, logger=v3._log.name):
            v3._fill_air_to_dem_surface(grid, gp, dem)
    assert "not finite" in caplog.text, (
        "the conform must report the cells it skipped")
    is_g = grid == v3.GROUND_CODE
    assert not is_g[3, 4].any(), "NaN column should be left unfilled"
    assert not is_g[5, 6].any(), "inf column should be left unfilled"
    # Every finite column still conforms normally.
    low, high = allowed_top_range(gp, 10.5, "levelset")
    tops = gp.n_z - 1 - np.argmax(np.flip(is_g, axis=2), axis=2)
    ok = np.ones((gp.n_rows, gp.n_cols), dtype=bool)
    ok[3, 4] = ok[5, 6] = False
    assert tops[ok].min() >= low and tops[ok].max() <= high


@pytest.mark.parametrize("path", TERRAIN_PATHS)
@pytest.mark.parametrize("phase", PHASES)
def test_building_on_terrain_touches(path, phase, monkeypatch):
    """A building based on the terrain surface has no air beneath it.

    For ``path="scanline"`` this also exercises the Numba building
    voxelizer (see the module docstring).
    """
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, path, monkeypatch)
    bverts, bfaces = box_building(z_base=z_t)
    v3._voxelize_building_solid(bverts, bfaces, gp, grid,
                                v3.BUILDING_CODE, True)
    assert max_air_gap_below_buildings(grid) == 0, f"phase {phase} {path}"


# ── Water carve: the conform's counterpart ──────────────────────────
#
# The DEM conform is raise-only by design, so a terrain TIN that plants
# ground ABOVE a flattened water DEM stays there -- that is the 5 m cliff
# inside the river measured on real PLATEAU LOD2.  These run the whole
# `voxelize_citygml_meshes` so the carve is asserted where it actually
# has to hold: after the conform, before the land-cover stamp.

WATER_TIN_Z = 6.0        # terrain TIN, well above the water DEM
WATER_DEM_Z = 3.0        # the flattened water surface
WATER_CODE = 9           # standard land-cover code for water
UNDERGROUND = 4.0        # drops the grid floor so the carve level isn't z=0


def water_rect():
    """Rectangle vertices (sw, nw, ne, se) that IdentityTransformer maps
    straight onto the [0, NXY]^2 local frame the terrain mesh lives in."""
    return [(0.0, 0.0), (0.0, NXY), (NXY, NXY), (NXY, 0.0)]


#: A bridge pier standing in the water, spanning the carve zone.  The
#: constraint that the carve touches GROUND_CODE only exists for real
#: piers in real rivers, so the pipeline fixture -- where the ORDERING is
#: also under test -- carries one rather than leaving that to unit scope.
PIER_X0 = PIER_Y0 = 18.0
PIER_W = 4.0


def pier_mesh():
    """A BRIDGE mesh punching from below the water DEM up through the TIN.

    Bridges voxelize as BUILDING_CODE with ``overwrite=True``, so this
    also puts BUILDING voxels where the terrain solid had put ground.
    """
    b = trimesh.creation.box(extents=[PIER_W, PIER_W, 3.0])
    b.apply_translation([PIER_X0 + PIER_W / 2, PIER_Y0 + PIER_W / 2,
                         WATER_DEM_Z + 2.0])
    # Mesh3D vertices are (lat, lon, z); the local frame is (x, y, z) and
    # swap_coordinates_3d + IdentityTransformer undo the swap.
    verts = np.asarray(b.vertices, float)[:, [1, 0, 2]]
    return Mesh3D(vertices=verts, faces=np.asarray(b.faces),
                  feature_type="bridge", feature_id="pier")


def run_water_pipeline(path, monkeypatch, flatten_water_dem):
    """Full voxelization of a flat TIN + bridge pier over all-water cover."""
    select_terrain_path(path, monkeypatch)
    monkeypatch.setattr(v3, "create_rectangle_frame_transformer",
                        lambda *a, **k: IdentityTransformer())
    n = int(round(NXY / VS))
    dem = np.full((n, n), WATER_DEM_Z, dtype=np.float64)
    collection = v3.CityGMLMeshCollection(
        terrain=[flat_terrain_mesh(WATER_TIN_Z)], bridges=[pier_mesh()])
    info = {}
    grid = v3.voxelize_citygml_meshes(
        collection,
        water_rect(),
        0.0, 0.0,
        VS,
        dem_grid=dem,
        land_cover_grid=np.full((n, n), WATER_CODE, dtype=np.int16),
        land_cover_source="CityGML",
        underground_depth=UNDERGROUND,
        flatten_water_dem=flatten_water_dem,
        info_out=info,
    )
    gp, _ = v3._compute_grid_params_3d(
        water_rect(), 0.0, 0.0, VS, collection,
        underground_depth=UNDERGROUND, dem_grid=dem,
    )
    # The recomputed gp must describe the grid that came back, or every
    # index this module asserts is measured against the wrong frame.
    assert float(info["voxel_min_z"]) == gp.min_z
    assert grid.shape == (gp.n_rows, gp.n_cols, gp.n_z)
    return gp, grid


def pier_columns(grid):
    """Boolean (n_rows, n_cols) mask of columns holding BUILDING voxels."""
    mask = (grid == v3.BUILDING_CODE).any(axis=2)
    assert mask.any(), "the bridge pier produced no BUILDING voxels"
    return mask


def surface_tops(gp, grid):
    """Topmost ground-surface voxel per column, land-cover codes included.

    ``ground_tops`` above looks for GROUND_CODE only; after the land-cover
    stamp the surface voxel is a positive code, so this mirrors
    ``_ground_surface_index``'s definition instead.
    """
    is_s = (grid == v3.GROUND_CODE) | (grid > 0)
    bare = int((~is_s.any(axis=2)).sum())
    assert bare == 0, f"{bare} columns have no ground surface"
    return gp.n_z - 1 - np.argmax(np.flip(is_s, axis=2), axis=2)


def containing_voxel(gp, z):
    return int(np.ceil(np.round((z - gp.min_z) / gp.voxel_size, 9))) - 1


@pytest.mark.parametrize("path", TERRAIN_PATHS)
def test_water_columns_carve_down_to_the_dem_surface(path, monkeypatch):
    """Ground above the flattened water DEM is removed, and the land-cover
    water voxel is stamped on the CARVED surface.

    The second half is the ordering pin: run the carve after
    ``_apply_land_cover`` instead and the water voxel is left orphaned in
    mid-air above the carved ground, because ``_ground_surface_index``'s
    two-call same-answer contract is broken between the land-cover stamp
    and the canopy seat.
    """
    gp, grid = run_water_pipeline(path, monkeypatch, flatten_water_dem=True)
    surface = containing_voxel(gp, WATER_DEM_Z)
    assert surface > 0, "fixture must not put the water surface on the floor"
    open_water = ~pier_columns(grid)

    assert not (grid[:, :, surface + 1:] == v3.GROUND_CODE).any(), (
        f"{path}: ground left standing above the flattened water surface")

    tops = surface_tops(gp, grid)[open_water]
    assert tops.min() == surface and tops.max() == surface, (
        f"{path}: carved water tops {tops.min()}..{tops.max()}, "
        f"expected {surface}")
    assert np.all(grid[:, :, surface][open_water] == WATER_CODE), (
        f"{path}: the water land-cover voxel must sit on the carved "
        f"surface, not on the uncarved terrain top")
    assert np.all(grid[:, :, :surface][open_water] == v3.GROUND_CODE), (
        f"{path}: the carve must not hollow out the ground below the DEM")


@pytest.mark.parametrize("path", TERRAIN_PATHS)
def test_water_carve_opt_out_leaves_the_tin(path, monkeypatch):
    """``flatten_water_dem=False`` restores the pre-carve behaviour."""
    gp, grid = run_water_pipeline(path, monkeypatch, flatten_water_dem=False)
    open_water = ~pier_columns(grid)
    tops = surface_tops(gp, grid)[open_water]
    low, high = allowed_solid_top_range(gp, WATER_TIN_Z, path)
    assert tops.min() >= low and tops.max() <= high, (
        f"{path}: opted-out tops {tops.min()}..{tops.max()}, "
        f"expected the TIN's [{low}, {high}]")
    assert tops.min() > containing_voxel(gp, WATER_DEM_Z), (
        f"{path}: opting out must leave the water columns high")


@pytest.mark.parametrize("path", TERRAIN_PATHS)
def test_water_carve_leaves_the_bridge_pier_standing(path, monkeypatch):
    """The carve is GROUND_CODE-only: every BUILDING voxel is preserved.

    Comparing the two runs voxel-for-voxel is the exact statement of the
    constraint -- a pier in a river must come out of the carved grid
    identical to the one in the uncarved grid, or the carve stranded
    real bridge geometry.
    """
    _, carved = run_water_pipeline(path, monkeypatch, flatten_water_dem=True)
    monkeypatch.undo()
    gp, uncarved = run_water_pipeline(path, monkeypatch,
                                      flatten_water_dem=False)

    assert np.array_equal(carved == v3.BUILDING_CODE,
                          uncarved == v3.BUILDING_CODE), (
        f"{path}: the carve changed the BUILDING voxel set")
    surface = containing_voxel(gp, WATER_DEM_Z)
    assert (carved[:, :, surface + 1:] == v3.BUILDING_CODE).any(), (
        f"{path}: the fixture's pier must reach above the carve surface, "
        f"or this test proves nothing")
