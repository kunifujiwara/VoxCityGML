"""Terrain surface placement and building ground contact.

Design: docs/superpowers/specs/2026-08-24-building-terrain-contact-design.md

A flat terrain TIN at elevation z_t must produce a terrain whose topmost
GROUND voxel is exactly the *surface voxel* ceil(t)-1 (t = (z_t-min_z)/vs)
at every fractional phase, on all three terrain paths (levelset, winding
fallback, Numba scanline fallback).  A box building whose base lies exactly
on that terrain must then touch it (zero air voxels below its lowest
building voxel in every footprint column).

The terrain is asserted at two levels, deliberately:

* ``test_terrain_solid_top_is_surface_voxel`` checks the terrain SOLID
  alone, before any DEM conform.  This is the only test here that can go
  red if the scoped-pre-shift fix is reverted: ``_fill_air_to_dem_surface``
  is raise-only and every flat-TIN path error is downward, so the conform
  launders a mis-shifted solid and no post-conform assertion can see it.
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


def inset_terrain_mesh(z_t, inset=0.4):
    """Flat TIN whose bounding box is deliberately OFF the grid lattice.

    ``_voxelize_meshlib_winding`` derives its SDF origin from the mesh
    bounding box, so a mesh spanning exactly the grid cannot detect a
    missing ``align_origin`` snap -- its bbox is already in phase.
    Insetting by a fraction of a voxel puts it out of phase.
    ``build_terrain_solid`` still covers the whole grid via its base box,
    so every column stays filled.
    """
    lo, hi = inset, NXY - inset
    verts = np.array([
        [lo, lo, z_t], [lo, hi, z_t], [hi, hi, z_t], [hi, lo, z_t],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh3D(vertices=verts, faces=faces,
                  feature_type="terrain", feature_id="t")


def box_building(z_base, x0=15.0, y0=15.0, w=8.0, h=9.0):
    b = trimesh.creation.box(extents=[w, w, h])
    b.apply_translation([x0 + w / 2, y0 + w / 2, z_base + h / 2])
    return np.asarray(b.vertices, float), np.asarray(b.faces)


def build_terrain_only(gp, grid, z_t, path, monkeypatch, mesh=None):
    """Voxelize the terrain solid via *path*, with no DEM conform."""
    tmesh = flat_terrain_mesh(z_t) if mesh is None else mesh
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
    """The terrain solid alone lands on the surface voxel, no DEM involved."""
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    build_terrain_only(gp, grid, z_t, path, monkeypatch)
    tops = ground_tops(gp, grid)
    low, high = allowed_top_range(gp, z_t, path)
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


@needs_meshlib
@pytest.mark.parametrize("phase", PHASES)
def test_winding_fallback_exact_on_off_lattice_mesh(phase, monkeypatch):
    """The winding fallback is phase-exact even when the mesh bbox is not.

    Covers the ``align_origin=True`` snap, which a mesh spanning exactly
    the grid cannot exercise.
    """
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    build_terrain_only(gp, grid, z_t, "winding", monkeypatch,
                       mesh=inset_terrain_mesh(z_t))
    tops = ground_tops(gp, grid)
    low, high = allowed_top_range(gp, z_t, "winding")
    assert tops.min() >= low and tops.max() <= high, (
        f"phase {phase}: off-lattice winding top {tops.min()}..{tops.max()}, "
        f"expected [{low}, {high}]")


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
