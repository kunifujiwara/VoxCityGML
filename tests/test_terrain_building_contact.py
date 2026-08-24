"""Terrain surface placement and building ground contact.

Design: docs/superpowers/specs/2026-08-24-building-terrain-contact-design.md

A flat terrain TIN at elevation z_t must produce a terrain whose topmost
GROUND voxel is exactly the *surface voxel* ceil(t)-1 (t = (z_t-min_z)/vs)
at every fractional phase, on all three terrain paths (levelset, winding
fallback, Numba scanline fallback).  A box building whose base lies exactly
on that terrain must then touch it (zero air voxels below its lowest
building voxel in every footprint column).
"""
import numpy as np
import pytest
import trimesh

from voxcitygml import voxelizer3d as v3
from voxcitygml.models import Mesh3D

VS = 1.0
NXY = 40.0
PHASES = [0.0, 0.1, 0.25, 0.5, 0.6, 0.7, 0.9]


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


def voxelize_terrain(gp, grid, z_t, path, monkeypatch):
    """Run the requested terrain path, then the DEM surface conform."""
    tmesh = flat_terrain_mesh(z_t)
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
    ok = v3._voxelize_terrain_solid([tmesh], IdentityTransformer(), gp, grid)
    assert ok
    dem = np.full((gp.n_rows, gp.n_cols), z_t, dtype=np.float64)
    v3._fill_air_to_dem_surface(grid, gp, dem)


def surface_voxel_index(gp, z_t):
    t = (z_t - gp.min_z) / gp.voxel_size
    return int(np.ceil(np.round(t, 9))) - 1


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
def test_terrain_top_is_surface_voxel(path, phase, monkeypatch):
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, path, monkeypatch)
    is_g = grid == v3.GROUND_CODE
    tops = gp.n_z - 1 - np.argmax(np.flip(is_g, axis=2), axis=2)
    interior = tops[5:-5, 5:-5]  # avoid boundary-column artifacts
    expected = surface_voxel_index(gp, z_t)
    # The conform is raise-only, so no path may sit LOW.  The levelset
    # stamp's accepted corner-sampling overfill may sit one voxel HIGH at
    # some phases (2026-08-24 design, "Consequences accepted"); the
    # centre-sampled winding/scanline paths must be exact.
    high = expected + 1 if path == "levelset" else expected
    assert interior.min() >= expected and interior.max() <= high, (
        f"phase {phase}: terrain top {interior.min()}..{interior.max()}, "
        f"expected [{expected}, {high}]")


@pytest.mark.parametrize("path", TERRAIN_PATHS)
@pytest.mark.parametrize("phase", PHASES)
def test_building_on_terrain_touches(path, phase, monkeypatch):
    z_t = 10.0 + phase
    gp, grid = make_grid(z_t)
    voxelize_terrain(gp, grid, z_t, path, monkeypatch)
    bverts, bfaces = box_building(z_base=z_t)
    v3._voxelize_building_solid(bverts, bfaces, gp, grid,
                                v3.BUILDING_CODE, True)
    assert max_air_gap_below_buildings(grid) == 0, f"phase {phase}"
