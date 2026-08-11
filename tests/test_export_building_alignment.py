"""The per-category OBJ export must voxelize buildings through the fixed seam.

Historically `export_per_category_voxels_obj`'s mesh_groups branch (the one
`pipeline_export.run_and_export` always takes) called
`_voxelize_meshlib_levelset` directly, so exported building voxels carried
the +half-voxel stamp displacement even after the main grid was fixed.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from voxcitygml.export_obj import export_per_category_voxels_obj
from voxcitygml.voxelizer3d import _MESHLIB_VOXEL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _MESHLIB_VOXEL_AVAILABLE, reason="meshlib required")


def test_export_building_voxels_are_grid_exact(tmp_path, monkeypatch):
    """A grid-aligned box building must fill exactly its analytic cells."""
    # ~200 m square near Tokyo; the local frame's origin lands at the
    # rectangle centre, so exact cell indices are derived from gp below.
    lon0, lat0 = 139.770, 35.695
    dlon = 200.0 / 111320.0 / np.cos(np.radians(lat0))
    dlat = 200.0 / 110540.0
    rect = [(lon0, lat0), (lon0, lat0 + dlat),
            (lon0 + dlon, lat0 + dlat), (lon0 + dlon, lat0)]
    center_lon, center_lat = lon0 + dlon / 2, lat0 + dlat / 2

    # collection only feeds the z-range of the grid
    fake = SimpleNamespace(vertices=np.array([[lat0, lon0, 0.0],
                                              [lat0, lon0, 20.0]]))
    coll = SimpleNamespace(buildings=[fake], bridges=[], vegetation=[],
                           terrain=[])

    # capture the gp and the building grid the export actually uses
    import voxcitygml.export_obj as eo
    seen = {}
    real = eo._voxelize_building_solid

    def spy(verts, faces, gp, grid, **kw):
        seen["gp"] = gp
        out = real(verts, faces, gp, grid, **kw)
        seen["grid"] = grid.copy()
        return out

    monkeypatch.setattr(eo, "_voxelize_building_solid", spy)

    ms = 2.0
    # place a 12x12x10 box aligned to the grid lattice once gp is known;
    # first run computes gp, second run uses it.  Cheaper: derive lattice
    # from a probe run with a tiny box, then place the real one.
    probe = trimesh.creation.box(extents=[4, 4, 4])
    probe.apply_translation([2, 2, 2])
    export_per_category_voxels_obj(
        coll, rect, center_lon, center_lat, ms, str(tmp_path),
        basename="probe",
        mesh_groups={"building": [(np.asarray(probe.vertices, float),
                                   np.asarray(probe.faces))]})
    gp = seen["gp"]

    # a box spanning whole cells of THIS lattice
    x0 = gp.min_x + 10 * ms
    y0 = gp.max_y - 40 * ms          # rows 34..39 will be inside
    z0 = gp.min_z + 4 * ms
    box = trimesh.creation.box(extents=[12, 12, 10])
    box.apply_translation([x0 + 6, y0 + 6, z0 + 5])
    export_per_category_voxels_obj(
        coll, rect, center_lon, center_lat, ms, str(tmp_path),
        basename="aligned",
        mesh_groups={"building": [(np.asarray(box.vertices, float),
                                   np.asarray(box.faces))]})
    grid = seen["grid"]
    cells = set(zip(*np.nonzero(grid != 0)))

    want = set()
    for row in range(gp.n_rows):
        for col in range(gp.n_cols):
            for zi in range(gp.n_z):
                x = gp.min_x + (col + 0.5) * ms
                y = gp.max_y - (row + 0.5) * ms
                z = gp.min_z + (zi + 0.5) * ms
                if (x0 < x < x0 + 12 and y0 < y < y0 + 12
                        and z0 < z < z0 + 10):
                    want.add((row, col, zi))
    assert len(want) == 180
    assert cells == want, (
        f"export building voxels misplaced: {len(cells)} cells, "
        f"{len(cells - want)} extra, {len(want - cells)} missing")
