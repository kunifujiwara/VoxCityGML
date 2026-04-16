"""Diagnose: is trimesh.contains() or watertight conversion the bottleneck?"""
import time
import numpy as np

from voxcitygml.citygml.parser import parse_citygml_directory, merge_terrain_meshes
from voxcitygml.citygml.coordinates import create_rectangle, swap_coordinates_3d
from voxcitygml.citygml.coordinates import create_local_transformer
from voxcitygml.terrain.processor import terrain_meshes_to_dem_grid
from voxcitygml.watertight import make_watertight_mesh
from voxcitygml.voxelizer3d import (
    _compute_grid_params_3d,
    _allocate_voxel_grid,
    _fill_terrain_from_dem,
    _resize_float_grid,
    _bbox_to_index_range,
    BUILDING_CODE,
)

cfg_path = "/path/to/citygml_dataset"  # Replace with your CityGML path
center_lon, center_lat = 139.7671, 35.6812
size_meters = 300
meshsize = 2.0

buffered = create_rectangle(center_lon, center_lat, size_meters + 100)
rectangle = create_rectangle(center_lon, center_lat, size_meters)
col = parse_citygml_directory(cfg_path, rectangle_vertices=buffered, n_workers=4, feature_types=["terrain", "building"])
if col.terrain:
    col.terrain = merge_terrain_meshes(col.terrain)

gp, transformer = _compute_grid_params_3d(rectangle, center_lon, center_lat, meshsize, col)

slow_indices = [7, 37, 38, 40, 41, 46, 48, 55, 60, 61, 65, 69, 78]

print(f"{'idx':>3}  {'faces':>5}  {'watertight':>10}  {'is_wt':>5}  {'bbox_voxels':>12}  {'contains':>10}  {'path':>10}")
print("-" * 80)

for bi in slow_indices:
    mesh = col.buildings[bi]
    verts_ll = swap_coordinates_3d(mesh.vertices)
    x_m, y_m = transformer.transform(verts_ll[:, 0], verts_ll[:, 1])
    verts = np.column_stack([x_m, y_m, verts_ll[:, 2]])
    faces = mesh.faces

    t0 = time.perf_counter()
    wt = make_watertight_mesh(verts, faces, voxel_size=gp.voxel_size)
    dt_wt = time.perf_counter() - t0

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    r0, r1, c0, c1, z0, z1 = _bbox_to_index_range(gp, vmin, vmax)
    n_voxels = (r1 - r0 + 1) * (c1 - c0 + 1) * (z1 - z0 + 1)

    path = wt.method
    dt_contains = 0.0
    if wt.is_watertight and len(wt.faces) > 0:
        import trimesh
        tm = trimesh.Trimesh(vertices=wt.vertices, faces=wt.faces, process=True)
        tm.fix_normals()
        if tm.is_volume:
            vs = gp.voxel_size
            wvmin = wt.vertices.min(axis=0)
            wvmax = wt.vertices.max(axis=0)
            wr0, wr1, wc0, wc1, wz0, wz1 = _bbox_to_index_range(gp, wvmin, wvmax)
            wn = (wr1 - wr0 + 1) * (wc1 - wc0 + 1) * (wz1 - wz0 + 1)
            # Generate centres
            nr = wr1 - wr0 + 1; nc = wc1 - wc0 + 1; nz = wz1 - wz0 + 1
            ri_g, ci_g, zi_g = np.mgrid[0:nr, 0:nc, 0:nz]
            centres = np.empty((wn, 3), dtype=np.float64)
            centres[:, 0] = gp.min_x + (wc0 + ci_g.ravel() + 0.5) * vs
            centres[:, 1] = gp.max_y - (wr0 + ri_g.ravel() + 0.5) * vs
            centres[:, 2] = gp.min_z + (wz0 + zi_g.ravel() + 0.5) * vs

            t0 = time.perf_counter()
            inside = tm.contains(centres)
            dt_contains = time.perf_counter() - t0
            n_voxels = wn
            path += f" -> occ({np.sum(inside)}/{wn})"
        else:
            path += " -> not_volume"

    print(f"{bi:3d}  {len(faces):5d}  {dt_wt:10.2f}s  {wt.is_watertight!s:>5}  {n_voxels:12,}  {dt_contains:10.2f}s  {path}")
