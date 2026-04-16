"""Profile WHERE time is spent inside the 3D voxelizer."""
import time
import numpy as np

from voxcitygml.citygml.parser import parse_citygml_directory, merge_terrain_meshes
from voxcitygml.citygml.coordinates import create_rectangle
from voxcitygml.terrain.processor import terrain_meshes_to_dem_grid
from voxcitygml.voxelizer3d import (
    _compute_grid_params_3d,
    _allocate_voxel_grid,
    _fill_terrain_from_dem,
    _voxelize_mesh_group,
    _resize_float_grid,
    BUILDING_CODE, TREE_CODE,
)

cfg_path = "/path/to/citygml_dataset"  # Replace with your CityGML path
center_lon, center_lat = 139.7671, 35.6812
size_meters = 300
meshsize = 2.0

# Parse
buffered = create_rectangle(center_lon, center_lat, size_meters + 100)
rectangle = create_rectangle(center_lon, center_lat, size_meters)
col = parse_citygml_directory(
    cfg_path, rectangle_vertices=buffered, n_workers=4,
    feature_types=["terrain", "building"],
)
if col.terrain:
    col.terrain = merge_terrain_meshes(col.terrain)

dem_grid = terrain_meshes_to_dem_grid(col.terrain, rectangle, meshsize)

# Step-by-step voxelization with timing
print("\n=== Voxelizer step-by-step profiling ===\n")

t0 = time.perf_counter()
gp, transformer = _compute_grid_params_3d(rectangle, center_lon, center_lat, meshsize, col)
print(f"  Grid params:     {time.perf_counter()-t0:.3f}s  -> {gp.n_rows}x{gp.n_cols}x{gp.n_z}")

t0 = time.perf_counter()
voxel_grid = _allocate_voxel_grid(gp, max_voxel_ram_mb=2000)
print(f"  Allocate grid:   {time.perf_counter()-t0:.3f}s")

dem_resized = _resize_float_grid(dem_grid, gp.n_rows, gp.n_cols)
t0 = time.perf_counter()
_fill_terrain_from_dem(voxel_grid, gp, dem_resized)
print(f"  Fill terrain:    {time.perf_counter()-t0:.3f}s")

# Buildings - the big one
print(f"\n  Buildings: {len(col.buildings)} meshes, {sum(len(m.faces) for m in col.buildings)} faces total")
for i, mesh in enumerate(col.buildings):
    t0 = time.perf_counter()
    _voxelize_mesh_group(
        [mesh], transformer, gp, voxel_grid,
        class_code=BUILDING_CODE, overwrite=True,
    )
    dt = time.perf_counter() - t0
    if dt > 1.0:
        print(f"    Building {i:3d}: {len(mesh.faces):5d} faces, {dt:.2f}s  <-- SLOW")
    elif dt > 0.3:
        print(f"    Building {i:3d}: {len(mesh.faces):5d} faces, {dt:.2f}s")

print("\nDone.")
