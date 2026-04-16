"""End-to-end voxelizer test using real CityGML sample data."""
import time
import numpy as np

from voxcitygml.citygml.parser import parse_citygml_directory
from voxcitygml.citygml.coordinates import create_rectangle
from voxcitygml.voxelizer3d import (
    voxelize_citygml_meshes,
    GROUND_CODE, TREE_CODE, BUILDING_CODE,
)
from voxcitygml.terrain.processor import terrain_meshes_to_dem_grid
from voxcitygml.citygml.parser import merge_terrain_meshes

cfg_path = "/path/to/citygml_dataset"  # Replace with your CityGML path
center_lon, center_lat = 139.7671, 35.6812
size_meters = 300
meshsize = 2.0

print("=" * 60)
print("End-to-end voxelizer test with real CityGML data")
print("=" * 60)

# ── Step 1: Parse CityGML ───────────────────────────────────────────
print("\n[1] Parsing CityGML...")
buffered = create_rectangle(center_lon, center_lat, size_meters + 100)
rectangle = create_rectangle(center_lon, center_lat, size_meters)

t0 = time.perf_counter()
col = parse_citygml_directory(
    cfg_path,
    rectangle_vertices=buffered,
    n_workers=4,
    feature_types=["terrain", "building"],
)
dt_parse = time.perf_counter() - t0

print(f"  Parsed in {dt_parse:.1f}s")
print(f"  Terrain meshes: {len(col.terrain)}")
print(f"  Building meshes: {len(col.buildings)}")
total_faces = sum(len(m.faces) for m in col.buildings)
print(f"  Total building faces: {total_faces}")

# ── Step 2: Create DEM ──────────────────────────────────────────────
print("\n[2] Creating DEM grid...")
if col.terrain:
    col.terrain = merge_terrain_meshes(col.terrain)

t0 = time.perf_counter()
dem_grid = terrain_meshes_to_dem_grid(col.terrain, rectangle, meshsize)
dt_dem = time.perf_counter() - t0
print(f"  DEM shape: {dem_grid.shape}, range: [{dem_grid.min():.1f}, {dem_grid.max():.1f}] m")
print(f"  DEM created in {dt_dem:.1f}s")

# ── Step 3: Full 3D voxelization ────────────────────────────────────
print("\n[3] Running 3D voxelization...")
t0 = time.perf_counter()
voxel_grid = voxelize_citygml_meshes(
    col,
    rectangle,
    center_lon,
    center_lat,
    meshsize,
    dem_grid=dem_grid,
    land_cover_grid=None,
    canopy_top=None,
    canopy_bottom=None,
    max_voxel_ram_mb=2000,
)
dt_vox = time.perf_counter() - t0

# ── Results ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Results")
print(f"{'=' * 60}")
print(f"  Voxel grid shape: {voxel_grid.shape}")
print(f"  Voxel grid dtype: {voxel_grid.dtype}")
print(f"  Memory: {voxel_grid.nbytes / 1024**2:.1f} MB")
print()

n_ground = np.count_nonzero(voxel_grid == GROUND_CODE)
n_building = np.count_nonzero(voxel_grid == BUILDING_CODE)
n_tree = np.count_nonzero(voxel_grid == TREE_CODE)
n_empty = np.count_nonzero(voxel_grid == 0)
n_total = voxel_grid.size

print(f"  Ground voxels:   {n_ground:>10,}  ({100*n_ground/n_total:.1f}%)")
print(f"  Building voxels: {n_building:>10,}  ({100*n_building/n_total:.1f}%)")
print(f"  Tree voxels:     {n_tree:>10,}  ({100*n_tree/n_total:.1f}%)")
print(f"  Empty voxels:    {n_empty:>10,}  ({100*n_empty/n_total:.1f}%)")
print(f"  Total voxels:    {n_total:>10,}")
print()
print(f"  Parse time:        {dt_parse:>6.1f}s")
print(f"  DEM time:          {dt_dem:>6.1f}s")
print(f"  Voxelization time: {dt_vox:>6.1f}s")
print(f"  Total:             {dt_parse + dt_dem + dt_vox:>6.1f}s")
print()

# Sanity checks
assert voxel_grid.shape[0] > 0 and voxel_grid.shape[1] > 0 and voxel_grid.shape[2] > 0
assert n_ground > 0, "No ground voxels produced"
assert n_building > 0, "No building voxels produced"
print("All sanity checks passed!")
