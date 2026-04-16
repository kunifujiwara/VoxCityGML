"""Test to verify buildings/bridges at rectangle edges are included."""
import numpy as np
from shapely.geometry import box as shapely_box
from voxcitygml.models import Mesh3D
from voxcitygml.citygml.coordinates import mesh_intersects_rectangle, rectangle_to_shapely

# Create a rectangle
rect_vertices = [
    (139.76, 35.68),   # SW
    (139.76, 35.69),   # NW
    (139.77, 35.69),   # NE
    (139.77, 35.68),   # SE
]
rect_poly = rectangle_to_shapely(rect_vertices)

# Create a building that partially overlaps (extends beyond the rectangle)
# Building is at the edge, partially inside
bldg_completely_outside = Mesh3D(
    vertices=np.array([
        [35.685, 139.780, 0],   # Outside to the east
        [35.685, 139.790, 10],
        [35.688, 139.785, 5],
    ], dtype=np.float64),
    faces=np.array([[0, 1, 2]], dtype=np.int32),
    feature_type='building',
    feature_id='bldg_outside'
)

# Building that partially overlaps
bldg_partially_overlaps = Mesh3D(
    vertices=np.array([
        [35.682, 139.768, 0],   # Inside southwest
        [35.682, 139.775, 10],  # Extends outside northeast
        [35.685, 139.772, 5],
    ], dtype=np.float64),
    faces=np.array([[0, 1, 2]], dtype=np.int32),
    feature_type='building',
    feature_id='bldg_partial'
)

# Building completely inside
bldg_completely_inside = Mesh3D(
    vertices=np.array([
        [35.682, 139.764, 0],
        [35.682, 139.766, 10],
        [35.684, 139.765, 5],
    ], dtype=np.float64),
    faces=np.array([[0, 1, 2]], dtype=np.int32),
    feature_type='building',
    feature_id='bldg_inside'
)

# Test intersections
print("Rectangle bounds:", rect_poly.bounds)  # (lon_min, lat_min, lon_max, lat_max)

print("\nBuilding completely outside:")
print(f"  Intersects: {mesh_intersects_rectangle(bldg_completely_outside, rect_poly)}")

print("\nBuilding partially overlaps:")
print(f"  Intersects: {mesh_intersects_rectangle(bldg_partially_overlaps, rect_poly)}")

print("\nBuilding completely inside:")
print(f"  Intersects: {mesh_intersects_rectangle(bldg_completely_inside, rect_poly)}")

# Check their bounding boxes
def print_bbox(mesh, name):
    if len(mesh.vertices) == 0:
        print(f"{name}: empty")
        return
    min_lat = mesh.vertices[:, 0].min()
    max_lat = mesh.vertices[:, 0].max()
    min_lon = mesh.vertices[:, 1].min()
    max_lon = mesh.vertices[:, 1].max()
    print(f"{name}: lon=[{min_lon:.5f}, {max_lon:.5f}], lat=[{min_lat:.5f}, {max_lat:.5f}]")

print("\nBounding boxes (note: vertices are [lat, lon, z]):")
print_bbox(bldg_completely_outside, "Outside")
print_bbox(bldg_partially_overlaps, "Partial")
print_bbox(bldg_completely_inside, "Inside")
