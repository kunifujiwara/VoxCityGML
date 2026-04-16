"""Quick test: parse Munich CityGML data to verify compatibility."""
import sys
sys.path.insert(0, '.')

from voxcitygml.citygml.parser import parse_citygml_directory
from voxcitygml.citygml.coordinates import create_rectangle

# Center of the Munich dataset tiles (Nuremberg area)
CENTER_LON = 11.082
CENTER_LAT = 49.445
SIZE_M = 500

rectangle = create_rectangle(CENTER_LON, CENTER_LAT, SIZE_M)
print(f"Rectangle: {rectangle}")

# Parse the Munich dataset (flat directory with .gml files)
# DEM is a separate GeoTIFF that must be passed explicitly.
DEM_PATH = "/path/to/munich/citygml_area_dgm1.tif"  # Replace with your DEM path

collection = parse_citygml_directory(
    citygml_path="/path/to/munich_citygml",  # Replace with your CityGML path
    rectangle_vertices=rectangle,
    feature_types=['building', 'terrain'],
    max_files=None,
    dem_path=DEM_PATH,
)

# Verify buildings
if collection.buildings:
    print(f"\n--- Sample building ---")
    b = collection.buildings[0]
    print(f"  ID: {b.feature_id}")
    print(f"  Vertices shape: {b.vertices.shape}")
    print(f"  Vertices[0]: {b.vertices[0]}  (should be lat, lon, z)")
    print(f"  Lat range: [{b.vertices[:,0].min():.6f}, {b.vertices[:,0].max():.6f}]")
    print(f"  Lon range: [{b.vertices[:,1].min():.6f}, {b.vertices[:,1].max():.6f}]")
    print(f"  Z range:   [{b.vertices[:,2].min():.1f}, {b.vertices[:,2].max():.1f}]")
else:
    print("WARNING: No buildings found!")

# Verify terrain
if collection.terrain:
    print(f"\n--- Terrain ---")
    t = collection.terrain[0]
    print(f"  ID: {t.feature_id}")
    print(f"  Vertices shape: {t.vertices.shape}")
    if len(t.vertices) > 0:
        print(f"  Vertices[0]: {t.vertices[0]}  (should be lat, lon, z)")
        print(f"  Lat range: [{t.vertices[:,0].min():.6f}, {t.vertices[:,0].max():.6f}]")
        print(f"  Lon range: [{t.vertices[:,1].min():.6f}, {t.vertices[:,1].max():.6f}]")
        print(f"  Z range:   [{t.vertices[:,2].min():.1f}, {t.vertices[:,2].max():.1f}]")
else:
    print("WARNING: No terrain found!")

print("\nDone!")
