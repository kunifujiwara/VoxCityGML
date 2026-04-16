"""Debug: test namespace and building extraction for Munich CityGML."""
import sys
sys.path.insert(0, '.')

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from voxcitygml.citygml.namespaces import build_namespaces, detect_crs_from_root
from voxcitygml.citygml.extractors import extract_buildings_from_root
from voxcitygml.citygml.coordinates import reproject_vertices

# Use the smallest file
gml_file = "/path/to/munich_citygml/652_5476.gml"  # Replace with your GML file path
print(f"Parsing: {gml_file}")

tree = ET.parse(gml_file)
root = tree.getroot()

# Check namespaces
ns = build_namespaces(root)
print(f"\nResolved namespaces:")
for k, v in ns.items():
    print(f"  {k}: {v}")

# Check CRS
epsg = detect_crs_from_root(root)
print(f"\nDetected CRS: {epsg}")

# Try to find buildings
buildings = root.findall('.//bldg:Building', ns)
print(f"\nBuildings found with bldg:Building: {len(buildings)}")

# If 0, try without namespace
if not buildings:
    # Try with iterfind
    buildings = list(root.iterfind('.//bldg:Building', ns))
    print(f"Buildings found with iterfind: {len(buildings)}")

# Also try the building namespace directly
bldg_ns = ns.get('bldg', '')
print(f"\nbldg namespace: {bldg_ns}")
buildings_direct = root.findall(f'.//{{{bldg_ns}}}Building')
print(f"Buildings found with direct NS: {len(buildings_direct)}")

# Extract buildings
if buildings or buildings_direct:
    meshes = extract_buildings_from_root(root, ns)
    print(f"\nExtracted meshes: {len(meshes)}")
    if meshes:
        m = meshes[0]
        print(f"  First mesh vertices shape: {m.vertices.shape}")
        print(f"  First vertex: {m.vertices[0]}")
        if epsg:
            reproj = reproject_vertices(m.vertices, epsg)
            print(f"  After reprojection: {reproj[0]} (should be lat, lon, z)")
