"""Quick test for tree parsing with prototype cache."""
from voxcitygml.citygml.namespaces import build_namespaces, detect_crs_from_root
from voxcitygml.citygml.extractors import extract_vegetation_from_root
import lxml.etree as ET

gml_file = "/path/to/tree_model.gml"  # Replace with your GML file path

print("Parsing GML...")
tree = ET.parse(gml_file)
root = tree.getroot()
ns = build_namespaces(root)

epsg = detect_crs_from_root(root)
print(f"CRS: {epsg}")

print("\nExtracting vegetation...")
meshes = extract_vegetation_from_root(root, ns)
print(f"Extracted {len(meshes)} vegetation meshes")

if meshes:
    m = meshes[0]
    print(f"\nFirst mesh: {m.feature_id}")
    print(f"  Vertices: {m.vertices.shape}")
    print(f"  Faces: {m.faces.shape}")
    print(f"  Vertex range X: [{m.vertices[:,0].min():.1f}, {m.vertices[:,0].max():.1f}]")
    print(f"  Vertex range Y: [{m.vertices[:,1].min():.1f}, {m.vertices[:,1].max():.1f}]")
    print(f"  Vertex range Z: [{m.vertices[:,2].min():.1f}, {m.vertices[:,2].max():.1f}]")
    print(f"  Attributes: {m.attributes}")

    # Sample trees from across the dataset
    for i in [100, 500, 1000, 2000, 3000]:
        if i < len(meshes):
            mi = meshes[i]
            print(f"  Tree[{i}]: {mi.feature_id}, verts={mi.vertices.shape}, "
                  f"X=[{mi.vertices[:,0].min():.0f},{mi.vertices[:,0].max():.0f}], "
                  f"Z=[{mi.vertices[:,2].min():.1f},{mi.vertices[:,2].max():.1f}]")

    # Count total vertices/faces
    total_v = sum(m.vertices.shape[0] for m in meshes)
    total_f = sum(m.faces.shape[0] for m in meshes)
    print(f"\nTotal: {total_v} vertices, {total_f} faces across {len(meshes)} trees")
