"""Verify inner-ring (hole) handling by comparing face counts."""
import numpy as np
import lxml.etree as ET
from pathlib import Path
from voxcitygml.citygml.namespaces import build_namespaces
from voxcitygml.citygml.geometry import parse_polygon_to_triangles, parse_pos_list

cfg_path = "/path/to/citygml_dataset"  # Replace with your CityGML path
gml_files = list(Path(cfg_path).rglob("udx/bldg/*.gml"))[:4]

total_holes = 0
total_hole_verts = 0
examples = []

for gf in gml_files:
    tree = ET.parse(str(gf))
    root = tree.getroot()
    ns = build_namespaces(root)
    for polygon in root.iterfind('.//gml:Polygon', ns):
        interiors = polygon.findall('gml:interior', ns)
        if not interiors:
            interiors = polygon.findall('.//gml:interior', ns)
        if interiors:
            total_holes += len(interiors)
            # Parse with our updated function
            verts, faces = parse_polygon_to_triangles(polygon, ns)
            
            # Count exterior-only coords for comparison
            ext = polygon.find('.//gml:exterior//gml:LinearRing//gml:posList', ns)
            ext_coords = parse_pos_list(ext.text) if ext is not None and ext.text else np.array([]).reshape(0,3)
            if len(ext_coords) > 3 and np.allclose(ext_coords[0], ext_coords[-1]):
                ext_coords = ext_coords[:-1]
            
            # Count hole vertices
            n_hole_verts = 0
            for interior in interiors:
                pl = interior.find('.//gml:LinearRing//gml:posList', ns)
                if pl is not None and pl.text:
                    hc = parse_pos_list(pl.text)
                    if len(hc) > 3 and np.allclose(hc[0], hc[-1]):
                        hc = hc[:-1]
                    n_hole_verts += len(hc)
                    total_hole_verts += len(hc)
            
            if len(examples) < 5:
                examples.append({
                    'ext_verts': len(ext_coords),
                    'hole_verts': n_hole_verts,
                    'total_verts': len(verts),
                    'n_faces': len(faces),
                    'n_holes': len(interiors),
                })

print(f"Total polygons with holes: {total_holes}")
print(f"Total hole vertices:       {total_hole_verts}")
print(f"\nExample polygons with holes:")
for i, ex in enumerate(examples):
    expected_total = ex['ext_verts'] + ex['hole_verts']
    print(f"  [{i}] ext={ex['ext_verts']} + holes={ex['hole_verts']} = {expected_total} total_verts | "
          f"parsed_verts={ex['total_verts']} | faces={ex['n_faces']} | holes={ex['n_holes']}")
    assert ex['total_verts'] == expected_total, f"Vertex count mismatch! Expected {expected_total}, got {ex['total_verts']}"
    
print("\nAll assertions passed — interior rings are correctly included in triangulation!")
