"""Export mesh + voxel OBJ files from the VoxCityGML pipeline.

Runs the full pipeline and writes two OBJ (+ MTL) files that share the
same local-metre coordinate system:

  output/export_obj/meshes.obj   – watertight-processed triangle meshes
  output/export_obj/voxels.obj   – voxelized model (surface voxels only)

Both files use the same Transverse Mercator projection centred on the
user-specified lon/lat, so they can be loaded into any 3-D viewer
together with matching positions.
"""
import os
os.environ["TI_LOG_LEVEL"] = "warn"  # suppress Taichi startup spam

from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline_export import run_and_export

if __name__ == "__main__":
    cfg = VoxelizerConfig(
        citygml_path="/path/to/citygml_dataset",  # Replace with your CityGML path
        center_lat=35.6946605621064, center_lon=139.75155741195408,
        size_meters=400,
        meshsize=5.0,
        land_cover_source="CityGML",
        gee_project="your-gee-project-id",  # Replace with your GEE project
        canopy_height_source='High Resolution 1m Global Canopy Height Maps',
        save_output=True,
        output_dir="output/export_obj",
        n_workers=4,
        max_voxel_ram_mb=4000,
        building_lod=2,
    )

    run_and_export(
        cfg,
        mesh_basename="meshes",
        voxel_basename="voxels",
        per_category_basename="mesh_voxels",
        landcover_basename="landcover",
        watertight_meshes=True,
    )
