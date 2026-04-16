"""Voxelize 1km x 1km area using LOD1 and LOD2 buildings and export OBJ/MTL."""
import os
os.environ["TI_LOG_LEVEL"] = "warn"

from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline_export import run_and_export

if __name__ == "__main__":
    base_kwargs = dict(
        citygml_path=[
            "/path/to/citygml_dataset_1",  # Replace with your CityGML path(s)
            "/path/to/citygml_dataset_2",
        ],
        center_lon=139.7671,
        center_lat=35.6812,
        size_meters=2000,
        meshsize=2.0,
        gee_project="your-gee-project-id",  # Replace with your GEE project
        canopy_height_source="Static",
        save_output=True,
        n_workers=4,
        max_voxel_ram_mb=4000,
    )

    print("\n=== Running Export for LOD1 ===")
    cfg_lod1 = VoxelizerConfig(
        **base_kwargs,
        building_lod=1,
        output_dir="output/vis_1km_lod1"
    )
    run_and_export(
        cfg_lod1,
        mesh_basename="meshes_lod1",
        voxel_basename="voxels_lod1",
        per_category_basename="mesh_voxels_lod1",
        landcover_basename="landcover_lod1",
        watertight_meshes=True,
    )

    print("\n=== Running Export for LOD2 ===")
    cfg_lod2 = VoxelizerConfig(
        **base_kwargs,
        building_lod=2,
        output_dir="output/vis_1km_lod2"
    )
    run_and_export(
        cfg_lod2,
        mesh_basename="meshes_lod2",
        voxel_basename="voxels_lod2",
        per_category_basename="mesh_voxels_lod2",
        landcover_basename="landcover_lod2",
        watertight_meshes=True,
    )
