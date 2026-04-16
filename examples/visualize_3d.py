"""Voxelize 1km x 1km area and visualize with voxcity."""
import os
os.environ["TI_LOG_LEVEL"] = "warn"  # suppress Taichi startup spam

from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline import VoxCityGML

if __name__ == "__main__":
    from voxcity.visualizer import visualize_voxcity
    cfg = VoxelizerConfig(
        citygml_path="/path/to/citygml_dataset",  # Replace with your CityGML path
        center_lon=139.7671,
        center_lat=35.6812,
        size_meters=1000,
        meshsize=2.0,
        gee_project="your-gee-project-id",  # Replace with your GEE project
        canopy_height_source="Static",
        save_output=True,
        output_dir="output/vis_1km",
        n_workers=4,
        max_voxel_ram_mb=4000,
        building_lod=2,
        # occupancy_threshold=0.5,
        # occupancy_subdivisions=4,
    )

    city = VoxCityGML(cfg).run()

    print("\nLaunching interactive visualisation...")
    visualize_voxcity(city, mode="interactive", downsample=1)
