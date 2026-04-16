"""End-to-end test: voxelize Munich CityGML data and visualize."""
from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline import VoxCityGML
from voxcity.visualizer import visualize_voxcity

if __name__ == "__main__":
    cfg = VoxelizerConfig(
        citygml_path="/path/to/munich/building",  # Replace with your CityGML path
        dem_path="/path/to/munich/terrain/merged.vrt",  # Replace with your DEM path
        tree_citygml_path="/path/to/munich/tree",  # Replace with your tree CityGML path
        center_lon=11.560,
        center_lat=48.135,
        size_meters=1000,
        meshsize=2.0,
        land_cover_source="OpenStreetMap",
        canopy_height_source="Static",
        save_output=True,
        output_dir="output/test_munich_e2e",
        n_workers=4,
        building_lod=2,
    )

    city = VoxCityGML(cfg).run()

    print("\nLaunching interactive visualisation...")
    visualize_voxcity(city, mode="interactive", downsample=1)
