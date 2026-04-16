"""Run voxcitygml pipeline and visualize the result."""
from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline import VoxCityGML
from voxcity.visualizer import visualize_voxcity

cfg = VoxelizerConfig(
    citygml_path="/path/to/citygml_dataset",  # Replace with your CityGML path
    center_lon=139.7671,
    center_lat=35.6812,
    size_meters=500,
    meshsize=1.0,
    save_output=True,  # save so we can reload if needed
    output_dir="output",
)

city = VoxCityGML(cfg).run()

print("\nLaunching interactive visualisation...")
visualize_voxcity(city, mode="interactive")
