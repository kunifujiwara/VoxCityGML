"""Osaka Station version of pedestrian solar irradiance pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from run_pedestrian_solar import run_pedestrian_solar

if __name__ == "__main__":
    run_pedestrian_solar(
        city_label="Osaka",
        citygml_path="/path/to/osaka_citygml_dataset",  # Replace with your CityGML path
        center_lon=135.4959,
        center_lat=34.7024,
        target_size=2000,
        buffer_meters=200,
        meshsize=2.0,
    )
