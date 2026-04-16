"""Calculate per-building Green View Index for Osaka Station area."""

import os
import sys
os.environ["TI_LOG_LEVEL"] = "warn"
sys.path.insert(0, os.path.dirname(__file__))

from run_building_gvi import run_city_gvi

if __name__ == "__main__":
    run_city_gvi(
        city_label="Osaka",
        citygml_path="/path/to/osaka_citygml_dataset",  # Replace with your CityGML path
        center_lon=135.4959,   # JR Osaka Station
        center_lat=34.7024,    # JR Osaka Station
    )
