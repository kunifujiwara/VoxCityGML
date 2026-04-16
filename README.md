# VoxCityGML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Convert [CityGML](https://www.citygml.org/) 3-D city model data into voxel-based semantic city models compatible with the [VoxCity](https://github.com/kunifujiwara/VoxCity) framework.

VoxCityGML parses CityGML datasets (including Japan's [PLATEAU](https://www.mlit.go.jp/plateau/) open data), extracts terrain, buildings, bridges, and vegetation, and voxelizes everything into a unified 3-D semantic grid.

## Features

- Parse CityGML datasets with terrain, buildings, bridges, and vegetation
- Create DEM elevation grids from terrain TIN meshes
- Download and integrate land-cover classification data (OpenStreetMap, Google Earth Engine, etc.)
- Download canopy-height models and merge with CityGML vegetation
- Voxelize all geometries into a shared 3-D grid (LOD1–LOD4 support)
- Export to OBJ, VoxCity `.voxcity` format, and more
- GPU-accelerated solar irradiance and visibility simulations (via VoxCity)
- Command-line interface for batch processing

## Installation

```bash
git clone https://github.com/kunifujiwara/VoxCityGML.git
cd VoxCityGML
pip install -e .
```

For optional mesh repair support:

```bash
pip install -e ".[mesh]"
```

## Quick Start

### Python API

```python
from voxcitygml import VoxCityGML, VoxelizerConfig

config = VoxelizerConfig(
    citygml_path="path/to/plateau_dataset",
    center_lon=139.7671,
    center_lat=35.6812,
    size_meters=500,
    meshsize=1.0,
)
city = VoxCityGML(config).run()
```

### Command Line

```bash
voxcitygml \
    --path /data/plateau/13101_chiyoda \
    --center-lon 139.7671 --center-lat 35.6812 \
    --size 500 --meshsize 1.0
```

Run `voxcitygml --help` for the full list of options including LOD selection,
output directory, land-cover source, canopy settings, and more.

## Examples

The [`examples/`](examples/) directory contains runnable scripts demonstrating
common workflows:

| Script | Description |
|--------|-------------|
| `export_obj.py` | Voxelize a CityGML dataset and export OBJ meshes |
| `visualize_3d.py` | Voxelize and launch an interactive 3-D viewer |
| `lod_comparison.py` | Compare LOD1 vs LOD2 voxelization with OBJ export |
| `run_building_gvi.py` | Compute per-building Green View Index |
| `run_pedestrian_solar.py` | Compute pedestrian-level solar irradiance |
| `compare_building_gvi.py` | Statistical comparison of GVI across cities |
| `compare_pedestrian_solar.py` | Statistical comparison of solar irradiance across cities |
| `lod_comparison_pedestrian_solar.py` | Paired LoD2 vs LoD1 solar irradiance analysis |

## Pipeline Overview

1. **Parse** CityGML → extract terrain, building, bridge, and vegetation meshes
2. **Create DEM** from terrain TIN and voxelize subsurface
3. **Download land-cover** grid; assign semantic labels
4. **Rasterize** building/bridge meshes into height grids
5. **Download canopy-height** data (merged with CityGML vegetation)
6. **Voxelize** all meshes into a shared 3-D grid, overlay land-cover & canopy
7. **Combine** all layers → VoxCity model

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
