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

## Re-applying a Revised Canopy

`reapply_canopy` overlays a new canopy onto a model's **existing** voxel grid,
in place — for example after refining canopy heights against an nDSM:

```python
from voxcitygml import generate_voxcity, reapply_canopy

city = generate_voxcity(config)
reapply_canopy(city, refined_canopy_top, refined_canopy_bottom)
```

Unlike `voxcity.generator.update.regenerate_voxels`, it does **not** rebuild
the grid from the 2.5-D component grids, so mesh-voxelized LOD2 roof and wall
geometry survives. Buildings, bridges, terrain and land cover are never
touched. Columns holding CityGML vegetation keep their crown geometry; canopy
fills the gaps around them. Either the whole update lands or none of it does.

**Orientation matters and is not checkable.** `canopy_top` / `canopy_bottom`
must be **north-up** (row 0 = north), matching `voxels.classes`,
`dem.elevation` and `tree_canopy.top`. `land_cover.classes` is south-up — a
canopy built in that frame must be `np.flipud`-ed first, or the result is
mirrored north-to-south and still looks plausible.

### Model extras

Models built with `use_3d_voxelizer=True` (the default) carry two extra keys
in `city.extras` that `reapply_canopy` depends on:

| Key | Meaning |
|-----|---------|
| `voxel_min_z` | `float` — elevation (m) of the bottom face of the `z=0` voxel layer, i.e. the grid's vertical datum. `None` on the legacy `use_3d_voxelizer=False` path, which exposes no datum; `reapply_canopy` raises without it. |
| `mesh_vegetation_mask` | `(n_rows, n_cols)` bool, north-up — columns whose tree voxels came from CityGML vegetation meshes rather than from the canopy overlay. Captured before any canopy voxel is written, so it cannot be recovered from a finished grid. |

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

## Parse Cache

Parsing CityGML XML dominates run time, so the first parse of each `.gml` file
is snapshotted in binary form and reused on every later run.

- **Where:** a `.voxcitygml_cache/` directory created **inside your CityGML
  dataset directory** (next to `udx/` for PLATEAU datasets), mirroring the
  dataset's own layout.
- **Safe to delete at any time** — the whole directory, or any file inside it.
  It is purely an accelerator: if an entry is missing, unreadable, or out of
  date, the GML is simply parsed again. Entries are invalidated automatically
  when the source file changes. A stray `.tmp` file, left behind only by a
  hard kill or power loss mid-write, is inert and can be deleted too.
- **Size:** budget roughly the size of the dataset itself, with individual
  entries in the tens of megabytes — terrain (DEM) tiles are much the largest,
  around 50 MB each. Each distinct cache key holds a full independent
  snapshot, so a dataset queried at LOD1, LOD2 *and* automatic LOD keeps three
  complete copies of its building meshes. For a full-city dataset this is a
  multi-gigabyte commitment.
- **No eviction.** Nothing is reclaimed automatically. A *changed* GML file is
  fine — its entry is overwritten in place — but entries belonging to GML
  files you later delete or rename are orphaned and remain until you remove
  them yourself.
- **Disabling:** set `VoxelizerConfig(use_parse_cache=False)` to always parse
  the XML (the same switch is available as `use_parse_cache=False` on
  `parse_citygml_directory` if you call the parser directly). If the dataset
  directory is read-only, caching disables itself after a few warnings and
  parsing continues unaffected.

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
