"""
Command-line interface for VoxCityGML.

Usage::

    python -m voxcitygml \\
        --path /data/plateau/13101_chiyoda \\
        --center-lon 139.7671 --center-lat 35.6812 \\
        --size 500 --meshsize 1.0
"""

import os
import argparse

from .models import VoxelizerConfig
from .pipeline import VoxCityGML


def main():
    default_workers = max(1, min(16, (os.cpu_count() or 4) * 3 // 4))

    parser = argparse.ArgumentParser(
        description="VoxCityGML – voxelise CityGML data into a VoxCity model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m voxcitygml -p ./plateau_dataset --center-lon 139.7671 --center-lat 35.6812 -s 500

  # Custom voxel size and land cover
  python -m voxcitygml -p ./data --center-lon 139.75 --center-lat 35.68 -s 300 \\
      --meshsize 2.0 --land-cover OpenEarthMapJapan

  # Static canopy with custom tree height
  python -m voxcitygml -p ./data --center-lon 139.75 --center-lat 35.68 -s 200 \\
      --canopy Static --tree-height 12
""",
    )

    parser.add_argument('--path', '-p', type=str, required=True,
                        help='Path to CityGML dataset directory')
    parser.add_argument('--center-lon', type=float, required=True,
                        help='Centre longitude of target area')
    parser.add_argument('--center-lat', type=float, required=True,
                        help='Centre latitude of target area')
    parser.add_argument('--size', '-s', type=float, default=500.0,
                        help='Target area side length in metres (default: 500)')
    parser.add_argument('--meshsize', type=float, default=1.0,
                        help='Voxel resolution in metres (default: 1.0)')
    parser.add_argument('--land-cover', type=str, default='OpenStreetMap',
                        help='Land cover source (default: OpenStreetMap)')
    parser.add_argument('--canopy', type=str,
                        default='High Resolution 1m Global Canopy Height Maps',
                        help='Canopy height source (default: High Resolution 1m Global Canopy Height Maps)')
    parser.add_argument('--tree-height', type=float, default=10.0,
                        help='Static tree height when --canopy=Static (default: 10)')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--workers', '-w', type=int, default=default_workers,
                        help=f'Parallel CityGML parsing workers (default: {default_workers})')
    parser.add_argument('--gridvis', action='store_true',
                        help='Show matplotlib grid visualisations')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the VoxCity model to disk')
    parser.add_argument('--buffer', type=float, default=50.0,
                        help='Buffer around target rectangle for CityGML parsing (metres, default: 50)')
    parser.add_argument('--legacy-2d', action='store_true',
                        help='Use legacy 2.5-D voxelization instead of shared 3-D meshes')
    parser.add_argument('--max-voxel-ram', type=float, default=None,
                        help='Max RAM for 3-D voxel grid in MB (optional)')
    parser.add_argument('--building-lod', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Preferred CityGML building LOD (1-4). Default: highest available.')
    parser.add_argument('--dem-path', type=str, default=None,
                        help='Path to a GeoTIFF DEM/DTM file for terrain (optional)')
    parser.add_argument('--occupancy-threshold', type=float, default=0.0,
                        help='Min volume overlap fraction (0-1) for boundary voxels (default: 0 = any contact)')
    parser.add_argument('--occupancy-subdivisions', type=int, default=3,
                        help='Sub-divisions per axis for occupancy estimation (default: 3)')

    args = parser.parse_args()

    config = VoxelizerConfig(
        citygml_path=args.path,
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        size_meters=args.size,
        meshsize=args.meshsize,
        land_cover_source=args.land_cover,
        canopy_height_source=args.canopy,
        output_dir=args.output,
        n_workers=args.workers,
        static_tree_height=args.tree_height,
        gridvis=args.gridvis,
        save_output=not args.no_save,
        buffer_meters=args.buffer,
        use_3d_voxelizer=not args.legacy_2d,
        max_voxel_ram_mb=args.max_voxel_ram,
        building_lod=args.building_lod,
        dem_path=args.dem_path,
        occupancy_threshold=args.occupancy_threshold,
        occupancy_subdivisions=args.occupancy_subdivisions,
    )

    city = VoxCityGML(config).run()
    print("\nDone!")
    return city


if __name__ == "__main__":
    main()
