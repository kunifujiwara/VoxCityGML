"""End-to-end test: voxelize Munich CityGML data, compute ground-level SVF, and visualize."""
import numpy as np
from voxcitygml.models import VoxelizerConfig
from voxcitygml.pipeline import VoxCityGML
from voxcity.simulator_gpu import get_sky_view_factor_map
from voxcity.visualizer import visualize_voxcity

if __name__ == "__main__":
    # ── 1. Build voxel city ──────────────────────────────────────────────
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
        output_dir="output/test_munich_svf",
        n_workers=4,
        building_lod=2,
    )

    city = VoxCityGML(cfg).run()

    # ── 2. Compute ground-level Sky View Factor (GPU-accelerated) ────────
    print("\nComputing Sky View Factor (GPU)...")
    svf_map = get_sky_view_factor_map(
        city,
        show_plot=False,
        view_point_height=1.5,
        N_azimuth=120,
        N_elevation=20,
    )
    print(f"SVF map shape: {svf_map.shape}, "
          f"range: [{svf_map[~np.isnan(svf_map)].min():.3f}, "
          f"{svf_map[~np.isnan(svf_map)].max():.3f}]")

    # ── 3. Interactive visualisation with SVF ground overlay ─────────────
    print("\nLaunching interactive visualisation with SVF overlay...")
    visualize_voxcity(
        city,
        mode="interactive",
        downsample=1,
        ground_sim_grid=svf_map,
        ground_colormap="BuPu_r",
        ground_vmin=0.0,
        ground_vmax=1.0,
        ground_view_point_height=1.5,
        sim_surface_opacity=0.95,
    )
