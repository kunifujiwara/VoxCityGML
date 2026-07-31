"""End-to-end LOD2 integration test against a local PLATEAU dataset.

Skipped automatically when the dataset is not present (e.g. CI).
"""
import os

import numpy as np
import pytest

DATASET = r"D:\03_Data\citygml\plateau\13102_chuo-ku_pref_2023_citygml_2_op"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATASET),
    reason="local PLATEAU dataset not available",
)


@pytest.mark.slow
def test_lod2_generate_voxcity_end_to_end(tmp_path):
    from voxcitygml import generate_voxcity, VoxelizerConfig
    from voxcitygml.citygml.coordinates import create_rectangle
    from voxcitygml.voxelizer3d import BUILDING_CODE

    # -3 is the building class code the VoxCity app and its renderer rely on;
    # pin it here so a change in either repo surfaces as a test failure rather
    # than as silently mis-classified voxels downstream.
    assert BUILDING_CODE == -3

    # A point inside PLATEAU tile 53393671 (Chuo-ku, Tokyo).
    rect = create_rectangle(139.7725, 35.6481, 200)
    cfg = VoxelizerConfig(
        citygml_path=DATASET,
        rectangle_vertices=rect,
        meshsize=2.0,
        building_lod=2,
        land_cover_source="OpenStreetMap",   # no GEE dependency
        canopy_height_source="Static",
        output_dir=str(tmp_path),
        save_output=False,
    )
    city = generate_voxcity(cfg)

    classes = city.voxels.classes
    assert classes.ndim == 3

    n_building = int(np.count_nonzero(classes == BUILDING_CODE))
    print(f"\nbuilding voxels: {n_building}, grid shape: {classes.shape}")
    assert n_building > 0, "expected building voxels in the grid"

    heights = city.buildings.heights
    assert np.any(heights > 0)
