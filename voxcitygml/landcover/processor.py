"""
Land-cover grid acquisition.

Uses ``voxcity.generator.grids.get_land_cover_grid`` to download a
land-cover classification grid matching the target rectangle and
resolution, or -- for ``land_cover_source='CityGML'`` -- rasterizes the
dataset's own ``luse:LandUse`` features.

Whatever the source, the grid is returned in the voxcity land-cover row
order: **row-reversed** relative to the north-up DEM / building grids
(row 0 = the southern edge for an unrotated rectangle).  Consumers
(``voxelizer3d._apply_land_cover``, ``export_obj``) apply ``np.flipud``
before use.

The land cover grid is used to set semantic labels on the topmost
terrain voxels (the ground surface layer) in the final VoxCity model.
"""

from typing import List, Tuple, Union
import numpy as np


def get_land_cover_grid(
    rectangle_vertices: List[Tuple[float, float]],
    meshsize: float,
    land_cover_source: str,
    output_dir: str = "output",
    *,
    citygml_path: Union[str, List[str]] = "",
    **kwargs,
) -> np.ndarray:
    """Download / generate a land-cover classification grid.

    Parameters
    ----------
    rectangle_vertices : list[(lon, lat)]
        [SW, NW, NE, SE] target rectangle.
    meshsize : float
        Grid resolution in metres.
    land_cover_source : str
        Data source name (e.g. ``'OpenStreetMap'``,
        ``'OpenEarthMapJapan'``, ``'ESA WorldCover'``,
        ``'CityGML'``, etc.).
    output_dir : str
        Working directory for cached downloads.
    citygml_path : str
        Path to CityGML dataset directory.  Required when
        *land_cover_source* is ``'CityGML'``.

    Returns
    -------
    np.ndarray
        2-D int32 grid of source-specific class indices
        (will be converted to VoxCity standard 1-based indices
        during voxelisation).
    """
    # CityGML land use – parsed directly from luse:LandUse features
    if land_cover_source == "CityGML":
        from .citygml_landcover import get_citygml_land_cover_grid

        if not citygml_path:
            raise ValueError(
                "citygml_path is required when land_cover_source='CityGML'"
            )
        print(f"Generating land cover from CityGML land use data...")
        lc_grid = get_citygml_land_cover_grid(
            citygml_path, rectangle_vertices, meshsize,
        )
        print(f"  Land cover grid shape: {lc_grid.shape}")
        return lc_grid

    import matplotlib as _mpl
    _prev_backend = _mpl.get_backend()
    _mpl.use("Agg")  # non-interactive – suppress plt.show() inside voxcity
    try:
        from voxcity.generator.grids import get_land_cover_grid as _voxcity_get_lc

        print(f"Downloading land cover ({land_cover_source})...")
        lc_grid = _voxcity_get_lc(
            rectangle_vertices,
            meshsize,
            land_cover_source,
            output_dir,
            **kwargs,
        )
    finally:
        _mpl.use(_prev_backend)  # restore original backend

    print(f"  Land cover grid shape: {lc_grid.shape}")
    return lc_grid
