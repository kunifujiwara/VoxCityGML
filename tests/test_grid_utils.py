"""compute_grid_params must match voxcity's canonical grid sizing."""
import pytest

from voxcitygml.grid_utils import compute_grid_params
from voxcitygml.citygml.coordinates import create_rectangle
from voxcity.geoprocessor.raster.core import compute_grid_geometry


@pytest.mark.parametrize("size,meshsize", [(500, 1.0), (500, 5.0),
                                           (333, 2.0), (1000, 10.0)])
def test_matches_voxcity_grid_size(size, meshsize):
    rect = create_rectangle(139.77, 35.65, size)
    gp = compute_grid_params(rect, meshsize)
    geom = compute_grid_geometry(rect, meshsize)
    # voxcity grid_size[0] is along side_1 (SW→NW, N-S → rows),
    # grid_size[1] along side_2 (SW→SE, E-W → cols)
    assert (gp.n_rows, gp.n_cols) == tuple(geom["grid_size"])


def test_delegation_is_actually_used(monkeypatch):
    """compute_grid_params must call voxcity, not its own arithmetic."""
    import voxcitygml.grid_utils as gu
    called = {}
    real = gu.compute_grid_geometry

    def spy(rect, meshsize):
        called['yes'] = True
        return real(rect, meshsize)

    monkeypatch.setattr(gu, 'compute_grid_geometry', spy)
    rect = create_rectangle(139.77, 35.65, 200)
    gu.compute_grid_params(rect, 2.0)
    assert called.get('yes')


def test_non_square_rectangle_axis_mapping():
    from voxcitygml.grid_utils import compute_grid_params
    # 200 m N-S x 600 m E-W rectangle built manually around a center
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lon0, lat0 = 139.77, 35.65
    # crude degree offsets: 1 deg lat ~ 110.95 km, 1 deg lon ~ 90.4 km at 35.65N
    dlat = 100 / 110950.0
    dlon = 300 / 90420.0
    sw = (lon0 - dlon, lat0 - dlat)
    nw = (lon0 - dlon, lat0 + dlat)
    ne = (lon0 + dlon, lat0 + dlat)
    se = (lon0 + dlon, lat0 - dlat)
    gp = compute_grid_params([sw, nw, ne, se], 2.0)
    # ~200 m / 2 m = ~100 rows (N-S), ~600 m / 2 m = ~300 cols (E-W)
    assert 95 <= gp.n_rows <= 105
    assert 290 <= gp.n_cols <= 310
