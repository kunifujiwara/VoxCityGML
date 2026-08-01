"""CityGML land-cover rasterisation in the affine GridParams frame.

``get_citygml_land_cover_grid`` used to build its own axis-aligned grid
from the rectangle's lon/lat bounding box with a flat-earth degree
approximation.  For a rotated rectangle that was wrong twice over: the
*shape* was bbox-sized rather than side-sized (so it disagreed with the
DEM / building grids and was silently nearest-neighbour resized), and the
*rasterisation* was axis-aligned.  These tests pin the affine behaviour.

Two things matter and are easy to get backwards:

* **Row order.**  ``GridParams`` is north-up (row 0 = the NW-anchored
  first row), but the land-cover contract is the voxcity one: the array
  is handed back row-reversed, because every consumer
  (``voxelizer3d._apply_land_cover``, ``export_obj``) applies
  ``np.flipud``.  A flip here would mirror land cover north-south on
  *every* dataset, rotated or not, and only show up as subtly wrong
  ground classes in the rendered model.
* **Rotation.**  Cell centres must come from the affine frame, not from
  ``linspace`` over the bounding box.

The offline tests below build a synthetic PLATEAU-shaped ``luse`` dataset
so they run in CI; the dataset-gated one cross-checks real PLATEAU data
against an independent *vector* reference.
"""
import os

import numpy as np
import pytest

from voxcitygml.grid_utils import compute_grid_params
from voxcitygml.landcover.citygml_landcover import get_citygml_land_cover_grid
from voxcitygml.citygml.coordinates import create_rectangle

from .geo_helpers import geodesic_rect

# PLATEAU Common_landUseType codes -> internal VoxCity codes, per the
# module's own table.  Picked to be far apart so a mix-up is obvious.
LUSE_FOREST, VOX_TREE = 203, 5
LUSE_WATER, VOX_WATER = 204, 9
VOX_NODATA = 14


# ---------------------------------------------------------------------------
# Synthetic dataset (offline)
# ---------------------------------------------------------------------------

def _write_luse_dataset(root, polygons):
    """Write a minimal PLATEAU-layout ``udx/luse/*.gml``.

    *polygons* is a list of ``(plateau_class, [(lon, lat), ...])``.  GML
    posLists are written as ``lat lon z`` triples, which is what PLATEAU
    uses and what the parser expects.
    """
    luse_dir = root / "udx" / "luse"
    luse_dir.mkdir(parents=True, exist_ok=True)

    all_lon = [lon for _c, ring in polygons for lon, _lat in ring]
    all_lat = [lat for _c, ring in polygons for _lon, lat in ring]
    blocks = []
    for cls, ring in polygons:
        closed = list(ring) + [ring[0]]
        pos = " ".join(f"{lat:.10f} {lon:.10f} 0.0" for lon, lat in closed)
        blocks.append(
            "<core:cityObjectMember><luse:LandUse gml:id='x'>"
            f"<luse:class>{cls}</luse:class>"
            "<luse:lod1MultiSurface><gml:MultiSurface><gml:surfaceMember>"
            "<gml:Polygon><gml:exterior><gml:LinearRing>"
            f"<gml:posList>{pos}</gml:posList>"
            "</gml:LinearRing></gml:exterior></gml:Polygon>"
            "</gml:surfaceMember></gml:MultiSurface></luse:lod1MultiSurface>"
            "</luse:LandUse></core:cityObjectMember>"
        )

    # The envelope drives ``_file_envelope_intersects``; write it for real
    # so that fast path is exercised rather than falling back to "include".
    doc = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<core:CityModel xmlns:core='http://www.opengis.net/citygml/2.0' "
        "xmlns:gml='http://www.opengis.net/gml' "
        "xmlns:luse='http://www.opengis.net/citygml/landuse/2.0'>"
        "<gml:boundedBy><gml:Envelope srsName='EPSG:6697'>"
        f"<gml:lowerCorner>{min(all_lat):.10f} {min(all_lon):.10f} 0.0</gml:lowerCorner>"
        f"<gml:upperCorner>{max(all_lat):.10f} {max(all_lon):.10f} 0.0</gml:upperCorner>"
        "</gml:Envelope></gml:boundedBy>"
        + "".join(blocks) +
        "</core:CityModel>"
    )
    (luse_dir / "53393671_luse_6697_op.gml").write_text(doc, encoding="utf-8")
    return str(root)


def _frame_quad(gp, row0, row1, col0, col1, pad=0.0):
    """Polygon corners (lon, lat) for a rectangle in *frame* coordinates.

    Coordinates are continuous cell-centre indices, so the outer corner of
    cell (0, 0) is ``(-0.5, -0.5)`` and the boundary between rows ``k-1``
    and ``k`` is at ``k - 0.5``.  *pad* pushes the edges outward (in cells)
    so a boundary the test does not care about cannot clip an edge cell.
    """
    corners = [(row0 - pad, col0 - pad), (row0 - pad, col1 + pad),
               (row1 + pad, col1 + pad), (row1 + pad, col0 - pad)]
    rows = np.array([r for r, _c in corners])
    cols = np.array([c for _r, c in corners])
    lon, lat = gp.rowcol_to_lonlat(rows, cols)
    return list(zip(lon.tolist(), lat.tolist()))


@pytest.mark.parametrize("rotation", [0.0, 30.0, 90.0, 137.0])
def test_grid_shape_follows_grid_params_not_the_bbox(tmp_path, rotation):
    """Shape must equal ``compute_grid_params``, at every rotation.

    The old implementation sized the grid from the lon/lat bounding box:
    at 30 deg a 200 x 150 m rectangle spans 248 x 230 m, giving (115, 124)
    cells instead of (75, 100).  It then reached the pipeline as a shape
    mismatch against the DEM grid and was silently ``zoom``-resized.
    """
    rect = geodesic_rect(139.7725, 35.6481, 200.0, 150.0, rotation)
    gp = compute_grid_params(rect, 2.0)
    root = _write_luse_dataset(
        tmp_path, [(LUSE_FOREST, _frame_quad(gp, -0.5, gp.n_rows - 0.5,
                                             -0.5, gp.n_cols - 0.5, pad=1.0))])

    grid = get_citygml_land_cover_grid(root, rect, 2.0)

    assert grid.shape == gp.shape
    # Fully covered: a bbox-sized frame would leave No Data in the corners.
    assert (grid == VOX_TREE).all()


@pytest.mark.parametrize("rotation", [0.0, 30.0, 90.0, 137.0])
def test_row_order_and_rotation_are_correct(tmp_path, rotation):
    """Two polygons placed in known parts of the *frame* land in known cells.

    Layout, in ``GridParams`` (north-up) coordinates:

        rows [0, H)      -> forest  over the full width
        rows [H, n_rows) -> water   over cols [0, W), No Data over the rest

    The expected array is asymmetric in both axes, so a north-south flip,
    an east-west flip, a 180 deg rotation and a row/col transpose all fail
    it.  Because it is expressed in *frame* coordinates it holds at every
    rotation -- which is exactly what the old bbox rasteriser could not do.
    """
    rect = geodesic_rect(139.7725, 35.6481, 200.0, 150.0, rotation)
    gp = compute_grid_params(rect, 2.0)
    H, W = gp.n_rows // 2, gp.n_cols // 2
    assert H >= 2 and W >= 2

    root = _write_luse_dataset(tmp_path, [
        # Northern band: rows 0 .. H-1 (boundary sits at H - 0.5, i.e.
        # exactly half a cell past the last included centre).
        (LUSE_FOREST, _frame_quad(gp, -0.5, H - 0.5, -0.5, gp.n_cols - 0.5,
                                  pad=0.0)),
        # South-west block: rows H .. end, cols 0 .. W-1.
        (LUSE_WATER, _frame_quad(gp, H - 0.5, gp.n_rows - 0.5, -0.5, W - 0.5,
                                 pad=0.0)),
    ])

    grid = get_citygml_land_cover_grid(root, rect, 2.0)
    assert grid.shape == gp.shape

    # The returned array is row-reversed by contract; undo that to compare
    # in the frame the polygons were expressed in.
    north_up = np.flipud(grid)

    expected = np.full(gp.shape, VOX_NODATA, dtype=np.int32)
    expected[:H, :] = VOX_TREE
    expected[H:, :W] = VOX_WATER

    # Only the polygon edges are padded to zero, so allow the one-cell ring
    # along each internal boundary to fall either way; everything else must
    # match exactly.
    mismatch = north_up != expected
    interior = np.ones(gp.shape, dtype=bool)
    interior[H - 1:H + 1, :] = False
    interior[:, W - 1:W + 1] = False
    assert not mismatch[interior].any(), (
        f"{int(mismatch[interior].sum())} interior cells misclassified -- "
        f"row order or rotation is wrong")

    # And the flipped readings must be *badly* wrong, so the assertion above
    # is not passing on a symmetric grid.
    for label, wrong in (("north-south", grid),
                         ("east-west", np.fliplr(north_up)),
                         ("180 deg", north_up[::-1, ::-1])):
        agreement = float((wrong == expected).mean())
        assert agreement < 0.9, (
            f"{label} flip agrees {agreement:.3f} with the expected grid -- "
            f"the layout is too symmetric to pin orientation")


def test_returned_grid_is_row_reversed_relative_to_the_frame(tmp_path):
    """Pins the voxcity land-cover row-order contract on its own.

    ``_apply_land_cover`` and ``export_obj`` both ``np.flipud`` the grid
    before use.  If this function ever returned north-up instead, land
    cover would be mirrored north-south everywhere and nothing else in the
    suite would notice.
    """
    rect = create_rectangle(139.7725, 35.6481, 200)
    gp = compute_grid_params(rect, 2.0)
    # A band across the northern third of the rectangle only.
    third = gp.n_rows // 3
    root = _write_luse_dataset(
        tmp_path,
        [(LUSE_WATER, _frame_quad(gp, -0.5, third - 0.5,
                                  -0.5, gp.n_cols - 0.5, pad=0.0))])

    grid = get_citygml_land_cover_grid(root, rect, 2.0)

    # Northern band -> *last* rows of the returned array.
    assert (grid[-third + 1:, :] == VOX_WATER).all()
    assert (grid[:-third - 1, :] == VOX_NODATA).all()


def test_missing_luse_directory_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="luse"):
        get_citygml_land_cover_grid(str(tmp_path),
                                    create_rectangle(139.77, 35.65, 200), 2.0)


# ---------------------------------------------------------------------------
# Real PLATEAU data, cross-checked against an independent vector reference
# ---------------------------------------------------------------------------

DATASET = os.environ.get(
    "VOXCITYGML_PLATEAU_TEST_DATA",
    r"D:\03_Data\citygml\plateau\13102_chuo-ku_pref_2023_citygml_2_op",
)

requires_luse = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATASET, "udx", "luse")),
    reason="local PLATEAU dataset with udx/luse not available",
)


def _vector_reference(rect, meshsize):
    """Per-cell class from the *vector* polygon path, north-up in the frame.

    Deliberately a different code path from the rasteriser under test:
    ``get_citygml_land_cover_polygons`` clips the polygons and resolves
    class priority by geometric differencing, so the result is a set of
    disjoint polygons that can be point-queried directly.  Agreement
    therefore checks the raster's *geolocation*, not just its internal
    consistency.
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    from voxcitygml.landcover.citygml_landcover import (
        get_citygml_land_cover_polygons)

    gp = compute_grid_params(rect, meshsize)
    entries = get_citygml_land_cover_polygons(DATASET, rect)
    geoms = [p for _c, p in entries]
    codes = [c for c, _p in entries]
    tree = STRtree(geoms)

    PX, PY = gp.cell_centres()
    ref = np.full(gp.shape, VOX_NODATA, dtype=np.int32)
    for r in range(gp.n_rows):
        for c in range(gp.n_cols):
            pt = Point(PX[r, c], PY[r, c])
            for i in tree.query(pt):
                if geoms[i].contains(pt):
                    ref[r, c] = codes[i]
                    break
    return gp, ref


@requires_luse
@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_real_dataset_matches_vector_reference(rotation):
    """Real PLATEAU luse: the raster must land where the vectors say.

    Measured on the reference rectangle (Chuo-ku, 200 x 150 m, 2 m cells):
    agreement 1.000 at both rotations, while the mis-oriented readings
    score 0.19-0.42.  The 0.99 threshold leaves room for a cell or two of
    boundary disagreement between the raster and the differencing-based
    vector path without going soft.
    """
    rect = geodesic_rect(139.7725, 35.6481, 200.0, 150.0, rotation)
    gp, ref = _vector_reference(rect, 2.0)

    grid = get_citygml_land_cover_grid(DATASET, rect, 2.0)
    assert grid.shape == gp.shape, (
        f"shape {grid.shape} != GridParams {gp.shape} -- the grid is still "
        f"sized from the lon/lat bounding box")

    north_up = np.flipud(grid)
    agreement = float((north_up == ref).mean())
    assert agreement > 0.99, (
        f"raster agrees with the vector reference on only {agreement:.3f} of "
        f"cells at rotation {rotation}")

    # The land cover must not be uniform, or the check above is vacuous.
    assert len(np.unique(ref)) >= 3, np.unique(ref)

    for label, wrong in (("north-south", grid),
                         ("east-west", np.fliplr(north_up)),
                         ("180 deg", north_up[::-1, ::-1])):
        assert float((wrong == ref).mean()) < 0.9, (
            f"{label} flip also agrees -- the scene is too symmetric")
