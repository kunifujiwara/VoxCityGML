"""Geometry helpers shared by several test modules.

Deliberately **not** named ``test_*`` so pytest does not collect it: it
holds constructors, not tests.  Import from here rather than reaching
into another test module.
"""
import math


def geodesic_rect(center_lon, center_lat, width_m, height_m, rotation_deg):
    """Rectangle built exactly as the app's /api/rectangle-from-dimensions
    does: each corner placed by ``Geod.fwd`` at the bearing/distance of its
    rotated local-frame offset.  This is the production construction, and
    unlike a degree-space rotation it is *not* an exact parallelogram in
    degree space — even at rotation 0 it carries ~1e-4 relative skew.

    Note the sign convention, which mirrors the endpoint: the corner
    offsets are rotated by ``-radians(rotation_deg)``, so a naive
    ``+radians`` helper builds the mirror-image rectangle.

    Returns the four corners as ``[SW, NW, NE, SE]`` ``(lon, lat)`` pairs.
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    hw, hh = width_m / 2.0, height_m / 2.0
    a = -math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for x, y in [(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)]:
        rx = x * ca - y * sa
        ry = x * sa + y * ca
        lon, lat, _ = geod.fwd(center_lon, center_lat,
                               math.degrees(math.atan2(rx, ry)),
                               math.hypot(rx, ry))
        out.append((lon, lat))
    return out
