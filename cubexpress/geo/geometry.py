"""Convert RasterTransforms and points into ee.Geometry search regions.

A RasterTransform describes the pixel grid you will DOWNLOAD. To DISCOVER which
images exist there, Earth Engine needs a search region (an ee.Geometry) to pass
to filterBounds. This module bridges the two.

Robustness note: the geometry is built in the RasterTransform's own CRS
(typically UTM) and is NOT reprojected to EPSG:4326. Reprojecting a rectangle to
4326 is safe at mid-latitudes but degenerates near the poles and the
antimeridian (self-intersecting, wrap-around polygons that break filterBounds
silently). Earth Engine reprojects internally per operation, so leaving the
geometry in UTM is both correct and safe at any latitude. evenOdd=False avoids
ambiguous polygon-fill interpretation in projected CRS.
"""

from __future__ import annotations

from cubexpress.geo.construct import point_to_rt
from cubexpress.geo.transform import RasterTransform


def rt_to_geometry(rt: RasterTransform):
    """Build an ee.Geometry rectangle covering a RasterTransform's extent.

    The rectangle is created in the RasterTransform's native CRS and is NOT
    reprojected, so it stays valid near poles and the antimeridian. It covers
    exactly the area the RasterTransform would download.

    Earth Engine must be initialized before calling this.

    Args:
        rt: The RasterTransform whose extent to cover.

    Returns:
        An ee.Geometry.Rectangle in rt.crs.
    """
    import ee

    xmin, ymin, xmax, ymax = rt.bbox()
    return ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax],
        proj=rt.crs,
        evenOdd=False,
    )


def point_to_geometry(
    lon: float,
    lat: float,
    width: int,
    height: int,
    scale: float,
):
    """Build an ee.Geometry rectangle around a point, sized in pixels.

    Convenience for the common case: take a lon/lat center and a patch size,
    and get the search region. Internally builds a RasterTransform (auto-UTM)
    and converts it, so the geometry aligns exactly with what point_to_rt would
    download for the same arguments.

    Earth Engine must be initialized before calling this.

    Args:
        lon: Longitude in decimal degrees, range [-180, 180].
        lat: Latitude in decimal degrees, range [-90, 90].
        width: Patch width in pixels (must be > 0).
        height: Patch height in pixels (must be > 0).
        scale: Pixel size in meters (must be > 0).

    Returns:
        An ee.Geometry.Rectangle in the appropriate UTM zone for (lon, lat).
    """
    rt = point_to_rt(lon=lon, lat=lat, width=width, height=height, scale=scale)
    return rt_to_geometry(rt)