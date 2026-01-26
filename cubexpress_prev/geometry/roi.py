"""ROI (Region of Interest) creation for Earth Engine queries."""

from __future__ import annotations

import ee

from cubexpress.core.config import METERS_PER_DEGREE_LAT, METERS_PER_DEGREE_LON


def _square_roi(lon: float, lat: float, edge_size: int | tuple[int, int], scale: int) -> ee.Geometry:
    """Create a square Earth Engine Geometry around a center point.

    Uses flat-earth approximation to convert meters to degrees.

    Args:
        lon: Longitude of center
        lat: Latitude of center
        edge_size: Size in pixels (int for square, tuple for rectangle)
        scale: Pixel resolution in meters

    Returns:
        Earth Engine Polygon geometry
    """
    if isinstance(edge_size, int):
        width = height = edge_size
    else:
        width, height = edge_size

    half_width_m = width * scale / 2
    half_height_m = height * scale / 2

    half_width_deg = half_width_m / METERS_PER_DEGREE_LON
    half_height_deg = half_height_m / METERS_PER_DEGREE_LAT

    coords = [
        [lon - half_width_deg, lat - half_height_deg],  # SW
        [lon - half_width_deg, lat + half_height_deg],  # NW
        [lon + half_width_deg, lat + half_height_deg],  # NE
        [lon + half_width_deg, lat - half_height_deg],  # SE
        [lon - half_width_deg, lat - half_height_deg],  # SW (close)
    ]

    return ee.Geometry.Polygon(coords)
