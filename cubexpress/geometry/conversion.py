"""Coordinate conversion and raster transform utilities."""

from __future__ import annotations

import utm
from pyproj import CRS, Transformer

from cubexpress.core.exceptions import ValidationError
from cubexpress.core.types import RasterTransform


def parse_edge_size(edge_size: int | tuple[int, int]) -> tuple[int, int]:
    """
    Parse edge_size input into (width, height) tuple.

    Args:
        edge_size: Size specification (int for square, tuple for rectangle)

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        ValidationError: If input is invalid
    """
    if isinstance(edge_size, int):
        if edge_size <= 0:
            raise ValidationError(f"edge_size must be positive, got {edge_size}")
        return (edge_size, edge_size)

    if len(edge_size) != 2:
        raise ValidationError(f"edge_size tuple must have 2 elements, got {len(edge_size)}")

    width, height = edge_size
    if width <= 0 or height <= 0:
        raise ValidationError(f"edge_size values must be positive, got {edge_size}")

    return (width, height)


def geo2utm(lon: float, lat: float) -> tuple[float, float, str]:
    """
    Convert lat/lon to UTM coordinates and EPSG code.

    Uses the utm library for standard conversion.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        Tuple of (x, y, epsg_code) where EPSG code is formatted as 'EPSG:XXXXX'

    Raises:
        utm.OutOfRangeError: If coordinates are outside valid UTM range
    """
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg_code = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return float(x), float(y), f"EPSG:{epsg_code}"


def lonlat2rt_utm_or_ups(lon: float, lat: float) -> tuple[float, float, str]:
    """
    Calculate UTM coordinates using pyproj (fallback for geo2utm).

    This method is more robust than the utm library and works globally,
    including near the poles. Uses standard UTM zones for all latitudes
    to match Google Earth Engine behavior.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees

    Returns:
        Tuple of (x, y, epsg_code)
    """
    zone = int((lon + 180) // 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    crs = CRS.from_epsg(epsg_code)

    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    return float(x), float(y), f"EPSG:{epsg_code}"


def lonlat2rt(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    scale: int,
    grid_reference: tuple[float, float, int] | None = None,
) -> RasterTransform:
    """
    Generate a RasterTransform from geographic coordinates.

    Converts (lon, lat) to UTM projection and builds geospatial metadata
    including affine transformation parameters. The Y-scale is negative
    because raster images have their origin at the top-left corner.

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        edge_size: Output raster size
            - int: creates square (width=height=edge_size)
            - tuple: specifies (width, height) in pixels
        scale: Spatial resolution in meters per pixel
        grid_reference: Tuple of (translate_x, translate_y, native_scale) from a
            reference image to align the output grid. If provided, the top-left
            corner will be snapped to the same grid as the reference image.

    Returns:
        RasterTransform with CRS, geotransform, and dimensions

    Examples:
        >>> # Without alignment - centered exactly on point
        >>> rt = lonlat2rt(lon=-76.0, lat=40.0, edge_size=512, scale=30)

        >>> # With grid alignment from reference image
        >>> rt = lonlat2rt(lon=-76.0, lat=40.0, edge_size=512, scale=30,
        ...                grid_reference=(502530.0, -1959510.0, 60))
    """
    try:
        x, y, crs = geo2utm(lon, lat)
    except Exception:
        x, y, crs = lonlat2rt_utm_or_ups(lon, lat)

    width, height = parse_edge_size(edge_size)

    half_width_m = (width * scale) / 2
    half_height_m = (height * scale) / 2

    # Calculate top-left corner (before alignment)
    ul_x = x - half_width_m
    ul_y = y + half_height_m

    # Align to reference grid if provided
    if grid_reference is not None:
        ref_x, ref_y, _ = grid_reference
        # Snap to the same grid as reference image
        # Formula: find nearest grid line that matches reference origin
        ul_x = ref_x + round((ul_x - ref_x) / scale) * scale
        ul_y = ref_y + round((ul_y - ref_y) / scale) * scale

    geotransform = {
        "scaleX": scale,
        "shearX": 0,
        "translateX": ul_x,
        "scaleY": -scale,
        "shearY": 0,
        "translateY": ul_y,
    }

    return RasterTransform(crs=crs, geotransform=geotransform, width=width, height=height)
