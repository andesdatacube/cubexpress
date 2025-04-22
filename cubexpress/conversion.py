import utm

from cubexpress.geotyping import RasterTransform

# Define your GeotransformDict type if not already defined
GeotransformDict = dict[str, float]


def geo2utm(lon: float, lat: float) -> tuple[float, float, str]:
    """
    Converts latitude and longitude coordinates to UTM coordinates and returns the EPSG code.

    Args:
        lon (float): Longitude.
        lat (float): Latitude.

    Returns:
        Tuple[float, float, str]: UTM coordinates (x, y) and the EPSG code.
    """
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg_code = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return float(x), float(y), f"EPSG:{epsg_code}"


def lonlat2rt(lon: float, lat: float, edge_size: int, scale: int) -> RasterTransform:
    """
    Generates a ``RasterTransform`` for a given point by converting geographic (lon, lat) coordinates
    to UTM projection and building the necessary geotransform metadata.

    This function:
      1. Converts the input (lon, lat) to UTM coordinates using :func:`geo2utm`.
      2. Defines the extent of the raster in UTM meters based on the specified ``edge_size`` (width/height in pixels)
         and ``scale`` (meters per pixel).
      3. Sets the Y-scale to be negative (``-scale``) because geospatial images typically consider the origin at
         the top-left corner, resulting in a downward Y axis.

    Args:
        lon (float): The longitude coordinate.
        lat (float): The latitude coordinate.
        edge_size (int): Width and height of the output raster in pixels.
        scale (int): Spatial resolution in meters per pixel.

    Returns:
        RasterTransform: A Pydantic model containing:
         - ``crs``: The EPSG code in the form ``"EPSG:XYZ"``,
         - ``geotransform``: A dictionary with the affine transform parameters,
         - ``width`` and ``height``.

    Example:
        >>> import cubexpress
        >>> rt = cubexpress.lonlat2rt(
        ...     lon=-76.0,
        ...     lat=40.0,
        ...     edge_size=512,
        ...     scale=30
        ... )
        >>> print(rt)
    """
    x, y, crs = geo2utm(lon, lat)
    half_extent = (edge_size * scale) / 2

    geotransform = GeotransformDict(
        scaleX=scale,
        shearX=0,
        translateX=x - half_extent,
        scaleY=-scale,  # Y-axis is inverted in geospatial images
        shearY=0,
        translateY=y + half_extent,
    )

    return RasterTransform(
        crs=crs, geotransform=geotransform, width=edge_size, height=edge_size
    )
