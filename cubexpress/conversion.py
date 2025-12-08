import utm
from pyproj import CRS, Transformer
from cubexpress.geotyping import RasterTransform

def parse_edge_size(edge_size: int | tuple[int, int]) -> tuple[int, int]:
    """
    Parse edge_size input into (width, height) tuple.
    
    Args:
        edge_size: Size specification
        
    Returns:
        tuple[int, int]: (width, height) in pixels
        
    Raises:
        ValueError: If tuple length != 2 or values <= 0
    """
    if isinstance(edge_size, int):
        if edge_size <= 0:
            raise ValueError(f"edge_size must be positive, got {edge_size}")
        return (edge_size, edge_size)
    else:
        if len(edge_size) != 2:
            raise ValueError(f"edge_size tuple must have 2 elements, got {len(edge_size)}")
        width, height = edge_size
        if width <= 0 or height <= 0:
            raise ValueError(f"edge_size values must be positive, got {edge_size}")
        return (width, height)

def geo2utm(
    lon: float, 
    lat: float
) -> tuple[float, float, str]:
    """
    Converts latitude and longitude coordinates to UTM coordinates and returns the EPSG code.

    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.

    Returns:
        tuple[float, float, str]: UTM coordinates (x, y) in meters and EPSG code as string.
        
    Raises:
        utm.OutOfRangeError: If coordinates are outside valid UTM range.
    """
    x, y, zone, _ = utm.from_latlon(lat, lon)
    epsg_code = f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"
    return float(x), float(y), f"EPSG:{epsg_code}"


def lonlat2rt_utm_or_ups(
    lon: float, 
    lat: float
) -> tuple[float, float, str]:
    """
    Calculate UTM coordinates using pyproj (fallback for geo2utm).
    
    Uses standard UTM zones for all latitudes, matching GEE behavior.
    This method is more robust than the utm library and works globally.
    
    Note:
        UTM is designed for [-80°, 84°] but works globally with 
        acceptable distortions for small tiles.
    
    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.
        
    Returns:
        tuple[float, float, str]: UTM coordinates (x, y) in meters and EPSG code as string.
    """
    zone = int((lon + 180) // 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    crs = CRS.from_epsg(epsg_code)
    
    to_xy = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = to_xy.transform(lon, lat)
    
    return float(x), float(y), f"EPSG:{epsg_code}"


def lonlat2rt(
    lon: float, 
    lat: float, 
    edge_size: int | tuple[int, int], 
    scale: int
) -> RasterTransform:
    """
    Generates a ``RasterTransform`` for a given point by converting geographic (lon, lat) coordinates
    to UTM projection and building the necessary geotransform metadata.

    This function:
      1. Converts the input (lon, lat) to UTM coordinates using :func:`geo2utm`.
      2. If that fails (e.g., near poles), falls back to pyproj-based calculation.
      3. Defines the extent of the raster in UTM meters based on the specified dimensions
         and ``scale`` (meters per pixel).
      4. Sets the Y-scale to be negative (``-scale``) because geospatial images typically consider 
         the origin at the top-left corner, resulting in a downward Y axis.

    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.
        edge_size (int | tuple[int, int]): Size of the output raster. 
            If int, creates a square (width=height=edge_size).
            If tuple, specifies (width, height) in pixels.
        scale (int): Spatial resolution in meters per pixel.

    Returns:
        RasterTransform: A Pydantic model containing:
         - ``crs``: The EPSG code in the form ``"EPSG:XYZ"``,
         - ``geotransform``: A dictionary with the affine transform parameters,
         - ``width`` and ``height``.

    Examples:
        Square raster:
        
        >>> rt = cubexpress.lonlat2rt(
        ...     lon=-76.0, lat=40.0,
        ...     edge_size=512, scale=30
        ... )
        >>> print(rt.width, rt.height)
        512 512
        
        Rectangular raster:
        
        >>> rt = cubexpress.lonlat2rt(
        ...     lon=-76.0, lat=40.0,
        ...     edge_size=(1024, 512), scale=30
        ... )
        >>> print(rt.width, rt.height)
        1024 512
    """
    try:
        x, y, crs = geo2utm(lon, lat)
    except Exception:
        x, y, crs = lonlat2rt_utm_or_ups(lon, lat)
    
    # Parse edge_size
    width, height = parse_edge_size(edge_size)
        
    half_width = (width * scale) / 2
    half_height = (height * scale) / 2

    geotransform = dict(
        scaleX=scale,
        shearX=0,
        translateX=x - half_width,
        scaleY=-scale,
        shearY=0,
        translateY=y + half_height,
    )

    return RasterTransform(
        crs=crs, geotransform=geotransform, width=width, height=height
    )