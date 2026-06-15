"""Geometry constructors that build RasterTransforms from various inputs."""

from __future__ import annotations

import math   

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

import shapely
from shapely.ops import transform as shp_transform

from cubexpress.geo.transform import RasterTransform


def _utm_zone_epsg(lon: float, lat: float) -> str:
    if not -180 <= lon <= 180:
        raise ValueError(f"lon must be in [-180, 180], got {lon}")
    if not -90 <= lat <= 90:
        raise ValueError(f"lat must be in [-90, 90], got {lat}")

    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    if not utm_crs_list:
        raise ValueError(f"No UTM zone found for (lon={lon}, lat={lat})")
    return f"EPSG:{utm_crs_list[0].code}"


def point_to_rt(
    lon: float,
    lat: float,
    width: int,
    height: int,
    scale: float,
) -> RasterTransform:
    """Build a RasterTransform centered on (lon, lat).

    The resulting patch is `width` x `height` pixels at `scale` meters/pixel,
    projected to the appropriate UTM zone for the given coordinates.

    Args:
        lon: Longitude in decimal degrees, range [-180, 180].
        lat: Latitude in decimal degrees, range [-90, 90].
        width: Patch width in pixels (must be > 0).
        height: Patch height in pixels (must be > 0).
        scale: Pixel size in meters (must be > 0).

    Returns:
        RasterTransform anchored so its bounding box is centered on (lon, lat).
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    crs = _utm_zone_epsg(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    ul_x = cx - (width * scale) / 2
    ul_y = cy + (height * scale) / 2

    return RasterTransform(
        crs=crs,
        translate_x=ul_x,
        translate_y=ul_y,
        scale_x=scale,
        scale_y=-scale,
        width=width,
        height=height,
    )


def bbox_to_rt(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs: str,
    scale: float,
) -> RasterTransform:
    """Build a RasterTransform covering the given bbox at `scale` units/pixel.

    The output raster's upper-left corner is anchored at (xmin, ymax). If the
    bbox dimensions aren't exact multiples of `scale`, the resulting raster
    extends slightly beyond `xmax` and below `ymin` (rounded up) to guarantee
    full coverage of the input bbox.

    Args:
        xmin, ymin, xmax, ymax: bounding box coordinates in `crs`.
        crs: Coordinate Reference System (EPSG code or WKT).
        scale: Pixel size in CRS units (typically meters for projected CRS).

    Returns:
        RasterTransform whose bbox contains the input bbox at the given scale.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if xmin >= xmax:
        raise ValueError(f"xmin must be < xmax, got xmin={xmin}, xmax={xmax}")
    if ymin >= ymax:
        raise ValueError(f"ymin must be < ymax, got ymin={ymin}, ymax={ymax}")

    width = math.ceil((xmax - xmin) / scale)
    height = math.ceil((ymax - ymin) / scale)

    return RasterTransform(
        crs=crs,
        translate_x=xmin,
        translate_y=ymax,
        scale_x=scale,
        scale_y=-scale,
        width=width,
        height=height,
    )


def to_polygon(
    geometry,
) -> "shapely.Polygon | shapely.MultiPolygon":
    """Normalize various polygon inputs to a shapely (Multi)Polygon.

    Accepts:
      - a shapely Polygon or MultiPolygon (returned as-is)
      - a WKT string: "POLYGON ((lon lat, ...))"
      - a GeoJSON geometry dict: {"type": "Polygon", "coordinates": [...]}
      - a GeoJSON Feature dict: {"type": "Feature", "geometry": {...}}
      - a GeoJSON FeatureCollection dict: uses the FIRST feature's geometry

    Lets the user pass whatever they have on hand without converting first.

    Args:
        geometry: a shapely (Multi)Polygon, a WKT string, or a GeoJSON dict.

    Returns:
        A shapely Polygon or MultiPolygon.

    Raises:
        TypeError: if the input type is unsupported or yields a non-polygon.
        ValueError: if a WKT string or GeoJSON dict can't be parsed.
    """
    # already shapely
    if isinstance(geometry, (shapely.Polygon, shapely.MultiPolygon)):
        return geometry

    # WKT string
    if isinstance(geometry, str):
        from shapely import wkt
        try:
            geom = wkt.loads(geometry)
        except Exception as exc:
            raise ValueError(f"could not parse WKT string: {exc}") from exc
        if not isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
            raise TypeError(
                f"WKT parsed to {geom.geom_type}, expected Polygon/MultiPolygon."
            )
        return geom

    # GeoJSON dict
    if isinstance(geometry, dict):
        from shapely.geometry import shape
        gtype = geometry.get("type")
        if gtype == "FeatureCollection":
            feats = geometry.get("features", [])
            if not feats:
                raise ValueError("FeatureCollection has no features.")
            geom_dict = feats[0]["geometry"]
        elif gtype == "Feature":
            geom_dict = geometry["geometry"]
        else:
            geom_dict = geometry            # assume it's a geometry dict
        geom = shape(geom_dict)
        if not isinstance(geom, (shapely.Polygon, shapely.MultiPolygon)):
            raise TypeError(
                f"GeoJSON is a {geom.geom_type}, expected Polygon/MultiPolygon."
            )
        return geom

    raise TypeError(
        f"unsupported geometry input: {type(geometry).__name__}. "
        f"Pass a shapely (Multi)Polygon, a WKT string, or a GeoJSON dict."
    )


def polygon_to_rt(
    geometry: shapely.Polygon,
    scale: float,
    crs: str = "EPSG:4326",
    target_crs: str | None = None,
) -> RasterTransform:
    """Build a RasterTransform that covers a polygon's bbox.

    Earth Engine downloads need an axis-aligned raster grid, so this function
    takes a Polygon and returns a RasterTransform fitted to it.

    Input must be a shapely.Polygon. Convert from other formats before calling:

        # From GeoJSON dict
        from shapely.geometry import shape
        poly = shape(geojson_dict)

        # From WKT
        from shapely import wkt
        poly = wkt.loads(wkt_string)

        # From GeoDataFrame (single feature)
        poly = gdf.geometry.iloc[0]

    Args:
        geometry: A shapely.Polygon. MultiPolygon is not supported — split via
            .geoms and call this function once per part.
        scale: Pixel size in units of target_crs (meters for UTM, degrees for 4326).
        crs: CRS of the input geometry. Default 'EPSG:4326'.
        target_crs: CRS of the output. None → auto-UTM by polygon centroid.

    Returns:
        RasterTransform in target_crs covering the polygon's bbox.

    Raises:
        TypeError: if geometry is not a shapely.Polygon.
        ValueError: if scale <= 0, the polygon is topologically invalid, or
            coordinates are inconsistent with the declared CRS.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    if isinstance(geometry, shapely.MultiPolygon):
        raise TypeError(
            "MultiPolygon not supported. Iterate .geoms and call polygon_to_rt per part:\n"
            "  for sub in multipoly.geoms:\n"
            "      rt = polygon_to_rt(sub, scale=..., crs=...)"
        )
    if not isinstance(geometry, shapely.Polygon):
        raise TypeError(
            f"geometry must be shapely.Polygon, got {type(geometry).__name__}. "
            f"Convert your input to a Polygon first "
            f"(e.g. shape(geojson_dict), wkt.loads(wkt_string), gdf.geometry.iloc[0])."
        )

    if not geometry.is_valid:
        from shapely.validation import explain_validity
        raise ValueError(f"Invalid polygon: {explain_validity(geometry)}")

    # Sanity check: coords vs declared CRS
    xmin, ymin, xmax, ymax = geometry.bounds
    if crs == "EPSG:4326":
        if not (-180 <= xmin <= 180 and -90 <= ymin <= 90
                and -180 <= xmax <= 180 and -90 <= ymax <= 90):
            raise ValueError(
                f"Declared crs='EPSG:4326' but bounds={geometry.bounds} look projected. "
                f"Did you forget to pass crs=? (e.g. crs='EPSG:32718')"
            )

    # Decide target_crs (auto-UTM by centroid if None)
    if target_crs is None:
        if crs == "EPSG:4326":
            lon_c, lat_c = geometry.centroid.x, geometry.centroid.y
        else:
            t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_c, lat_c = t.transform(geometry.centroid.x, geometry.centroid.y)
        target_crs = _utm_zone_epsg(lon_c, lat_c)

    # Reproject FULL polygon (not just bounds) to target_crs
    if crs == target_crs:
        bxmin, bymin, bxmax, bymax = geometry.bounds
    else:
        transformer = Transformer.from_crs(crs, target_crs, always_xy=True)
        poly_proj = shp_transform(transformer.transform, geometry)
        bxmin, bymin, bxmax, bymax = poly_proj.bounds

    return bbox_to_rt(bxmin, bymin, bxmax, bymax, crs=target_crs, scale=scale)


def asset_to_rt(
    image,
    scale: float | None = None,
) -> RasterTransform:
    """Build a RasterTransform from a GEE asset or ee.Image, in its native CRS.

    Queries Earth Engine for the asset's native projection, transform and
    dimensions, and constructs a RasterTransform that covers the full image.

    Earth Engine must be initialized before calling this:
        >>> import ee
        >>> ee.Initialize(project='your-project')

    Args:
        image: Either an Earth Engine asset path (str) or an ee.Image instance.
            Accepting ee.Image lets you build complex computed images
            (e.g. img.clip(), img1.add(img2), ic.median()) and pass them
            directly without serializing.
        scale: Pixel size in meters of the native CRS. None → use the asset's
            native scale (10 m for S2 B2/B3/B4/B8, 30 m for Landsat, etc.).

    Returns:
        RasterTransform in the image's native CRS, covering its full footprint.

    Raises:
        TypeError: if image is not str or ee.Image.
        ValueError: if asset has no bands or scale <= 0.
    """
    import ee

    if isinstance(image, str):
        if not image:
            raise TypeError("image string must be non-empty")
        img = ee.Image(image)
    elif isinstance(image, ee.Image):
        img = image
    else:
        raise TypeError(
            f"image must be str (asset id) or ee.Image, got {type(image).__name__}"
        )

    info = img.getInfo()

    bands = info.get("bands", [])
    if not bands:
        raise ValueError(f"Image has no bands: {image!r}")

    band0 = bands[0]
    native_crs = band0["crs"]
    native_transform = band0["crs_transform"]
    native_width, native_height = band0["dimensions"]

    # Native scale: return the exact RT of the file on disk
    if scale is None:
        return RasterTransform(
            crs=native_crs,
            translate_x=native_transform[2],
            translate_y=native_transform[5],
            scale_x=native_transform[0],
            scale_y=native_transform[4],
            width=native_width,
            height=native_height,
        )

    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    # Custom scale: recompute dimensions over the native bbox
    xmin = native_transform[2]
    ymax = native_transform[5]
    xmax = xmin + native_width * native_transform[0]
    ymin = ymax + native_height * native_transform[4]  # native_transform[4] is negative

    return bbox_to_rt(xmin, ymin, xmax, ymax, crs=native_crs, scale=scale)