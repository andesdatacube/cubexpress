"""Quick single-image download utilities."""

from __future__ import annotations

import pathlib
from typing import Any

import ee

from cubexpress.core.exceptions import ValidationError
from cubexpress.core.types import RasterTransform, Request, RequestSet
from cubexpress.download.batch import get_geotiff
from cubexpress.geometry.conversion import lonlat2rt
from cubexpress.metadata.scene import get_scene_info
from cubexpress.request.transforms import _apply_toa_to_single, _should_apply_mss_toa


def _extract_image_geometry(
    image: ee.Image,
    bands: list[str],
    scale: int,
) -> dict[str, Any]:
    """Extract geometry metadata from ee.Image."""

    bounds = image.geometry().bounds().getInfo()
    coords = bounds["coordinates"][0]

    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)

    # Reject unbounded images
    if (max_lon - min_lon) > 180 or (max_lat - min_lat) > 90:
        raise ValidationError(
            "Cannot download unbounded mosaic.\n\n"
            "Use ROI mode instead:\n"
            "  get_image(mosaic, bands, scale, lon=x, lat=y, edge_size=512, outfile=...)\n\n"
            "Or clip before downloading:\n"
            "  area = ee.Geometry.Point([x, y]).buffer(100000)\n"
            "  mosaic = collection.mosaic().clip(area)"
        )

    # Calculate center
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Get UTM CRS
    from cubexpress.geometry.conversion import geo2utm

    try:
        _, _, crs_result = geo2utm(center_lon, center_lat)
        crs = crs_result if isinstance(crs_result, str) and crs_result.startswith("EPSG:") else f"EPSG:{crs_result}"
    except Exception:
        from cubexpress.geometry.conversion import lonlat2rt_utm_or_ups

        _, _, crs_result = lonlat2rt_utm_or_ups(center_lon, center_lat)
        crs = crs_result if isinstance(crs_result, str) and crs_result.startswith("EPSG:") else f"EPSG:{crs_result}"

    # Transform to UTM
    from pyproj import CRS as ProjCRS
    from pyproj import Transformer

    src_crs = ProjCRS.from_epsg(4326)
    dst_crs = ProjCRS.from_string(crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    x_coords = []
    y_coords = []
    for corner in [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]:
        x, y = transformer.transform(corner[0], corner[1])
        x_coords.append(x)
        y_coords.append(y)

    width = int(round((max(x_coords) - min(x_coords)) / scale))
    height = int(round((max(y_coords) - min(y_coords)) / scale))

    return {
        "crs": crs,
        "geotransform": {
            "scaleX": float(scale),
            "shearX": 0.0,
            "translateX": min(x_coords),
            "scaleY": float(-scale),
            "shearY": 0.0,
            "translateY": max(y_coords),
        },
        "width": width,
        "height": height,
    }


def get_image(
    image: str | ee.Image,
    outfile: str | pathlib.Path,
    bands: list[str] | None = None,
    scale: int | None = None,
    lon: float | None = None,
    lat: float | None = None,
    edge_size: int | tuple[int, int] | None = None,
    toa: bool | None = None,
    export_format: Any = None,
    nworks: int | None = None,
) -> None:
    """Quick download of a single image.

    Args:
        image: Asset ID (string) or ee.Image object
        outfile: Output file path
        bands: Bands to download. If None, auto-detects.
        scale: Resolution in meters. If None, auto-detects.
        lon, lat, edge_size: If provided, crops to ROI.
        toa: TOA processing (asset IDs only)
        export_format: Output format
        nworks: Number of parallel workers
    """
    outfile = pathlib.Path(outfile)

    # Convert to ee.Image
    if isinstance(image, str):
        ee_image = ee.Image(image)
        asset_id = image
    else:
        ee_image = image
        asset_id = None

    # Auto-detect
    if bands is None:
        bands = ee_image.bandNames().getInfo()

    if scale is None:
        proj_info = ee_image.select(bands[0]).projection().getInfo()
        scale = int(abs(proj_info["transform"][0]))

    # ROI or full scene
    is_roi_mode = lon is not None and lat is not None and edge_size is not None

    if is_roi_mode:
        rt = lonlat2rt(lon, lat, edge_size, scale)

        if asset_id:
            if _should_apply_mss_toa(asset_id, toa):
                image_source = _apply_toa_to_single(asset_id, bands)
            else:
                image_source = asset_id
        else:
            image_source = ee_image
    else:
        if asset_id:
            scene_info = get_scene_info(asset_id, scale=scale, cache=True)

            rt = RasterTransform(
                crs=scene_info["crs"],
                geotransform=scene_info["geotransform"],
                width=scene_info["width"],
                height=scene_info["height"],
            )

            if _should_apply_mss_toa(asset_id, toa):
                image_source = _apply_toa_to_single(asset_id, bands)
            else:
                image_source = asset_id
        else:
            # ee.Image - must have valid bounds
            scene_info = _extract_image_geometry(ee_image, bands, scale)

            rt = RasterTransform(
                crs=scene_info["crs"],
                geotransform=scene_info["geotransform"],
                width=scene_info["width"],
                height=scene_info["height"],
            )

            image_source = ee_image

    # Build and download
    req = Request(
        id=outfile.stem,
        raster_transform=rt,
        image=image_source,
        bands=bands,
    )

    requests = RequestSet(requestset=[req])
    manifest = requests._dataframe.iloc[0]["manifest"]

    get_geotiff(
        manifest=manifest,
        full_outname=outfile,
        nworks=nworks,
        export_format=export_format,
    )


def get_images(
    images: list[str | ee.Image],
    outfolder: str | pathlib.Path,
    bands: list[str] | None = None,
    scale: int | None = None,
    lon: float | None = None,
    lat: float | None = None,
    edge_size: int | tuple[int, int] | None = None,
    toa: bool | None = None,
    export_format: Any = None,
    nworks: int | None = None,
) -> None:
    """Quick download of multiple images."""

    outfolder = pathlib.Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)

    if nworks is None:
        from cubexpress.core.config import CONFIG

        nworks = CONFIG.default_workers

    # Auto-detect
    first_img = images[0]
    ee_first = ee.Image(first_img) if isinstance(first_img, str) else first_img

    if bands is None:
        bands = ee_first.bandNames().getInfo()

    if scale is None:
        proj_info = ee_first.select(bands[0]).projection().getInfo()
        scale = int(abs(proj_info["transform"][0]))

    # Download
    for i, img in enumerate(images):
        img_id = img.split("/")[-1] if isinstance(img, str) else f"image_{i:04d}"
        outfile = outfolder / f"{img_id}.tif"

        get_image(
            image=img,
            outfile=outfile,
            bands=bands,
            scale=scale,
            lon=lon,
            lat=lat,
            edge_size=edge_size,
            toa=toa,
            export_format=export_format,
            nworks=nworks,
        )
