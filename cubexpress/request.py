"""Build Earth Engine request sets from cloud score tables."""

from __future__ import annotations

import ee
import pandas as pd
import pygeohash as pgh

from cubexpress.conversion import lonlat2rt
from cubexpress.exceptions import ValidationError
from cubexpress.geotyping import Request, RequestSet


def _apply_toa_if_needed(
    image_source: str | ee.Image,
    toa: bool,
    sensor_prefix: str = ""
) -> str | ee.Image:
    """
    Apply TOA calibration to MSS images ONLY if requested.
    
    TM, ETM+, and OLI/TIRS have dedicated _TOA collections that are already TOA-calibrated,
    so they should NOT use this function. Only MSS needs ee.Algorithms.Landsat.TOA().
    
    Args:
        image_source: EE Image object or asset ID string
        toa: If True, applies Landsat TOA calibration (for MSS only)
        sensor_prefix: Sensor identifier to check if TOA conversion is needed
        
    Returns:
        Original image or TOA-calibrated image (MSS only)
    """
    # Only apply TOA conversion for MSS sensors
    if not toa or not sensor_prefix.startswith("MSS"):
        return image_source
    
    # Convert string asset ID to ee.Image if needed
    if isinstance(image_source, str):
        image_source = ee.Image(image_source)
    
    # Apply TOA calibration for MSS only
    return ee.Algorithms.Landsat.TOA(image_source)


def _get_tile_suffix(full_id: str) -> str:
    """
    Extract tile identifier from full Earth Engine asset ID.
    
    Args:
        full_id: Full EE asset path
        
    Returns:
        Tile suffix (e.g., '33VUC' from Sentinel-2 ID)
    """
    filename = full_id.split("/")[-1]
    suffix = filename.split("_")[-1]
    # Heuristic: S2 tiles start with T
    if suffix.startswith("T") and len(suffix) == 6:
        return suffix[1:]
    return suffix


def table_to_requestset(
    table: pd.DataFrame, 
    mosaic: bool = True
) -> RequestSet:
    """
    Converts a cloud score table into Earth Engine requests.
    
    If table.attrs['toa'] is True, applies TOA calibration to Landsat images.

    Args:
        table: DataFrame with metadata (columns: 'id', 'date', cloud metric)
            and required .attrs metadata (lon, lat, collection, bands, toa).
            Note: The 'id' column must contain the full Earth Engine asset ID.
        mosaic: If True, composites images from the same day into a single
            mosaic. If False, requests each image individually.

    Returns:
        RequestSet containing the generated Request objects.

    Raises:
        ValidationError: If input table is empty or missing required metadata.
    """
    if table.empty:
        raise ValidationError(
            "Input table is empty. Check dates, location, or cloud criteria."
        )
    
    required_attrs = {"lon", "lat", "edge_size", "scale", "collection", "bands"}
    missing_attrs = required_attrs - set(table.attrs.keys())
    if missing_attrs:
        raise ValidationError(f"Missing required attributes: {missing_attrs}")
    
    df = table.copy()
    meta = df.attrs
    
    rt = lonlat2rt(
        lon=meta["lon"],
        lat=meta["lat"],
        edge_size=meta["edge_size"],
        scale=meta["scale"],
    )
    
    centre_hash = pgh.encode(meta["lat"], meta["lon"], precision=5)
    bands = meta["bands"]
    toa = meta.get("toa", False)
    
    # Identify cloud metric column dynamically
    metric_col = None
    for candidate in ["cs_cdf", "CLOUD_COVER", "cloud_cover"]:
        if candidate in df.columns:
            metric_col = candidate
            break
            
    if metric_col is None:
        metric_col = "cloud_metric_dummy"
        df[metric_col] = 0.0
    
    reqs = []

    if mosaic:
        grouped = (
            df.groupby('date')
            .agg(
                id_list=('id', list),
                tiles=(
                    'id',
                    lambda ids: ','.join(
                        sorted({_get_tile_suffix(i) for i in ids})
                    )
                ),
                cloud_metric=(metric_col, lambda x: round(x.mean(), 2))
            )
        )

        for day, row in grouped.iterrows():
            img_ids = row["id_list"]
            metric_val = row["cloud_metric"]
            
            if len(img_ids) > 1:
                req_id = f"{day}_{centre_hash}_{metric_val:.2f}"
                image_source = ee.ImageCollection(
                    [ee.Image(img) for img in img_ids]
                ).mosaic()
                image_source = _apply_toa_if_needed(image_source, toa)
            else:
                tile = _get_tile_suffix(img_ids[0])
                req_id = f"{day}_{tile}_{metric_val:.2f}"
                image_source = _apply_toa_if_needed(img_ids[0], toa)

            reqs.append(
                Request(
                    id=req_id,
                    raster_transform=rt,
                    image=image_source,
                    bands=bands,
                )
            )
    else:
        for _, row in df.iterrows():
            full_id = row["id"]
            tile = _get_tile_suffix(full_id)
            day = row["date"]
            val = row.get(metric_col, 0)
            metric_val = round(val, 2)
            
            image_source = _apply_toa_if_needed(full_id, toa)
            
            reqs.append(
                Request(
                    id=f"{day}_{tile}_{metric_val:.2f}",
                    raster_transform=rt,
                    image=image_source,
                    bands=bands,
                )
            )

    return RequestSet(requestset=reqs)