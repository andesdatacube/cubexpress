"""Build Earth Engine request sets from cloud score tables."""

from __future__ import annotations

import ee
import pandas as pd
import pygeohash as pgh

from cubexpress.conversion import lonlat2rt
from cubexpress.exceptions import ValidationError
from cubexpress.geotyping import Request, RequestSet


def table_to_requestset(
    table: pd.DataFrame, 
    mosaic: bool = True
) -> RequestSet:
    """
    Convert a cloud score table into Earth Engine requests.

    Args:
        table: DataFrame with Sentinel-2 metadata (columns: 'id', 'date', 
            'cs_cdf') and required .attrs metadata (lon, lat, collection, bands)
        mosaic: If True, composite images from the same day into a single
            mosaic. If False, request each image individually

    Returns:
        RequestSet containing the generated Request objects

    Raises:
        ValidationError: If input table is empty or missing required metadata
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
    collection = meta["collection"]
    bands = meta["bands"]
    
    reqs = []

    if mosaic:
        grouped = (
            df.groupby('date')
            .agg(
                id_list=('id', list),
                tiles=(
                    'id',
                    lambda ids: ','.join(
                        sorted({i.split('_')[-1][1:] for i in ids})
                    )
                ),
                cs_cdf_mean=('cs_cdf', lambda x: round(x.mean(), 2))
            )
        )

        for day, row in grouped.iterrows():
            img_ids = row["id_list"]
            cdf = row["cs_cdf_mean"]
            
            if len(img_ids) > 1:
                req_id = f"{day}_{centre_hash}_{cdf:.2f}"
                image_source = ee.ImageCollection(
                    [ee.Image(f"{collection}/{img}") for img in img_ids]
                ).mosaic()
            else:
                tile = img_ids[0].split('_')[-1][1:]
                req_id = f"{day}_{tile}_{cdf:.2f}"
                image_source = f"{collection}/{img_ids[0]}"

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
            img_id = row["id"]
            tile = img_id.split("_")[-1][1:]
            day = row["date"]
            cdf = round(row["cs_cdf"], 2)
            
            reqs.append(
                Request(
                    id=f"{day}_{tile}_{cdf:.2f}",
                    raster_transform=rt,
                    image=f"{collection}/{img_id}",
                    bands=bands,
                )
            )

    return RequestSet(requestset=reqs)