from __future__ import annotations

import ee
import pandas as pd
import pygeohash as pgh

from cubexpress.geotyping import Request, RequestSet
from cubexpress.conversion import lonlat2rt


def table_to_requestset(
    table: pd.DataFrame, 
    mosaic: bool = True
) -> RequestSet:
    """
    Converts a cloud score table into a list of Earth Engine requests.

    Iterates through the provided DataFrame to construct Request objects.
    Can either mosaic images from the same day or request them individually.

    Args:
        table (pd.DataFrame): Input DataFrame containing Sentinel-2 metadata
            (columns: 'id', 'date', 'cs_cdf') and required ``.attrs`` metadata
            (lon, lat, collection, bands).
        mosaic (bool, optional): If True, composites images from the same day
            into a single mosaic. If False, requests each image individually.
            Defaults to True.

    Returns:
        RequestSet: A container object holding the list of generated Requests.

    Raises:
        ValueError: If the input table is empty.
    """

    if table.empty:
        raise ValueError(
            "Input table is empty. Check dates, location, or cloud criteria."
        )
        
    # Extract metadata once to improve readability
    df = table.copy()
    meta = df.attrs
    
    rt = lonlat2rt(
        lon=meta["lon"],
        lat=meta["lat"],
        edge_size=meta["edge_size"],
        scale=meta["scale"],
    )
    
    # Generate a geohash for the center point (used in naming)
    centre_hash = pgh.encode(meta["lat"], meta["lon"], precision=5)
    collection = meta["collection"]
    bands = meta["bands"]
    
    reqs = []

    if mosaic:
        # Group by date to handle daily mosaics
        grouped = (
            df.groupby('date')
            .agg(
                id_list = ('id', list),
                tiles = (
                    'id',
                    lambda ids: ','.join(
                        sorted({i.split('_')[-1][1:] for i in ids})
                    )
                ),
                cs_cdf_mean = (
                    'cs_cdf',
                    lambda x: round(x.mean(), 2)
                )
            )
        )

        for day, row in grouped.iterrows():
            img_ids = row["id_list"]
            cdf = row["cs_cdf_mean"]
            
            if len(img_ids) > 1:
                # Multiple images: create mosaic and use geohash
                req_id = f"{day}_{centre_hash}_{cdf:.2f}"
                image_source = ee.ImageCollection(
                    [ee.Image(f"{collection}/{img}") for img in img_ids]
                ).mosaic()
            else:
                # Single image: use tile name directly
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
        # Process every image individually
        for _, row in df.iterrows():
            img_id = row["id"]
            # Extract tile ID from the Sentinel ID string
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