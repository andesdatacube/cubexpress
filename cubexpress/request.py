"""Convert cloud_table output into a RequestSet."""

from __future__ import annotations

import ee
import pandas as pd
import pygeohash as pgh
from typing import List

from cubexpress.geotyping import Request, RequestSet
from cubexpress.conversion import lonlat2rt


def table_to_requestset(df: pd.DataFrame, *, mosaic: bool = True) -> RequestSet:
    """Return a :class:`RequestSet` built from *df* (cloud_table result).

    Parameters
    ----------
    df
        DataFrame with *day* and *images* columns plus attrs created by
        :pyfunc:`cubexpress.cloud_table`.
    mosaic
        If ``True`` a single mosaic per day is requested; otherwise each
        individual asset becomes its own request.

    Raises
    ------
    ValueError
        If *df* is empty after filtering.

    """
    if df.empty:
        raise ValueError("cloud_table returned no rows; nothing to request.")

    rt = lonlat2rt(
        lon=df.attrs["lon"],
        lat=df.attrs["lat"],
        edge_size=df.attrs["edge_size"],
        scale=df.attrs["scale"],
    )
    centre_hash = pgh.encode(df.attrs["lat"], df.attrs["lon"], precision=5)
    reqs: List[Request] = []

    if mosaic:
        # group all asset IDs per day
        grouped = (
            df.assign(img=lambda x: x.images.str.split("-"))
              .explode("img")
              .groupby("day")["img"]
              .apply(list)
        )

        for day, img_ids in grouped.items():
            ee_img = ee.ImageCollection(
                [ee.Image(f"{df.attrs['collection']}/{img}") for img in img_ids]
            ).mosaic()
            reqs.append(
                Request(
                    id=f"{day}_{centre_hash}_mosaic",
                    raster_transform=rt,
                    image=ee_img,
                    bands=df.attrs["bands"],
                )
            )
    else:  # one request per asset
        for _, row in df.iterrows():
            for img_id in row["images"].split("-"):
                reqs.append(
                    Request(
                        id=f"{row['day']}_{centre_hash}_{img_id}",
                        raster_transform=rt,
                        image=f"{df.attrs['collection']}/{img_id}",
                        bands=df.attrs["bands"],
                    )
                )

    return RequestSet(requestset=reqs)
