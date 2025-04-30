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


    df_ = df.copy()

    if df_.empty:
        raise ValueError("cloud_table returned no rows; nothing to request.")

    rt = lonlat2rt(
        lon=df_.attrs["lon"],
        lat=df_.attrs["lat"],
        edge_size=df_.attrs["edge_size"],
        scale=df_.attrs["scale"],
    )
    centre_hash = pgh.encode(df_.attrs["lat"], df_.attrs["lon"], precision=5)
    reqs: list[Request] = []

    if mosaic:
        # group all asset IDs per day
        grouped = (
            df_.groupby("date")["id"]   # Series con listas de ids por día
            .apply(list)
        )

        for day, img_ids in grouped.items():
            ee_img = ee.ImageCollection(
                [ee.Image(f"{df_.attrs['collection']}/{img}") for img in img_ids]
            ).mosaic()

            reqs.append(
                Request(
                    id=f"{day}_{centre_hash}",
                    raster_transform=rt,
                    image=ee_img,
                    bands=df_.attrs["bands"],
                )
            )
    else:  # one request per asset
        for _, row in df_.iterrows():
            img_id = row["id"]
            day    = row["date"]

            reqs.append(
                Request(
                    id=f"{day}_{centre_hash}_{img_id}",
                    raster_transform=rt,
                    image=f"{df_.attrs['collection']}/{img_id}",
                    bands=df_.attrs["bands"],
                )
            )

    return RequestSet(requestset=reqs)
