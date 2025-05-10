"""Convert cloud_table output into a RequestSet."""

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

    
    df = table.copy()

    if df.empty:
        raise ValueError("cloud_table returned no rows; nothing to request.")

    rt = lonlat2rt(
        lon=df.attrs["lon"],
        lat=df.attrs["lat"],
        edge_size=df.attrs["edge_size"],
        scale=df.attrs["scale"],
    )
    centre_hash = pgh.encode(df.attrs["lat"], df.attrs["lon"], precision=5)
    reqs: list[Request] = []



    if mosaic:
        grouped = (
        df.groupby('date')
            .agg(
                id_list      = ('id', list),
                cs_cdf_mean  = ('cs_cdf', lambda x: int(round(x.mean(), 2) * 100))
            )
        )

        for day, row in grouped.iterrows():
            
            img_ids   = row["id_list"]
            cdf  = row["cs_cdf_mean"]
            
            if len(img_ids) > 1:

                ee_img = ee.ImageCollection(
                    [ee.Image(f"{df.attrs['collection']}/{img}") for img in img_ids]
                ).mosaic()

                reqs.append(
                    Request(
                        id=f"{day}_{centre_hash}_{cdf}",
                        raster_transform=rt,
                        image=ee_img,
                        bands=df.attrs["bands"],
                    )
                )
            else:
                for img_id in img_ids:
                    tile = img_id.split("_")[-1][1:]
                    reqs.append(
                        Request(
                            id=f"{day}_{centre_hash}_{tile}_{cdf}",
                            raster_transform=rt,
                            image=f"{df.attrs['collection']}/{img_id}",
                            bands=df.attrs["bands"],
                        )
                    )
    else:
        for _, row in df.iterrows():
            img_id = row["id"]
            tile = img_id.split("_")[-1][1:]
            day = row["date"]
            cdf = int(round(row["cs_cdf"], 2) * 100)

            reqs.append(
                Request(
                    id=f"{day}_{centre_hash}_{tile}_{cdf}",
                    raster_transform=rt,
                    image=f"{df.attrs['collection']}/{img_id}",
                    bands=df.attrs["bands"],
                )
            )

    return RequestSet(requestset=reqs)
