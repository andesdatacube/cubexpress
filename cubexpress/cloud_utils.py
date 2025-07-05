"""Cloud-coverage tables for Sentinel-2 over a square ROI.

Two helpers are exposed:

* :func:`_cloud_table_single_range` – query Earth Engine for one date-range.
* :func:`cloud_table` – smart wrapper that adds on-disk caching, automatic
  back-filling, and cloud-percentage filtering.

Both return a ``pandas.DataFrame`` with the columns **day**, **cloudPct** and
**images** plus useful ``.attrs`` metadata for downstream functions.
"""

from __future__ import annotations

import datetime as dt
import ee
import pandas as pd
from cubexpress.cache import _cache_key
import datetime as dt
from cubexpress.geospatial import _square_roi
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def _cloud_table_single_range(
    lon: float,
    lat: float,
    edge_size: int,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Build a daily cloud-score table for a square Sentinel-2 footprint.

    Parameters
    ----------
    lon, lat : float
        Point at the centre of the requested region (°).
    edge_size : int
        Side length of the square region in Sentinel-2 pixels (10 m each).
    start, end : str
        ISO-8601 dates delimiting the period, e.g. ``"2024-06-01"``.

    Returns
    -------
    pandas.DataFrame
        One row per image with columns:
        * ``id`` – Sentinel-2 ID
        * ``cs_cdf`` – Cloud Score Plus CDF (0–1)
        * ``date`` – acquisition date (YYYY-MM-DD)
        * ``null_flag`` – 1 if cloud score missing

    Notes
    -----
    Missing ``cs_cdf`` values are filled with the mean of the same day.
    """

    center = ee.Geometry.Point([lon, lat])
    roi = _square_roi(lon, lat, edge_size, 10)
    
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
    )
    ic = (
        s2
        .linkCollection(
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), 
            ["cs_cdf"]
        )
        .select(["cs_cdf"])
    )
    ids_inside = (
        ic                              
        .map(                           
            lambda img: img.set(
                'roi_inside_scene',
                img.geometry().contains(roi, maxError=10)
            )
        )
        .filter(ee.Filter.eq('roi_inside_scene', True))
        .aggregate_array('system:index')               
        .getInfo()
    )
    
    try:
        raw = ic.getRegion(
            geometry=center,
            scale=(edge_size) * 11
        ).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands in collection" in str(e):
            return pd.DataFrame(
                columns=["id", "longitude", "latitude", "time", "cs_cdf", "inside"]
            )
        raise e
    
    df_raw = (
        pd.DataFrame(raw[1:], columns=raw[0])
        .drop(columns=["longitude", "latitude"])
        .assign(
            date=lambda d: pd.to_datetime(d["id"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        )
    )
    df_raw["inside"] = df_raw["id"].isin(set(ids_inside)).astype(int)
    df_raw['cs_cdf'] = df_raw.groupby('date').apply(
        lambda group: group['cs_cdf'].transform(
            lambda _: group[group['inside'] == 1]['cs_cdf'].iloc[0] 
            if (group['inside'] == 1).any() 
            else group['cs_cdf'].mean()
        )
    ).reset_index(drop=True)
    
    return df_raw
    
def s2_table(
    lon: float,
    lat: float,
    edge_size: int,
    start: str,
    end: str,
    max_cscore: float = 1.0,
    min_cscore: float = 0.0,
    cache: bool = False
) -> pd.DataFrame:
    """Build (and cache) a per-day cloud-table for the requested ROI.

    The function first checks an on-disk parquet cache keyed on location and
    parameters.  If parts of the requested date-range are missing, it fetches
    only those gaps from Earth Engine, merges, updates the cache and finally
    filters by *cloud_max*.

    Parameters
    ----------
    lon, lat
        Centre coordinates.
    edge_size, scale
        Square size (pixels) and resolution (metres).
    start, end
        ISO start/end dates.
    cloud_max
        Maximum allowed cloud percentage (0-100). Rows above this threshold are
        dropped.
    bands
        List of spectral bands to embed as metadata.  If *None* the full
        Sentinel-2 set is used.
    collection
        Sentinel-2 collection to query.
    output_path
        Downstream path hint stored in ``result.attrs``; not used internally.
    cache
        Toggle parquet caching.
    
    Returns
    -------
    pandas.DataFrame
        Filtered cloud table with ``.attrs`` containing the call parameters.
    """

    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    collection = "COPERNICUS/S2_HARMONIZED"
    scale = 10
    cache_file = _cache_key(lon, lat, edge_size, scale, collection)

    # Load cached data if present
    if cache and cache_file.exists():
        print("📂  Loading cached metadata …")
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()

        cached_start = have_idx.min().date()
        cached_end = have_idx.max().date()

        if (
            dt.date.fromisoformat(start) >= cached_start
            and dt.date.fromisoformat(end) <= cached_end
        ):
            print("✅  Served entirely from metadata.")
            df_full = df_cached
        else:
            # Identify missing segments and fetch only those.
            df_new_parts = []
            if dt.date.fromisoformat(start) < cached_start:
                a1, b1 = start, cached_start.isoformat()
                df_new_parts.append(
                    _cloud_table_single_range(
                        lon=lon, 
                        lat=lat, 
                        edge_size=edge_size, 
                        start=a1, 
                        end=b1
                    )
                )
            if dt.date.fromisoformat(end) > cached_end:
                a2, b2 = cached_end.isoformat(), end
                df_new_parts.append(
                    _cloud_table_single_range(
                        lon=lon, 
                        lat=lat, 
                        edge_size=edge_size, 
                        start=a2, 
                        end=b2
                    )
                )
            df_new_parts = [df for df in df_new_parts if not df.empty]
            
            if df_new_parts:
                
                df_new = pd.concat(df_new_parts, ignore_index=True)
                df_full = (
                    pd.concat([df_cached, df_new], ignore_index=True)
                    .sort_values("date", kind="mergesort")
                )
            else:
                df_full = df_cached
    else:
        print("⏳ Generating metadata…")
        df_full = _cloud_table_single_range(
            lon=lon, 
            lat=lat, 
            edge_size=edge_size, 
            start=start, 
            end=end
        )

    # Save cache
    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # Filter by cloud cover and requested date window 
    result = (
        df_full.query("@start <= date <= @end")
        .query("@min_cscore <= cs_cdf <= @max_cscore")
        .reset_index(drop=True)
    )

    # Attach metadata for downstream helpers
    result.attrs.update(
        {
            "lon": lon,
            "lat": lat,
            "edge_size": edge_size,
            "scale": scale,
            "bands": bands,
            "collection": collection
        }
    )
    return result

