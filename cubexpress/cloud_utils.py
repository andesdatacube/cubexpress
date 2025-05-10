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
from cubexpress.geospatial import _square_roi


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
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
    )

    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    ic = (
        s2
        .linkCollection(csp, ["cs_cdf"])
        .select(["cs_cdf"])
    )

    # image IDs for every expected date
    ids = ic.aggregate_array("system:index").getInfo()
    df_ids = pd.DataFrame({"id": ids})


    region_scale = edge_size * 10 / 2


    try:
        raw = ic.getRegion(geometry=center, scale=region_scale).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands in collection" in str(e):
            return pd.DataFrame(
                columns=["id", "cs_cdf", "date", "null_flag"]
            )
        raise 

    df_raw = pd.DataFrame(raw[1:], columns=raw[0])


    df = (
        df_ids
        .merge(df_raw, on="id", how="left")
        .assign(
            date=lambda d: pd.to_datetime(d["id"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
            null_flag=lambda d: d["cs_cdf"].isna().astype(int),
        )
        .drop(columns=["longitude", "latitude", "time"])
    )

    # fill missing scores with daily mean
    df["cs_cdf"] = df["cs_cdf"].fillna(df.groupby("date")["cs_cdf"].transform("mean"))

    return df


def s2_cloud_table(
    lon: float,
    lat: float,
    edge_size: int,
    start: str,
    end: str,
    max_cscore: float = 1.0,
    min_cscore: float = 0.0,
    cache: bool = False,
    verbose: bool = True,
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
    verbose
        If *True* prints cache info/progress.

    Returns
    -------
    pandas.DataFrame
        Filtered cloud table with ``.attrs`` containing the call parameters.
    """

    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    collection = "COPERNICUS/S2_HARMONIZED"
    scale = 10
    cache_file = _cache_key(lon, lat, edge_size, scale, collection)

    # ─── 1. Load cached data if present ────────────────────────────────────
    if cache and cache_file.exists():
        if verbose:
            print("📂  Loading cached metadata …")
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()

        cached_start = have_idx.min().date()
        cached_end = have_idx.max().date()

        if (
            dt.date.fromisoformat(start) >= cached_start
            and dt.date.fromisoformat(end) <= cached_end
        ):
            if verbose:
                print("✅  Served entirely from metadata.")
            df_full = df_cached
        else:
            # Identify missing segments and fetch only those.
            df_new_parts = []
            if dt.date.fromisoformat(start) < cached_start:
                a1, b1 = start, cached_start.isoformat()
                df_new_parts.append(
                    _cloud_table_single_range(
                        lon, lat, edge_size, a1, b1
                    )
                )
            if dt.date.fromisoformat(end) > cached_end:
                a2, b2 = cached_end.isoformat(), end
                df_new_parts.append(
                    _cloud_table_single_range(
                        lon, lat, edge_size, a2, b2
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

        if verbose:
            msg = "Generating metadata (no cache found)…" if cache else "Generating metadata…"
            print("⏳", msg)
        df_full = _cloud_table_single_range(
            lon, lat, edge_size, start, end
        )
        

    # ─── 2. Save cache ─────────────────────────────────────────────────────
    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # ─── 3. Filter by cloud cover and requested date window ────────────────
    
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

