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
import json
import pathlib
from typing import List, Optional

import ee
import pandas as pd

from cubexpress.cache import _cache_key
from cubexpress.geospatial import _square_roi


def _cloud_table_single_range(
    lon: float,
    lat: float,
    edge_size: int,
    scale: int,
    start: str,
    end: str,
    collection: str = "COPERNICUS/S2_HARMONIZED",
) -> pd.DataFrame:
    """Return raw cloud-table rows for a single *start–end* interval.

    Parameters
    ----------
    lon, lat
        Centre coordinates in decimal degrees.
    edge_size, scale
        ROI size in pixels (*edge_size*) and pixel resolution in metres
        (*scale*), fed into :pyfunc:`cubexpress.geospatial._square_roi`.
    start, end
        ISO-dates (``YYYY-MM-DD``) delimiting the query.
    collection
        Sentinel-2 collection name to query.

    Returns
    -------
    pandas.DataFrame
        Columns: **day** (str), **cloudPct** (float), **images** (str
        concatenation of asset IDs separated by ``-``). No filtering applied.
    """
    roi = _square_roi(lon, lat, edge_size, scale)
    s2 = ee.ImageCollection(collection)

    if collection in (
        "COPERNICUS/S2_HARMONIZED",
        "COPERNICUS/S2_SR_HARMONIZED",
    ):
        qa_band = "cs_cdf"
        csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    else:
        qa_band, csp = None, None

    def _add_props(img):
        day = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        imgid = img.get("system:index")

        if qa_band:
            score = (
                img.linkCollection(csp, [qa_band])
                .select([qa_band])
                .reduceRegion(ee.Reducer.mean(), roi, scale)
                .get(qa_band)
            )
            # If score is null assume completely clear (score=1 → cloudPct=0)
            score_safe = ee.Algorithms.If(score, score, -1)
            cloud_pct = (
                ee.Number(1)
                .subtract(ee.Number(score_safe))
                .multiply(10000)
                .round()
                .divide(100)
            )
        else:
            cloud_pct = ee.Number(-1)

        return ee.Feature(
            None,
            {
                "day": day,
                "cloudPct": cloud_pct,
                "images": imgid,
            },
        )

    triples = (
        s2.filterDate(start, end)
        .filterBounds(roi)
        .map(_add_props)
        .reduceColumns(ee.Reducer.toList(3), ["day", "cloudPct", "images"])
        .get("list")
        .getInfo()
    )

    df = pd.DataFrame(triples, columns=["day", "cloudPct", "images"]).dropna()
    df["cloudPct"] = df["cloudPct"].astype(float)
    df["images"] = df["images"].astype(str)
    return df


def cloud_table(
    lon: float,
    lat: float,
    edge_size: int = 2048,
    scale: int = 10,
    start: str = "2017-01-01",
    end: str = "2024-12-31",
    cloud_max: float = 7.0,
    bands: Optional[List[str]] = None,
    collection: str = "COPERNICUS/S2_HARMONIZED",
    output_path: str | pathlib.Path | None = None,
    cache: bool = True,
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
    if bands is None:
        bands = [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12",
        ]

    cache_file = _cache_key(lon, lat, edge_size, scale, collection)

    # ─── 1. Load cached data if present ────────────────────────────────────
    if cache and cache_file.exists():
        if verbose:
            print("📂  Loading cached table …")
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["day"], errors="coerce").dropna()

        cached_start = have_idx.min().date()
        cached_end = have_idx.max().date()

        if (
            dt.date.fromisoformat(start) >= cached_start
            and dt.date.fromisoformat(end) <= cached_end
        ):
            if verbose:
                print("✅  Served entirely from cache.")
            df_full = df_cached
        else:
            # Identify missing segments and fetch only those.
            df_new_parts = []
            if dt.date.fromisoformat(start) < cached_start:
                a1, b1 = start, cached_start.isoformat()
                df_new_parts.append(
                    _cloud_table_single_range(
                        lon, lat, edge_size, scale, a1, b1, collection
                    )
                )
            if dt.date.fromisoformat(end) > cached_end:
                a2, b2 = cached_end.isoformat(), end
                df_new_parts.append(
                    _cloud_table_single_range(
                        lon, lat, edge_size, scale, a2, b2, collection
                    )
                )
            df_new = pd.concat(df_new_parts, ignore_index=True)
            df_full = (
                pd.concat([df_cached, df_new], ignore_index=True)
                .drop_duplicates("day")
                .sort_values("day", kind="mergesort")
            )
    else:
        # No cache or caching disabled: fetch full range.
        if verbose:
            msg = "Generating table (no cache found)…" if cache else "Generating table…"
            print("⏳", msg)
        df_full = _cloud_table_single_range(
            lon, lat, edge_size, scale, start, end, collection
        )

    # ─── 2. Save cache ─────────────────────────────────────────────────────
    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # ─── 3. Filter by cloud cover and requested date window ────────────────
    result = (
        df_full.query("@start <= day <= @end")
        .query("cloudPct < @cloud_max")
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
            "collection": collection,
            "cloud_max": cloud_max,
            "output_path": str(output_path) if output_path else "",
        }
    )
    return result
