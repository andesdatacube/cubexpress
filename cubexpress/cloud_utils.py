from __future__ import annotations

import datetime as dt
import sys
import time
import ee
import pandas as pd
from cubexpress.cache import _cache_key
import datetime as dt
from cubexpress.geospatial import _square_roi
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


# --- CONFIGURATION CONSTANTS ---
S2_COLLECTION = "COPERNICUS/S2_HARMONIZED"
S2_CLOUD_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S2_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
]
S2_PIXEL_SCALE = 10  # meters
# -------------------------------

def _cloud_table_single_range(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Build a daily cloud-score table for a square Sentinel-2 footprint.

    Query Earth Engine for a specific date range, identifying which images
    fully contain the ROI and filling missing cloud scores with daily means.

    Args:
        lon (float): Longitude of the center point.
        lat (float): Latitude of the center point.
        edge_size (int | tuple[int, int]): Side length of the square region 
            in Sentinel-2 pixels (10 m each).
        start (str): ISO-8601 start date (inclusive), e.g. "2024-06-01".
        end (str): ISO-8601 end date (inclusive).

    Returns:
        pd.DataFrame: A DataFrame with one row per image. Columns include:
            * id: Sentinel-2 ID.
            * cs_cdf: Cloud Score Plus CDF (0—1).
            * date: Acquisition date (YYYY-MM-DD).
            * inside: 1 if the image fully contains the ROI, 0 otherwise.
            
            Note: Missing ``cs_cdf`` values are filled with the mean of the 
            same day if a full-coverage image is not available.

    Raises:
        ee.ee_exception.EEException: If Earth Engine fails for reasons other
            than an empty collection (e.g., quota exceeded, bad request).
    """
    # Define ROI (bbox around point)
    center = ee.Geometry.Point([lon, lat])
    roi = _square_roi(lon, lat, edge_size, 10)
    
    # Query S2
    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
    )
    
    # Cloud Score Plus collection
    ic = (
        s2
        .linkCollection(
            ee.ImageCollection(S2_CLOUD_COLLECTION), 
            ["cs_cdf"]
        )
        .select(["cs_cdf"])
    )
    
    # Identify images whose footprint contains the ROI
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
    
    # Generate % cloud of each image over the ROI
    try:
        raw = ic.getRegion(
            geometry=center,
            scale=(edge_size) * 11 # 10 m pixels plus margin (it's a tricky calculation)
        ).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands in collection" in str(e):
            return pd.DataFrame(
                columns=["id", "longitude", "latitude", "time", "cs_cdf", "inside"]
            )
        raise e
    
    # Convert raw data to DataFrame
    df_raw = (
        pd.DataFrame(raw[1:], columns=raw[0])
        .drop(columns=["longitude", "latitude"])
        .assign(
            date=lambda d: pd.to_datetime(d["id"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        )
    )
    
    # Mark images whose ROI is fully inside the scene
    df_raw["inside"] = df_raw["id"].isin(set(ids_inside)).astype(int)
    
    # Fill missing cloud scores with daily mean (mosaic approach)
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
    edge_size: int | tuple[int, int],
    start: str,
    end: str,
    max_cscore: float = 1.0,
    min_cscore: float = 0.0,
    cache: bool = False
) -> pd.DataFrame:
    """
    Build (and cache) a per-day cloud-table for the requested ROI.

    The function checks an on-disk parquet cache keyed on location and
    parameters. If parts of the requested date-range are missing, it fetches
    only those gaps from Earth Engine, merges them, updates the cache, and 
    finally filters by cloud score thresholds.

    Args:
        lon (float): Longitude of the center point.
        lat (float): Latitude of the center point.
        edge_size (int | tuple[int, int]): Side length of the square region 
            in Sentinel-2 pixels (10 m each).
        start (str): ISO-8601 start date, e.g. "2024-06-01".
        end (str): ISO-8601 end date.
        max_cscore (float, optional): Maximum allowed cloud score CDF (0.0 to 1.0). 
            Rows above this threshold are dropped. Defaults to 1.0.
        min_cscore (float, optional): Minimum allowed cloud score CDF (0.0 to 1.0).
            Defaults to 0.0.
        cache (bool, optional): If True, enables on-disk parquet caching to 
            avoid re-fetching data for the same parameters. Defaults to False.
    
    Returns:
        pd.DataFrame: Filtered cloud table. The DataFrame contains useful 
            metadata in ``.attrs`` (bands, collection, scale, etc.) needed
            for downstream functions.
    """
    cache_file = _cache_key(lon, lat, edge_size, S2_PIXEL_SCALE, S2_COLLECTION)

    # Load cached data if present
    if cache and cache_file.exists():
        print("📂 Loading cached metadata...", end='', flush=True)
        t0 = time.time()
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()

        cached_start = have_idx.min().date()
        cached_end = have_idx.max().date()
        elapsed = time.time() - t0

        if (
            dt.date.fromisoformat(start) >= cached_start
            and dt.date.fromisoformat(end) <= cached_end
        ):
            print(f"\r✅ Loaded {len(df_cached)} images from cache ({elapsed:.2f}s)")
            df_full = df_cached
        else:
            print(f"\r📂 Cache loaded ({len(df_cached)} images, {elapsed:.2f}s)")
            
            # Identify missing segments and fetch only those.
            print("⏳ Fetching missing date ranges...", end='', flush=True)
            t0 = time.time()
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
                elapsed = time.time() - t0
                print(f"\r✅ Fetched {len(df_new)} new images ({elapsed:.2f}s)      ")
                
                df_full = (
                    pd.concat([df_cached, df_new], ignore_index=True)
                    .sort_values("date", kind="mergesort")
                )
            else:
                elapsed = time.time() - t0
                print(f"\r✅ No new images needed ({elapsed:.2f}s)      ")
                df_full = df_cached
    else:
        print("⏳ Querying Earth Engine metadata...", end='', flush=True)
        t0 = time.time()
        df_full = _cloud_table_single_range(
            lon=lon, 
            lat=lat, 
            edge_size=edge_size, 
            start=start, 
            end=end
        )
        elapsed = time.time() - t0
        n_images = len(df_full)
        date_range = f"{start} to {end}"
        print(f"\r✅ Retrieved {n_images} images from {date_range} ({elapsed:.2f}s)")

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
            "scale": S2_PIXEL_SCALE,
            "bands": S2_BANDS,
            "collection": S2_COLLECTION
        }
    )
    return result