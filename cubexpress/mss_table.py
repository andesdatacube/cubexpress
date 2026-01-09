"""Extended cloud_utils.py to include Landsat MSS support."""

from __future__ import annotations

import datetime as dt
import time
import warnings

import ee
import pandas as pd

from cubexpress.cache import _cache_key
from cubexpress.geospatial import _square_roi

warnings.filterwarnings('ignore', category=DeprecationWarning)


# --- MSS CONFIGURATION CONSTANTS ---
MSS_COLLECTION = "LANDSAT/LM01/C02/T1"  # Landsat 1 MSS
MSS_BANDS = ["B4", "B5", "B6", "B7"]
MSS_PIXEL_SCALE = 60  # meters
# -----------------------------------


def _mss_table_single_range(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Build a daily cloud-cover table for a square Landsat MSS footprint.

    Query Earth Engine for a specific date range, identifying which images
    fully contain the ROI. Uses CLOUD_COVER property instead of Cloud Score Plus.

    Args:
        lon (float): Longitude of the center point.
        lat (float): Latitude of the center point.
        edge_size (int | tuple[int, int]): Side length of the square region 
            in MSS pixels (60 m each).
        start (str): ISO-8601 start date (inclusive), e.g. "1972-07-26".
        end (str): ISO-8601 end date (inclusive).

    Returns:
        pd.DataFrame: A DataFrame with one row per image. Columns include:
            * id: Landsat scene ID.
            * cloud_cover: Cloud cover percentage (0-100).
            * date: Acquisition date (YYYY-MM-DD).
            * inside: 1 if the image fully contains the ROI, 0 otherwise.
            * path: WRS path number.
            * row: WRS row number.

    Raises:
        ee.ee_exception.EEException: If Earth Engine fails.
    """
    # Define ROI (bbox around point)
    roi = _square_roi(lon, lat, edge_size, MSS_PIXEL_SCALE)
    
    # Query Landsat MSS
    mss = (
        ee.ImageCollection(MSS_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
    )
    
    # Get all images and their metadata
    try:
        # Get basic info first
        img_list = mss.toList(mss.size())
        size = img_list.size().getInfo()
        
        if size == 0:
            return pd.DataFrame(
                columns=["id", "cloud_cover", "date", "inside", "path", "row"]
            )
        
        # Extract metadata for each image
        records = []
        for i in range(size):
            img = ee.Image(img_list.get(i))
            info = img.getInfo()
            props = info['properties']
            
            # Check if ROI is inside scene
            inside = img.geometry().contains(roi, maxError=10).getInfo()
            
            # Extract relevant metadata
            # Use system:index as the ID (this is the GEE asset ID)
            asset_id = props.get('system:index', '')
            cloud_cover = props.get('CLOUD_COVER', -1)
            date_acquired = props.get('DATE_ACQUIRED', '')
            wrs_path = props.get('WRS_PATH', -1)
            wrs_row = props.get('WRS_ROW', -1)
            
            records.append({
                'id': asset_id,  # Use system:index for asset reference
                'cloud_cover': cloud_cover,
                'date': date_acquired,
                'inside': 1 if inside else 0,
                'path': wrs_path,
                'row': wrs_row
            })
        
        df_raw = pd.DataFrame(records)
        
        # Ensure date is in correct format
        if not df_raw.empty and 'date' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['date']).dt.strftime('%Y-%m-%d')
        
        return df_raw
        
    except ee.ee_exception.EEException as e:
        if "Collection.toList" in str(e) or "No bands" in str(e):
            return pd.DataFrame(
                columns=["id", "cloud_cover", "date", "inside", "path", "row"]
            )
        raise e


def mss_table(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str,
    max_cloud_cover: float = 100.0,
    min_cloud_cover: float = 0.0,
    cache: bool = False
) -> pd.DataFrame:
    """
    Build (and cache) a per-day cloud-table for Landsat MSS at the requested ROI.

    Similar to s2_table but for Landsat MSS data. Uses CLOUD_COVER property
    instead of Cloud Score Plus.

    Args:
        lon (float): Longitude of the center point.
        lat (float): Latitude of the center point.
        edge_size (int | tuple[int, int]): Side length of the square region 
            in MSS pixels (60 m each).
        start (str): ISO-8601 start date, e.g. "1972-07-26".
        end (str): ISO-8601 end date.
        max_cloud_cover (float, optional): Maximum allowed cloud cover (0-100). 
            Rows above this threshold are dropped. Defaults to 100.0.
        min_cloud_cover (float, optional): Minimum allowed cloud cover (0-100).
            Defaults to 0.0.
        cache (bool, optional): If True, enables on-disk parquet caching to 
            avoid re-fetching data for the same parameters. Defaults to False.
    
    Returns:
        pd.DataFrame: Filtered cloud table. The DataFrame contains useful 
            metadata in .attrs (bands, collection, scale, etc.) needed
            for downstream functions.
    """
    cache_file = _cache_key(lon, lat, edge_size, MSS_PIXEL_SCALE, MSS_COLLECTION)

    # Load cached data if present
    if cache and cache_file.exists():
        print("📂 Loading cached MSS metadata...", end='', flush=True)
        t0 = time.time()
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()
        elapsed = time.time() - t0

        # Handle empty cache or all NaT dates
        if have_idx.empty:
            print(f"\r⚠️  Cache is empty, fetching fresh data ({elapsed:.2f}s)")
            df_cached = pd.DataFrame(
                columns=["id", "cloud_cover", "date", "inside", "path", "row"]
            )
            cached_start = None
            cached_end = None
        else:
            cached_start = have_idx.min().date()
            cached_end = have_idx.max().date()

        if (
            cached_start is not None 
            and cached_end is not None
            and dt.date.fromisoformat(start) >= cached_start
            and dt.date.fromisoformat(end) <= cached_end
        ):
            print(f"\r✅ Loaded {len(df_cached)} images from cache ({elapsed:.2f}s)")
            df_full = df_cached
        else:
            if cached_start is not None and cached_end is not None:
                print(f"\r📂 Cache loaded ({len(df_cached)} images, {elapsed:.2f}s)")
            else:
                print(f"\r📂 Empty cache loaded ({elapsed:.2f}s)")
            
            # Identify missing segments and fetch only those
            print("⏳ Fetching missing date ranges...", end='', flush=True)
            t0 = time.time()
            df_new_parts = []
            
            # If cache is empty, fetch entire range
            if cached_start is None or cached_end is None:
                df_new_parts.append(
                    _mss_table_single_range(
                        lon=lon, 
                        lat=lat, 
                        edge_size=edge_size, 
                        start=start, 
                        end=end
                    )
                )
            else:
                # Fetch missing segments at the start
                if dt.date.fromisoformat(start) < cached_start:
                    a1, b1 = start, cached_start.isoformat()
                    df_new_parts.append(
                        _mss_table_single_range(
                            lon=lon, 
                            lat=lat, 
                            edge_size=edge_size, 
                            start=a1, 
                            end=b1
                        )
                    )
                # Fetch missing segments at the end
                if dt.date.fromisoformat(end) > cached_end:
                    a2, b2 = cached_end.isoformat(), end
                    df_new_parts.append(
                        _mss_table_single_range(
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
        print("⏳ Querying Earth Engine MSS metadata...", end='', flush=True)
        t0 = time.time()
        df_full = _mss_table_single_range(
            lon=lon, 
            lat=lat, 
            edge_size=edge_size, 
            start=start, 
            end=end
        )
        elapsed = time.time() - t0
        n_images = len(df_full)
        if n_images > 0:
            actual_start = df_full['date'].min()
            actual_end = df_full['date'].max()
            print(f"\r✅ Retrieved {n_images} images from {actual_start} to {actual_end} ({elapsed:.2f}s)")
        else:
            print(f"\r✅ Retrieved 0 images ({elapsed:.2f}s)")

    # Save cache
    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # Filter by cloud cover and requested date window
    result = (
        df_full.query("@start <= date <= @end")
        .query("@min_cloud_cover <= cloud_cover <= @max_cloud_cover")
        .reset_index(drop=True)
    )

    # Attach metadata for downstream helpers
    result.attrs.update(
        {
            "lon": lon,
            "lat": lat,
            "edge_size": edge_size,
            "scale": MSS_PIXEL_SCALE,
            "bands": MSS_BANDS,
            "collection": MSS_COLLECTION
        }
    )
    return result