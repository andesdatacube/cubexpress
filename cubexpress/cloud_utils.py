"""Multi-sensor cloud metadata extraction for Earth Engine collections."""

from __future__ import annotations

import datetime as dt
import time
import warnings
from dataclasses import dataclass

import ee
import pandas as pd

from cubexpress.cache import _cache_key
from cubexpress.geospatial import _square_roi

warnings.filterwarnings('ignore', category=DeprecationWarning)


# --- SENSOR CONFIGURATIONS ---

@dataclass
class SensorConfig:
    """Configuration for a satellite sensor.

    Attributes:
        collection: The Earth Engine collection ID(s). Can be a string or list.
        bands: List of band names to be used.
        pixel_scale: Native resolution of the sensor in meters.
        cloud_property: Metadata property name used for cloud filtering.
        cloud_range: Tuple of (min, max) valid values for the cloud property.
        default_dates: Tuple of (start, end) dates. 'end' can be 'today'.
        has_cloud_score_plus: Boolean indicating if Cloud Score Plus is supported.
    """
    collection: str | list[str]
    bands: list[str]
    pixel_scale: int
    cloud_property: str
    cloud_range: tuple[float, float]
    default_dates: tuple[str, str]
    has_cloud_score_plus: bool = False
    toa: bool = False


def _get_ee_collection(config: SensorConfig) -> ee.ImageCollection:
    """Resolves the Earth Engine collection, merging if necessary.

    Args:
        config: The sensor configuration object.

    Returns:
        An ee.ImageCollection (merged if the config has multiple IDs).
    """
    if isinstance(config.collection, list):
        coll = ee.ImageCollection(config.collection[0])
        for c in config.collection[1:]:
            coll = coll.merge(ee.ImageCollection(c))
        return coll
    else:
        return ee.ImageCollection(config.collection)


# --- CONFIGURATION DICTIONARY ---

def _define_mss(
    t1_id: str,
    t2_id: str,
    bands: list[str],
    dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Helper to generate MSS configuration variants (DN, T1, T2)."""
    base_config = {
        "bands": bands,
        "pixel_scale": 60,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False
    }
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], **base_config),
        "T1": SensorConfig(collection=t1_id, **base_config),
        "T2": SensorConfig(collection=t2_id, **base_config),
        "TOA": SensorConfig(collection=[t1_id, t2_id], toa=True, **base_config),
        "T1_TOA": SensorConfig(collection=t1_id, toa=True, **base_config),
        "T2_TOA": SensorConfig(collection=t2_id, toa=True, **base_config)
    }
    
def _define_tm(
    t1_id: str,
    t2_id: str,
    t1_toa_id: str,
    t2_toa_id: str,
    t1_l2_id: str,
    t2_l2_id: str,
    bands_base: list[str],
    bands_sr: list[str],
    dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Helper to generate TM configuration variants (DN, T1, T2, TOA, BOA)."""
    base_config = {
        "pixel_scale": 30,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False
    }
    
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], bands=bands_base, **base_config),
        "T1": SensorConfig(collection=t1_id, bands=bands_base, **base_config),
        "T2": SensorConfig(collection=t2_id, bands=bands_base, **base_config),
        "TOA": SensorConfig(collection=[t1_toa_id, t2_toa_id], bands=bands_base, toa=True, **base_config),
        "T1_TOA": SensorConfig(collection=t1_toa_id, bands=bands_base, toa=True, **base_config),
        "T2_TOA": SensorConfig(collection=t2_toa_id, bands=bands_base, toa=True, **base_config),
        "BOA": SensorConfig(collection=[t1_l2_id, t2_l2_id], bands=bands_sr, **base_config), 
        "T1_BOA": SensorConfig(collection=t1_l2_id, bands=bands_sr, **base_config),
        "T2_BOA": SensorConfig(collection=t2_l2_id, bands=bands_sr, **base_config),
    }

def _define_etm(
    t1_id: str,
    t2_id: str,
    t1_toa_id: str,
    t2_toa_id: str,
    t1_l2_id: str,
    t2_l2_id: str,
    bands_base: list[str],
    bands_sr: list[str],
    dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Helper to generate ETM+ configuration variants (DN, T1, T2, TOA, BOA)."""
    base_config = {
        "pixel_scale": 30,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False
    }
    
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], bands=bands_base, **base_config),
        "T1": SensorConfig(collection=t1_id, bands=bands_base, **base_config),
        "T2": SensorConfig(collection=t2_id, bands=bands_base, **base_config),
        "TOA": SensorConfig(collection=[t1_toa_id, t2_toa_id], bands=bands_base, toa=True, **base_config),
        "T1_TOA": SensorConfig(collection=t1_toa_id, bands=bands_base, toa=True, **base_config),
        "T2_TOA": SensorConfig(collection=t2_toa_id, bands=bands_base, toa=True, **base_config),
        "BOA": SensorConfig(collection=[t1_l2_id, t2_l2_id], bands=bands_sr, **base_config),
        "T1_BOA": SensorConfig(collection=t1_l2_id, bands=bands_sr, **base_config),
        "T2_BOA": SensorConfig(collection=t2_l2_id, bands=bands_sr, **base_config),
    }

def _define_oli(
    t1_id: str,
    t2_id: str,
    t1_toa_id: str,
    t2_toa_id: str,
    t1_l2_id: str,
    t2_l2_id: str,
    bands_base: list[str],
    bands_sr: list[str],
    dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Helper to generate OLI/TIRS configuration variants (DN, T1, T2, TOA, BOA)."""
    base_config = {
        "pixel_scale": 30,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False
    }
    
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], bands=bands_base, **base_config),
        "T1": SensorConfig(collection=t1_id, bands=bands_base, **base_config),
        "T2": SensorConfig(collection=t2_id, bands=bands_base, **base_config),
        "TOA": SensorConfig(collection=[t1_toa_id, t2_toa_id], bands=bands_base, toa=True, **base_config),
        "T1_TOA": SensorConfig(collection=t1_toa_id, bands=bands_base, toa=True, **base_config),
        "T2_TOA": SensorConfig(collection=t2_toa_id, bands=bands_base, toa=True, **base_config),
        "BOA": SensorConfig(collection=[t1_l2_id, t2_l2_id], bands=bands_sr, **base_config),
        "T1_BOA": SensorConfig(collection=t1_l2_id, bands=bands_sr, **base_config),
        "T2_BOA": SensorConfig(collection=t2_l2_id, bands=bands_sr, **base_config),
    }
    
# Pre-define MSS variants
# Pre-define MSS variants
_m1 = _define_mss("LANDSAT/LM01/C02/T1", "LANDSAT/LM01/C02/T2", ["B4", "B5", "B6", "B7"], ("1972-07-23", "1978-01-06"))
_m2 = _define_mss("LANDSAT/LM02/C02/T1", "LANDSAT/LM02/C02/T2", ["B4", "B5", "B6", "B7"], ("1975-01-22", "1982-02-25"))
_m3 = _define_mss("LANDSAT/LM03/C02/T1", "LANDSAT/LM03/C02/T2", ["B4", "B5", "B6", "B7"], ("1978-03-05", "1983-03-31"))
_m4 = _define_mss("LANDSAT/LM04/C02/T1", "LANDSAT/LM04/C02/T2", ["B1", "B2", "B3", "B4"], ("1982-07-16", "1993-12-14"))
_m5 = _define_mss("LANDSAT/LM05/C02/T1", "LANDSAT/LM05/C02/T2", ["B1", "B2", "B3", "B4"], ("1984-03-01", "2013-01-05"))

# Pre-define TM variants (Landsat 4 & 5)
_tm4 = _define_tm(
    "LANDSAT/LT04/C02/T1", "LANDSAT/LT04/C02/T2",
    "LANDSAT/LT04/C02/T1_TOA", "LANDSAT/LT04/C02/T2_TOA",
    "LANDSAT/LT04/C02/T1_L2", "LANDSAT/LT04/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],  # DN/TOA bands (includes thermal)
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],  # L2 surface reflectance only
    ("1982-08-22", "1993-12-14")
)

_tm5 = _define_tm(
    "LANDSAT/LT05/C02/T1", "LANDSAT/LT05/C02/T2",
    "LANDSAT/LT05/C02/T1_TOA", "LANDSAT/LT05/C02/T2_TOA",
    "LANDSAT/LT05/C02/T1_L2", "LANDSAT/LT05/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    ("1984-03-16", "2012-05-05")
)

# Pre-define ETM+ variants (Landsat 7)
_etm = _define_etm(
    "LANDSAT/LE07/C02/T1", "LANDSAT/LE07/C02/T2",
    "LANDSAT/LE07/C02/T1_TOA", "LANDSAT/LE07/C02/T2_TOA",
    "LANDSAT/LE07/C02/T1_L2", "LANDSAT/LE07/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7", "B8"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    ("1999-05-28", "today")
)

# Pre-define OLI/TIRS variants (Landsat 8 & 9)
_oli8 = _define_oli(
    "LANDSAT/LC08/C02/T1", "LANDSAT/LC08/C02/T2",
    "LANDSAT/LC08/C02/T1_TOA", "LANDSAT/LC08/C02/T2_TOA",
    "LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC08/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    ("2013-03-18", "today")
)

_oli9 = _define_oli(
    "LANDSAT/LC09/C02/T1", "LANDSAT/LC09/C02/T2",
    "LANDSAT/LC09/C02/T1_TOA", "LANDSAT/LC09/C02/T2_TOA",
    "LANDSAT/LC09/C02/T1_L2", "LANDSAT/LC09/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    ("2021-10-31", "today")
)

# Helper to extract collections
def _extract_collections(sensor_dict: dict, key: str) -> list[str]:
    """Extract collection IDs from sensor config."""
    coll = sensor_dict[key].collection
    return coll if isinstance(coll, list) else [coll]

SENSORS = {
    # --- Sentinel-2 ---
    "S2": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
        pixel_scale=10,
        cloud_property="cs_cdf",
        cloud_range=(0.0, 1.0),
        default_dates=("2015-06-23", "today"),
        has_cloud_score_plus=True
    ),

    # --- Landsat MSS Variants ---
    "MSS1": _m1["DN"], "MSS1_T1": _m1["T1"], "MSS1_T2": _m1["T2"], 
    "MSS1_TOA": _m1["TOA"], "MSS1_T1_TOA": _m1["T1_TOA"], "MSS1_T2_TOA": _m1["T2_TOA"],
    "MSS2": _m2["DN"], "MSS2_T1": _m2["T1"], "MSS2_T2": _m2["T2"], 
    "MSS2_TOA": _m2["TOA"], "MSS2_T1_TOA": _m2["T1_TOA"], "MSS2_T2_TOA": _m2["T2_TOA"],
    "MSS3": _m3["DN"], "MSS3_T1": _m3["T1"], "MSS3_T2": _m3["T2"], 
    "MSS3_TOA": _m3["TOA"], "MSS3_T1_TOA": _m3["T1_TOA"], "MSS3_T2_TOA": _m3["T2_TOA"],
    "MSS4": _m4["DN"], "MSS4_T1": _m4["T1"], "MSS4_T2": _m4["T2"], 
    "MSS4_TOA": _m4["TOA"], "MSS4_T1_TOA": _m4["T1_TOA"], "MSS4_T2_TOA": _m4["T2_TOA"],
    "MSS5": _m5["DN"], "MSS5_T1": _m5["T1"], "MSS5_T2": _m5["T2"], 
    "MSS5_TOA": _m5["TOA"], "MSS5_T1_TOA": _m5["T1_TOA"], "MSS5_T2_TOA": _m5["T2_TOA"],
    
    # --- Landsat TM Variants (4 & 5) ---
    "TM4": _tm4["DN"], 
    "TM4_T1": _tm4["T1"], 
    "TM4_T2": _tm4["T2"],
    
    "TM4_TOA": _tm4["TOA"], 
    "TM4_T1_TOA": _tm4["T1_TOA"], 
    "TM4_T2_TOA": _tm4["T2_TOA"],
    
    "TM4_BOA": _tm4["BOA"], 
    "TM4_T1_BOA": _tm4["T1_BOA"], 
    "TM4_T2_BOA": _tm4["T2_BOA"],
    
    "TM5": _tm5["DN"], 
    "TM5_T1": _tm5["T1"], 
    "TM5_T2": _tm5["T2"],
    
    "TM5_TOA": _tm5["TOA"], 
    "TM5_T1_TOA": _tm5["T1_TOA"], 
    "TM5_T2_TOA": _tm5["T2_TOA"],
    
    "TM5_BOA": _tm5["BOA"], 
    "TM5_T1_BOA": _tm5["T1_BOA"], 
    "TM5_T2_BOA": _tm5["T2_BOA"],
    
    # --- Landsat ETM+ Variants (7) ---
    "ETM": _etm["DN"], "ETM_T1": _etm["T1"], "ETM_T2": _etm["T2"],
    "ETM_TOA": _etm["TOA"], "ETM_T1_TOA": _etm["T1_TOA"], "ETM_T2_TOA": _etm["T2_TOA"],
    "ETM_BOA": _etm["BOA"], "ETM_T1_BOA": _etm["T1_BOA"], "ETM_T2_BOA": _etm["T2_BOA"],
    
    # --- Landsat OLI/TIRS Variants (8 & 9) ---
    "OLI8": _oli8["DN"], "OLI8_T1": _oli8["T1"], "OLI8_T2": _oli8["T2"],
    "OLI8_TOA": _oli8["TOA"], "OLI8_T1_TOA": _oli8["T1_TOA"], "OLI8_T2_TOA": _oli8["T2_TOA"],
    "OLI8_BOA": _oli8["BOA"], "OLI8_T1_BOA": _oli8["T1_BOA"], "OLI8_T2_BOA": _oli8["T2_BOA"],
    
    "OLI9": _oli9["DN"], "OLI9_T1": _oli9["T1"], "OLI9_T2": _oli9["T2"],
    "OLI9_TOA": _oli9["TOA"], "OLI9_T1_TOA": _oli9["T1_TOA"], "OLI9_T2_TOA": _oli9["T2_TOA"],
    "OLI9_BOA": _oli9["BOA"], "OLI9_T1_BOA": _oli9["T1_BOA"], "OLI9_T2_BOA": _oli9["T2_BOA"],
    
    "LANDSAT": SensorConfig(
        collection=[
            *_extract_collections(_m1, "DN"),
            *_extract_collections(_m2, "DN"),
            *_extract_collections(_m3, "DN"),
            *_extract_collections(_m4, "DN"),
            *_extract_collections(_m5, "DN"),
            *_extract_collections(_tm4, "DN"),
            *_extract_collections(_tm5, "DN"),
            *_extract_collections(_etm, "DN"),
            *_extract_collections(_oli8, "DN"),
            *_extract_collections(_oli9, "DN"),
        ],
        bands=[],
        pixel_scale=30,
        cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0),
        default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False,
        toa=False
    ),

    "LANDSAT_TOA": SensorConfig(
        collection=[
            *_extract_collections(_m1, "TOA"),
            *_extract_collections(_m2, "TOA"),
            *_extract_collections(_m3, "TOA"),
            *_extract_collections(_m4, "TOA"),
            *_extract_collections(_m5, "TOA"),
            *_extract_collections(_tm4, "TOA"),
            *_extract_collections(_tm5, "TOA"),
            *_extract_collections(_etm, "TOA"),
            *_extract_collections(_oli8, "TOA"),
            *_extract_collections(_oli9, "TOA"),
        ],
        bands=[],
        pixel_scale=30,
        cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0),
        default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False,
        toa=True
    ),

    "LANDSAT_BOA": SensorConfig(
        collection=[
            *_extract_collections(_tm4, "BOA"),
            *_extract_collections(_tm5, "BOA"),
            *_extract_collections(_etm, "BOA"),
            *_extract_collections(_oli8, "BOA"),
            *_extract_collections(_oli9, "BOA"),
        ],
        bands=[],  
        pixel_scale=30,
        cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0),
        default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False,
        toa=False
    ),
}


# --- SENTINEL-2 SPECIFIC (Cloud Score Plus) ---

def _s2_cloud_table_single_range(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str,
    config: SensorConfig,
    scale: int
) -> pd.DataFrame:
    """Builds a cloud-score table for Sentinel-2 using Cloud Score Plus.

    Args:
        lon: Longitude of the center point.
        lat: Latitude of the center point.
        edge_size: Side length of the square region (in pixels relative to scale).
        start: ISO-8601 start date (inclusive).
        end: ISO-8601 end date (inclusive).
        config: Sensor configuration object.
        scale: The scale in meters to use for the ROI and extraction.

    Returns:
        pd.DataFrame: DataFrame with full 'id' (collection/index), dates,
        cloud scores, and inside status.
    """
    center = ee.Geometry.Point([lon, lat])
    roi = _square_roi(lon, lat, edge_size, scale)
    
    # Query S2
    s2 = (
        ee.ImageCollection(config.collection)
        .filterBounds(roi)
        .filterDate(start, end)
    )
    
    # Cloud Score Plus collection
    cloud_collection = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    ic = (
        s2
        .linkCollection(
            ee.ImageCollection(cloud_collection), 
            [config.cloud_property]
        )
        .select([config.cloud_property])
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
    
    # Generate cloud score of each image over the ROI
    try:
        raw = ic.getRegion(
            geometry=center,
            scale=scale * 1.1
        ).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands in collection" in str(e):
            return pd.DataFrame(
                columns=["id", "longitude", "latitude", "time", config.cloud_property, "date", "inside"]
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
    
    # Construct full ID (Collection + System Index) for S2
    # S2 config.collection is always a single string in this setup
    if isinstance(config.collection, str):
        df_raw['id'] = config.collection + "/" + df_raw['id']

    # Mark images whose ROI is fully inside the scene
    df_raw["inside"] = df_raw["id"].apply(lambda x: x.split("/")[-1]).isin(set(ids_inside)).astype(int)
    
    # Fill missing cloud scores with daily mean
    df_raw[config.cloud_property] = df_raw.groupby('date').apply(
        lambda group: group[config.cloud_property].transform(
            lambda _: group[group['inside'] == 1][config.cloud_property].iloc[0] 
            if (group['inside'] == 1).any() 
            else group[config.cloud_property].mean()
        )
    ).reset_index(drop=True)

    return df_raw


# --- GENERIC METADATA EXTRACTION ---

def _generic_metadata_table_single_range(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str,
    end: str,
    config: SensorConfig,
    scale: int
) -> pd.DataFrame:
    """Builds a metadata table for sensors using standard Earth Engine properties.

    Optimized to fetch all metadata in a single server-side request to avoid 
    timeouts with large collections like MSS.

    Args:
        lon: Longitude of the center point.
        lat: Latitude of the center point.
        edge_size: Side length of the square region (in pixels relative to scale).
        start: ISO-8601 start date (inclusive).
        end: ISO-8601 end date (inclusive).
        config: Sensor configuration object.
        scale: The scale in meters to use for the ROI definition.

    Returns:
        pd.DataFrame: DataFrame containing full asset IDs, dates, cloud metadata,
        and footprint status.
    """
    roi = _square_roi(lon, lat, edge_size, scale)
    
    # Get collection (handles merging T1+T2)
    collection = (
        _get_ee_collection(config)
        .filterBounds(roi)
        .filterDate(start, end)
    )
    
    # Use server-side mapping to extract properties efficiently
    def extract_props(img):
        # Check if the ROI is fully contained within the image footprint
        inside = img.geometry().contains(roi, 10)
        
        # Return a Feature with only the necessary metadata properties
        # 'system:id' gives the full ID (collection/asset_id)
        return ee.Feature(None, {
            'id': img.get('system:id'), 
            config.cloud_property: img.get(config.cloud_property),
            'date': img.get('DATE_ACQUIRED'),
            'inside': inside,
            'path': img.get('WRS_PATH'),
            'row': img.get('WRS_ROW')
        })

    # Map the extraction function over the collection
    meta_fc = collection.map(extract_props)
    
    try:
        # Fetch all metadata in a single request
        data = meta_fc.getInfo()
    except ee.ee_exception.EEException as e:
        # Handle cases where the collection might be empty or malformed
        if "No bands" in str(e):
            return pd.DataFrame(
                columns=["id", config.cloud_property, "date", "inside", "path", "row"]
            )
        raise e
        
    # Process the returned JSON data
    features = data.get('features', [])
    
    if not features:
        return pd.DataFrame(
            columns=["id", config.cloud_property, "date", "inside", "path", "row"]
        )
        
    # Extract properties from the feature list
    records = [feat['properties'] for feat in features]
    df_raw = pd.DataFrame(records)
    
    if not df_raw.empty:
        # Ensure all columns exist even if some records returned nulls
        expected_cols = ["id", config.cloud_property, "date", "inside", "path", "row"]
        for col in expected_cols:
            if col not in df_raw.columns:
                df_raw[col] = None
                
        # Format date column
        if 'date' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['date']).dt.strftime('%Y-%m-%d')
            
        # Ensure boolean/integer consistency for the 'inside' flag
        if 'inside' in df_raw.columns:
            df_raw['inside'] = df_raw['inside'].fillna(0).astype(int)

    return df_raw

# --- MAIN TABLE FUNCTION ---

def _sensor_table(
    sensor: str,
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    max_cloud: float | None = None,
    min_cloud: float | None = None,
    scale: int | None = None,
    bands: list[str] | None = None,
    cache: bool = False
) -> pd.DataFrame:
    """Generic coordinator to build metadata tables for any sensor.

    Args:
        sensor: Sensor key (e.g., "S2", "MSS1").
        lon: Longitude of the center point.
        lat: Latitude of the center point.
        edge_size: Side length of the square region.
        start: Start date. If None, uses sensor defaults.
        end: End date. If None, uses sensor defaults.
        max_cloud: Max cloud threshold.
        min_cloud: Min cloud threshold.
        scale: Resolution in meters.
        cache: If True, uses local parquet caching.

    Returns:
        pd.DataFrame: Filtered metadata table with .attrs attached.
    """
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Available: {list(SENSORS.keys())}")
    
    config = SENSORS[sensor]
    
    # --- HANDLE DEFAULTS ---
    if start is None:
        start = config.default_dates[0]
    if end is None:
        raw_end = config.default_dates[1]
        end = dt.date.today().strftime("%Y-%m-%d") if raw_end == "today" else raw_end
    if max_cloud is None:
        max_cloud = config.cloud_range[1]
    if min_cloud is None:
        min_cloud = config.cloud_range[0]
        
    effective_scale = scale if scale is not None else config.pixel_scale
    effective_bands = bands if bands is not None else config.bands
    
    cache_file = _cache_key(lon, lat, edge_size, effective_scale, str(config.collection))

    if config.has_cloud_score_plus:
        extract_fn = _s2_cloud_table_single_range
    else:
        extract_fn = _generic_metadata_table_single_range

    # --- CACHING & EXTRACTION ---
    if cache and cache_file.exists():
        print(f"📂 Loading cached {sensor} metadata...", end='', flush=True)
        t0 = time.time()
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()
        elapsed = time.time() - t0

        if have_idx.empty:
            df_cached = pd.DataFrame()
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
            df_full = df_cached
        else:
            print(f"\r📂 Cache loaded ({len(df_cached)} imgs)... checking missing ranges", end='', flush=True)
            df_new_parts = []
            
            if cached_start is None:
                df_new_parts.append(
                    extract_fn(lon, lat, edge_size, start, end, config, effective_scale)
                )
            else:
                if dt.date.fromisoformat(start) < cached_start:
                    df_new_parts.append(
                        extract_fn(lon, lat, edge_size, start, cached_start.isoformat(), config, effective_scale)
                    )
                if dt.date.fromisoformat(end) > cached_end:
                    df_new_parts.append(
                        extract_fn(lon, lat, edge_size, cached_end.isoformat(), end, config, effective_scale)
                    )
            
            df_new_parts = [df for df in df_new_parts if not df.empty]
            
            if df_new_parts:
                df_new = pd.concat(df_new_parts, ignore_index=True)
                df_full = pd.concat([df_cached, df_new], ignore_index=True).sort_values("date")
            else:
                df_full = df_cached
    else:
        print(f"⏳ Querying {sensor} (Scale: {effective_scale}m)...", end='', flush=True)
        t0 = time.time()
        df_full = extract_fn(lon, lat, edge_size, start, end, config, effective_scale)
        elapsed = time.time() - t0

    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # Apply filters
    result = (
        df_full.query("@start <= date <= @end")
        .query(f"@min_cloud <= {config.cloud_property} <= @max_cloud")
        .sort_values("date")
        .reset_index(drop=True)
    )
    
    # Print AFTER filtering
    print(f"\r✅ Retrieved {len(result)} images ({elapsed:.2f}s)")

    result.attrs.update({
        "lon": lon,
        "lat": lat,
        "edge_size": edge_size,
        "scale": effective_scale,
        "bands": effective_bands,
        "collection": config.collection,
        "start": start,
        "end": end,
        "toa": config.toa
    })
    return result

# --- PUBLIC API FUNCTIONS ---

def sensor_table(
    sensor: str,
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    scale: int | None = None,
    max_cloud: float | None = None,
    min_cloud: float | None = None,
    bands: list[str] | None = None,  # NEW PARAMETER
    cache: bool = False
) -> pd.DataFrame:
    """Builds (and caches) a metadata table for any supported sensor.

    Args:
        sensor: Sensor identifier (e.g., "S2", "MSS1_TOA", "TM5_BOA").
        lon: Longitude of the center point.
        lat: Latitude of the center point.
        edge_size: Box size in pixels (relative to scale).
        start: Start date (YYYY-MM-DD). If None, uses sensor defaults.
        end: End date (YYYY-MM-DD). If None, uses sensor defaults.
        scale: Resolution in meters. If None, uses sensor native resolution.
        max_cloud: Maximum cloud cover/score threshold.
        min_cloud: Minimum cloud cover/score threshold.
        bands: List of bands to include. If None, uses sensor defaults.
               **Ignored for aggregated sensors (LANDSAT, LANDSAT_TOA, LANDSAT_BOA).**
        cache: Enable local parquet caching for faster subsequent queries.

    Returns:
        pd.DataFrame: Metadata table with columns: id, date, cloud metric, inside, path, row.
        
    Examples:
        >>> # Sentinel-2 with custom bands
        >>> df = sensor_table("S2", lon=-0.09, lat=51.5, edge_size=256, 
        ...                   bands=["B2", "B3", "B4", "B8"])
        
        >>> # Landsat 5 MSS with custom bands
        >>> df = sensor_table("MSS5", lon=-0.09, lat=51.5, edge_size=172,
        ...                   bands=["B1", "B2", "B3"])
        
        >>> # Aggregated LANDSAT (bands parameter ignored)
        >>> df = sensor_table("LANDSAT_TOA", lon=-0.09, lat=51.5, edge_size=256)
    """
    # Aggregated sensors that do NOT allow band override
    AGGREGATED_SENSORS = {"LANDSAT", "LANDSAT_TOA", "LANDSAT_BOA"}
    
    if sensor in AGGREGATED_SENSORS and bands is not None:
        import warnings
        warnings.warn(
            f"Parameter 'bands' is ignored for aggregated sensor '{sensor}'. "
            f"All available bands will be downloaded.",
            UserWarning
        )
        bands = None
    
    return _sensor_table(
        sensor=sensor,
        lon=lon,
        lat=lat,
        edge_size=edge_size,
        start=start,
        end=end,
        scale=scale,
        max_cloud=max_cloud,
        min_cloud=min_cloud,
        bands=bands,
        cache=cache
    )


def s2_table(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    scale: int | None = None,
    max_cscore: float | None = None,
    min_cscore: float | None = None,
    cache: bool = False
) -> pd.DataFrame:
    """Builds (and caches) a per-day cloud-table for Sentinel-2.
    
    Convenience wrapper for sensor_table(sensor="S2", ...).

    Args:
        lon: Longitude.
        lat: Latitude.
        edge_size: Box size in pixels (relative to scale).
        start: Start date (optional).
        end: End date (optional).
        scale: Resolution in meters (optional).
        max_cscore: Max cloud score (0-1).
        min_cscore: Min cloud score (0-1).
        cache: Enable caching.

    Returns:
        pd.DataFrame: Sentinel-2 metadata.
    """
    return _sensor_table(
        sensor="S2", lon=lon, lat=lat, edge_size=edge_size, start=start, end=end,
        scale=scale, max_cloud=max_cscore, min_cloud=min_cscore, cache=cache
    )


def mss_table(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    start: str | None = None,
    end: str | None = None,
    sensor: str = "MSS1",
    scale: int | None = None,
    max_cloud_cover: float | None = None,
    min_cloud_cover: float | None = None,
    cache: bool = False,
) -> pd.DataFrame:
    """Builds (and caches) a per-day cloud-table for Landsat MSS.
    
    Convenience wrapper for sensor_table(sensor=..., ...).

    Args:
        lon: Longitude.
        lat: Latitude.
        edge_size: Box size in pixels (relative to scale).
        start: Start date (optional).
        end: End date (optional).
        sensor: MSS sensor key (e.g., "MSS1", "MSS5", "MSS5_TOA").
        scale: Resolution in meters (optional).
        max_cloud_cover: Max cloud cover (0-100).
        min_cloud_cover: Min cloud cover (0-100).
        cache: Enable caching.

    Returns:
        pd.DataFrame: Landsat MSS metadata.
    """
    return _sensor_table(
        sensor=sensor, lon=lon, lat=lat, edge_size=edge_size, start=start, end=end,
        scale=scale, max_cloud=max_cloud_cover, min_cloud=min_cloud_cover, cache=cache
    )