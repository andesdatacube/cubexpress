"""Multi-sensor cloud metadata extraction for Earth Engine collections."""

from __future__ import annotations

import datetime as dt
import time
import warnings
from dataclasses import dataclass
from typing import Final

import ee
import pandas as pd

from cubexpress.cache import _cache_key
from cubexpress.geospatial import _square_roi

warnings.filterwarnings('ignore', category=DeprecationWarning)


# --- LANDSAT METADATA CONSTANTS ---

# Base properties always extracted for Landsat sensors
LANDSAT_BASE_PROPS: Final[list[str]] = [
    'CLOUD_COVER', 'DATE_ACQUIRED', 'WRS_PATH', 'WRS_ROW'
]

# Optional properties common to ALL Landsat sensors and processing levels
# (MSS through OLI, DN/TOA/BOA). Users can request via extra_properties.
# Note: WRS_PATH and WRS_ROW are always included in base columns.
LANDSAT_COMMON_OPTIONAL: Final[set[str]] = {
    'CLOUD_COVER_LAND',
    'SUN_AZIMUTH',
    'SUN_ELEVATION',
    'EARTH_SUN_DISTANCE',
    'SPACECRAFT_ID',
    'SENSOR_ID',
    'IMAGE_QUALITY',
    'COLLECTION_CATEGORY',
    'LANDSAT_PRODUCT_ID',
    'LANDSAT_SCENE_ID',
    'GEOMETRIC_RMSE_MODEL',
    'GROUND_CONTROL_POINTS_MODEL',
}

# Aggregated sensor keys (for validation)
AGGREGATED_SENSORS: Final[set[str]] = {
    "LANDSAT", "LANDSAT_TOA", "LANDSAT_BOA",
    "MULTISPECTRAL_TOA", "MULTISPECTRAL_BOA"
}

# Native scale for sensors (for align_to_grid validation)
SENSOR_NATIVE_SCALE: Final[dict[str, int]] = {
    "MSS1": 60, "MSS2": 60, "MSS3": 60, "MSS4": 60, "MSS5": 60,
    "TM4": 30, "TM5": 30, "ETM+": 30, "OLI8": 30, "OLI9": 30,
    "S2": 10,
}

# Mapping from collection path prefix to sensor name
ASSET_ID_TO_SENSOR: Final[dict[str, str]] = {
    "LANDSAT/LM01": "MSS1",
    "LANDSAT/LM02": "MSS2",
    "LANDSAT/LM03": "MSS3",
    "LANDSAT/LM04": "MSS4",
    "LANDSAT/LM05": "MSS5",
    "LANDSAT/LT04": "TM4",
    "LANDSAT/LT05": "TM5",
    "LANDSAT/LE07": "ETM+",
    "LANDSAT/LC08": "OLI8",
    "LANDSAT/LC09": "OLI9",
}

def _parse_sensor_from_id(asset_id: str) -> str | None:
    """Extract sensor name with tier from Earth Engine asset ID.
    
    Returns format: "SENSOR-TIER" (e.g., "OLI8-T1", "MSS1-T2", "OLI9-RT", "S2-MSI")
    """
    # Handle Sentinel-2 (same instrument for both TOA and BOA)
    if asset_id.startswith("COPERNICUS/S2"):
        return "S2-MSI"
    
    # Handle Landsat
    base_sensor = None
    for prefix, sensor in ASSET_ID_TO_SENSOR.items():
        if asset_id.startswith(prefix):
            base_sensor = sensor
            break
    
    if base_sensor is None:
        return None
    
    # Identify the tier from the collection path
    if "/T1_RT" in asset_id or "/T1_RT_TOA" in asset_id:
        tier = "RT"
    elif "/T2" in asset_id or "/T2_TOA" in asset_id or "/T2_L2" in asset_id:
        tier = "T2"
    elif "/T1" in asset_id or "/T1_TOA" in asset_id or "/T1_L2" in asset_id:
        tier = "T1"
    else:
        tier = "T1"
    
    return f"{base_sensor}-{tier}"

def _get_grid_reference(
    reference: str,
    lon: float,
    lat: float,
    scale: int
) -> tuple[float, float, int]:
    """
    Get grid reference coordinates from a reference sensor or asset ID.
    
    Returns coordinates in the same UTM zone as the target point (lon, lat).
    """
    from pyproj import CRS, Transformer

    from cubexpress.conversion import geo2utm, lonlat2rt_utm_or_ups

    # Determine target CRS (same logic as lonlat2rt)
    try:
        _, _, target_crs = geo2utm(lon, lat)
    except Exception:
        _, _, target_crs = lonlat2rt_utm_or_ups(lon, lat)
    
    # Determine if reference is asset ID or sensor key
    if reference.startswith("LANDSAT/"):
        asset_id = reference
        is_mss = "/LM0" in reference
        native_scale = 60 if is_mss else 30
    elif reference.startswith("COPERNICUS/S2"):
        asset_id = reference
        native_scale = 10
    else:
        if reference not in SENSORS:
            raise ValueError(f"Unknown sensor '{reference}' for align_to_grid")
        
        config = SENSORS[reference]
        native_scale = config.pixel_scale
        roi = _square_roi(lon, lat, 1, scale)
        
        collection = _get_ee_collection(config).filterBounds(roi).limit(1)
        asset_id = collection.first().get('system:id').getInfo()
        
        if asset_id is None:
            raise ValueError(
                f"No images found for sensor '{reference}' at ({lon}, {lat})."
            )
    
    # Get projection info from asset
    proj = ee.Image(asset_id).select(0).projection()
    proj_info = proj.getInfo()
    transform = proj_info['transform']
    source_crs_wkt = proj_info.get('wkt') or proj_info.get('crs')
    
    # Extract translate coordinates in source CRS
    src_x = transform[2]
    src_y = transform[5]
    
    # Transform to target CRS if different
    try:
        source_crs = CRS.from_wkt(source_crs_wkt) if 'PROJCS' in str(source_crs_wkt) else CRS.from_string(source_crs_wkt)
        target_crs_obj = CRS.from_string(target_crs)
        
        if source_crs != target_crs_obj:
            transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
            ref_x, ref_y = transformer.transform(src_x, src_y)
        else:
            ref_x, ref_y = src_x, src_y
    except Exception as e:
        warnings.warn(
            f"Could not transform grid reference CRS: {e}. Using raw coordinates.",
            UserWarning
        )
        ref_x, ref_y = src_x, src_y
    
    if native_scale != scale:
        warnings.warn(
            f"Asset native scale {native_scale}m differs from requested {scale}m.",
            UserWarning
        )
    
    return (ref_x, ref_y, native_scale)


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
    """Resolves the Earth Engine collection, merging if necessary."""
    if isinstance(config.collection, list):
        coll = ee.ImageCollection(config.collection[0])
        for c in config.collection[1:]:
            coll = coll.merge(ee.ImageCollection(c))
        return coll
    return ee.ImageCollection(config.collection)


# --- CONFIGURATION HELPERS ---

def _define_mss(
    t1_id: str, t2_id: str, bands: list[str], dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Generate MSS configuration variants (DN, T1, T2, TOA)."""
    base = {
        "bands": bands, "pixel_scale": 60, "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0), "default_dates": dates, "has_cloud_score_plus": False
    }
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], **base),
        "T1": SensorConfig(collection=t1_id, **base),
        "T2": SensorConfig(collection=t2_id, **base),
        "TOA": SensorConfig(collection=[t1_id, t2_id], toa=True, **base),
        "T1_TOA": SensorConfig(collection=t1_id, toa=True, **base),
        "T2_TOA": SensorConfig(collection=t2_id, toa=True, **base)
    }

def _define_tm(
    t1_id: str, t2_id: str, t1_toa_id: str, t2_toa_id: str,
    t1_l2_id: str, t2_l2_id: str, bands_base: list[str],
    bands_sr: list[str], dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Generate TM configuration variants (DN, T1, T2, TOA, BOA)."""
    base = {
        "pixel_scale": 30, "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0), "default_dates": dates, "has_cloud_score_plus": False
    }
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], bands=bands_base, **base),
        "T1": SensorConfig(collection=t1_id, bands=bands_base, **base),
        "T2": SensorConfig(collection=t2_id, bands=bands_base, **base),
        "TOA": SensorConfig(collection=[t1_toa_id, t2_toa_id], bands=bands_base, toa=True, **base),
        "T1_TOA": SensorConfig(collection=t1_toa_id, bands=bands_base, toa=True, **base),
        "T2_TOA": SensorConfig(collection=t2_toa_id, bands=bands_base, toa=True, **base),
        "BOA": SensorConfig(collection=[t1_l2_id, t2_l2_id], bands=bands_sr, **base),
        "T1_BOA": SensorConfig(collection=t1_l2_id, bands=bands_sr, **base),
        "T2_BOA": SensorConfig(collection=t2_l2_id, bands=bands_sr, **base),
    }

def _define_etm(
    t1_id: str, t2_id: str, t1_toa_id: str, t2_toa_id: str,
    t1_l2_id: str, t2_l2_id: str, bands_base: list[str],
    bands_sr: list[str], dates: tuple[str, str]
) -> dict[str, SensorConfig]:
    """Generate ETM+ configuration variants (DN, T1, T2, TOA, BOA)."""
    base = {
        "pixel_scale": 30, "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0), "default_dates": dates, "has_cloud_score_plus": False
    }
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], bands=bands_base, **base),
        "T1": SensorConfig(collection=t1_id, bands=bands_base, **base),
        "T2": SensorConfig(collection=t2_id, bands=bands_base, **base),
        "TOA": SensorConfig(collection=[t1_toa_id, t2_toa_id], bands=bands_base, toa=True, **base),
        "T1_TOA": SensorConfig(collection=t1_toa_id, bands=bands_base, toa=True, **base),
        "T2_TOA": SensorConfig(collection=t2_toa_id, bands=bands_base, toa=True, **base),
        "BOA": SensorConfig(collection=[t1_l2_id, t2_l2_id], bands=bands_sr, **base),
        "T1_BOA": SensorConfig(collection=t1_l2_id, bands=bands_sr, **base),
        "T2_BOA": SensorConfig(collection=t2_l2_id, bands=bands_sr, **base),
    }

def _define_oli(
    t1_id: str, t2_id: str, t1_toa_id: str, t2_toa_id: str,
    t1_l2_id: str, t2_l2_id: str, bands_base: list[str],
    bands_sr: list[str], dates: tuple[str, str],
    rt_id: str | None = None, rt_toa_id: str | None = None
) -> dict[str, SensorConfig]:
    """Generate OLI/TIRS configuration variants (DN, T1, T2, TOA, BOA, RT)."""
    base = {
        "pixel_scale": 30, "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0), "default_dates": dates, "has_cloud_score_plus": False
    }
    
    configs = {
        "DN": SensorConfig(collection=[t1_id, t2_id], bands=bands_base, **base),
        "T1": SensorConfig(collection=t1_id, bands=bands_base, **base),
        "T2": SensorConfig(collection=t2_id, bands=bands_base, **base),
        "TOA": SensorConfig(collection=[t1_toa_id, t2_toa_id], bands=bands_base, toa=True, **base),
        "T1_TOA": SensorConfig(collection=t1_toa_id, bands=bands_base, toa=True, **base),
        "T2_TOA": SensorConfig(collection=t2_toa_id, bands=bands_base, toa=True, **base),
        "BOA": SensorConfig(collection=[t1_l2_id, t2_l2_id], bands=bands_sr, **base),
        "T1_BOA": SensorConfig(collection=t1_l2_id, bands=bands_sr, **base),
        "T2_BOA": SensorConfig(collection=t2_l2_id, bands=bands_sr, **base),
    }
    
    # Add RT variants if available
    if rt_id is not None:
        configs["RT"] = SensorConfig(collection=rt_id, bands=bands_base, **base)
    if rt_toa_id is not None:
        configs["RT_TOA"] = SensorConfig(collection=rt_toa_id, bands=bands_base, toa=True, **base)
    
    return configs

# --- PRE-DEFINED SENSOR VARIANTS ---

_m1 = _define_mss("LANDSAT/LM01/C02/T1", "LANDSAT/LM01/C02/T2", ["B4", "B5", "B6", "B7"], ("1972-07-23", "1978-01-06"))
_m2 = _define_mss("LANDSAT/LM02/C02/T1", "LANDSAT/LM02/C02/T2", ["B4", "B5", "B6", "B7"], ("1975-01-22", "1982-02-25"))
_m3 = _define_mss("LANDSAT/LM03/C02/T1", "LANDSAT/LM03/C02/T2", ["B4", "B5", "B6", "B7"], ("1978-03-05", "1983-03-31"))
_m4 = _define_mss("LANDSAT/LM04/C02/T1", "LANDSAT/LM04/C02/T2", ["B1", "B2", "B3", "B4"], ("1982-07-16", "1993-12-14"))
_m5 = _define_mss("LANDSAT/LM05/C02/T1", "LANDSAT/LM05/C02/T2", ["B1", "B2", "B3", "B4"], ("1984-03-01", "2013-01-05"))

_tm4 = _define_tm(
    "LANDSAT/LT04/C02/T1", "LANDSAT/LT04/C02/T2",
    "LANDSAT/LT04/C02/T1_TOA", "LANDSAT/LT04/C02/T2_TOA",
    "LANDSAT/LT04/C02/T1_L2", "LANDSAT/LT04/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
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

_etm = _define_etm(
    "LANDSAT/LE07/C02/T1", "LANDSAT/LE07/C02/T2",
    "LANDSAT/LE07/C02/T1_TOA", "LANDSAT/LE07/C02/T2_TOA",
    "LANDSAT/LE07/C02/T1_L2", "LANDSAT/LE07/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7", "B8"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    ("1999-05-28", "today")
)

_oli8 = _define_oli(
    "LANDSAT/LC08/C02/T1", "LANDSAT/LC08/C02/T2",
    "LANDSAT/LC08/C02/T1_TOA", "LANDSAT/LC08/C02/T2_TOA",
    "LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC08/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    ("2013-03-18", "today"),
    rt_id="LANDSAT/LC08/C02/T1_RT",
    rt_toa_id="LANDSAT/LC08/C02/T1_RT_TOA"
)

_oli9 = _define_oli(
    "LANDSAT/LC09/C02/T1", "LANDSAT/LC09/C02/T2",
    "LANDSAT/LC09/C02/T1_TOA", "LANDSAT/LC09/C02/T2_TOA",
    "LANDSAT/LC09/C02/T1_L2", "LANDSAT/LC09/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    ("2021-10-31", "today")
)

def _extract_collections(sensor_dict: dict, key: str) -> list[str]:
    """Extract collection IDs from sensor config."""
    coll = sensor_dict[key].collection
    return coll if isinstance(coll, list) else [coll]

# --- SENTINEL-2 BANDS ---

S2_TOA_BANDS: Final[list[str]] = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
]
S2_BOA_BANDS: Final[list[str]] = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
]

# Sentinel-2 common optional properties (available for all S2 products)
S2_COMMON_OPTIONAL: Final[set[str]] = {
    'CLOUDY_PIXEL_PERCENTAGE',
    'CLOUD_COVERAGE_ASSESSMENT',
    'MEAN_SOLAR_AZIMUTH_ANGLE',
    'MEAN_SOLAR_ZENITH_ANGLE',
    'SPACECRAFT_NAME',
    'MGRS_TILE',
    'SENSING_ORBIT_NUMBER',
    'SENSING_ORBIT_DIRECTION',
    'PRODUCT_ID',
    'GRANULE_ID',
    'DATASTRIP_ID',
    'PROCESSING_BASELINE',
}

# --- SENSORS DICTIONARY ---

SENSORS = {
    # Sentinel-2
    "S2": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
        pixel_scale=10, cloud_property="cs_cdf", cloud_range=(0.0, 1.0),
        default_dates=("2015-06-23", "today"), has_cloud_score_plus=True
    ),
    "S2_TOA": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=S2_TOA_BANDS,
        pixel_scale=10, cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0), default_dates=("2015-06-23", "today"),
        has_cloud_score_plus=False, toa=True
    ),
    "S2_BOA": SensorConfig(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=S2_BOA_BANDS,
        pixel_scale=10, cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0), default_dates=("2017-03-28", "today"),
        has_cloud_score_plus=False, toa=False
    ),

    # Landsat MSS
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

    # Landsat TM (4 & 5)
    "TM4": _tm4["DN"], "TM4_T1": _tm4["T1"], "TM4_T2": _tm4["T2"],
    "TM4_TOA": _tm4["TOA"], "TM4_T1_TOA": _tm4["T1_TOA"], "TM4_T2_TOA": _tm4["T2_TOA"],
    "TM4_BOA": _tm4["BOA"], "TM4_T1_BOA": _tm4["T1_BOA"], "TM4_T2_BOA": _tm4["T2_BOA"],
    "TM5": _tm5["DN"], "TM5_T1": _tm5["T1"], "TM5_T2": _tm5["T2"],
    "TM5_TOA": _tm5["TOA"], "TM5_T1_TOA": _tm5["T1_TOA"], "TM5_T2_TOA": _tm5["T2_TOA"],
    "TM5_BOA": _tm5["BOA"], "TM5_T1_BOA": _tm5["T1_BOA"], "TM5_T2_BOA": _tm5["T2_BOA"],

    # Landsat ETM+ (7)
    "ETM+": _etm["DN"], "ETM+_T1": _etm["T1"], "ETM+_T2": _etm["T2"],
    "ETM+_TOA": _etm["TOA"], "ETM+_T1_TOA": _etm["T1_TOA"], "ETM+_T2_TOA": _etm["T2_TOA"],
    "ETM+_BOA": _etm["BOA"], "ETM+_T1_BOA": _etm["T1_BOA"], "ETM+_T2_BOA": _etm["T2_BOA"],

    # Landsat OLI/TIRS (8 & 9)
    "OLI8": _oli8["DN"], "OLI8_T1": _oli8["T1"], "OLI8_T2": _oli8["T2"],
    "OLI8_TOA": _oli8["TOA"], "OLI8_T1_TOA": _oli8["T1_TOA"], "OLI8_T2_TOA": _oli8["T2_TOA"],
    "OLI8_BOA": _oli8["BOA"], "OLI8_T1_BOA": _oli8["T1_BOA"], "OLI8_T2_BOA": _oli8["T2_BOA"],
    "OLI8_RT": _oli8["RT"], "OLI8_RT_TOA": _oli8["RT_TOA"],
    "OLI9": _oli9["DN"], "OLI9_T1": _oli9["T1"], "OLI9_T2": _oli9["T2"],
    "OLI9_TOA": _oli9["TOA"], "OLI9_T1_TOA": _oli9["T1_TOA"], "OLI9_T2_TOA": _oli9["T2_TOA"],
    "OLI9_BOA": _oli9["BOA"], "OLI9_T1_BOA": _oli9["T1_BOA"], "OLI9_T2_BOA": _oli9["T2_BOA"],

    # Aggregated Landsat
    "LANDSAT": SensorConfig(
        collection=[
            *_extract_collections(_m1, "DN"), *_extract_collections(_m2, "DN"),
            *_extract_collections(_m3, "DN"), *_extract_collections(_m4, "DN"),
            *_extract_collections(_m5, "DN"), *_extract_collections(_tm4, "DN"),
            *_extract_collections(_tm5, "DN"), *_extract_collections(_etm, "DN"),
            *_extract_collections(_oli8, "DN"), *_extract_collections(_oli9, "DN"),
            *_extract_collections(_oli8, "RT"),
        ],
        bands=[], pixel_scale=30, cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0), default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False, toa=False
    ),
    "LANDSAT_TOA": SensorConfig(
        collection=[
            *_extract_collections(_m1, "TOA"), *_extract_collections(_m2, "TOA"),
            *_extract_collections(_m3, "TOA"), *_extract_collections(_m4, "TOA"),
            *_extract_collections(_m5, "TOA"), *_extract_collections(_tm4, "TOA"),
            *_extract_collections(_tm5, "TOA"), *_extract_collections(_etm, "TOA"),
            *_extract_collections(_oli8, "TOA"), *_extract_collections(_oli9, "TOA"),
            *_extract_collections(_oli8, "RT_TOA"),
        ],
        bands=[], pixel_scale=30, cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0), default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False, toa=True
    ),
    "LANDSAT_BOA": SensorConfig(
        collection=[
            *_extract_collections(_tm4, "BOA"), *_extract_collections(_tm5, "BOA"),
            *_extract_collections(_etm, "BOA"), *_extract_collections(_oli8, "BOA"),
            *_extract_collections(_oli9, "BOA"),
        ],
        bands=[], pixel_scale=30, cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0), default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False, toa=False
    ),
    
    # Aggregated Sentinel-2 (MULTISPECTRAL - uses native cloud property)
    "MULTISPECTRAL_TOA": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=[], pixel_scale=10, cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0), default_dates=("2015-06-23", "today"),
        has_cloud_score_plus=False, toa=True
    ),
    "MULTISPECTRAL_BOA": SensorConfig(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=[], pixel_scale=10, cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0), default_dates=("2017-03-28", "today"),
        has_cloud_score_plus=False, toa=False
    ),
}


# --- SENTINEL-2 SPECIFIC (Cloud Score Plus) ---

def _s2_cloud_table_single_range(
    lon: float, lat: float, edge_size: int | tuple[int, int],
    start: str, end: str, config: SensorConfig, scale: int,
    extra_properties: list[str] | None = None,
    include_sensor_column: bool = False
) -> pd.DataFrame:
    """Builds a cloud-score table for Sentinel-2 using Cloud Score Plus."""
    center = ee.Geometry.Point([lon, lat])
    roi = _square_roi(lon, lat, edge_size, scale)

    s2 = (
        ee.ImageCollection(config.collection)
        .filterBounds(roi)
        .filterDate(start, end)
    )

    cloud_collection = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
    ic = (
        s2.linkCollection(ee.ImageCollection(cloud_collection), [config.cloud_property])
        .select([config.cloud_property])
    )

    ids_inside = (
        ic.map(lambda img: img.set('roi_inside_scene', img.geometry().contains(roi, maxError=10)))
        .filter(ee.Filter.eq('roi_inside_scene', True))
        .aggregate_array('system:index')
        .getInfo()
    )

    try:
        raw = ic.getRegion(geometry=center, scale=scale * 1.1).getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands in collection" in str(e):
            return pd.DataFrame(columns=["id", "date", config.cloud_property, "inside", "tile"])
        raise e

    df_raw = (
        pd.DataFrame(raw[1:], columns=raw[0])
        .drop(columns=["longitude", "latitude", "time"], errors="ignore")
        .assign(date=lambda d: pd.to_datetime(d["id"].str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d"))
    )

    if isinstance(config.collection, str):
        df_raw['id'] = config.collection + "/" + df_raw['id']

    df_raw["inside"] = df_raw["id"].apply(lambda x: x.split("/")[-1]).isin(set(ids_inside)).astype(int)
    
    # Extract tile from id (format: .../20151022T002722_..._T55KEV -> 55KEV)
    df_raw['tile'] = df_raw['id'].apply(lambda x: x.split("_")[-1][1:] if x.split("_")[-1].startswith("T") else x.split("_")[-1])

    # Keep cs_cdf as column name (don't rename to cloud_cover)
    df_raw[config.cloud_property] = df_raw.groupby('date').apply(
        lambda g: g[config.cloud_property].transform(
            lambda _: g[g['inside'] == 1][config.cloud_property].iloc[0]
            if (g['inside'] == 1).any() else g[config.cloud_property].mean()
        )
    ).reset_index(drop=True)

    # Add sensor column for aggregated queries
    if include_sensor_column and 'id' in df_raw.columns:
        df_raw['sensor'] = df_raw['id'].apply(_parse_sensor_from_id)

    # Reorder columns - keep cs_cdf name
    if include_sensor_column:
        base_cols = ['id', 'date', 'sensor', config.cloud_property, 'inside', 'tile']
    else:
        base_cols = ['id', 'date', config.cloud_property, 'inside', 'tile']
    other_cols = [c for c in df_raw.columns if c not in base_cols]
    df_raw = df_raw[[c for c in base_cols + other_cols if c in df_raw.columns]]

    # Extract extra properties if requested
    if extra_properties and not df_raw.empty:
        extra_props = extra_properties
        
        def extract_extra(img):
            feat_props = {'system_index': img.get('system:index')}
            for prop in extra_props:
                feat_props[prop.lower()] = img.get(prop)
            return ee.Feature(None, feat_props)
        
        extra_fc = s2.map(extract_extra)
        extra_data = extra_fc.getInfo()
        
        if extra_data.get('features'):
            extra_records = [f['properties'] for f in extra_data['features']]
            df_extra = pd.DataFrame(extra_records)
            
            # Create merge key from id
            df_raw['_merge_key'] = df_raw['id'].apply(lambda x: x.split('/')[-1])
            df_extra['_merge_key'] = df_extra['system_index']
            
            # Merge extra columns
            extra_cols = [p.lower() for p in extra_props]
            df_raw = df_raw.merge(
                df_extra[['_merge_key'] + extra_cols],
                on='_merge_key',
                how='left'
            ).drop(columns=['_merge_key'])

    return df_raw


# --- GENERIC METADATA EXTRACTION ---

def _generic_metadata_table_single_range(
    lon: float, lat: float, edge_size: int | tuple[int, int],
    start: str, end: str, config: SensorConfig, scale: int,
    extra_properties: list[str] | None = None,
    include_sensor_column: bool = False
) -> pd.DataFrame:
    """Builds a metadata table for Landsat and Sentinel-2 sensors.

    Extracts base properties plus any requested extra_properties in a single
    server-side request for efficiency.
    """
    roi = _square_roi(lon, lat, edge_size, scale)
    collection = _get_ee_collection(config).filterBounds(roi).filterDate(start, end)

    # Detect if this is Sentinel-2
    coll_str = config.collection if isinstance(config.collection, str) else config.collection[0]
    is_sentinel2 = coll_str.startswith("COPERNICUS/S2")

    # Build property extraction dict based on sensor type
    if is_sentinel2:
        props_to_extract = {
            'id': 'system:id',
            'cloud_cover': config.cloud_property,
            'date': 'system:time_start',  # S2 uses system:time_start
            'tile': 'MGRS_TILE',
        }
    else:
        props_to_extract = {
            'id': 'system:id',
            'cloud_cover': config.cloud_property,
            'date': 'DATE_ACQUIRED',
            'path': 'WRS_PATH',
            'row': 'WRS_ROW',
        }

    # Add extra properties if requested
    extra_props = extra_properties or []
    for prop in extra_props:
        props_to_extract[prop.lower()] = prop

    def extract_props(img):
        inside = img.geometry().contains(roi, 10)
        feat_props = {'inside': inside}
        for out_key, ee_prop in props_to_extract.items():
            if ee_prop == 'system:time_start':
                feat_props[out_key] = ee.Date(img.get(ee_prop)).format('YYYY-MM-dd')
            else:
                feat_props[out_key] = img.get(ee_prop)
        return ee.Feature(None, feat_props)

    meta_fc = collection.map(extract_props)

    try:
        data = meta_fc.getInfo()
    except ee.ee_exception.EEException as e:
        if "No bands" in str(e):
            cols = ["id", "date"]
            if include_sensor_column:
                cols.append("sensor")
            cols += ["cloud_cover", "inside"]
            if is_sentinel2:
                cols.append("tile")
            else:
                cols += ["path", "row"]
            cols += [p.lower() for p in extra_props]
            return pd.DataFrame(columns=cols)
        raise e

    features = data.get('features', [])
    if not features:
        cols = ["id", "date"]
        if include_sensor_column:
            cols.append("sensor")
        cols += ["cloud_cover", "inside"]
        if is_sentinel2:
            cols.append("tile")
        else:
            cols += ["path", "row"]
        cols += [p.lower() for p in extra_props]
        return pd.DataFrame(columns=cols)

    records = [feat['properties'] for feat in features]
    df_raw = pd.DataFrame(records)

    # Ensure base columns exist based on sensor type
    if is_sentinel2:
        base_cols = ["id", "date", "cloud_cover", "inside", "tile"]
    else:
        base_cols = ["id", "date", "cloud_cover", "inside", "path", "row"]
    
    for col in base_cols:
        if col not in df_raw.columns:
            df_raw[col] = None

    # Add sensor column for aggregated queries
    if include_sensor_column and 'id' in df_raw.columns:
        df_raw['sensor'] = df_raw['id'].apply(_parse_sensor_from_id)
        
    # Check for missing extra properties and warn with collection context
    if extra_props and not df_raw.empty:
        for prop in extra_props:
            col_name = prop.lower()
            if col_name in df_raw.columns:
                null_count = df_raw[col_name].isna().sum()
                if null_count == len(df_raw):
                    warnings.warn(
                        f"Property '{prop}' returned all null values for collection "
                        f"'{coll_str}'. Check GEE documentation for valid properties.",
                        UserWarning
                    )
                elif null_count > 0:
                    warnings.warn(
                        f"Property '{prop}' has {null_count}/{len(df_raw)} null values. "
                        f"This property may not exist for all sensors in the collection.",
                        UserWarning
                    )

    # Format date and inside columns
    if 'date' in df_raw.columns:
        df_raw['date'] = pd.to_datetime(df_raw['date']).dt.strftime('%Y-%m-%d')
    if 'inside' in df_raw.columns:
        df_raw['inside'] = df_raw['inside'].fillna(0).astype(int)

    # Reorder columns: base first, sensor (if present), then extras
    if is_sentinel2:
        if include_sensor_column:
            final_base = ["id", "date", "sensor", "cloud_cover", "inside", "tile"]
        else:
            final_base = ["id", "date", "cloud_cover", "inside", "tile"]
    else:
        if include_sensor_column:
            final_base = ["id", "date", "sensor", "cloud_cover", "inside", "path", "row"]
        else:
            final_base = ["id", "date", "cloud_cover", "inside", "path", "row"]
    
    extra_cols = [p.lower() for p in extra_props if p.lower() in df_raw.columns]
    final_cols = final_base + extra_cols
    df_raw = df_raw[[c for c in final_cols if c in df_raw.columns]]
    
    return df_raw


# --- MAIN TABLE FUNCTION ---

def _sensor_table(
    sensor: str, lon: float, lat: float, edge_size: int | tuple[int, int],
    start: str | None = None, end: str | None = None,
    max_cloud: float | None = None, min_cloud: float | None = None,
    scale: int | None = None, bands: list[str] | None = None,
    extra_properties: list[str] | None = None, cache: bool = False,
    align_to_grid: bool | str = False
) -> pd.DataFrame:
    """Generic coordinator to build metadata tables for any sensor."""
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Available: {list(SENSORS.keys())}")

    config = SENSORS[sensor]
    
    # Determine if we should include sensor column (only for aggregated sensors)
    include_sensor_column = sensor in AGGREGATED_SENSORS

    # Handle defaults
    start = start or config.default_dates[0]
    if end is None:
        raw_end = config.default_dates[1]
        end = dt.date.today().strftime("%Y-%m-%d") if raw_end == "today" else raw_end
    max_cloud = max_cloud if max_cloud is not None else config.cloud_range[1]
    min_cloud = min_cloud if min_cloud is not None else config.cloud_range[0]
    
    # Determine effective scale - consider align_to_grid if scale not specified
    if scale is not None:
        effective_scale = scale
    elif align_to_grid is not False and align_to_grid in SENSORS:
        # Use native scale of reference sensor
        effective_scale = SENSORS[align_to_grid].pixel_scale
    elif align_to_grid is not False and isinstance(align_to_grid, str) and align_to_grid.startswith("LANDSAT/"):
        # Asset ID - detect scale from path
        is_mss = "/LM0" in align_to_grid
        effective_scale = 60 if is_mss else 30
    else:
        effective_scale = config.pixel_scale
        
    # Determine effective bands
    effective_bands = bands if bands is not None else config.bands

    cache_file = _cache_key(lon, lat, edge_size, effective_scale, str(config.collection))

    extract_fn = _s2_cloud_table_single_range if config.has_cloud_score_plus else _generic_metadata_table_single_range

    # Caching logic
    if cache and cache_file.exists():
        print(f"📂 Loading cached {sensor} metadata...", end='', flush=True)
        t0 = time.time()
        df_cached = pd.read_parquet(cache_file)
        have_idx = pd.to_datetime(df_cached["date"], errors="coerce").dropna()
        elapsed = time.time() - t0

        if have_idx.empty:
            df_cached = pd.DataFrame()
            cached_start = cached_end = None
        else:
            cached_start, cached_end = have_idx.min().date(), have_idx.max().date()

        if (cached_start and cached_end and
            dt.date.fromisoformat(start) >= cached_start and
            dt.date.fromisoformat(end) <= cached_end):
            df_full = df_cached
        else:
            print(f"\r📂 Cache loaded ({len(df_cached)} imgs)... checking missing ranges", end='', flush=True)
            df_new_parts = []

            if cached_start is None:
                df_new_parts.append(extract_fn(lon, lat, edge_size, start, end, config, effective_scale, extra_properties, include_sensor_column))
            else:
                if dt.date.fromisoformat(start) < cached_start:
                    df_new_parts.append(extract_fn(lon, lat, edge_size, start, cached_start.isoformat(), config, effective_scale, extra_properties, include_sensor_column))
                if dt.date.fromisoformat(end) > cached_end:
                    df_new_parts.append(extract_fn(lon, lat, edge_size, cached_end.isoformat(), end, config, effective_scale, extra_properties, include_sensor_column))
            df_new_parts = [df for df in df_new_parts if not df.empty]
            if df_new_parts:
                df_new = pd.concat(df_new_parts, ignore_index=True)
                df_full = pd.concat([df_cached, df_new], ignore_index=True).sort_values("date")
            else:
                df_full = df_cached
    else:
        print(f"⏳ Querying {sensor} (Scale: {effective_scale}m)...", end='', flush=True)
        t0 = time.time()
        df_full = extract_fn(lon, lat, edge_size, start, end, config, effective_scale, extra_properties, include_sensor_column)
        elapsed = time.time() - t0

    if cache:
        df_full.to_parquet(cache_file, compression="zstd")

    # Apply filters - detect actual cloud column name in DataFrame
    if config.cloud_property in df_full.columns:
        cloud_col = config.cloud_property  # cs_cdf for S2 with Cloud Score Plus
    else:
        cloud_col = 'cloud_cover'  # generic metadata uses cloud_cover
    
    result = (
        df_full.query("@start <= date <= @end")
        .query(f"@min_cloud <= {cloud_col} <= @max_cloud")
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(f"\r✅ Retrieved {len(result)} images ({elapsed:.2f}s)")

    # Handle grid alignment
    grid_reference = None
    grid_offset = None
    if align_to_grid is not False and not result.empty:
        print(f"🔧 Calculating grid alignment...", end='', flush=True)
        t0_align = time.time()
        
        if align_to_grid is True:
            # Auto mode: determine reference based on sensor and scale
            if sensor in AGGREGATED_SENSORS:
                if sensor.startswith("MULTISPECTRAL"):
                    # For MULTISPECTRAL, use S2_TOA as reference
                    ref_sensor = "S2_TOA"
                elif effective_scale <= 30:
                    # For aggregated LANDSAT at 30m or less, use TM5
                    ref_sensor = "TM5"
                else:
                    # For aggregated LANDSAT at >30m, use MSS5
                    ref_sensor = "MSS5"
                grid_reference = _get_grid_reference(ref_sensor, lon, lat, effective_scale)
            else:
                # For individual sensors, use first image from result
                # For Sentinel-2: prefer images from 2016+ due to early georeferencing issues
                coll_str = config.collection if isinstance(config.collection, str) else config.collection[0]
                if coll_str.startswith("COPERNICUS/S2"):
                    # Filter to 2016+ if possible
                    s2_2016_plus = result[result['date'] >= '2016-01-01']
                    if not s2_2016_plus.empty:
                        ref_asset = s2_2016_plus.iloc[0]['id']
                    elif len(result) > 1:
                        # Only 2015 images: use second image (index 1)
                        ref_asset = result.iloc[1]['id']
                    else:
                        # Single image: use it
                        ref_asset = result.iloc[0]['id']
                else:
                    ref_asset = result.iloc[0]['id']
                
                grid_reference = _get_grid_reference(ref_asset, lon, lat, effective_scale)
        
        elif isinstance(align_to_grid, str) and align_to_grid.startswith("LANDSAT/"):
            # Direct asset ID
            grid_reference = _get_grid_reference(align_to_grid, lon, lat, effective_scale)
        
        elif align_to_grid in SENSORS:
            # Sensor key as reference
            grid_reference = _get_grid_reference(align_to_grid, lon, lat, effective_scale)
        
        else:
            raise ValueError(
                f"Invalid align_to_grid value: '{align_to_grid}'. "
                f"Use True, False, a sensor key (e.g., 'TM5'), or an asset ID."
            )
        
        elapsed_align = time.time() - t0_align

        # Calculate offset from original center
        from cubexpress.conversion import geo2utm, lonlat2rt_utm_or_ups, parse_edge_size
        try:
            cx, cy, _ = geo2utm(lon, lat)
        except:
            cx, cy, _ = lonlat2rt_utm_or_ups(lon, lat)
        w, h = parse_edge_size(edge_size)
        ul_orig_x, ul_orig_y = cx - w * effective_scale / 2, cy + h * effective_scale / 2
        ref_x, ref_y, _ = grid_reference
        ul_snap_x = ref_x + round((ul_orig_x - ref_x) / effective_scale) * effective_scale
        ul_snap_y = ref_y + round((ul_orig_y - ref_y) / effective_scale) * effective_scale
        grid_offset = (ul_snap_x - ul_orig_x, ul_snap_y - ul_orig_y)
        
        print(f"\r✅ Grid aligned: offset ({grid_offset[0]:+.1f}, {grid_offset[1]:+.1f})m ({elapsed_align:.2f}s)")

    result.attrs.update({
        "lon": lon, "lat": lat, "edge_size": edge_size, "scale": effective_scale,
        "bands": effective_bands, "collection": config.collection,
        "start": start, "end": end, "toa": config.toa,
        "grid_reference": grid_reference, "grid_offset": grid_offset  # (translate_x, translate_y, native_scale) or None
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
    bands: list[str] | None = None,
    extra_properties: list[str] | None = None,
    cache: bool = False,
    align_to_grid: bool | str = False
) -> pd.DataFrame:
    """Builds (and caches) a metadata table for any supported sensor.

    Args:
        sensor: Sensor identifier (e.g., "S2", "MSS1_TOA", "TM5_BOA", "LANDSAT").
        lon: Longitude of the center point.
        lat: Latitude of the center point.
        edge_size: Box size in pixels (relative to scale).
        start: Start date (YYYY-MM-DD). If None, uses sensor defaults.
        end: End date (YYYY-MM-DD). If None, uses sensor defaults.
        scale: Resolution in meters. If None, uses sensor native resolution.
        max_cloud: Maximum cloud cover threshold (0-100 for Landsat, 0-1 for S2).
        min_cloud: Minimum cloud cover threshold.
        bands: List of bands to include. If None, uses sensor defaults.
            **Ignored for aggregated sensors (LANDSAT, LANDSAT_TOA, LANDSAT_BOA).**
        extra_properties: List of additional Earth Engine properties to extract.
            Properties are returned as lowercase column names. You can request
            ANY valid Earth Engine property - not limited to the common list below.
            
            **Common Landsat properties (available for ALL sensors and levels):**
            - CLOUD_COVER_LAND: Cloud cover over land (%)
            - SUN_AZIMUTH: Solar azimuth angle (degrees)
            - SUN_ELEVATION: Solar elevation angle (degrees)
            - EARTH_SUN_DISTANCE: Earth-Sun distance (AU)
            - SPACECRAFT_ID: Satellite identifier
            - SENSOR_ID: Sensor identifier
            - IMAGE_QUALITY: Quality score (0-9)
            - COLLECTION_CATEGORY: Tier (T1/T2)
            - LANDSAT_PRODUCT_ID: Full product ID
            - LANDSAT_SCENE_ID: Scene ID
            - GEOMETRIC_RMSE_MODEL: Geometric accuracy (meters)
            - GROUND_CONTROL_POINTS_MODEL: Number of GCPs used
            
            Note: WRS_PATH and WRS_ROW are always included in base columns.
            
            **Sensor-specific properties** (only for individual sensors, not aggregated):
            For TM/ETM+/OLI: K1_CONSTANT_BAND_*, K2_CONSTANT_BAND_*,
            RADIANCE_ADD_BAND_*, RADIANCE_MULT_BAND_*, REFLECTANCE_ADD_BAND_*, etc.
            
            See GEE documentation for complete property lists per collection.
            
            **Note:** For aggregated sensors (LANDSAT, LANDSAT_TOA, LANDSAT_BOA),
            sensor-specific properties may return null for some images.
        cache: Enable local parquet caching for faster subsequent queries.
        align_to_grid: Controls pixel grid alignment for time series consistency.
            
            - False (default): No alignment, pixels centered on (lon, lat).
            - True: Auto-alignment based on sensor and scale:
                - For LANDSAT/LANDSAT_TOA/LANDSAT_BOA at scale<=30: uses TM5 grid
                - For LANDSAT/LANDSAT_TOA/LANDSAT_BOA at scale>30: uses MSS5 grid
                - For individual sensors: uses first image from results
            - Sensor key (e.g., "TM5", "MSS1", "OLI8"): Uses that sensor's grid.
            - Asset ID (e.g., "LANDSAT/LT05/C02/T1/..."): Uses that specific image's grid.
              Warning issued if asset's native scale differs from requested scale.
            
            Adds ~1-2s for the grid offset calculation (one getInfo call).

    Returns:
        pd.DataFrame: Metadata table with columns:
            - id: Full Earth Engine asset ID
            - date: Acquisition date (YYYY-MM-DD)
            - cloud_cover: Cloud coverage metric
            - inside: 1 if ROI fully inside scene, 0 otherwise
            - path: WRS path number
            - row: WRS row number
            - [extra_properties]: Any additional requested properties (lowercase)

    Examples:
        >>> # Basic Landsat query (no alignment)
        >>> df = sensor_table("TM5", lon=-0.09, lat=51.5, edge_size=256)

        >>> # With auto grid alignment for time series
        >>> df = sensor_table(
        ...     "LANDSAT", lon=-76.0, lat=40.0, edge_size=256, scale=30,
        ...     align_to_grid=True  # Uses TM5 grid automatically
        ... )

        >>> # Align to specific sensor grid
        >>> df = sensor_table(
        ...     "LANDSAT", lon=-76.0, lat=40.0, edge_size=128, scale=60,
        ...     align_to_grid="MSS5"  # Explicit MSS5 reference
        ... )

        >>> # Align to specific asset
        >>> df = sensor_table(
        ...     "MSS1", lon=147.7, lat=-18.3, edge_size=86, scale=60,
        ...     align_to_grid="LANDSAT/LM01/C02/T2/LM01_100073_19731123"
        ... )
    """
    # Validate bands parameter for aggregated sensors
    if sensor in AGGREGATED_SENSORS and bands is not None:
        warnings.warn(
            f"Parameter 'bands' is ignored for aggregated sensor '{sensor}'. "
            f"All available bands will be downloaded.",
            UserWarning
        )
        bands = None

    # Warn about non-common properties on aggregated sensors
    if sensor in AGGREGATED_SENSORS and extra_properties:
        non_common = set(extra_properties) - LANDSAT_COMMON_OPTIONAL
        if non_common:
            warnings.warn(
                f"Properties {non_common} are not in LANDSAT_COMMON_OPTIONAL. "
                f"For aggregated sensor '{sensor}', these may return null for "
                f"some images (e.g., MSS doesn't have thermal calibration constants). "
                f"Consider using a specific sensor like 'OLI8' or 'TM5' instead.",
                UserWarning
            )

    return _sensor_table(
        sensor=sensor, lon=lon, lat=lat, edge_size=edge_size,
        start=start, end=end, scale=scale, max_cloud=max_cloud,
        min_cloud=min_cloud, bands=bands, extra_properties=extra_properties,
        cache=cache, align_to_grid=align_to_grid
    )


def s2_table(
    lon: float, lat: float, edge_size: int | tuple[int, int],
    start: str | None = None, end: str | None = None,
    scale: int | None = None, max_cscore: float | None = None,
    min_cscore: float | None = None, cache: bool = False,
    align_to_grid: bool | str = False, extra_properties: list[str] | None = None
) -> pd.DataFrame:
    """Builds (and caches) a per-day cloud-table for Sentinel-2.

    Convenience wrapper for sensor_table(sensor="S2", ...).
    """
    return _sensor_table(
        sensor="S2", lon=lon, lat=lat, edge_size=edge_size,
        start=start, end=end, scale=scale, max_cloud=max_cscore,
        min_cloud=min_cscore, cache=cache, align_to_grid=align_to_grid,
        extra_properties=extra_properties
    )


def mss_table(
    lon: float, lat: float, edge_size: int | tuple[int, int],
    start: str | None = None, end: str | None = None,
    sensor: str = "MSS1", scale: int | None = None,
    max_cloud_cover: float | None = None, min_cloud_cover: float | None = None,
    cache: bool = False, align_to_grid: bool | str = False
) -> pd.DataFrame:
    """Builds (and caches) a per-day cloud-table for Landsat MSS.

    Convenience wrapper for sensor_table(sensor=..., ...).
    """
    return _sensor_table(
        sensor=sensor, lon=lon, lat=lat, edge_size=edge_size,
        start=start, end=end, scale=scale, max_cloud=max_cloud_cover,
        min_cloud=min_cloud_cover, cache=cache, align_to_grid=align_to_grid
    )