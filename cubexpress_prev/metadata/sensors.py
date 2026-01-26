"""Sensor configurations and constants for Earth Engine collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import ee

# --- LANDSAT METADATA CONSTANTS ---

LANDSAT_BASE_PROPS: Final[list[str]] = ["CLOUD_COVER", "DATE_ACQUIRED", "WRS_PATH", "WRS_ROW"]

LANDSAT_COMMON_OPTIONAL: Final[set[str]] = {
    "CLOUD_COVER_LAND",
    "SUN_AZIMUTH",
    "SUN_ELEVATION",
    "EARTH_SUN_DISTANCE",
    "SPACECRAFT_ID",
    "SENSOR_ID",
    "IMAGE_QUALITY",
    "COLLECTION_CATEGORY",
    "LANDSAT_PRODUCT_ID",
    "LANDSAT_SCENE_ID",
    "GEOMETRIC_RMSE_MODEL",
    "GROUND_CONTROL_POINTS_MODEL",
}

AGGREGATED_SENSORS: Final[set[str]] = {
    "LANDSAT",
    "LANDSAT_TOA",
    "LANDSAT_BOA",
    "MULTISPECTRAL_TOA",
    "MULTISPECTRAL_BOA",
}

SENSOR_NATIVE_SCALE: Final[dict[str, int]] = {
    "MSS1": 60,
    "MSS2": 60,
    "MSS3": 60,
    "MSS4": 60,
    "MSS5": 60,
    "TM4": 30,
    "TM5": 30,
    "ETM+": 30,
    "OLI8": 30,
    "OLI9": 30,
    "S2": 10,
}

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

# --- SENTINEL-2 BANDS ---

S2_TOA_BANDS: Final[list[str]] = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_BOA_BANDS: Final[list[str]] = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

S2_COMMON_OPTIONAL: Final[set[str]] = {
    "CLOUDY_PIXEL_PERCENTAGE",
    "CLOUD_COVERAGE_ASSESSMENT",
    "MEAN_SOLAR_AZIMUTH_ANGLE",
    "MEAN_SOLAR_ZENITH_ANGLE",
    "SPACECRAFT_NAME",
    "MGRS_TILE",
    "SENSING_ORBIT_NUMBER",
    "SENSING_ORBIT_DIRECTION",
    "PRODUCT_ID",
    "GRANULE_ID",
    "DATASTRIP_ID",
    "PROCESSING_BASELINE",
}


# --- SENSOR CONFIGURATION ---


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


def _define_mss(t1_id: str, t2_id: str, bands: list[str], dates: tuple[str, str]) -> dict[str, SensorConfig]:
    """Generate MSS configuration variants (DN, T1, T2, TOA)."""
    base = {
        "bands": bands,
        "pixel_scale": 60,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False,
    }
    return {
        "DN": SensorConfig(collection=[t1_id, t2_id], **base),
        "T1": SensorConfig(collection=t1_id, **base),
        "T2": SensorConfig(collection=t2_id, **base),
        "TOA": SensorConfig(collection=[t1_id, t2_id], toa=True, **base),
        "T1_TOA": SensorConfig(collection=t1_id, toa=True, **base),
        "T2_TOA": SensorConfig(collection=t2_id, toa=True, **base),
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
    dates: tuple[str, str],
) -> dict[str, SensorConfig]:
    """Generate TM configuration variants (DN, T1, T2, TOA, BOA)."""
    base = {
        "pixel_scale": 30,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False,
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
    t1_id: str,
    t2_id: str,
    t1_toa_id: str,
    t2_toa_id: str,
    t1_l2_id: str,
    t2_l2_id: str,
    bands_base: list[str],
    bands_sr: list[str],
    dates: tuple[str, str],
) -> dict[str, SensorConfig]:
    """Generate ETM+ configuration variants (DN, T1, T2, TOA, BOA)."""
    base = {
        "pixel_scale": 30,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False,
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
    t1_id: str,
    t2_id: str,
    t1_toa_id: str,
    t2_toa_id: str,
    t1_l2_id: str,
    t2_l2_id: str,
    bands_base: list[str],
    bands_sr: list[str],
    dates: tuple[str, str],
    rt_id: str | None = None,
    rt_toa_id: str | None = None,
) -> dict[str, SensorConfig]:
    """Generate OLI/TIRS configuration variants (DN, T1, T2, TOA, BOA, RT)."""
    base = {
        "pixel_scale": 30,
        "cloud_property": "CLOUD_COVER",
        "cloud_range": (0.0, 100.0),
        "default_dates": dates,
        "has_cloud_score_plus": False,
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

    if rt_id is not None:
        configs["RT"] = SensorConfig(collection=rt_id, bands=bands_base, **base)
    if rt_toa_id is not None:
        configs["RT_TOA"] = SensorConfig(collection=rt_toa_id, bands=bands_base, toa=True, **base)

    return configs


def _extract_collections(sensor_dict: dict, key: str) -> list[str]:
    """Extract collection IDs from sensor config."""
    coll = sensor_dict[key].collection
    return coll if isinstance(coll, list) else [coll]


# --- PRE-DEFINED SENSOR VARIANTS ---

_m1 = _define_mss("LANDSAT/LM01/C02/T1", "LANDSAT/LM01/C02/T2", ["B4", "B5", "B6", "B7"], ("1972-07-23", "1978-01-06"))
_m2 = _define_mss("LANDSAT/LM02/C02/T1", "LANDSAT/LM02/C02/T2", ["B4", "B5", "B6", "B7"], ("1975-01-22", "1982-02-25"))
_m3 = _define_mss("LANDSAT/LM03/C02/T1", "LANDSAT/LM03/C02/T2", ["B4", "B5", "B6", "B7"], ("1978-03-05", "1983-03-31"))
_m4 = _define_mss("LANDSAT/LM04/C02/T1", "LANDSAT/LM04/C02/T2", ["B1", "B2", "B3", "B4"], ("1982-07-16", "1993-12-14"))
_m5 = _define_mss("LANDSAT/LM05/C02/T1", "LANDSAT/LM05/C02/T2", ["B1", "B2", "B3", "B4"], ("1984-03-01", "2013-01-05"))

_tm4 = _define_tm(
    "LANDSAT/LT04/C02/T1",
    "LANDSAT/LT04/C02/T2",
    "LANDSAT/LT04/C02/T1_TOA",
    "LANDSAT/LT04/C02/T2_TOA",
    "LANDSAT/LT04/C02/T1_L2",
    "LANDSAT/LT04/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    ("1982-08-22", "1993-12-14"),
)

_tm5 = _define_tm(
    "LANDSAT/LT05/C02/T1",
    "LANDSAT/LT05/C02/T2",
    "LANDSAT/LT05/C02/T1_TOA",
    "LANDSAT/LT05/C02/T2_TOA",
    "LANDSAT/LT05/C02/T1_L2",
    "LANDSAT/LT05/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    ("1984-03-16", "2012-05-05"),
)

_etm = _define_etm(
    "LANDSAT/LE07/C02/T1",
    "LANDSAT/LE07/C02/T2",
    "LANDSAT/LE07/C02/T1_TOA",
    "LANDSAT/LE07/C02/T2_TOA",
    "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT/LE07/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7", "B8"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    ("1999-05-28", "today"),
)

_oli8 = _define_oli(
    "LANDSAT/LC08/C02/T1",
    "LANDSAT/LC08/C02/T2",
    "LANDSAT/LC08/C02/T1_TOA",
    "LANDSAT/LC08/C02/T2_TOA",
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC08/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    ("2013-03-18", "today"),
    rt_id="LANDSAT/LC08/C02/T1_RT",
    rt_toa_id="LANDSAT/LC08/C02/T1_RT_TOA",
)

_oli9 = _define_oli(
    "LANDSAT/LC09/C02/T1",
    "LANDSAT/LC09/C02/T2",
    "LANDSAT/LC09/C02/T1_TOA",
    "LANDSAT/LC09/C02/T2_TOA",
    "LANDSAT/LC09/C02/T1_L2",
    "LANDSAT/LC09/C02/T2_L2",
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    ("2021-10-31", "today"),
)


# --- SENSORS DICTIONARY ---

SENSORS = {
    # Sentinel-2
    "S2": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
        pixel_scale=10,
        cloud_property="cs_cdf",
        cloud_range=(0.0, 1.0),
        default_dates=("2015-06-23", "today"),
        has_cloud_score_plus=True,
    ),
    "S2_TOA": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=S2_TOA_BANDS,
        pixel_scale=10,
        cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0),
        default_dates=("2015-06-23", "today"),
        has_cloud_score_plus=False,
        toa=True,
    ),
    "S2_BOA": SensorConfig(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=S2_BOA_BANDS,
        pixel_scale=10,
        cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0),
        default_dates=("2017-03-28", "today"),
        has_cloud_score_plus=False,
        toa=False,
    ),
    # Landsat MSS
    "MSS1": _m1["DN"],
    "MSS1_T1": _m1["T1"],
    "MSS1_T2": _m1["T2"],
    "MSS1_TOA": _m1["TOA"],
    "MSS1_T1_TOA": _m1["T1_TOA"],
    "MSS1_T2_TOA": _m1["T2_TOA"],
    "MSS2": _m2["DN"],
    "MSS2_T1": _m2["T1"],
    "MSS2_T2": _m2["T2"],
    "MSS2_TOA": _m2["TOA"],
    "MSS2_T1_TOA": _m2["T1_TOA"],
    "MSS2_T2_TOA": _m2["T2_TOA"],
    "MSS3": _m3["DN"],
    "MSS3_T1": _m3["T1"],
    "MSS3_T2": _m3["T2"],
    "MSS3_TOA": _m3["TOA"],
    "MSS3_T1_TOA": _m3["T1_TOA"],
    "MSS3_T2_TOA": _m3["T2_TOA"],
    "MSS4": _m4["DN"],
    "MSS4_T1": _m4["T1"],
    "MSS4_T2": _m4["T2"],
    "MSS4_TOA": _m4["TOA"],
    "MSS4_T1_TOA": _m4["T1_TOA"],
    "MSS4_T2_TOA": _m4["T2_TOA"],
    "MSS5": _m5["DN"],
    "MSS5_T1": _m5["T1"],
    "MSS5_T2": _m5["T2"],
    "MSS5_TOA": _m5["TOA"],
    "MSS5_T1_TOA": _m5["T1_TOA"],
    "MSS5_T2_TOA": _m5["T2_TOA"],
    # Landsat TM (4 & 5)
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
    # Landsat ETM+ (7)
    "ETM+": _etm["DN"],
    "ETM+_T1": _etm["T1"],
    "ETM+_T2": _etm["T2"],
    "ETM+_TOA": _etm["TOA"],
    "ETM+_T1_TOA": _etm["T1_TOA"],
    "ETM+_T2_TOA": _etm["T2_TOA"],
    "ETM+_BOA": _etm["BOA"],
    "ETM+_T1_BOA": _etm["T1_BOA"],
    "ETM+_T2_BOA": _etm["T2_BOA"],
    # Landsat OLI/TIRS (8 & 9)
    "OLI8": _oli8["DN"],
    "OLI8_T1": _oli8["T1"],
    "OLI8_T2": _oli8["T2"],
    "OLI8_TOA": _oli8["TOA"],
    "OLI8_T1_TOA": _oli8["T1_TOA"],
    "OLI8_T2_TOA": _oli8["T2_TOA"],
    "OLI8_BOA": _oli8["BOA"],
    "OLI8_T1_BOA": _oli8["T1_BOA"],
    "OLI8_T2_BOA": _oli8["T2_BOA"],
    "OLI8_RT": _oli8["RT"],
    "OLI8_RT_TOA": _oli8["RT_TOA"],
    "OLI9": _oli9["DN"],
    "OLI9_T1": _oli9["T1"],
    "OLI9_T2": _oli9["T2"],
    "OLI9_TOA": _oli9["TOA"],
    "OLI9_T1_TOA": _oli9["T1_TOA"],
    "OLI9_T2_TOA": _oli9["T2_TOA"],
    "OLI9_BOA": _oli9["BOA"],
    "OLI9_T1_BOA": _oli9["T1_BOA"],
    "OLI9_T2_BOA": _oli9["T2_BOA"],
    # Aggregated Landsat
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
            *_extract_collections(_oli8, "RT"),
        ],
        bands=[],
        pixel_scale=30,
        cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0),
        default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False,
        toa=False,
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
            *_extract_collections(_oli8, "RT_TOA"),
        ],
        bands=[],
        pixel_scale=30,
        cloud_property="CLOUD_COVER",
        cloud_range=(0.0, 100.0),
        default_dates=("1972-07-23", "today"),
        has_cloud_score_plus=False,
        toa=True,
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
        toa=False,
    ),
    # Aggregated Sentinel-2
    "MULTISPECTRAL_TOA": SensorConfig(
        collection="COPERNICUS/S2_HARMONIZED",
        bands=[],
        pixel_scale=10,
        cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0),
        default_dates=("2015-06-23", "today"),
        has_cloud_score_plus=False,
        toa=True,
    ),
    "MULTISPECTRAL_BOA": SensorConfig(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=[],
        pixel_scale=10,
        cloud_property="CLOUDY_PIXEL_PERCENTAGE",
        cloud_range=(0.0, 100.0),
        default_dates=("2017-03-28", "today"),
        has_cloud_score_plus=False,
        toa=False,
    ),
}
