"""Earth Engine metadata extraction and caching."""

from __future__ import annotations

from cubexpress.metadata.cache import clear_cache, get_cache_size
from cubexpress.metadata.scene import clear_scene_cache, get_batch_scene_info, get_scene_info
from cubexpress.metadata.sensors import (
    AGGREGATED_SENSORS,
    LANDSAT_COMMON_OPTIONAL,
    S2_BOA_BANDS,
    S2_COMMON_OPTIONAL,
    S2_TOA_BANDS,
    SENSORS,
    SensorConfig,
)
from cubexpress.metadata.tables import mss_table, s2_table, sensor_table

__all__ = [
    # Tables
    "sensor_table",
    "s2_table",
    "mss_table",
    # Sensors
    "SENSORS",
    "SensorConfig",
    "AGGREGATED_SENSORS",
    "LANDSAT_COMMON_OPTIONAL",
    "S2_COMMON_OPTIONAL",
    "S2_TOA_BANDS",
    "S2_BOA_BANDS",
    # Cache
    "clear_cache",
    "get_cache_size",
    # Scene geometry
    "get_scene_info",
    "get_batch_scene_info",
    "clear_scene_cache",
]
