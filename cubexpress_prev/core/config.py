"""Configuration constants for cubexpress."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Final

# Earth Engine Collections
S2_COLLECTION: Final[str] = "COPERNICUS/S2_HARMONIZED"
S2_CLOUD_COLLECTION: Final[str] = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S2_BANDS: Final[list[str]] = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_PIXEL_SCALE: Final[int] = 10  # meters

# Geospatial Constants
METERS_PER_DEGREE_LON: Final[float] = 111320.0  # at equator
METERS_PER_DEGREE_LAT: Final[float] = 110540.0  # constant globally

# Earth Engine scale buffer for getRegion() API calls
# EE requires ~10% extra buffer due to imprecise scale calculation
EE_SCALE_BUFFER_FACTOR: Final[float] = 1.1

# Cache directory (configurable via environment variable)
CACHE_DIR: Final[pathlib.Path] = pathlib.Path(os.getenv("CUBEXPRESS_CACHE", "~/.cubexpress_cache")).expanduser()


@dataclass(frozen=True)
class CubExpressConfig:
    """Runtime configuration for cubexpress operations."""

    # Threading
    default_workers: int = 4

    # Merging
    default_nodata: int = 0  # 65535
    gdal_threads: int = 8

    # Caching
    cache_enabled: bool = False
    coordinate_precision: int = 4  # decimal places for coordinate rounding


# Global config instance
CONFIG = CubExpressConfig()
