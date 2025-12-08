import hashlib
import json
import os
import pathlib
from typing import Final

# Directory for storing cached metadata files (configurable via env var)
_CACHE_DIR: Final[pathlib.Path] = pathlib.Path(
    os.getenv("CUBEXPRESS_CACHE", "~/.cubexpress_cache")
).expanduser()
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(
    lon: float,
    lat: float,
    edge_size: int | tuple[int, int],
    scale: int,
    collection: str,
) -> pathlib.Path:
    """
    Generates a deterministic file path for caching query results.

    Hashes the query parameters to create a unique filename. Coordinates
    are rounded to 4 decimals to ensure cache hits on equivalent locations.

    Args:
        lon (float): Longitude of the center point.
        lat (float): Latitude of the center point.
        edge_size (int | Tuple[int, int]): Size of the ROI in pixels.
        scale (int): Pixel resolution in meters.
        collection (str): Earth Engine collection ID.

    Returns:
        pathlib.Path: Full path to the hashed .parquet cache file.
    """
    # Round coordinates to ~11m precision to group nearby requests
    lon_r, lat_r = round(lon, 4), round(lat, 4)
    
    # Normalize edge_size to tuple for consistent hashing
    if isinstance(edge_size, int):
        edge_tuple = (edge_size, edge_size)
    else:
        edge_tuple = edge_size
    
    # Create a unique signature for this request configuration
    signature = [lon_r, lat_r, edge_tuple, scale, collection]
    
    # Use MD5 to generate a short, filesystem-friendly filename
    raw = json.dumps(signature).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    return _CACHE_DIR / f"{digest}.parquet"