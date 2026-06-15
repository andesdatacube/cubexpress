"""band_names: write band descriptions (e.g. B4/B3/B2) into a GeoTIFF header.

EE downloads and the tile merge produce GeoTIFFs whose bands are in the right
ORDER but unnamed (descriptions=None). This writes the band names into the file
header so downstream tools (and users opening the file) see B4/B3/B2 instead of
"Band 1/2/3". It only touches metadata — pixels are untouched, cost is trivial.
"""

from __future__ import annotations

import pathlib
from typing import Sequence, Union


def set_band_descriptions(
    tif_path: Union[str, pathlib.Path],
    names: Sequence[str],
) -> pathlib.Path:
    """Write band names into a GeoTIFF's header (metadata only, no pixel I/O).

    Args:
        tif_path: the GeoTIFF to annotate (modified in place).
        names: band names in order; length must match the file's band count.

    Returns:
        Path to the annotated GeoTIFF.

    Raises:
        ValueError: if len(names) != the file's band count.
    """
    import rasterio

    tif_path = pathlib.Path(tif_path)
    with rasterio.open(tif_path, "r+") as src:
        if len(names) != src.count:
            raise ValueError(
                f"got {len(names)} names for a {src.count}-band file: {list(names)}"
            )
        for i, name in enumerate(names, start=1):    # rasterio bands are 1-indexed
            src.set_band_description(i, name)
    return tif_path