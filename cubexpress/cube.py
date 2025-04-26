"""High-level helpers for tiled GeoTIFF downloads.

The module provides two thread-friendly wrappers:

* **get_geotiff** – download a single manifest, auto-tiling on EE pixel-count
  errors.
* **get_cube** – iterate over a ``RequestSet`` (or similar) and build a local
  raster “cube” in parallel.

The core download/split logic lives in *cubexpress.downloader* and
*cubexpress.geospatial*; here we merely orchestrate it.
"""

from __future__ import annotations

import pathlib
import concurrent.futures
from typing import Dict, Any

import ee
from cubexpress.downloader import download_manifest, download_manifests
from cubexpress.geospatial import quadsplit_manifest, calculate_cell_size
from cubexpress.geotyping import RequestSet


def get_geotiff(
    manifest: Dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int = 4,
) -> None:
    """Download *manifest* to *full_outname*, retrying with tiled requests.

    Parameters
    ----------
    manifest
        Earth Engine download manifest returned by cubexpress.
    full_outname
        Final ``.tif`` path (created/overwritten).
    nworks
        Maximum worker threads when the image must be split; default **4**.
    """
    try:
        download_manifest(manifest, pathlib.Path(full_outname))
    except ee.ee_exception.EEException as err:
        # Handle EE “too many pixels” error by recursive tiling.
        size = manifest["grid"]["dimensions"]["width"]  # square images assumed
        cell_w, cell_h, power = calculate_cell_size(str(err), size)
        tiled = quadsplit_manifest(manifest, cell_w, cell_h, power)
        download_manifests(tiled, max_workers=nworks, full_outname=pathlib.Path(full_outname))


def get_cube(
    requests: RequestSet,
    outfolder: pathlib.Path | str,
    nworks: int = 4,
) -> None:
    """Download every request in *requests* to *outfolder* using a thread pool.

    Each row in ``requests._dataframe`` must expose ``manifest`` and ``id``.
    Resulting files are named ``{id}.tif``.

    Parameters
    ----------
    requests
        A ``RequestSet`` or object with an internal ``_dataframe`` attribute.
    outfolder
        Folder where the GeoTIFFs will be written (created if absent).
    nworks
        Pool size for concurrent downloads; default **4**.
    """
    out = pathlib.Path(outfolder)

    with concurrent.futures.ThreadPoolExecutor(max_workers=nworks) as pool:
        futures = []
        for _, row in requests._dataframe.iterrows():
            outname = out / f"{row.id}.tif"
            outname.parent.mkdir(parents=True, exist_ok=True)
            futures.append(pool.submit(get_geotiff, row.manifest, outname, nworks))

        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 – log and keep going
                print(f"Download error: {exc}")
