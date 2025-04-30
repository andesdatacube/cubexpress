"""Low-level download helpers for Earth Engine manifests.

Only two public callables are exposed:

* :func:`download_manifest` – fetch a single manifest and write one GeoTIFF.
* :func:`download_manifests` – convenience wrapper to parallel-download a list
  of manifests with a thread pool.

Both functions are fully I/O bound; no return value is expected.
"""

from __future__ import annotations

import json
import pathlib
import concurrent.futures
from copy import deepcopy
from typing import Any, Dict, List

import ee
import rasterio as rio
from rasterio.io import MemoryFile
import logging
import os

os.environ['CPL_LOG_ERRORS'] = 'OFF'
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

def download_manifest(ulist: Dict[str, Any], full_outname: pathlib.Path) -> None:
    """Download *ulist* and save it as *full_outname*.

    The manifest must include either an ``assetId`` or an ``expression``
    (serialized EE image). RasterIO is used to write a tiled, compressed
    GeoTIFF; the function is silent apart from the final ``print``.
    """
    if "assetId" in ulist:
        images_bytes = ee.data.getPixels(ulist)
    elif "expression" in ulist:
        ee_image = ee.deserializer.decode(json.loads(ulist["expression"]))
        ulist_deep = deepcopy(ulist)
        ulist_deep["expression"] = ee_image
        images_bytes = ee.data.computePixels(ulist_deep)
    else:  # pragma: no cover
        raise ValueError("Manifest does not contain 'assetId' or 'expression'")

    with MemoryFile(images_bytes) as memfile:
        with memfile.open() as src:
            profile = src.profile
            profile.update(
                driver="GTiff", 
                tiled=True,
                interleave="band",
                blockxsize=256,
                blockysize=256,
                compress="ZSTD",
                zstd_level=13,
                predictor=2,
                num_threads=20,
                nodata=65535,
                dtype="uint16",
                count=13,
                photometric="MINISBLACK"
            )

            with rio.open(full_outname, "w", **profile) as dst:
                dst.write(src.read())

    print(f"{full_outname} downloaded successfully.")  # noqa: T201


def download_manifests(
    manifests: List[Dict[str, Any]],
    max_workers: int,
    full_outname: pathlib.Path,
) -> None:
    """Download every manifest in *manifests* concurrently.

    Each output file is saved in the folder
    ``full_outname.parent/full_outname.stem`` with names ``000000.tif``,
    ``000001.tif`` … according to the list order.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for index, umanifest in enumerate(manifests):
            folder = full_outname.parent / full_outname.stem
            folder.mkdir(parents=True, exist_ok=True)
            outname = folder / f"{index:06d}.tif"
            futures.append(executor.submit(download_manifest, umanifest, outname))

        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"Error en una de las descargas: {exc}")  # noqa: T201
