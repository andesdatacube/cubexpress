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
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict

import ee
import rasterio as rio
from rasterio.io import MemoryFile
import logging
import os
import shutil
import tempfile
from cubexpress.geospatial import merge_tifs

os.environ['CPL_LOG_ERRORS'] = 'OFF'
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

def download_manifest(
    ulist: Dict[str, Any], 
    full_outname: pathlib.Path
) -> None:
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
    else:
        raise ValueError("Manifest does not contain 'assetId' or 'expression'")
    
    with open(full_outname, "wb") as src:
        src.write(images_bytes)

    # with MemoryFile(images_bytes) as memfile:
    #     with memfile.open() as src:
    #         profile = src.profile
    #         profile.update(
    #             driver="GTiff", 
    #             tiled=True,
    #             interleave="band",
    #             blockxsize=256,
    #             blockysize=256,
    #             compress="ZSTD",
    #             zstd_level=13,
    #             predictor=2,
    #             num_threads=20,
    #             nodata=65535,
    #             dtype="uint16",
    #             count=12,
    #             photometric="MINISBLACK"
    #         )

    #         with rio.open(full_outname, "w", **profile) as dst:
    #             dst.write(src.read())

def download_manifests(
    manifests: list[Dict[str, Any]],
    full_outname: pathlib.Path,
    max_workers: int,
) -> None:
    """Download every manifest in *manifests* concurrently.

    Each output file is saved in the folder
    ``full_outname.parent/full_outname.stem`` with names ``000000.tif``,
    ``000001.tif`` … according to the list order.
    """
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="cubexpress_"))
    full_outname_temp = tmp_dir / full_outname.stem
    full_outname_temp.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as exe: # -
        futures = {
            exe.submit(
                download_manifest, 
                ulist=umanifest, 
                full_outname=full_outname_temp / f"{index:06d}.tif" 
            ): umanifest for index, umanifest in enumerate(manifests)               
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Error in one of the downloads: {exc}")
        
    if full_outname_temp.exists():
        input_files = sorted(full_outname_temp.glob("*.tif"))
        merge_tifs(input_files, full_outname)
        shutil.rmtree(full_outname_temp) 
    else:
        raise ValueError(f"Error in {full_outname}")