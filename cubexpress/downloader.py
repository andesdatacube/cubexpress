from __future__ import annotations

import json
import logging
import os
import ee
import pathlib
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict
from cubexpress.geospatial import merge_tifs

# Suppress RasterIO/GDAL logging
os.environ['CPL_LOG_ERRORS'] = 'OFF'
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

def download_manifest(
    ulist: Dict[str, Any], 
    full_outname: pathlib.Path
) -> None:
    """
    Downloads data from Earth Engine based on a manifest dictionary.

    Handles both direct asset IDs and serialized EE expressions. The resulting
    bytes are written directly to the specified output path.

    Args:
        ulist (Dict[str, Any]): The export manifest containing 'assetId' or 
            'expression' and format parameters.
        full_outname (pathlib.Path): Destination path for the downloaded file.

    Raises:
        ValueError: If the manifest lacks both 'assetId' and 'expression'.
    """
    if "assetId" in ulist:
        images_bytes = ee.data.getPixels(ulist)
    elif "expression" in ulist:
        # Decode serialized expression before request
        ee_image = ee.deserializer.decode(json.loads(ulist["expression"]))
        ulist_deep = deepcopy(ulist)
        ulist_deep["expression"] = ee_image
        images_bytes = ee.data.computePixels(ulist_deep)
    else:
        raise ValueError("Manifest does not contain 'assetId' or 'expression'")
    
    with open(full_outname, "wb") as src:
        src.write(images_bytes)
        
def download_manifests(
    manifests: list[Dict[str, Any]],
    full_outname: pathlib.Path,
    max_workers: int = 1,
) -> None:
    """
    Downloads multiple manifests concurrently and merges them into one file.

    Creates a temporary directory to store individual tiles, downloads them
    in parallel, merges them using `merge_tifs`, and cleans up temporary files.

    Args:
        manifests (list[Dict[str, Any]]): List of Earth Engine manifests.
        full_outname (pathlib.Path): Final destination path for the merged TIFF.
        max_workers (int, optional): Number of parallel download threads. 
            Defaults to 1.

    Raises:
        ValueError: If the temporary directory was not created or processed.
    """
    # Create temporary directory for tile storage
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="cubexpress_"))
    full_outname_temp = tmp_dir / full_outname.stem
    full_outname_temp.mkdir(parents=True, exist_ok=True)

    # Download tiles in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
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
    
    # Merge tiles and cleanup
    if full_outname_temp.exists():
        input_files = sorted(full_outname_temp.glob("*.tif"))
        merge_tifs(input_files, full_outname)
        shutil.rmtree(full_outname_temp) 
    else:
        raise ValueError(f"Error in {full_outname}")