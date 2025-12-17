from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator

import ee

from cubexpress.geospatial import merge_tifs


@contextmanager
def temp_workspace(prefix: str = "cubexpress_") -> Iterator[pathlib.Path]:
    """
    Create a temporary directory with automatic cleanup.
    
    Args:
        prefix: Prefix for the temporary directory name
        
    Yields:
        Path to temporary directory
    """
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp_dir
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def download_manifest(
    ulist: dict[str, Any], 
    full_outname: pathlib.Path
) -> None:
    """
    Download data from Earth Engine based on a manifest dictionary.

    Handles both direct asset IDs and serialized EE expressions.

    Args:
        ulist: Export manifest containing 'assetId' or 'expression'
        full_outname: Destination path for the downloaded file

    Raises:
        ValueError: If manifest is invalid
        ee.ee_exception.EEException: If Earth Engine request fails
    """
    if "assetId" in ulist:
        images_bytes = ee.data.getPixels(ulist)
    elif "expression" in ulist:
        ee_image = ee.deserializer.decode(json.loads(ulist["expression"]))
        ulist_deep = deepcopy(ulist)
        ulist_deep["expression"] = ee_image
        images_bytes = ee.data.computePixels(ulist_deep)
    else:
        raise ValueError("Manifest must contain 'assetId' or 'expression'")
    
    full_outname.parent.mkdir(parents=True, exist_ok=True)
    with open(full_outname, "wb") as f:
        f.write(images_bytes)


def download_manifests(
    manifests: list[dict[str, Any]],
    full_outname: pathlib.Path,
    max_workers: int = 1,
) -> None:
    """
    Download multiple manifests concurrently and merge into one file.

    Uses a temporary workspace that is automatically cleaned up.

    Args:
        manifests: List of Earth Engine manifests
        full_outname: Final destination path for merged TIFF
        max_workers: Number of parallel download threads

    Raises:
        ee.ee_exception.EEException: If any download fails
        ValueError: If merge fails
    """
    with temp_workspace() as tmp_dir:
        tile_dir = tmp_dir / full_outname.stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Download tiles in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    download_manifest, 
                    ulist=manifest, 
                    full_outname=tile_dir / f"{idx:06d}.tif"
                ): idx 
                for idx, manifest in enumerate(manifests)
            }
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    idx = futures[future]
                    print(f"Error downloading tile {idx}: {exc}")
                    raise
        
        # Merge tiles
        input_files = sorted(tile_dir.glob("*.tif"))
        if not input_files:
            raise ValueError(f"No tiles downloaded in {tile_dir}")
        
        merge_tifs(input_files, full_outname)