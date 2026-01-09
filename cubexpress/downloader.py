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
    """Create a temporary directory with automatic cleanup."""
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
    """Download data from Earth Engine based on a manifest dictionary."""
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
    """Download multiple manifests concurrently and merge into one file."""
    with temp_workspace() as tmp_dir:
        tile_dir = tmp_dir / full_outname.stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        errors = []
        
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
                    errors.append(exc)
        
        if errors:
            raise errors[0]
        
        input_files = sorted(tile_dir.glob("*.tif"))
        if not input_files:
            raise ValueError(f"No tiles downloaded in {tile_dir}")
        
        merge_tifs(input_files, full_outname)