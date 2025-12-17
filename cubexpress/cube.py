from __future__ import annotations

import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

import ee
import pandas as pd
from tqdm import tqdm

from cubexpress.downloader import download_manifest, download_manifests
from cubexpress.geospatial import calculate_cell_size, quadsplit_manifest
from cubexpress.geotyping import RequestSet
from cubexpress.logging_config import setup_logger

logger = setup_logger(__name__)


def _test_manifest_tiling(manifest: dict[str, Any]) -> int:
    """
    Test if a manifest requires tiling without downloading data.
    
    Args:
        manifest: Earth Engine download manifest
        
    Returns:
        Number of tiles required (1 if no tiling needed)
    """
    try:
        if "assetId" in manifest:
            _ = ee.data.getPixels(manifest)
        elif "expression" in manifest:
            ee_image = ee.deserializer.decode(json.loads(manifest["expression"]))
            manifest_copy = deepcopy(manifest)
            manifest_copy["expression"] = ee_image
            _ = ee.data.computePixels(manifest_copy)
        
        return 1
        
    except ee.ee_exception.EEException as err:
        size = manifest["grid"]["dimensions"]["width"]
        cell_w, cell_h, power = calculate_cell_size(str(err), size)
        n_tiles = (2 ** power) ** 2
        return n_tiles


def get_geotiff(
    manifest: dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int,
    return_tile_info: bool = False,
) -> int | None:
    """
    Download a single GeoTIFF with automatic tiling if needed.

    Args:
        manifest: Earth Engine download manifest
        full_outname: Output path for final GeoTIFF
        nworks: Number of worker threads for tiling
        return_tile_info: If True, return number of tiles created

    Returns:
        Number of tiles if return_tile_info=True, otherwise None
    """
    try:
        download_manifest(ulist=manifest, full_outname=full_outname)
        return 1 if return_tile_info else None
        
    except ee.ee_exception.EEException as err:
        size = manifest["grid"]["dimensions"]["width"]
        cell_w, cell_h, power = calculate_cell_size(str(err), size)
        
        tiled = quadsplit_manifest(manifest, cell_w, cell_h, power)
        n_tiles = len(tiled)
        
        # Silent tiling - no log spam
        download_manifests(
            manifests=tiled,
            full_outname=full_outname,
            max_workers=nworks
        )
        
        return n_tiles if return_tile_info else None


def _detect_optimal_workers(
    first_manifest: dict[str, Any],
    total_workers: int
) -> tuple[int, int]:
    """
    Detect optimal worker distribution by testing first image.
    
    Args:
        first_manifest: Manifest of first image
        total_workers: Total workers to distribute
        
    Returns:
        Tuple of (outer_workers, inner_workers)
    """
    n_tiles = _test_manifest_tiling(first_manifest)
    
    if n_tiles == 1:
        outer, inner = total_workers, 1
        logger.debug(f"No tiling needed - using {outer} parallel images")
    else:
        inner = min(n_tiles, max(1, total_workers // 2))
        outer = max(1, total_workers // inner)
        logger.info(
            f"Auto-detected tiling required ({n_tiles} tiles/image) - "
            f"using outer={outer}, inner={inner}"
        )
    
    return outer, inner


def get_cube(
    requests: pd.DataFrame | RequestSet,
    outfolder: pathlib.Path | str,
    nworks: int | tuple[int, int] = 4,
    auto_workers: bool = True
) -> None:
    """
    Download a set of Earth Engine requests in parallel.

    Args:
        requests: Collection of requests (DataFrame or RequestSet)
        outfolder: Destination directory
        nworks: Worker configuration (int or tuple of (outer, inner))
        auto_workers: If True, automatically detect optimal distribution

    Raises:
        ValueError: If nworks configuration is invalid
        TypeError: If nworks has wrong type
    """
    outfolder = pathlib.Path(outfolder).expanduser().resolve()
    outfolder.mkdir(parents=True, exist_ok=True)
    
    dataframe = (
        requests._dataframe if isinstance(requests, RequestSet) 
        else requests
    )
    
    if dataframe.empty:
        logger.warning("Request set is empty")
        return
    
    # Determine worker configuration
    if isinstance(nworks, int):
        if nworks <= 0:
            raise ValueError(f"nworks must be positive, got {nworks}")
        
        if auto_workers:
            first_row = dataframe.iloc[0]
            nworks_outer, nworks_inner = _detect_optimal_workers(
                first_manifest=first_row.manifest,
                total_workers=nworks
            )
        else:
            nworks_outer, nworks_inner = nworks, 1
            
    elif isinstance(nworks, (list, tuple)):
        if len(nworks) != 2:
            raise ValueError(f"nworks tuple must have 2 elements, got {len(nworks)}")
        
        nworks_outer, nworks_inner = nworks
        
        if not all(isinstance(n, int) for n in (nworks_outer, nworks_inner)):
            raise TypeError(f"nworks elements must be integers")
        
        if nworks_outer <= 0 or nworks_inner <= 0:
            raise ValueError(f"nworks values must be positive")
    else:
        raise TypeError(f"nworks must be int or tuple, got {type(nworks)}")
    
    # Execute downloads
    failed = []
    with ThreadPoolExecutor(max_workers=nworks_outer) as executor:
        futures = {
            executor.submit(
                get_geotiff,
                manifest=row.manifest,
                full_outname=outfolder / f"{row.id}.tif",
                nworks=nworks_inner,
                return_tile_info=False
            ): row.id 
            for _, row in dataframe.iterrows()
        }

        for future in tqdm(
            as_completed(futures), 
            total=len(futures),
            desc=f"Downloading (outer={nworks_outer}, inner={nworks_inner})",
            unit="image",
            leave=True
        ):
            img_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Failed {img_id}: {exc}")
                failed.append(img_id)
    
    # Summary
    if failed:
        logger.warning(f"{len(failed)}/{len(dataframe)} downloads failed")
    else:
        logger.info(f"✓ Downloaded {len(dataframe)} images to {outfolder}")