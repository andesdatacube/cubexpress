from __future__ import annotations

import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, Any, Tuple
import ee
from tqdm import tqdm


from cubexpress.downloader import download_manifest, download_manifests
from cubexpress.geospatial import quadsplit_manifest, calculate_cell_size
import pandas as pd
from cubexpress.geotyping import RequestSet


def _test_manifest_tiling(manifest: Dict[str, Any]) -> int:
    """
    Tests if a manifest requires tiling WITHOUT downloading data.
    
    Makes a request to Earth Engine to check if the pixel limit error occurs.
    Does not write any data to disk - only captures the error to determine
    tiling requirements.
    
    Args:
        manifest: The Earth Engine download manifest to test
        
    Returns:
        int: Number of tiles required (1 if no tiling needed)
    """
    try:
        # Attempt the request to trigger potential pixel limit error
        if "assetId" in manifest:
            # Just make the request to see if it fails - don't store result
            _ = ee.data.getPixels(manifest)
        elif "expression" in manifest:
            # Decode and make request
            ee_image = ee.deserializer.decode(json.loads(manifest["expression"]))
            manifest_copy = deepcopy(manifest)
            manifest_copy["expression"] = ee_image
            _ = ee.data.computePixels(manifest_copy)
        
        # No error - no tiling needed
        return 1
        
    except ee.ee_exception.EEException as err:
        # Pixel limit error - calculate required tiling
        size = manifest["grid"]["dimensions"]["width"]
        cell_w, cell_h, power = calculate_cell_size(str(err), size)
        
        # Calculate number of tiles that would be created
        n_tiles = (2 ** power) ** 2
        return n_tiles
        
def get_geotiff(
    manifest: Dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int,
    return_tile_info: bool = False,
) -> int | None:
    """
    Downloads a single GeoTIFF, auto-tiling if pixel limits are exceeded.

    Attempts a direct download first. If Earth Engine raises a pixel count
    error, it calculates the necessary split depth, tiles the request,
    downloads tiles in parallel, and merges them.

    Args:
        manifest (Dict[str, Any]): The Earth Engine download manifest.
        full_outname (pathlib.Path | str): Path where the final GeoTIFF 
            will be saved.
        nworks (int): Number of worker threads to use if tiling is required.
        return_tile_info (bool): If True, returns the number of tiles created.

    Returns:
        int | None: Number of tiles if return_tile_info=True and tiling occurred,
            otherwise None.
    """
    try:
        download_manifest(
            ulist=manifest, 
            full_outname=full_outname
        )
        return 1 if return_tile_info else None
    except ee.ee_exception.EEException as err:
        # Check dimensions to determine split strategy
        size = manifest["grid"]["dimensions"]["width"]
        cell_w, cell_h, power = calculate_cell_size(str(err), size)
        
        # Generate sub-manifests (tiles)
        tiled = quadsplit_manifest(manifest, cell_w, cell_h, power)
        n_tiles = len(tiled)
        
        # Download tiles concurrently and merge
        download_manifests(
            manifests=tiled,
            full_outname=full_outname,
            max_workers=nworks
        )
        
        return n_tiles if return_tile_info else None
    

def _detect_optimal_workers(
    first_manifest: Dict[str, Any],
    total_workers: int
) -> Tuple[int, int]:
    """
    Detects optimal worker configuration by testing first image.
    
    Tests the first manifest to see if it requires tiling WITHOUT downloading
    any data. Then calculates the optimal distribution of workers between 
    outer (parallel images) and inner (tiles within an image).
    
    Args:
        first_manifest: Manifest of the first image to test
        total_workers: Total number of workers to distribute
        
    Returns:
        Tuple[int, int]: (outer_workers, inner_workers)
    """
    # Test manifest WITHOUT downloading - just check for tiling
    n_tiles = _test_manifest_tiling(first_manifest)
    
    if n_tiles == 1:
        # No tiling - use all workers for outer
        outer, inner = total_workers, 1
    else:
        # Tiling detected - distribute workers intelligently
        # Prioritize inner workers up to n_tiles, remaining for outer
        inner = min(n_tiles, max(1, total_workers // 2))
        outer = max(1, total_workers // inner)
    
    return outer, inner


def get_cube(
    requests: pd.DataFrame | RequestSet,
    outfolder: pathlib.Path | str,
    nworks: int | tuple[int, int] = 4,
    auto_workers: bool = True
) -> None:
    """
    Downloads a set of Earth Engine requests in parallel.

    Orchestrates the download of multiple images. Supports nested parallelism
    configuration for handling both the queue of images and the potential
    tiling of large individual images.

    Args:
        requests (pd.DataFrame | RequestSet): Collection of requests. Must
            expose ``manifest`` and ``id`` attributes/columns.
        outfolder (pathlib.Path | str): Destination directory.
        nworks (int | Tuple[int, int], optional): Concurrency configuration.
            - If int: Total number of workers to use. Distribution depends on auto_workers.
            - If tuple (outer, inner): Manual configuration (auto_workers ignored).
            Defaults to 4.
        auto_workers (bool, optional): If True and nworks is int, automatically 
            tests first image to detect if tiling is needed and distributes workers
            optimally. If no tiling: uses all workers for outer (classic behavior).
            If tiling detected: distributes between outer and inner intelligently.
            If False: uses all workers for outer, inner=1. Defaults to True.

    Raises:
        ValueError: If nworks tuple is invalid or contains non-positive numbers.
        TypeError: If nworks is not an int or tuple of ints.
    """
    
    # Setup output directory
    outfolder = pathlib.Path(outfolder).expanduser().resolve()
    outfolder.mkdir(parents=True, exist_ok=True)
    
    # Normalize input to DataFrame
    dataframe = requests._dataframe if isinstance(requests, RequestSet) else requests
    
    # Determine worker configuration
    if isinstance(nworks, int):
        if nworks <= 0:
            raise ValueError(f"nworks must be positive, got {nworks}")
            
        if auto_workers:
            # Auto-detect optimal distribution by testing first image (no download)
            first_row = dataframe.iloc[0]
            
            nworks_outer, nworks_inner = _detect_optimal_workers(
                first_manifest=first_row.manifest,
                total_workers=nworks
            )
        else:
            # Classic behavior: all workers for outer
            nworks_outer = nworks
            nworks_inner = 1
            
    elif isinstance(nworks, (list, tuple)):
        # Manual configuration - ignore auto_workers
        if len(nworks) != 2:
            raise ValueError(
                f"nworks must have exactly 2 elements (outer, inner), got {len(nworks)}"
            )
        nworks_outer, nworks_inner = nworks
        
        # Validate positive integers
        if not (isinstance(nworks_outer, int) and isinstance(nworks_inner, int)):
            raise TypeError(
                f"nworks elements must be integers, got ({type(nworks_outer)}, {type(nworks_inner)})"
            )
        if nworks_outer <= 0 or nworks_inner <= 0:
            raise ValueError(
                f"nworks values must be positive, got ({nworks_outer}, {nworks_inner})"
            )
    else:
        raise TypeError(
            f"nworks must be int or tuple[int, int], got {type(nworks)}"
        )
    
    # Execute downloads with progress tracking
    with ThreadPoolExecutor(max_workers=nworks_outer) as executor:
        futures = {
            executor.submit(
                get_geotiff,
                manifest=row.manifest,
                full_outname=pathlib.Path(outfolder) / f"{row.id}.tif",
                nworks=nworks_inner,
                return_tile_info=False
            ): row.id for _, row in dataframe.iterrows()
        }

        # Show progress bar
        for future in tqdm(
            as_completed(futures), 
            total=len(futures),
            desc=f"Downloading images (outer={nworks_outer}, inner={nworks_inner})",
            unit="image",
            leave=True
        ):
            try:
                future.result()
            except Exception as exc:
                print(f"Download error for {futures[future]}: {exc}")