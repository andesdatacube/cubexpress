"""Earth Engine data cube download with optimal parallelization."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm

from cubexpress.config import CONFIG
from cubexpress.downloader import download_manifest, temp_workspace
from cubexpress.geospatial import merge_tifs
from cubexpress.geotyping import RequestSet
from cubexpress.logging_config import setup_logger
from cubexpress.tiling import (
    calculate_tiling_from_error,
    generate_tile_manifests,
    get_manifest_group_key,
)

logger = setup_logger(__name__)


def _is_size_error(error: Exception) -> bool:
    """Check if error is a GEE size limit error."""
    msg = str(error).lower()
    return "must be less" in msg or "limit" in msg or "size" in msg


def get_geotiff(
    manifest: dict[str, Any],
    full_outname: pathlib.Path | str,
    nworks: int | None = None,
) -> int:
    """Download a single GeoTIFF with reactive tiling on error."""
    if nworks is None:
        nworks = CONFIG.default_workers
        
    full_outname = pathlib.Path(full_outname)
    
    try:
        download_manifest(ulist=manifest, full_outname=full_outname)
        return 1
    except Exception as e:
        if not _is_size_error(e):
            raise
        err_msg = str(e)
    
    width = manifest["grid"]["dimensions"]["width"]
    height = manifest["grid"]["dimensions"]["height"]
    strategy = calculate_tiling_from_error(err_msg, width, height)
    
    tiles = generate_tile_manifests(manifest, strategy)
    
    with temp_workspace() as tmp_dir:
        tile_dir = tmp_dir / full_outname.stem
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=nworks) as executor:
            futures = {
                executor.submit(
                    download_manifest,
                    ulist=tile,
                    full_outname=tile_dir / f"{idx:06d}.tif"
                ): idx
                for idx, tile in enumerate(tiles)
            }
            
            errors = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            
            if errors:
                raise errors[0]
        
        input_files = sorted(tile_dir.glob("*.tif"))
        merge_tifs(input_files, full_outname)
    
    return strategy.total_tiles


def get_cube(
    requests: pd.DataFrame | RequestSet,
    outfolder: pathlib.Path | str,
    nworks: int | None = None,
) -> None:
    """Download batch of Earth Engine requests with optimal parallelization."""
    if nworks is None:
        nworks = CONFIG.default_workers
        
    outfolder = pathlib.Path(outfolder).expanduser().resolve()
    outfolder.mkdir(parents=True, exist_ok=True)

    dataframe = (
        requests._dataframe if isinstance(requests, RequestSet)
        else requests
    )

    if dataframe.empty:
        logger.warning("Request set is empty")
        return

    n_images = len(dataframe)
    
    # Group by pattern (same bands + dimensions = same tiling strategy)
    groups = defaultdict(list)
    for idx, row in dataframe.iterrows():
        key = get_manifest_group_key(row.manifest)
        groups[key].append(row)
    
    failed = []
    
    with tqdm(total=n_images, desc="Downloading", unit="img") as pbar:
        for group_key, rows in groups.items():
            first_row = rows[0]
            first_manifest = first_row.manifest
            first_outpath = outfolder / f"{first_row.id}.tif"
            
            # Test first image of group
            try:
                download_manifest(ulist=first_manifest, full_outname=first_outpath)
                needs_tiling = False
                strategy = None
                pbar.update(1)
            except Exception as e:
                if not _is_size_error(e):
                    logger.error(f"Failed {first_row.id}: {e}")
                    failed.append(first_row.id)
                    pbar.update(1)
                    continue
                
                needs_tiling = True
                width = first_manifest["grid"]["dimensions"]["width"]
                height = first_manifest["grid"]["dimensions"]["height"]
                strategy = calculate_tiling_from_error(str(e), width, height)
                first_outpath.unlink(missing_ok=True)
            
            remaining_rows = rows[1:] if not needs_tiling else rows
            
            if not remaining_rows:
                continue
            
            if not needs_tiling:
                # Direct download for rest of group
                with ThreadPoolExecutor(max_workers=nworks) as executor:
                    futures = {
                        executor.submit(
                            download_manifest,
                            ulist=row.manifest,
                            full_outname=outfolder / f"{row.id}.tif"
                        ): row.id
                        for row in remaining_rows
                    }
                    
                    for future in as_completed(futures):
                        img_id = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            logger.error(f"Failed {img_id}: {exc}")
                            failed.append(img_id)
                        pbar.update(1)
            else:
                # Tiled download with global pool
                with temp_workspace() as tmp_dir:
                    tile_tasks = []
                    img_tile_map = defaultdict(list)
                    
                    for row in remaining_rows:
                        img_id = row.id
                        manifest = row.manifest
                        tiles = generate_tile_manifests(manifest, strategy)
                        
                        img_dir = tmp_dir / img_id
                        img_dir.mkdir(parents=True, exist_ok=True)
                        
                        for idx, tile in enumerate(tiles):
                            tile_path = img_dir / f"{idx:06d}.tif"
                            tile_tasks.append((tile, tile_path, img_id))
                            img_tile_map[img_id].append(tile_path)
                    
                    completed_tiles = defaultdict(int)
                    total_tiles_per_img = strategy.total_tiles
                    merged = set()
                    
                    with ThreadPoolExecutor(max_workers=nworks) as executor:
                        futures = {
                            executor.submit(download_manifest, ulist=tile, full_outname=path): (path, img_id)
                            for tile, path, img_id in tile_tasks
                        }
                        
                        for future in as_completed(futures):
                            path, img_id = futures[future]
                            
                            if img_id in failed or img_id in merged:
                                continue
                                
                            try:
                                future.result()
                                completed_tiles[img_id] += 1
                                
                                if completed_tiles[img_id] == total_tiles_per_img:
                                    tile_files = sorted(img_tile_map[img_id])
                                    merge_tifs(tile_files, outfolder / f"{img_id}.tif")
                                    merged.add(img_id)
                                    pbar.update(1)
                                    
                            except Exception as exc:
                                if img_id not in failed:
                                    logger.error(f"Failed {img_id}: {exc}")
                                    failed.append(img_id)
                                    pbar.update(1)

    if failed:
        logger.warning(f"{len(failed)}/{n_images} downloads failed")