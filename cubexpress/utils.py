import pathlib
import ee
import re
import rasterio as rio
from rasterio.io import MemoryFile
import concurrent.futures
from copy import deepcopy
from typing import Dict








def download_manifests(manifests: list[dict], max_workers: int, full_outname: pathlib.Path) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_list = []
        
        for index, umanifest in enumerate(manifests):
            folder = full_outname.parent / full_outname.stem
            folder.mkdir(parents=True, exist_ok=True)
            outname = folder / f"{index:06d}.tif"
            future = executor.submit(download_manifest, umanifest, outname)
            future_list.append(future)
        
        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()
            except Exception as e:
                print(f"Error en una de las descargas: {e}")

def quadsplit_manifest(manifest: Dict, cell_width: int, cell_height: int, power: int) -> list[Dict]:
    manifest_copy = deepcopy(manifest)
    
    manifest_copy["grid"]["dimensions"]["width"] = cell_width
    manifest_copy["grid"]["dimensions"]["height"] = cell_height
    x = manifest_copy["grid"]["affineTransform"]["translateX"]
    y = manifest_copy["grid"]["affineTransform"]["translateY"]
    scale_x = manifest_copy["grid"]["affineTransform"]["scaleX"]
    scale_y = manifest_copy["grid"]["affineTransform"]["scaleY"]

    manifests = []

    for columny in range(2**power):
        for rowx in range(2**power):
            new_x = x + (rowx * cell_width) * scale_x
            new_y = y - (columny * cell_height) * scale_y
            new_manifest = deepcopy(manifest_copy)
            new_manifest["grid"]["affineTransform"]["translateX"] = new_x
            new_manifest["grid"]["affineTransform"]["translateY"] = new_y
            manifests.append(new_manifest)

    return manifests

def calculate_cell_size(ee_error_message: str, size: int) -> tuple[int, int]:
    match = re.findall(r'\d+', ee_error_message)
    image_pixel = int(match[0])
    max_pixel = int(match[1])
    
    images = image_pixel / max_pixel
    power = 0
    
    while images > 1:
        power += 1
        images = image_pixel / (max_pixel * 4 ** power)
    
    cell_width = size // 2 ** power
    cell_height = size // 2 ** power
    
    return cell_width, cell_height, power


def download_manifest(ulist, full_outname):
    
    if "assetId" in ulist:
        images_bytes = ee.data.getPixels(ulist)
    elif "expression" in ulist:
        images_bytes = ee.data.computePixels(ulist)
    else:
        raise ValueError("Manifest does not contain 'assetId' or 'expression'")

    with MemoryFile(images_bytes) as memfile:
        with memfile.open() as src:
            profile = src.profile
            metadata_rio = {
                "driver": "Gtiff",
                "tiled": "yes",
                "interleave": "band",
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "ZSTD",
                "predictor": 2,
                "num_threads": 20,
                "nodata": 65535,
                "dtype": "uint16",
                "count": 12,
                "lztd_level": 13,
                "copy_src_overviews": True,
                "overviews": "AUTO"
            }
            profile.update(metadata_rio)
            all_bands = src.read()
    with rio.open(full_outname, "w", **profile) as dst:
        dst.write(all_bands)
    
    print(f"{full_outname} downloaded successfully.")



def get_geotiff(manifest, full_outname, nworks):
    try:
        # Si no hay error, se descarga normalmente
        download_manifest(manifest, full_outname)
    except ee.ee_exception.EEException as ee_error:
        # Si ocurre un error, se maneja y se divide la imagen
        ee_error_message = str(ee_error)
        size = manifest["grid"]["dimensions"]["width"] # Assuming width and height are the same
        cell_width, cell_height, power = calculate_cell_size(ee_error_message, size)
        manifests = quadsplit_manifest(manifest, cell_width, cell_height, power)
        download_manifests(manifests=manifests, max_workers=nworks, full_outname=full_outname)

def get_cube(requests, outfolder, nworks):

    with concurrent.futures.ThreadPoolExecutor(max_workers=nworks) as executor:
        future_list = []
        for _, row in requests._dataframe.iterrows():
            manifest = row.manifest
            full_outname = outfolder / f"{row.id}.tif"
            full_outname.parent.mkdir(parents=True, exist_ok=True)
            future = executor.submit(get_geotiff, manifest, full_outname, nworks)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()
            except Exception as e:
                print(f"Error en una de las descargas: {e}")