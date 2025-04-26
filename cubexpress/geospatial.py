import ee
import re
from copy import deepcopy
from typing import Dict


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
            new_y = y + (columny * cell_height) * scale_y
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



def _square_roi(lon: float, lat: float, edge_size: int, scale: int) -> ee.Geometry:
    """Return a square `ee.Geometry` centred on (*lon*, *lat*)."""
    half = edge_size * scale / 2
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(half).bounds()
