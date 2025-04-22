import cubexpress
import pathlib
import ee
import re
import rasterio as rio
from rasterio.io import MemoryFile
import concurrent.futures

ee.Initialize(project='ee-julius013199')


def fetch_and_save(ulist, full_outname, index):
    
    images_bytes = ee.data.getPixels(ulist) # computePixels

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
        print(index)

lon = -0.000027
lat = 40.901171
outfolder = pathlib.Path("raw")
scale = 10 
nworks = 5
x, y, epsg = cubexpress.geo2utm(lon, lat)
size = 8192 # The trouble is this size is not multiple of 4 (10240 % 4 != 0)
bands = ['B1', 'B2', 'B3', "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
assetId = 'COPERNICUS/S2_SR_HARMONIZED/20210927T104719_20210927T105809_T30TYL'


request = {
    'assetId': assetId,
    'fileFormat': 'GEO_TIFF',
    'bandIds': bands,
    'grid': {
        'dimensions': {
            'width': size,
            'height': size               
        },
        'affineTransform': {
            'scaleX': scale,
            'scaleY': -scale,
            'translateX': x,
            'translateY': y,
            'shearX': 0,
            'shearY': 0
        },
        'crsCode': epsg
    }
}

try:
    image = ee.data.getPixels(request) # computePixels, align it with fetch_and_save
except ee.ee_exception.EEException as ee_error:
    ee_error_message = str(ee_error)

# 
outfolder.mkdir(parents=True, exist_ok=True)

match = re.findall(r'\d+', ee_error_message)
image_pixel = match[0]
max_pixel = match[1]
images = int(image_pixel) / int(max_pixel)

power = 0  # Starting with 4^0
while images > 1:
    power += 1
    images = int(image_pixel) / (int(max_pixel) * 4**power)
    val_split = 4**power

print(f"Generating {val_split} geotransforms")

request_list = []
cell_width = size // 2**power  # 640
cell_height = size // 2**power  # 640

for v in range(2**power):  # Rows
    print(v)
    for j in range(2**power):  # Columns
        # New x and y for each subrequest
        new_x = x + (j * cell_width) * scale
        new_y = y - (v * cell_height) * scale # Negative because Y is inverted in geospatial images

        # Subrequest for each tile
        subrequest = {
            'assetId': assetId,
            'fileFormat': 'GEO_TIFF',
            'bandIds': bands,
            'grid': {
                'dimensions': {
                    'width': cell_width,
                    'height': cell_height
                },
                'affineTransform': {
                    'scaleX': scale,
                    'shearX': 0,
                    'translateX': new_x,
                    'shearY': 0,
                    'scaleY': -scale,
                    'translateY': new_y
                },
                'crsCode': epsg,
            }
        }

        # Add the subrequest to the list
        request_list.append(subrequest)


with concurrent.futures.ThreadPoolExecutor(max_workers=nworks) as executor:
    future_list = []
    
    for index, ulist in enumerate(request_list):
        # Calcula el path de salida

        full_outname = outfolder / "GeoHash" / f"{index}.tif"
        full_outname.parent.mkdir(parents=True, exist_ok=True)

        # Envía la tarea al ThreadPool
        future = executor.submit(fetch_and_save, ulist, full_outname, index)
        future_list.append(future)

    # Opcional: esperar a que terminen todas y/o manejar errores
    for future in concurrent.futures.as_completed(future_list):
        # Si quieres capturar excepciones individualmente:
        try:
            future.result()  
        except Exception as e:
            print(f"Error en una de las descargas: {e}")








