import ee
import pathlib
import cubexpress

ee.Initialize(project='ee-julius013199')

lon = -0.000027
lat = 40.901171
outfolder = pathlib.Path("raw")
scale = 10 
nworks = 5
x, y, epsg = cubexpress.geo2utm(lon, lat)
size = 2048 # The trouble is this size is not multiple of 4 (10240 % 4 != 0)
bands = ['B1', 'B2', 'B3', "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
assetId = 'COPERNICUS/S2_SR_HARMONIZED/20210927T104719_20210927T105809_T30TYL'


geotransform = cubexpress.lonlat2rt(
    lon=lon,
    lat=lat,
    edge_size=size,
    scale=scale
)

request = cubexpress.Request(
    id="test",
    raster_transform=geotransform,
    bands=bands,
    image=assetId
)

requests = cubexpress.RequestSet(requestset=[request])

cubexpress.get_cube(requests, outfolder, nworks)



















