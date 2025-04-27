import ee # pygeohash
import cubexpress

ee.Initialize()

lon=-17.000027
lat=68.1017
cloud_max=40
edge_size = 2_048
scale = 10
start= "2018-03-01"
end="2018-04-01"
output = "raw"
nworks = 4


df = cubexpress.cloud_table( # Generanting table .... Además guardarla para luego (por tiempo y cloud_maz)
    lon=lon,
    lat=lat,
    edge_size = edge_size,
    scale = scale,
    cloud_max=cloud_max,
    start= start,
    end=end
)

requests = cubexpress.table_to_requestset(df, mosaic=True)

cubexpress.get_cube(requests, output, nworks)
t 










