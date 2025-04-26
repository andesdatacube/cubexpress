import ee
import cubexpress
import time
# start= "2017-01-01"
# end="2017-03-01"
# start= "2017-03-02"
# end="2017-04-01"
ee.Initialize()

lon=-2.000027
lat=41.0017
cloud_max=70
edge_size = 2_048
scale = 10
start= "2018-01-01"
end="2018-02-01"
output = "images"
nworks = 4
bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
              "B8", "B8A", "B9", "B10","B11", "B12"]

# collection = "COPERNICUS/S2_HARMONIZED"


# Medir tiempo de ejecución

start_time = time.time()
df = cubexpress.cloud_table( # Generanting table .... Además guardarla para luego (por tiempo y cloud_maz)
    lon=lon,
    lat=lat,
    edge_size = edge_size,
    scale = scale,
    bands=bands,
    cloud_max=cloud_max,
    start= start,
    end=end
)
df.attrs
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")


requests = cubexpress.table_to_requestset(
    df=df,
    mosaic=True
)
requests._dataframe.iloc[0]["manifest"]

cubexpress.get_cube(requests, output, nworks)











