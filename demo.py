import ee # pygeohash
import cubexpress

ee.Initialize()


df = cubexpress.s2_cloud_table( # Generanting table .... Además guardarla para luego (por tiempo y cloud_maz)
    lon=-75,
    lat=-11,
    edge_size=2_048,
    cscore=0,
    start= "2017-03-01",
    end="2017-05-01"
)

requests = cubexpress.table_to_requestset(df)

requests._dataframe

cubexpress.get_cube(
    requests= requests, 
    outfolder = "cubexpress_test"
)