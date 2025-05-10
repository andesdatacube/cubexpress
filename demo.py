import ee
import cubexpress

ee.Initialize()


init_df = cubexpress.s2_cloud_table(
    lon=-75, # Adicionar si no quiere datos nulos, es decir que no caiga fuera de un foorprint
    lat=-11,
    edge_size=2048,
    min_cscore=0.6,
    start= "2017-03-01",
    end="2017-09-01"
)

download_df = cubexpress.get_cube(
    table = init_df, # validator que si no contiene datos decirlo, tus parametros no son suficiente para generar un cubo
    outfolder = "cubexpress_test", # Podria la salida ser un tabla de lo que se descargo, porque no seria siempre una tabla completa
)




adfdsaf["outname"] + "asfasdf"


adfdsaf["outname"] = outfolder / adfdsaf["outname"]  + "sadf"



download_df.drop(columns=["manifest", "scale_x", "scale_y", "lon", "lat", "x", "y"])