import cubexpress
import ee
import datetime
import pandas as pd

ee.Initialize(project="ee-julius013199")
       
# Define the variables to aggregate
# These variables are 'instantaneous' and will be averaged

mean_vars = [
    'temperature_2m',
    # 'dewpoint_temperature_2m',
    # 'u_component_of_wind_10m',
    # 'v_component_of_wind_10m',
    # 'surface_pressure'
]

# These variables are 'cumulative' (fluxes) and will be summed
# In ERA5 hourly, they represent the accumulation over the *past hour*
# 
sum_vars = [
    'total_precipitation'
    # 'surface_net_solar_radiation',
    # 'surface_net_thermal_radiation',
]

vars = mean_vars + sum_vars

affineTransform = {
            'scaleX': 0.25,
            'shearX': 0,
            'translateX': -180.125,
            'shearY': 0,
            'scaleY': -0.25,
            'translateY': 90.125
        }


raster_transform = cubexpress.RasterTransform(
        crs="EPSG:4326", 
        geotransform=affineTransform,
        width=340, 
        height=340
)

# geotransform = cubexpress.lonlat2rt(
#     lon=16.5,
#     lat=41.5,
#     edge_size=170,
#     scale=27830
# )

# --- Define the Daily Aggregation Functions ---
def aggregate_daily_sum(dataset):    
    # Calculate the sum for the 'sum' variables
    sum_image = dataset.sum()    
    # Set the time properties for the new daily image
    return sum_image

def aggregate_daily_mean(dataset):    
    # Calculate the sum for the 'sum' variables
    mean_image = dataset.mean()    
    # Set the time properties for the new daily image
    return mean_image                
                      
i=0

for i in range(len(vars)):
    
    
    list_requests = []    
    var = vars[i]    
    # generate list of days between start and end  (start="1970-01-01", end="2020-12-31", freq="D")
    list_dates = pd.date_range(start="1990-01-01", end="1990-01-02", freq="D")
        
    for t in range(0, len(list_dates)):
        
        #print(f"Processing variable {var} for {list_dates[t]}")
        start = list_dates[t]
        end = list_dates[t] + datetime.timedelta(days=1)
        # set time
        dataset = ee.ImageCollection('ECMWF/ERA5/HOURLY') \
            .filter(ee.Filter.date(start, end)) \
            .select([var])
        
        if var in sum_vars:
            var_dataset = aggregate_daily_sum(dataset)
        elif var in mean_vars:
            var_dataset = aggregate_daily_mean(dataset)      
            
        
        names = var_dataset.bandNames().getInfo()
        #var_dataset.getInfo()
        
        request = cubexpress.Request(
            id=f"{var}_{start.strftime('%Y_%m_%d')}",
            raster_transform=raster_transform,
            bands=names,
            image=var_dataset
        )
        list_requests.append(request)

    cube_requests = cubexpress.RequestSet(requestset=list_requests)

    cubexpress.get_cube(
        requests=cube_requests,
        # outfolder=f"../../../../data/databases/ERA5/Mediterranean/{var}",
        outfolder=f"data/{var}",
        nworks=8
    )