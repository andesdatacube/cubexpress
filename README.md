<h1></h1>

<p align="center">
  <img src="./docs/logo_cubexpress.png" width="39%">
</p>

<p align="center">
    <em>A Python package for efficient processing of cubic earth observation (EO) data</em> 🚀
</p>

<p align="center">
<a href='https://pypi.python.org/pypi/cubexpress'>
    <img src='https://img.shields.io/pypi/v/cubexpress.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
</p>

---

**GitHub**: [https://github.com/andesdatacube/cubexpress/](https://github.com/andesdatacube/cubexpress/) 🌐

**PyPI**: [https://pypi.org/project/cubexpress/](https://pypi.org/project/cubexpress/) 🛠️

---

## **Overview**

**CubeXpress** is a Python package designed to **simplify and accelerate** the process of working with Google Earth Engine (GEE) data cubes. With features like multi-threaded downloads, automatic subdivision of large requests, and direct pixel-level computations on GEE, **CubeXpress** helps you handle massive datasets with ease.

## **Key Features**
- **Fast Image and Collection Downloads**  
  Retrieve single images or entire collections at once, taking advantage of multi-threaded requests.
- **Automatic Tiling**  
  Large images are split ("quadsplit") into smaller sub-tiles, preventing errors with GEE’s size limits.
- **Direct Pixel Computations**  
  Perform computations (e.g., band math) directly on GEE, then fetch results in a single step.
- **Scalable & Efficient**  
  Optimized memory usage and parallelism let you handle complex tasks in big data environments.

## **Installation**
Install the latest version from PyPI:

```bash
pip install cubexpress
```

> **Note**: You need a valid Google Earth Engine account and `earthengine-api` installed (`pip install earthengine-api`). Also run `ee.Initialize()` before using CubeXpress.

---

## **Basic Usage**

### **Download a single `ee.Image`**

```python
import ee
import cubexpress

# Initialize Earth Engine
ee.Initialize(project="your-project-id")

# Create a raster transform
geotransform = cubexpress.lonlat2rt(
    lon=-76.5,
    lat=-9.5,
    edge_size=128,  # Width=Height=128 pixels
    scale=90        # 90m resolution
)

# Define a single Request
request = cubexpress.Request(
    id="dem_test",
    raster_transform=geotransform,
    bands=["elevation"],
    image="NASA/NASADEM_HGT/001" # Note: you can wrap with ee.Image("NASA/NASADEM_HGT/001").divide(10000) if needed

# Build the RequestSet
cube_requests = cubexpress.RequestSet(requestset=[request])

# Download with multi-threading
cubexpress.getcube(
    request=cube_requests,
    output_path="output_dem",
    nworkers=4,
    max_deep_level=5
)
```

This will create a GeoTIFF named `dem_test.tif` in the `output_dem` folder, containing the elevation band.

---


### **Download pixel values from an ee.ImageCollection**

You can fetch multiple images by constructing a `RequestSet` with several `Request` objects. For example, filter Sentinel-2 images near a point:

```python
import ee
import cubexpress

ee.Initialize(project="your-project-id")

# Filter a Sentinel-2 collection
point = ee.Geometry.Point([-97.59, 33.37])
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
               .filterBounds(point) \
               .filterDate('2024-01-01', '2024-01-31')

# Extract image IDs
image_ids = collection.aggregate_array('system:id').getInfo()

# Set the geotransform
geotransform = cubexpress.lonlat2rt(
    lon=-97.59, 
    lat=33.37, 
    edge_size=512, 
    scale=10
)

# Build multiple requests
requests = [
    cubexpress.Request(
        id=f"s2test_{i}",
        raster_transform=geotransform,
        bands=["B4", "B3", "B2"],
        image=image_id  # Note: you can wrap with ee.Image(image_id).divide(10000) if needed
    )
    for i, image_id in enumerate(image_ids)
]

# Create the RequestSet
cube_requests = cubexpress.RequestSet(requestset=requests)

# Download
cubexpress.getcube(
    request=cube_requests,
    output_path="output_sentinel",
    nworkers=4,
    max_deep_level=5
)
```

---

### **Process and extract a pixel from an ee.Image**
If you provide an `ee.Image` with custom calculations (e.g., `.divide(10000)`, `.normalizedDifference(...)`), CubeXpress can run those on GEE, then download the result. For large results, it automatically splits the image into sub-tiles.

```python
import ee
import cubexpress

ee.Initialize(project="your-project-id")

# Example: NDVI from Sentinel-2
image = ee.Image("COPERNICUS/S2_HARMONIZED/20170804T154911_20170804T155116_T18SUJ") \
           .normalizedDifference(["B8", "B4"]) \
           .rename("NDVI")

geotransform = cubexpress.lonlat2rt(
    lon=-76.59, 
    lat=38.89, 
    edge_size=256, 
    scale=10
)

request = cubexpress.Request(
    id="ndvi_test",
    raster_transform=geotransform,
    bands=["NDVI"],
    image=image  # custom expression
)

cube_requests = cubexpress.RequestSet(requestset=[request])

cubexpress.getcube(
    request=cube_requests,
    output_path="output_ndvi",
    nworkers=2,
    max_deep_level=5
)
```

---

## **Advanced Usage**

### **Same Set of Sentinel-2 Images for Multiple Points**

Below is a **advanced example** demonstrating how to work with **multiple points** and a **Sentinel-2** image collection in one script. We first create a global collection but then filter it on a point-by-point basis, extracting only the images that intersect each coordinate. Finally, we download them in parallel using **CubeXpress**.


```python
import ee
import cubexpress

# Initialize Earth Engine with your project
ee.Initialize(project="your-project-id")

# Define multiple points (longitude, latitude)
points = [
    (-97.64, 33.37),
    (-97.59, 33.37)
]

# Start with a broad Sentinel-2 collection
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate("2024-01-01", "2024-01-31")
)

# Build a list of Request objects
requestset = []
for i, (lon, lat) in enumerate(points):
    # Create a point geometry for the current coordinates
    point_geom = ee.Geometry.Point([lon, lat])
    collection_filtered = collection.filterBounds(point_geom)
    
    # Convert the filtered collection into a list of asset IDs
    image_ids = collection_filtered.aggregate_array("system:id").getInfo()
    
    # Define a geotransform for this point
    geotransform = cubexpress.lonlat2rt(
        lon=lon,
        lat=lat,
        edge_size=512,  # Adjust the image size in pixels
        scale=10        # 10m resolution for Sentinel-2
    )
    
    # Create one Request per image found for this point
    requestset.extend([
        cubexpress.Request(
            id=f"s2test_{i}_{idx}",
            raster_transform=geotransform,
            bands=["B4", "B3", "B2"],
            image=image_id
        )
        for idx, image_id in enumerate(image_ids)
    ])

# Combine into a RequestSet
cube_requests = cubexpress.RequestSet(requestset=requestset)

# Download everything in parallel
results = cubexpress.getcube(
    request=cube_requests,
    nworkers=4,
    output_path="images_s2",
    max_deep_level=5
)

print("Downloaded files:", results)
```


**How it works**:  

1. **Points:** We define multiple coordinates in `points`.  
2. **Global collection:** We retrieve a broad Sentinel-2 collection covering the desired date range.  
3. **Per-point filter:** For each point, we call `.filterBounds(...)` to get only images intersecting that location.  
4. **Geotransform:** We create a local geotransform (`edge_size`, `scale`) defining the spatial extent and resolution around each point.  
5. **Requests:** Each point-image pair becomes a `Request`, stored in a single list.  
6. **Parallel download:** With `cubexpress.getcube()`, all requests are fetched simultaneously, automatically splitting large outputs into sub-tiles if needed (up to `max_deep_level`).  



## **License**
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<p align="center">
  Built with 🌎 and ❤️ by the <strong>CubeXpress</strong> team
</p>
