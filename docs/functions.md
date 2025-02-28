# **Main Classes and Functions**


## **`lonlat2rt`**  
Generates a **`RasterTransform`** for a given geographic point by converting **longitude/latitude** coordinates to **UTM projection**. This transformation defines the spatial extent, resolution, and coordinate reference system (CRS) needed for geospatial processing.

- **Arguments**:
  - `lon`: Longitude coordinate.
  - `lat`: Latitude coordinate.
  - `edge_size`: Width/height of the raster in pixels.
  - `scale`: Spatial resolution in meters per pixel.

- **Returns**:  
  - A **`RasterTransform`** object containing the CRS, affine transformation parameters, and raster dimensions.

- **Example**:
  ```python
  import cubexpress

  geotransform = cubexpress.lonlat2rt(
      lon=-76.5,
      lat=40.0,
      edge_size=512,
      scale=30
  )
  
  print(geotransform)
  ```
## **`RasterTransform`**

Defines the spatial metadata required for geospatial operations, including the **coordinate reference system (CRS)** and **affine transformation matrix**.

- **Attributes**:
  - `crs`: EPSG code of the UTM projection.
  - `geotransform`: Affine transformation parameters (scale, translation, shear).
  - `width`: Raster width in pixels.
  - `height`: Raster height in pixels.

- **Example**:
  ```python
  from cubexpress.geotyping import RasterTransform

  rt = RasterTransform(
      crs="EPSG:32617",
      geotransform={
          "scaleX": 30, 
          "shearX": 0, 
          "translateX": 500000,
          "scaleY": -30, 
          "shearY": 0, 
          "translateY": 4000000
      },
      width=512,
      height=512
  )
  print(rt)
  ```


## **`Request`**
A single image download specification.

- **Parameters**:  
  - `id`: Unique identifier for the request (used for naming output files).  
  - `raster_transform`: Spatial metadata, typically created via `lonlat2rt(...)`.  
  - `image`: Can be an `ee.Image` (serialized internally) or a string asset ID.  
  - `bands`: List of band names to extract.  

- **Example**:

  ```python
  request = cubexpress.Request(
      id="my_image",
      raster_transform=geotransform,
      bands=["B4", "B3", "B2"],
      image="COPERNICUS/S2_HARMONIZED/20170804T154911_20170804T155116_T18SUJ"
  )
  ```

## **`RequestSet`**
A container for multiple `Request` objects, ensuring validation and organization before processing.

- **Automatically generates** an internal **DataFrame** (the "manifest") containing all request details.

- **Example**:
  ```python
  requests = [request1, request2, ...]
  request_set = cubexpress.RequestSet(requestset=requests)
  ```

- **Viewing the RequestSet DataFrame**:  
  To inspect the structured request details, you can print the internal DataFrame:

  ```python
  print(request_set._dataframe)
  ```


## **`getcube`**
The main download function. It reads the manifest from a `RequestSet`, calls GEE’s internal APIs (`getPixels`/`computePixels`), and writes GeoTIFFs to disk.  

- **Arguments**:
  - `request`: The `RequestSet` to process.
  - `output_path`: Directory for saving the resulting GeoTIFF files.
  - `nworkers`: Number of parallel threads (workers).
  - `max_deep_level`: Maximum recursion depth if sub-tiling is required.

- **Returns**: A list of `pathlib.Path` objects pointing to the downloaded files.
