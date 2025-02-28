import concurrent.futures
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Optional

import ee
import numpy as np
import pandas as pd

from cubexpress.geotyping import RequestSet


def check_not_found_error(error_message: str) -> bool:
    """
    Checks if the error message indicates that the image was not found.

    Args:
        error_message (str): The error message to check.

    Returns:
        bool: True if the error message indicates "not found", False otherwise.

    Example:
        >>> check_not_found_error("Total request size must be less than or equal to...")
        True
    """
    return (
        "Total request size" in error_message
        and "must be less than or equal to" in error_message
    )


def quadsplit_manifest(manifest: dict) -> list[dict]:
    """
    Splits a manifest into four smaller ones by dividing the grid dimensions.

    Args:
        manifest (dict): The original manifest to split.

    Returns:
        List[dict]: A list of four smaller manifests with updated grid transformations.

    Example:
        >>> manifest = {'grid': {'dimensions': {'width': 100, 'height': 100}, 'affineTransform': {'scaleX': 0.1, 'scaleY': 0.1, 'translateX': 0, 'translateY': 0}}}
        >>> quadsplit_manifest(manifest)
        [{'grid': {'dimensions': {'width': 50, 'height': 50}, 'affineTransform': {'scaleX': 0.1, 'scaleY': 0.1, 'translateX': 0, 'translateY': 0}}}, {'grid': {'dimensions': {'width': 50, 'height': 50}, 'affineTransform': {'scaleX': 0.1, 'scaleY': 0.1, 'translateX': 5.0, 'translateY': 0}}}, ...]
    """
    manifest_copy = deepcopy(manifest)
    new_width = manifest["grid"]["dimensions"]["width"] // 2
    new_height = manifest["grid"]["dimensions"]["height"] // 2
    manifest_copy["grid"]["dimensions"]["width"] = new_width
    manifest_copy["grid"]["dimensions"]["height"] = new_height

    manifests = []
    for idx in range(4):
        new_manifest = deepcopy(manifest_copy)
        res_x = manifest["grid"]["affineTransform"]["scaleX"]
        res_y = manifest["grid"]["affineTransform"]["scaleY"]

        add_x, add_y = (0, 0)
        if idx == 1:
            add_x = new_width * res_x
        elif idx == 2:
            add_y = new_height * res_y
        elif idx == 3:
            add_x = new_width * res_x
            add_y = new_height * res_y

        new_manifest["grid"]["affineTransform"]["translateX"] += add_x
        new_manifest["grid"]["affineTransform"]["translateY"] += add_y

        manifests.append(new_manifest)

    return manifests


def getGeoTIFFbatch(
    manifest_dict: dict,
    full_outname: pathlib.Path,
    max_deep_level: Optional[int] = 5,
    method: Optional[str] = "getPixels",
) -> Optional[np.ndarray]:
    """
    Downloads a GeoTIFF image from Google Earth Engine using either the `getPixels` or `computePixels` method.
    If the requested area exceeds the size limit, the image is recursively split into smaller tiles until the
    download succeeds or the maximum recursion depth is reached.

    Args:
        manifest_dict (dict): A dictionary containing image metadata, including grid dimensions, affine transformations,
                              and either an `assetId` or `expression` for the image source.
        full_outname (pathlib.Path): The full path where the downloaded GeoTIFF file will be saved.
        max_deep_level (Optional[int]): Maximum recursion depth for splitting large requests. Defaults to 5.
        method (Optional[str]): Method for retrieving image data. Can be 'getPixels' for asset-based requests or
                                'computePixels' for expressions. Defaults to 'getPixels'.

    Returns:
        Optional[pathlib.Path]: The path to the downloaded GeoTIFF file. Returns `None` if the download fails.

    Raises:
        ValueError: If the method is not 'getPixels' or 'computePixels', or if the image cannot be found.

    Example:
        >>> import ee
        >>> import pathlib
        >>> ee.Initialize()
        >>> manifest_dict = {
        ...     "assetId": "COPERNICUS/S2_HARMONIZED/20160816T153912_20160816T154443_T18TYN",
        ...     "fileFormat": "GEO_TIFF",
        ...     "bandIds": ["B4", "B3", "B2"],
        ...     "grid": {
        ...         "dimensions": {
        ...             "width": 512,
        ...             "height": 512
        ...         },
        ...         "affineTransform": {
        ...             "scaleX": 10,
        ...             "shearX": 0,
        ...             "translateX": 725260.108545126,
        ...             "scaleY": -10,
        ...             "shearY": 0,
        ...             "translateY": 4701550.38712196
        ...         },
        ...         "crsCode": "EPSG:32618"
        ...     }
        ... }

        >>> getGeoTIFFbatch(manifest_dict pathlib.Path('output/sentinel_image.tif'))
        PosixPath('output/sentinel_image.tif')
    """

    # Check if the maximum recursion depth has been reached
    if max_deep_level == 0:
        raise ValueError("Max recursion depth reached.")

    try:
        # Get the image bytes
        if method == "getPixels":
            image_bytes: bytes = ee.data.getPixels(manifest_dict)
        elif method == "computePixels":
            image_bytes: bytes = ee.data.computePixels(manifest_dict)
        else:
            raise ValueError("Method must be either 'getPixels' or 'computePixels'")

        # Write the image bytes to a file
        with open(full_outname, "wb") as src:
            src.write(image_bytes)
    except Exception as e:
        # TODO: This is a workaround when the image is not found, as it is a message from the server
        # it is not possible to check the type of the exception
        if not check_not_found_error(str(e)):
            raise ValueError(
                f"Error downloading the GeoTIFF file from Earth Engine: {e}"
            )

        # Create the output directory if it doesn't exist
        child_folder: pathlib.Path = full_outname.parent / full_outname.stem
        pathlib.Path(child_folder).mkdir(parents=True, exist_ok=True)

        # Split the manifest into four smaller manifests
        manifest_dicts = quadsplit_manifest(manifest_dict)

        for idx, manifest_dict_batch in enumerate(manifest_dicts):
            # Recursively download the image
            getGeoTIFFbatch(
                full_outname=child_folder / ("%s__%02d.tif" % (full_outname.stem, idx)),
                manifest_dict=manifest_dict_batch,
                max_deep_level=max_deep_level - 1,
                method=method,
            )

    return full_outname


def getGeoTIFF(
    manifest_dict: dict, full_outname: pathlib.Path, max_deep_level: Optional[int] = 5
) -> Optional[np.ndarray]:
    """
    Retrieves an image from Earth Engine using the appropriate method based on the manifest type.

    This function downloads a GeoTIFF image from Google Earth Engine (GEE). Depending on the content of
    the provided manifest (`manifest_dict`), the function will either use the `getPixels` method (for
    asset-based requests) or the `computePixels` method (for expressions). If the requested area exceeds
    the size limit, the image will be recursively split into smaller tiles until the download succeeds or
    the maximum recursion depth is reached.

    Args:
        manifest_dict (dict): A dictionary containing the image metadata. This should include either:
            - `assetId`: The identifier of a GEE asset (e.g., satellite imagery).
            - `expression`: A serialized string representing a GEE image expression (e.g., an image computation).
            Additionally, the manifest should include grid information such as the image dimensions and affine transformations.

        full_outname (pathlib.Path): The full path where the downloaded GeoTIFF file will be saved.

        max_deep_level (Optional[int]): The maximum recursion depth for splitting large requests into smaller tiles if needed.
            Defaults to 5.

    Returns:
        Optional[np.ndarray]: The downloaded image as a `numpy` array, or `None` if the download fails. It will
            also return the full file path to the saved GeoTIFF image.

    Raises:
        ValueError: If the manifest does not contain either an `assetId` or `expression`, or if there is an error during download.

    Example 1: Downloading an image using an `assetId`:
        >>> import ee
        >>> import pathlib
        >>> ee.Initialize()
        >>> manifest_dict = {
        ...     "assetId": "COPERNICUS/S2_HARMONIZED/20160816T153912_20160816T154443_T18TYN",
        ...     "fileFormat": "GEO_TIFF",
        ...     "bandIds": ["B4", "B3", "B2"],
        ...     "grid": {
        ...         "dimensions": {"width": 512, "height": 512},
        ...         "affineTransform": {
        ...             "scaleX": 10,
        ...             "shearX": 0,
        ...             "translateX": 725260.108545126,
        ...             "scaleY": -10,
        ...             "shearY": 0,
        ...             "translateY": 4701550.38712196
        ...         },
        ...         "crsCode": "EPSG:32618"
        ...     }
        ... }
        >>> getGeoTIFF(manifest_dict, pathlib.Path('output/sentinel_image.tif'))
        PosixPath('output/sentinel_image.tif')

    Example 2: Downloading an image using an `expression`:
        >>> image = ee.Image("COPERNICUS/S2_HARMONIZED/20160816T153912_20160816T154443_T18TYN") \
        ...           .divide(10_000) \
        ...           .select(["B4", "B3", "B2"])
        >>> expression = image.serialize()
        >>> manifest_dict = {
        ...     "expression": expression,
        ...     "fileFormat": "GEO_TIFF",
        ...     "grid": {
        ...         "dimensions": {"width": 512, "height": 512},
        ...         "affineTransform": {
        ...             "scaleX": 10,
        ...             "shearX": 0,
        ...             "translateX": 725260.108545126,
        ...             "scaleY": -10,
        ...             "shearY": 0,
        ...             "translateY": 4701550.38712196
        ...         },
        ...         "crsCode": "EPSG:32618"
        ...     }
        ... }
        >>> getGeoTIFF(manifest_dict, pathlib.Path('output/expression_image.tif'))
        PosixPath('output/expression_image.tif')
    """
    if "assetId" in manifest_dict:
        return getGeoTIFFbatch(
            manifest_dict=manifest_dict,
            full_outname=full_outname,
            max_deep_level=max_deep_level,
            method="getPixels",
        )
    elif "expression" in manifest_dict:
        if isinstance(
            manifest_dict["expression"], str
        ):  # Decode only if the expression is still a string.
            # From a string to a ee.Image object
            manifest_dict["expression"] = ee.deserializer.decode(
                json.loads(manifest_dict["expression"])
            )

        return getGeoTIFFbatch(
            manifest_dict=manifest_dict,
            full_outname=full_outname,
            max_deep_level=max_deep_level,
            method="computePixels",
        )
    else:
        raise ValueError("Manifest does not contain 'assetId' or 'expression'")


def getcube(
    request: RequestSet,
    output_path: str | pathlib.Path,
    nworkers: Optional[int] = None,
    max_deep_level: Optional[int] = 5,
) -> list[pathlib.Path]:
    """
    Downloads multiple GeoTIFF images in parallel from Google Earth Engine (GEE) based on the provided request set.

    Args:
        request (RequestSet): A collection of image requests containing metadata and processing parameters.
        output_path (Union[str, pathlib.Path]): Directory where the downloaded images will be saved.
        nworkers (Optional[int], default=None): Number of parallel threads. If None, runs sequentially.
        max_deep_level (Optional[int], default=5): Maximum recursion depth for image subdivision if exceeding GEE limits.

    Returns:
        List[pathlib.Path]: List of paths to the downloaded GeoTIFF files.

    Example:
        >>> import ee, cubexpress
        >>> ee.Initialize()
        >>> point = ee.Geometry.Point([-97.59, 33.37])
        >>> collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        ...                 .filterBounds(point) \
        ...                 .filterDate('2024-01-01', '2024-01-31')
        >>> image_ids = collection.aggregate_array('system:id').getInfo()
        >>> geotransform = cubexpress.lonlat2rt(lon=-97.59, lat=33.37, edge_size=128, scale=10)
        >>> requests = [cubexpress.Request(id=f"s2_{i}", raster_transform=geotransform, bands=["B4", "B3", "B2"], image=ee.Image(img_id)) for i, img_id in enumerate(image_ids)]
        >>> cube_requests = cubexpress.RequestSet(requestset=requests)
        >>> cubexpress.getcube(request=cube_requests, nworkers=4, output_path="output", max_deep_level=5)
        [PosixPath('output/s2_0.tif'), PosixPath('output/s2_1.tif'), ...]
    """

    # Check that _dataframe exists and is not empty
    if request._dataframe is None or request._dataframe.empty:
        raise ValueError(
            "The request's _dataframe is None or empty. "
            "There are no valid requests to process."
        )
    
    # **Revalidate** the DataFrame structure, in case the user manipulated it.
    request._validate_dataframe_schema()

    # Get the table
    table: pd.DataFrame = request._dataframe

    # Create the output directory if it doesn't exist
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        futures = {
            executor.submit(
                getGeoTIFF, row.manifest, output_path / row.outname, max_deep_level
            ): row
            for _, row in table.iterrows()
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                # TODO add this into the log
                print(f"Error processing {futures[future].outname}: {e}")

    return results
