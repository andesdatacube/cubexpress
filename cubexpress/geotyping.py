from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final, List, Set, TypeAlias

import ee
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from pyproj import CRS, Transformer
from typing_extensions import TypedDict

# Type definitions
NumberType: TypeAlias = int | float

# Constants for required keys in the geotransform
REQUIRED_KEYS: Final[Set[str]] = {
    "scaleX",
    "shearX",
    "translateX",
    "scaleY",
    "shearY",
    "translateY",
}


def get_transformer(source_crs_wkt: str) -> Transformer:
    """Get cached transformer from source CRS to WGS84 (EPSG:4326)"""
    target_crs = CRS.from_epsg(4326)
    return Transformer.from_crs(
        CRS.from_wkt(source_crs_wkt),
        target_crs,
        always_xy=True,  # Ensures consistent x,y order
    )


def rt2lonlat(raster: "RasterTransform") -> tuple[float, float]:
    """
    Calculate the geographic centroid in WGS84 with optimized performance.

    Args:
        raster: RasterTransform instance with geospatial metadata

    Returns:
        Tuple of (longitude, latitude) in WGS84 coordinates
    """
    # Calculate pixel coordinates of raster center
    col_center = (raster.width - 1) / 2.0
    row_center = (raster.height - 1) / 2.0

    # Extract geotransform parameters as local variables for faster access
    gt = raster.geotransform
    tx = gt["translateX"]
    sx = gt["scaleX"]
    shx = gt["shearX"]
    ty = gt["translateY"]
    shy = gt["shearY"]
    sy = gt["scaleY"]

    # Apply affine transformation
    x = tx + sx * col_center + shx * row_center
    y = ty + shy * col_center + sy * row_center

    # Check if transformation is needed
    source_crs = CRS.from_user_input(raster.crs)
    target_crs = CRS.from_epsg(4326)

    if source_crs == target_crs:
        return (x, y)

    # Perform the transformation
    transformer = get_transformer(source_crs.to_wkt())
    lon, lat = transformer.transform(x, y)

    return lon, lat, x, y


class GeotransformDict(TypedDict):
    """
    Type definition for a geotransform dictionary containing spatial transformation parameters.

    Attributes:
        scaleX (NumberType): The scaling factor in the X direction.
        shearX (NumberType): The shear factor in the X direction.
        translateX (NumberType): The translation in the X direction.
        scaleY (NumberType): The scaling factor in the Y direction.
        shearY (NumberType): The shear factor in the Y direction.
        translateY (NumberType): The translation in the Y direction.
    """

    scaleX: NumberType
    shearX: NumberType
    translateX: NumberType
    scaleY: NumberType
    shearY: NumberType
    translateY: NumberType


class RasterTransform(BaseModel):
    """
    Represents a single geospatial metadata entry with CRS and transformation information.

    Attributes:
        id (str): The unique identifier for the raster metadata.
        crs (str): The Coordinate Reference System string (EPSG code or WKT).
        geotransform (GeotransformDict): A dictionary containing spatial transformation parameters.
        width (int): Raster width in pixels.
        height (int): Raster height in pixels.

    Example:
        >>> metadata = RasterTransform(
        ...     id="image1",
        ...     crs="EPSG:4326",
        ...     geotransform={
        ...         'scaleX': 1.0, 'shearX': 0, 'translateX': 100.0,
        ...         'scaleY': 1.0, 'shearY': 0, 'translateY': 200.0
        ...     },
        ...     width=1000,
        ...     height=1000
        ... )
        RasterTransform(crs='EPSG:4326', width=1000, height=1000)
    """

    crs: str
    geotransform: GeotransformDict
    width: int
    height: int

    @model_validator(mode="before")
    def validate_geotransform(cls, values):
        """
        Validates the geotransform dictionary to ensure it has the required keys
        and that all values are numeric and non-zero where applicable.

        Args:
            values (dict): The input values being validated.

        Returns:
            dict: The validated values.

        Raises:
            ValueError: If validation fails for missing keys, invalid types, or zero scale values.
        """
        geotransform = values.get("geotransform")

        if not isinstance(geotransform, dict):
            raise ValueError(
                f"Expected geotransform to be a dictionary, got {type(geotransform)}"
            )

        missing_keys = REQUIRED_KEYS - set(geotransform.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        extra_keys = set(geotransform.keys()) - REQUIRED_KEYS
        if extra_keys:
            raise ValueError(f"Unexpected keys found: {extra_keys}")

        for key in REQUIRED_KEYS:
            if not isinstance(geotransform[key], (int, float)):
                raise ValueError(f"Value for '{key}' must be numeric (int or float)")

        if not (geotransform["scaleX"] and geotransform["scaleY"]):
            raise ValueError("Scale values cannot be zero")

        return values

    @field_validator("width", "height")
    @classmethod
    def validate_positive(cls, value: int, field) -> int:
        """
        Validates that width and height are positive integers.

        Args:
            value (int): The value of width or height to validate.
            field: The field being validated (width or height).

        Returns:
            int: The validated value.

        Raises:
            ValueError: If the value is not positive.
        """
        if value <= 0:
            raise ValueError(
                f"{field.field_name} must be positive and greater than zero, but got {value}"
            )
        return value

    def __str__(self) -> str:
        """
        Provides a string representation of the RasterTransform instance with a table
        format for the geotransform parameters.

        Returns:
            str: A formatted string showing the raster metadata and geotransform.
        """
        geotransform_data = {
            "Parameter": [
                "scaleX",
                "shearX",
                "translateX",
                "scaleY",
                "shearY",
                "translateY",
            ],
            "Value": [
                self.geotransform["scaleX"],
                self.geotransform["shearX"],
                self.geotransform["translateX"],
                self.geotransform["scaleY"],
                self.geotransform["shearY"],
                self.geotransform["translateY"],
            ],
        }
        geotransform_df = pd.DataFrame(geotransform_data)
        return f"RasterTransform(crs={self.crs}, width={self.width}, height={self.height})\n\nGeotransform:\n{geotransform_df}"


class Request(BaseModel):
    id: str
    raster_transform: RasterTransform
    image: Any
    bands: List[str]
    _expression_key: str = None

    @model_validator(mode="after")
    def validate_image(self):

        if isinstance(self.image, ee.Image):
            self.image = self.image.serialize()
            self._expression_key = "expression"
        # to avoid reading serialization of an ee.Image as str in RequestSet
        elif isinstance(self.image, str) and self.image.strip().startswith("{"):
            self._expression_key = "expression"
        else:
            self.image = self.image
            self._expression_key = "assetId"

        return self


class RequestSet(BaseModel):
    """
    Container for multiple RasterTransform instances with bulk validation capabilities.

    Attributes:
        rastertransformset (List[RasterTransform]): A list of RasterTransform metadata entries.

    Example:
        >>> metadatas = RasterTransformSet(rastertransformset=[metadata1, metadata2])
        >>> df = metadatas.export_df()
    """

    requestset: List[Request]
    _dataframe: pd.DataFrame | None = None



    def create_manifests(self) -> pd.DataFrame:
        """
        Exports the raster metadata to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the metadata for all entries.

        Example:
            >>> df = raster_transform_set.export_df()
            >>> print(df)
        """
        # Use ProcessPoolExecutor for CPU-bound tasks to convert raster transforms to lon/lat
        with ProcessPoolExecutor(max_workers=None) as executor:
            # Submit all tasks to the executor
            futures = [
                executor.submit(rt2lonlat, rt.raster_transform)
                for rt in self.requestset
            ]

            # Collect results as they complete
            points = [future.result() for future in futures]
            lon, lat, x, y = zip(*points)

        return pd.DataFrame(
            [
                {
                    "id": meta.id, # add clud
                    "lon": lon[index],
                    "lat": lat[index],
                    "x": x[index],
                    "y": y[index],
                    "crs": meta.raster_transform.crs,
                    "width": meta.raster_transform.width,
                    "height": meta.raster_transform.height,
                    "geotransform": meta.raster_transform.geotransform,
                    "scale_x": meta.raster_transform.geotransform["scaleX"],
                    "scale_y": meta.raster_transform.geotransform["scaleY"],
                    "manifest": {
                        meta._expression_key: meta.image,
                        "fileFormat": "GEO_TIFF",
                        "bandIds": meta.bands,
                        "grid": {
                            "dimensions": {
                                "width": meta.raster_transform.width,
                                "height": meta.raster_transform.height,
                            },
                            "affineTransform": meta.raster_transform.geotransform,
                            "crsCode": meta.raster_transform.crs,
                        },
                    },
                    "cs_cdf": int(meta.id.split("_")[-1]) / 100,
                    "date": meta.id.split("_")[0],
                    "outname": f"{meta.id}.tif",
                }
                
                for index, meta in enumerate(self.requestset)
            ]
        )



    def _validate_dataframe_schema(self) -> None:
        """
        Checks that the `_dataframe` contains the required columns and that each column
        has the expected data type. Also verifies that the `manifest` field has the
        necessary minimum structure.
        """

        # A) Required columns and expected data types
        required_columns = {
            "id": str,
            "lon": float,
            "lat": float,
            "x": float,
            "y": float,
            "crs": str,
            "width": int,
            "height": int,
            "geotransform": dict,
            "scale_x": (int, float),
            "scale_y": (int, float),
            "manifest": dict,
            "outname": str,
        }

        df_cols = set(self._dataframe.columns)
        required_cols = set(required_columns.keys())

        # 1. Check for missing columns
        missing_cols = required_cols - df_cols
        if missing_cols:
            raise ValueError(f"Missing required columns in dataframe: {missing_cols}")

        # 2. (Optional) Check for extra columns
        # extra_cols = df_cols - required_cols
        # if extra_cols:
        #     raise ValueError(f"Unexpected extra columns in dataframe: {extra_cols}")

        # 3. Verify data types (basic check)
        for col_name, expected_type in required_columns.items():
            for i, value in enumerate(self._dataframe[col_name]):
                if not isinstance(value, expected_type):
                    # For cases like (int, float), you can check with isinstance(value, (int, float))
                    # or directly use the tuple in `expected_type`
                    if isinstance(expected_type, tuple):
                        if not any(isinstance(value, t) for t in expected_type):
                            raise ValueError(
                                f"Column '{col_name}' has an invalid type in row {i}. "
                                f"Expected one of {expected_type}, got {type(value)}"
                            )
                    else:
                        raise ValueError(
                            f"Column '{col_name}' has an invalid type in row {i}. "
                            f"Expected {expected_type}, got {type(value)}"
                        )
                    
        for i, row in self._dataframe.iterrows():
            manifest = row["manifest"]

            # Main required keys
            for key in ["fileFormat", "bandIds", "grid"]:
                if key not in manifest:
                    raise ValueError(
                        f"Missing key '{key}' in 'manifest' for row index {i}"
                    )

            # At least one of 'assetId' or 'expression'
            if not any(k in manifest for k in ["assetId", "expression"]):
                raise ValueError(
                    f"Manifest in row {i} does not contain 'assetId' or 'expression'"
                )

            # Basic validation of 'grid'
            grid = manifest["grid"]
            for subkey in ["dimensions", "affineTransform", "crsCode"]:
                if subkey not in grid:
                    raise ValueError(
                        f"Missing key '{subkey}' in 'manifest.grid' for row index {i}"
                    )

            # Basic validation of 'dimensions'
            dims = grid["dimensions"]
            for dim_key in ["width", "height"]:
                if dim_key not in dims:
                    raise ValueError(
                        f"Missing '{dim_key}' in 'manifest.grid.dimensions' for row {i}"
                    )
                if not isinstance(dims[dim_key], int) or dims[dim_key] <= 0:
                    raise ValueError(
                        f"'{dim_key}' in 'manifest.grid.dimensions' must be a positive integer. "
                        f"Row {i} has value {dims[dim_key]}"
                    )

            # Basic validation of 'affineTransform'
            aff = grid["affineTransform"]
            for a_key in ["scaleX", "shearX", "translateX", "scaleY", "shearY", "translateY"]:
                if a_key not in aff:
                    raise ValueError(
                        f"Missing '{a_key}' in 'manifest.grid.affineTransform' for row {i}"
                    )
                if not isinstance(aff[a_key], (int, float)):
                    raise ValueError(
                        f"Value for '{a_key}' in 'manifest.grid.affineTransform' must be numeric. "
                        f"Row {i} has {type(aff[a_key])}."
                    )



    @model_validator(mode="after")
    def validate_metadata(self) -> RequestSet:
        """
        Validates that all entries have consistent and valid CRS formats.

        Returns:
            RasterTransformSet: The validated instance.

        Raises:
            ValueError: If any CRS is invalid or inconsistent.
        """
        # 1. Pre-consistency validation (CRS, IDs, etc.)
        crs_set: Set[str] = {meta.raster_transform.crs for meta in self.requestset}
        validated_crs: Set[str] = set()

        # Validate CRS formats
        for crs in crs_set:
            if crs not in validated_crs:
                try:
                    CRS.from_string(crs)
                    validated_crs.add(crs)
                except Exception as e:
                    raise ValueError(f"Invalid CRS format: {crs}") from e

        # Validate ids, they must be unique
        ids = {meta.id for meta in self.requestset}
        if len(ids) != len(self.requestset):
            raise ValueError("All entries must have unique IDs")

        # Upgrade same_coordinates to True if all coordinates are the same
        # 2. We create the dataframe
        self._dataframe = self.create_manifests()
        
        # 3. We validate the structure of the dataframe
        self._validate_dataframe_schema()

        return self
    


    def __repr__(self) -> str:
        """
        Provides a string representation of the metadata set including a table of all entries.

        Returns:
            str: A string representation of the entire RasterTransformSet.
        """
        num_entries = len(self.requestset)
        return f"RequestSet({num_entries} entries)"

    def __str__(self):
        return super().__repr__()
