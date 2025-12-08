from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final, List, Set

import ee
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from pyproj import CRS, Transformer

# Constants for geotransform validation
REQUIRED_KEYS: Final[Set[str]] = {
    "scaleX",
    "shearX",
    "translateX",
    "scaleY",
    "shearY",
    "translateY",
}

def rt2lonlat(raster) -> tuple[float, float, float, float]:
    """
    Calculate geographic centroid from raster transform.
    
    Args:
        raster: Object with .crs, .geotransform, .width, .height
        
    Returns:
        tuple[float, float, float, float]: (lon, lat, x, y)
    """
    # Calculate pixel coordinates of raster center
    col_center = raster.width / 2.0
    row_center = raster.height / 2.0

    # Extract geotransform parameters
    gt = raster.geotransform
    tx, sx, shx = gt["translateX"], gt["scaleX"], gt["shearX"]
    ty, shy, sy = gt["translateY"], gt["shearY"], gt["scaleY"]

    # Apply affine transformation to get projected center coordinates
    x = tx + sx * col_center + shx * row_center
    y = ty + shy * col_center + sy * row_center

    # Transform to WGS84
    source_crs = CRS.from_user_input(raster.crs)
    target_crs = CRS.from_epsg(4326)

    if source_crs == target_crs:
        return (x, y, x, y)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)

    return lon, lat, x, y
class RasterTransform(BaseModel):
    """
    Geospatial metadata with CRS and affine transformation.

    The geotransform dictionary must contain these keys:
        - scaleX: X-axis pixel size (meters per pixel)
        - shearX: X-axis rotation/shear
        - translateX: X coordinate of upper-left corner
        - scaleY: Y-axis pixel size (typically negative)
        - shearY: Y-axis rotation/shear  
        - translateY: Y coordinate of upper-left corner

    Attributes:
        crs: Coordinate Reference System (EPSG code or WKT)
        geotransform: Affine transformation parameters
        width: Raster width in pixels
        height: Raster height in pixels

    Example:
        >>> rt = RasterTransform(
        ...     crs="EPSG:32618",
        ...     geotransform={
        ...         'scaleX': 10, 'shearX': 0, 'translateX': 500000,
        ...         'scaleY': -10, 'shearY': 0, 'translateY': 4500000
        ...     },
        ...     width=1024,
        ...     height=1024
        ... )
    """

    crs: str
    geotransform: dict[str, int | float]
    width: int
    height: int

    @model_validator(mode="before")
    @classmethod
    def validate_geotransform(cls, values: dict) -> dict:
        """Validate geotransform structure and values."""
        geotransform = values.get("geotransform")

        # Validate type
        if not isinstance(geotransform, dict):
            raise ValueError(
                f"geotransform must be dict, got {type(geotransform)}"
            )
        
        # Validate all keys are strings
        for key in geotransform.keys():
            if not isinstance(key, str):
                raise ValueError(
                    f"geotransform keys must be strings, got key {key!r} of type {type(key).__name__}"
                )

        # Validate required keys
        missing = REQUIRED_KEYS - set(geotransform.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        # Validate unexpected keys
        extra = set(geotransform.keys()) - REQUIRED_KEYS
        if extra:
            raise ValueError(f"Unexpected keys: {extra}")

        # Validate numeric values
        for key in REQUIRED_KEYS:
            val = geotransform[key]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"'{key}' must be numeric, got {type(val)}"
                )

        # Scale values cannot be zero
        if geotransform["scaleX"] == 0 or geotransform["scaleY"] == 0:
            raise ValueError("Scale values (scaleX, scaleY) cannot be zero")

        return values

    @field_validator("width", "height")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        """Validate positive dimensions."""
        if value <= 0:
            raise ValueError(f"Dimensions must be positive, got {value}")
        return value

    def __str__(self) -> str:
        """Human-readable representation."""
        data = {
            "Property": [
                "CRS", "Width", "Height",
                "scaleX", "shearX", "translateX",
                "scaleY", "shearY", "translateY"
            ],
            "Value": [
                self.crs, self.width, self.height,
                self.geotransform["scaleX"],
                self.geotransform["shearX"],
                self.geotransform["translateX"],
                self.geotransform["scaleY"],
                self.geotransform["shearY"],
                self.geotransform["translateY"]
            ]
        }
        df = pd.DataFrame(data)
        return f"RasterTransform\n{'-' * 30}\n{df.to_string(index=False)}"


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
        """
        # Calculate lon/lat for each request
        points = [rt2lonlat(rt.raster_transform) for rt in self.requestset]
        lon, lat, x, y = zip(*points)

        return pd.DataFrame(
            [
                {
                    "id": meta.id,
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
            "lon": (float, type(None)),
            "lat": (float, type(None)),
            "x": (float, type(None)),
            "y": (float, type(None)),
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
        crs_set: Set[str] = {meta.raster_transform.crs for meta in self.requestset}
        validated_crs: Set[str] = set()

        for crs in crs_set:
            if crs not in validated_crs:
                try:
                    CRS.from_string(crs)
                    validated_crs.add(crs)
                except Exception as e:
                    raise ValueError(f"Invalid CRS format: {crs}") from e

        ids = {meta.id for meta in self.requestset}
        if len(ids) != len(self.requestset):
            raise ValueError("All entries must have unique IDs")

        self._dataframe = self.create_manifests()
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
