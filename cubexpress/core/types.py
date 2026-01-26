from __future__ import annotations

from typing import Any, Final

import ee
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from pyproj import CRS, Transformer

from cubexpress.core.exceptions import ValidationError

REQUIRED_KEYS: Final[set[str]] = {
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
        Tuple of (lon, lat, x, y) in WGS84
    """
    col_center = raster.width / 2.0
    row_center = raster.height / 2.0

    gt = raster.geotransform
    tx, sx, shx = gt["translateX"], gt["scaleX"], gt["shearX"]
    ty, shy, sy = gt["translateY"], gt["shearY"], gt["scaleY"]

    x = tx + sx * col_center + shx * row_center
    y = ty + shy * col_center + sy * row_center

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

    Attributes:
        crs: Coordinate Reference System (EPSG code or WKT)
        geotransform: Affine transformation parameters
        width: Raster width in pixels
        height: Raster height in pixels
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

        if not isinstance(geotransform, dict):
            raise ValidationError(f"geotransform must be dict, got {type(geotransform)}")

        for key in geotransform:
            if not isinstance(key, str):
                raise ValidationError(f"geotransform keys must be strings, got {type(key)}")

        missing = REQUIRED_KEYS - set(geotransform.keys())
        if missing:
            raise ValidationError(f"Missing required keys: {missing}")

        extra = set(geotransform.keys()) - REQUIRED_KEYS
        if extra:
            raise ValidationError(f"Unexpected keys: {extra}")

        for key in REQUIRED_KEYS:
            val = geotransform[key]
            if not isinstance(val, int | float):
                raise ValidationError(f"'{key}' must be numeric, got {type(val)}")

        if geotransform["scaleX"] == 0 or geotransform["scaleY"] == 0:
            raise ValidationError("Scale values cannot be zero")

        return values

    @field_validator("width", "height")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        """Validate positive dimensions."""
        if value <= 0:
            raise ValidationError(f"Dimensions must be positive, got {value}")
        return value


class Request(BaseModel):
    """
    Single Earth Engine request with raster transform.

    Attributes:
        id: Unique identifier for the request
        raster_transform: Geospatial metadata
        image: EE Image or asset ID
        bands: List of band names to export
    """

    id: str
    raster_transform: RasterTransform
    image: Any
    bands: list[str]
    _expression_key: str = None

    @model_validator(mode="after")
    def validate_image(self):
        if isinstance(self.image, ee.Image):
            self.image = self.image.serialize()
            self._expression_key = "expression"
        elif isinstance(self.image, str) and self.image.strip().startswith("{"):
            self._expression_key = "expression"
        else:
            self._expression_key = "assetId"
        return self


class RequestSet(BaseModel):
    """
    Container for multiple Request instances with bulk validation.

    Attributes:
        requestset: List of Request objects
    """

    requestset: list[Request]
    _dataframe: pd.DataFrame | None = None

    def create_manifests(self) -> pd.DataFrame:
        """Export raster metadata to pandas DataFrame."""
        points = [rt2lonlat(rt.raster_transform) for rt in self.requestset]
        lon, lat, x, y = zip(*points, strict=False)

        return pd.DataFrame(
            [
                {
                    "id": meta.id,
                    "lon": lon[i],
                    "lat": lat[i],
                    "x": x[i],
                    "y": y[i],
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
                for i, meta in enumerate(self.requestset)
            ]
        )

    def _validate_columns(self) -> None:
        """Validate required columns exist."""
        required = {
            "id",
            "lon",
            "lat",
            "x",
            "y",
            "crs",
            "width",
            "height",
            "geotransform",
            "scale_x",
            "scale_y",
            "manifest",
            "outname",
        }
        missing = required - set(self._dataframe.columns)
        if missing:
            raise ValidationError(f"Missing columns: {missing}")

    def _validate_types(self) -> None:
        """Validate column data types."""
        type_map = {
            "id": str,
            "lon": (float, type(None)),
            "lat": (float, type(None)),
            "x": (float, type(None)),
            "y": (float, type(None)),
            "crs": str,
            "width": int,
            "height": int,
            "geotransform": dict,
            "scale_x": int | float,
            "scale_y": int | float,
            "manifest": dict,
            "outname": str,
        }

        for col, expected in type_map.items():
            for i, val in enumerate(self._dataframe[col]):
                if isinstance(expected, tuple):
                    if not any(isinstance(val, t) for t in expected):
                        raise ValidationError(f"Column '{col}' row {i}: " f"expected {expected}, got {type(val)}")
                else:
                    if not isinstance(val, expected):
                        raise ValidationError(f"Column '{col}' row {i}: " f"expected {expected}, got {type(val)}")

    def _validate_manifest_structure(self) -> None:
        """Validate manifest dictionary structure."""
        for i, row in self._dataframe.iterrows():
            manifest = row["manifest"]

            for key in ["fileFormat", "bandIds", "grid"]:
                if key not in manifest:
                    raise ValidationError(f"Missing '{key}' in manifest for row {i}")

            if not any(k in manifest for k in ["assetId", "expression"]):
                raise ValidationError(f"Manifest row {i} needs 'assetId' or 'expression'")

            self._validate_grid_structure(manifest["grid"], i)

    def _validate_grid_structure(self, grid: dict, row_idx: int) -> None:
        """Validate grid structure within manifest."""
        for key in ["dimensions", "affineTransform", "crsCode"]:
            if key not in grid:
                raise ValidationError(f"Missing '{key}' in grid for row {row_idx}")

        dims = grid["dimensions"]
        for dim in ["width", "height"]:
            if dim not in dims:
                raise ValidationError(f"Missing '{dim}' in dimensions for row {row_idx}")
            if not isinstance(dims[dim], int) or dims[dim] <= 0:
                raise ValidationError(f"'{dim}' must be positive integer, row {row_idx}")

        aff = grid["affineTransform"]
        for key in REQUIRED_KEYS:
            if key not in aff:
                raise ValidationError(f"Missing '{key}' in affineTransform for row {row_idx}")
            if not isinstance(aff[key], int | float):
                raise ValidationError(f"'{key}' must be numeric, row {row_idx}")

    def _validate_dataframe_schema(self) -> None:
        """Orchestrate all dataframe validations."""
        self._validate_columns()
        self._validate_types()
        self._validate_manifest_structure()

    @model_validator(mode="after")
    def validate_metadata(self) -> RequestSet:
        """Validate CRS consistency and unique IDs."""
        crs_set = {meta.raster_transform.crs for meta in self.requestset}

        for crs in crs_set:
            try:
                CRS.from_string(crs)
            except Exception as e:
                raise ValidationError(f"Invalid CRS: {crs}") from e

        ids = [meta.id for meta in self.requestset]
        if len(set(ids)) != len(ids):
            raise ValidationError("Request IDs must be unique")

        self._dataframe = self.create_manifests()
        self._validate_dataframe_schema()

        return self

    def __repr__(self) -> str:
        return f"RequestSet({len(self.requestset)} entries)"
