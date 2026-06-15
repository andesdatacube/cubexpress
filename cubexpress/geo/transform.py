"""RasterTransform: a rectangle of pixels in a CRS. Central type of the package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RasterTransform:
    """Describes a rectangle of pixels in a coordinate reference system.

    GDAL convention: scale_x > 0, scale_y < 0.
    The origin (translate_x, translate_y) is the upper-left corner.

    shear_x / shear_y default to 0.0 (axis-aligned raster, the common case for
    Sentinel-2, Landsat, etc.). Non-zero shear describes a rotated raster; it is
    supported for completeness but rarely needed in Earth observation.
    """

    crs: str
    translate_x: float
    translate_y: float
    scale_x: float
    scale_y: float
    width: int
    height: int
    shear_x: float = 0.0
    shear_y: float = 0.0

    def __post_init__(self) -> None:
        if not self.crs:
            raise ValueError("crs cannot be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height must be positive, got {self.width}x{self.height}")
        if self.scale_x <= 0:
            raise ValueError(f"scale_x must be > 0, got {self.scale_x}")
        if self.scale_y >= 0:
            raise ValueError(f"scale_y must be < 0 (GDAL convention), got {self.scale_y}")

    def area_pixels(self) -> int:
        """Total number of pixels."""
        return self.width * self.height

    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box (xmin, ymin, xmax, ymax) in the raster CRS.

        Note: assumes axis-aligned (shear = 0). For sheared rasters the bbox is
        an approximation of the unrotated extent.
        """
        xmin = self.translate_x
        ymax = self.translate_y
        xmax = xmin + self.width * self.scale_x
        ymin = ymax + self.height * self.scale_y  # scale_y is negative
        return (xmin, ymin, xmax, ymax)

    def size_meters(self) -> tuple[float, float]:
        """Dimensions in meters: (width, height)."""
        return (self.width * self.scale_x, self.height * abs(self.scale_y))

    def to_ee_dict(self) -> dict[str, float]:
        """Earth Engine compatible affineTransform dictionary."""
        return {
            "scaleX": self.scale_x,
            "shearX": self.shear_x,
            "translateX": self.translate_x,
            "scaleY": self.scale_y,
            "shearY": self.shear_y,
            "translateY": self.translate_y,
        }
