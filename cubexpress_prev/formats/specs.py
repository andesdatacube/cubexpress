"""Format specifications and data classes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final


class EEFileFormat(str, Enum):
    """Earth Engine supported file formats for getPixels/computePixels."""

    GEO_TIFF = "GEO_TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    AUTO_JPEG_PNG = "AUTO_JPEG_PNG"
    NPY = "NPY"
    NUMPY_NDARRAY = "NUMPY_NDARRAY"


FORMAT_EXTENSIONS: Final[dict[EEFileFormat, str]] = {
    EEFileFormat.GEO_TIFF: ".tif",
    EEFileFormat.PNG: ".png",
    EEFileFormat.JPEG: ".jpg",
    EEFileFormat.AUTO_JPEG_PNG: ".png",
    EEFileFormat.NPY: ".npy",
    EEFileFormat.NUMPY_NDARRAY: ".npy",
}

VISUALIZATION_FORMATS: Final[set[EEFileFormat]] = {
    EEFileFormat.PNG,
    EEFileFormat.JPEG,
    EEFileFormat.AUTO_JPEG_PNG,
}

BINARY_FORMATS: Final[set[EEFileFormat]] = {
    EEFileFormat.GEO_TIFF,
    EEFileFormat.PNG,
    EEFileFormat.JPEG,
    EEFileFormat.AUTO_JPEG_PNG,
    EEFileFormat.NPY,
}


@dataclass
class VisualizationOptions:
    """Visualization parameters for RGB output formats (PNG/JPEG).

    Note: 'bands' is handled separately - it goes to bandIds in the request,
    not to visualizationOptions. The GEE API expects exactly 3 bands for
    RGB visualization.

    Attributes:
        bands: List of 3 band names for RGB visualization (goes to bandIds)
        min: Minimum value(s) for stretching (single value or per-band list)
        max: Maximum value(s) for stretching (single value or per-band list)
        gain: Gain multiplier(s) (alternative to min/max)
        bias: Bias offset(s) (alternative to min/max)
        gamma: Gamma correction value(s)
        palette: Color palette for single-band visualization (list of hex colors)
    """

    bands: list[str] | None = None
    min: float | list[float] | None = None
    max: float | list[float] | None = None
    gain: float | list[float] | None = None
    bias: float | list[float] | None = None
    gamma: float | list[float] | None = None
    palette: list[str] | None = None

    def to_ee_dict(self) -> dict[str, Any]:
        """Convert to Earth Engine visualizationOptions dict.

        Note: 'bands' is NOT included here - it must be set as bandIds
        in the main request.
        """
        opts: dict[str, Any] = {}

        if self.min is not None or self.max is not None:
            if isinstance(self.min, list):
                n_bands = len(self.min)
            elif isinstance(self.max, list):
                n_bands = len(self.max)
            else:
                n_bands = 3 if self.bands and len(self.bands) == 3 else 1

            ranges = []
            for i in range(n_bands):
                mn = self.min[i] if isinstance(self.min, list) else (self.min or 0)
                mx = self.max[i] if isinstance(self.max, list) else (self.max or 1)
                ranges.append({"min": mn, "max": mx})
            opts["ranges"] = ranges

        if self.gain is not None:
            opts["gain"] = self.gain if isinstance(self.gain, list) else [self.gain]
        if self.bias is not None:
            opts["bias"] = self.bias if isinstance(self.bias, list) else [self.bias]

        if self.gamma is not None:
            opts["gamma"] = self.gamma
        if self.palette:
            opts["palette"] = self.palette

        return opts

    def get_bands(self) -> list[str] | None:
        """Get bands for bandIds in request."""
        return self.bands


@dataclass
class ExportFormat:
    """Complete export format specification.

    Attributes:
        file_format: Earth Engine file format
        visualization: Visualization options (for PNG/JPEG)
        cog_profile: COG conversion profile (for GeoTIFF post-processing)
    """

    file_format: EEFileFormat = EEFileFormat.GEO_TIFF
    visualization: VisualizationOptions | None = None
    cog_profile: dict[str, Any] | None = None

    def __post_init__(self):
        if self.visualization and self.file_format not in VISUALIZATION_FORMATS:
            raise ValueError(
                f"Visualization options not supported for {self.file_format}. "
                f"Use one of: {[f.value for f in VISUALIZATION_FORMATS]}"
            )

        if self.cog_profile and self.file_format != EEFileFormat.GEO_TIFF:
            raise ValueError("COG profile only applies to GEO_TIFF format")

    @property
    def extension(self) -> str:
        """Get file extension for this format."""
        return FORMAT_EXTENSIONS[self.file_format]

    @property
    def needs_visualization(self) -> bool:
        """Check if format requires visualization options."""
        return self.file_format in VISUALIZATION_FORMATS
