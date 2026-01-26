"""Format specifications, presets and visualization options for cubexpress."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


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

    Note: 'bands' goes to bandIds in the request, not to visualizationOptions.
    The GEE API expects exactly 3 bands for RGB visualization.

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

        Note: 'bands' is NOT included — it must be set as bandIds in the request.
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
        return FORMAT_EXTENSIONS[self.file_format]

    @property
    def needs_visualization(self) -> bool:
        return self.file_format in VISUALIZATION_FORMATS


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class Formats:
    """Pre-configured export format presets."""

    GEOTIFF = ExportFormat(file_format=EEFileFormat.GEO_TIFF)
    COG = ExportFormat(
        file_format=EEFileFormat.GEO_TIFF,
        cog_profile={"driver": "COG", "compress": "DEFLATE", "predictor": 2},
    )
    COG_LZW = ExportFormat(
        file_format=EEFileFormat.GEO_TIFF,
        cog_profile={"driver": "COG", "compress": "LZW", "predictor": 2},
    )
    COG_ZSTD = ExportFormat(
        file_format=EEFileFormat.GEO_TIFF,
        cog_profile={"driver": "COG", "compress": "ZSTD", "predictor": 2},
    )
    NPY = ExportFormat(file_format=EEFileFormat.NPY)
    NUMPY = ExportFormat(file_format=EEFileFormat.NUMPY_NDARRAY)

    @staticmethod
    def png_rgb(
        bands: list[str],
        min_val: float | list[float] = 0,
        max_val: float | list[float] = 3000,
        gamma: float | None = None,
    ) -> ExportFormat:
        """PNG with RGB visualization. Requires exactly 3 bands."""
        if len(bands) != 3:
            raise ValueError(f"RGB visualization requires exactly 3 bands, got {len(bands)}")
        return ExportFormat(
            file_format=EEFileFormat.PNG,
            visualization=VisualizationOptions(bands=bands, min=min_val, max=max_val, gamma=gamma),
        )

    @staticmethod
    def jpeg_rgb(
        bands: list[str],
        min_val: float | list[float] = 0,
        max_val: float | list[float] = 3000,
        gamma: float | None = None,
    ) -> ExportFormat:
        """JPEG with RGB visualization. Requires exactly 3 bands."""
        if len(bands) != 3:
            raise ValueError(f"RGB visualization requires exactly 3 bands, got {len(bands)}")
        return ExportFormat(
            file_format=EEFileFormat.JPEG,
            visualization=VisualizationOptions(bands=bands, min=min_val, max=max_val, gamma=gamma),
        )

    @staticmethod
    def png_palette(
        band: str,
        min_val: float = 0,
        max_val: float = 1,
        palette: list[str] | None = None,
    ) -> ExportFormat:
        """PNG with palette visualization for single band."""
        if palette is None:
            palette = ["0000FF", "00FF00", "FFFF00", "FF0000"]
        return ExportFormat(
            file_format=EEFileFormat.PNG,
            visualization=VisualizationOptions(bands=[band], min=min_val, max=max_val, palette=palette),
        )

    @staticmethod
    def auto_rgb(
        bands: list[str],
        min_val: float | list[float] = 0,
        max_val: float | list[float] = 3000,
        gamma: float | None = None,
    ) -> ExportFormat:
        """AUTO_JPEG_PNG: GEE decides format based on transparency."""
        if len(bands) != 3:
            raise ValueError(f"RGB visualization requires exactly 3 bands, got {len(bands)}")
        return ExportFormat(
            file_format=EEFileFormat.AUTO_JPEG_PNG,
            visualization=VisualizationOptions(bands=bands, min=min_val, max=max_val, gamma=gamma),
        )


class VisPresets:
    """Pre-configured visualization presets for common sensors."""

    @staticmethod
    def s2_truecolor(min_val: int = 0, max_val: int = 3000) -> VisualizationOptions:
        """Sentinel-2 true color (B4, B3, B2)."""
        return VisualizationOptions(bands=["B4", "B3", "B2"], min=min_val, max=max_val)

    @staticmethod
    def s2_falsecolor(min_val: int = 0, max_val: int = 5000) -> VisualizationOptions:
        """Sentinel-2 false color infrared (B8, B4, B3)."""
        return VisualizationOptions(bands=["B8", "B4", "B3"], min=min_val, max=max_val)

    @staticmethod
    def s2_agriculture(min_val: int = 0, max_val: int = 5000) -> VisualizationOptions:
        """Sentinel-2 agriculture (B11, B8, B2)."""
        return VisualizationOptions(bands=["B11", "B8", "B2"], min=min_val, max=max_val)

    @staticmethod
    def landsat_truecolor_toa(min_val: float = 0, max_val: float = 0.4) -> VisualizationOptions:
        """Landsat TOA true color (B4, B3, B2)."""
        return VisualizationOptions(bands=["B4", "B3", "B2"], min=min_val, max=max_val)

    @staticmethod
    def landsat_truecolor_sr(min_val: float = 0, max_val: float = 0.3) -> VisualizationOptions:
        """Landsat SR/BOA true color (SR_B4, SR_B3, SR_B2)."""
        return VisualizationOptions(bands=["SR_B4", "SR_B3", "SR_B2"], min=min_val, max=max_val)

    @staticmethod
    def ndvi(min_val: float = -1, max_val: float = 1) -> VisualizationOptions:
        """NDVI with green palette."""
        return VisualizationOptions(min=min_val, max=max_val, palette=["FF0000", "FFFF00", "00FF00", "006400"])
