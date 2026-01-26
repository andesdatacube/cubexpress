"""Pre-configured format and visualization presets."""

from __future__ import annotations

from cubexpress.formats.specs import EEFileFormat, ExportFormat, VisualizationOptions


class Formats:
    """Pre-configured export format presets."""

    GEOTIFF = ExportFormat(file_format=EEFileFormat.GEO_TIFF)

    COG = ExportFormat(
        file_format=EEFileFormat.GEO_TIFF, cog_profile={"driver": "COG", "compress": "DEFLATE", "predictor": 2}
    )

    COG_LZW = ExportFormat(
        file_format=EEFileFormat.GEO_TIFF, cog_profile={"driver": "COG", "compress": "LZW", "predictor": 2}
    )

    COG_ZSTD = ExportFormat(
        file_format=EEFileFormat.GEO_TIFF, cog_profile={"driver": "COG", "compress": "ZSTD", "predictor": 2}
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
        """Create PNG format with RGB visualization.

        Args:
            bands: List of exactly 3 band names for RGB
            min_val: Min value(s) for stretching
            max_val: Max value(s) for stretching
            gamma: Optional gamma correction
        """
        if len(bands) != 3:
            raise ValueError(f"RGB visualization requires exactly 3 bands, got {len(bands)}")
        return ExportFormat(
            file_format=EEFileFormat.PNG,
            visualization=VisualizationOptions(
                bands=bands,
                min=min_val,
                max=max_val,
                gamma=gamma,
            ),
        )

    @staticmethod
    def jpeg_rgb(
        bands: list[str],
        min_val: float | list[float] = 0,
        max_val: float | list[float] = 3000,
        gamma: float | None = None,
    ) -> ExportFormat:
        """Create JPEG format with RGB visualization."""
        if len(bands) != 3:
            raise ValueError(f"RGB visualization requires exactly 3 bands, got {len(bands)}")
        return ExportFormat(
            file_format=EEFileFormat.JPEG,
            visualization=VisualizationOptions(
                bands=bands,
                min=min_val,
                max=max_val,
                gamma=gamma,
            ),
        )

    @staticmethod
    def png_palette(
        band: str,
        min_val: float = 0,
        max_val: float = 1,
        palette: list[str] | None = None,
    ) -> ExportFormat:
        """Create PNG format with palette visualization for single band."""
        if palette is None:
            palette = ["0000FF", "00FF00", "FFFF00", "FF0000"]
        return ExportFormat(
            file_format=EEFileFormat.PNG,
            visualization=VisualizationOptions(
                bands=[band],
                min=min_val,
                max=max_val,
                palette=palette,
            ),
        )

    @staticmethod
    def auto_rgb(
        bands: list[str],
        min_val: float | list[float] = 0,
        max_val: float | list[float] = 3000,
        gamma: float | None = None,
    ) -> ExportFormat:
        """Create AUTO_JPEG_PNG format (GEE decides based on transparency).

        Uses JPEG if no transparency, PNG if there is.
        """
        if len(bands) != 3:
            raise ValueError(f"RGB visualization requires exactly 3 bands, got {len(bands)}")
        return ExportFormat(
            file_format=EEFileFormat.AUTO_JPEG_PNG,
            visualization=VisualizationOptions(
                bands=bands,
                min=min_val,
                max=max_val,
                gamma=gamma,
            ),
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
        """NDVI visualization with green palette."""
        return VisualizationOptions(min=min_val, max=max_val, palette=["FF0000", "FFFF00", "00FF00", "006400"])
