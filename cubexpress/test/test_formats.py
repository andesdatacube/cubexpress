"""Tests for formats module."""

import pytest

from cubexpress.formats.presets import VisPresets
from cubexpress.formats.specs import FORMAT_EXTENSIONS, EEFileFormat, ExportFormat, Formats, VisualizationOptions


class TestEEFileFormat:
    """Tests for EEFileFormat enum."""

    def test_geotiff_value(self):
        assert EEFileFormat.GEO_TIFF.value == "GEO_TIFF"

    def test_png_value(self):
        assert EEFileFormat.PNG.value == "PNG"

    def test_all_formats_have_extensions(self):
        for fmt in EEFileFormat:
            assert fmt in FORMAT_EXTENSIONS


class TestVisualizationOptions:
    """Tests for VisualizationOptions dataclass."""

    def test_basic_creation(self):
        vis = VisualizationOptions(bands=["B4", "B3", "B2"], min=0, max=3000)
        assert vis.bands == ["B4", "B3", "B2"]
        assert vis.min == 0
        assert vis.max == 3000

    def test_to_ee_dict_with_ranges(self):
        vis = VisualizationOptions(bands=["B4", "B3", "B2"], min=0, max=3000)
        ee_dict = vis.to_ee_dict()
        assert "ranges" in ee_dict
        assert len(ee_dict["ranges"]) == 3
        assert ee_dict["ranges"][0] == {"min": 0, "max": 3000}

    def test_to_ee_dict_with_palette(self):
        vis = VisualizationOptions(min=0, max=1, palette=["blue", "green", "red"])
        ee_dict = vis.to_ee_dict()
        assert ee_dict["palette"] == ["blue", "green", "red"]

    def test_to_ee_dict_excludes_bands(self):
        """Bands should NOT be in visualizationOptions (goes to bandIds)."""
        vis = VisualizationOptions(bands=["B4", "B3", "B2"], min=0, max=3000)
        ee_dict = vis.to_ee_dict()
        assert "bands" not in ee_dict

    def test_get_bands(self):
        vis = VisualizationOptions(bands=["B4", "B3", "B2"])
        assert vis.get_bands() == ["B4", "B3", "B2"]

    def test_per_band_min_max(self):
        vis = VisualizationOptions(bands=["B4", "B3", "B2"], min=[0, 100, 200], max=[3000, 3500, 4000])
        ee_dict = vis.to_ee_dict()
        assert ee_dict["ranges"][0] == {"min": 0, "max": 3000}
        assert ee_dict["ranges"][1] == {"min": 100, "max": 3500}
        assert ee_dict["ranges"][2] == {"min": 200, "max": 4000}


class TestExportFormat:
    """Tests for ExportFormat dataclass."""

    def test_default_format(self):
        fmt = ExportFormat()
        assert fmt.file_format == EEFileFormat.GEO_TIFF

    def test_extension_property(self):
        fmt = ExportFormat(file_format=EEFileFormat.PNG)
        assert fmt.extension == ".png"

    def test_needs_visualization_png(self):
        fmt = ExportFormat(file_format=EEFileFormat.PNG)
        assert fmt.needs_visualization is True

    def test_needs_visualization_geotiff(self):
        fmt = ExportFormat(file_format=EEFileFormat.GEO_TIFF)
        assert fmt.needs_visualization is False

    def test_visualization_on_geotiff_raises(self):
        vis = VisualizationOptions(bands=["B4", "B3", "B2"], min=0, max=3000)
        with pytest.raises(ValueError, match="not supported"):
            ExportFormat(file_format=EEFileFormat.GEO_TIFF, visualization=vis)

    def test_cog_profile_on_png_raises(self):
        with pytest.raises(ValueError, match="COG profile"):
            ExportFormat(file_format=EEFileFormat.PNG, cog_profile={"compress": "DEFLATE"})


class TestFormatsPresets:
    """Tests for Formats preset class."""

    def test_geotiff_preset(self):
        assert Formats.GEOTIFF.file_format == EEFileFormat.GEO_TIFF
        assert Formats.GEOTIFF.cog_profile is None

    def test_cog_preset(self):
        assert Formats.COG.file_format == EEFileFormat.GEO_TIFF
        assert Formats.COG.cog_profile is not None
        assert Formats.COG.cog_profile["driver"] == "COG"

    def test_npy_preset(self):
        assert Formats.NPY.file_format == EEFileFormat.NPY

    def test_png_rgb_factory(self):
        fmt = Formats.png_rgb(bands=["B4", "B3", "B2"], min_val=0, max_val=3000)
        assert fmt.file_format == EEFileFormat.PNG
        assert fmt.visualization is not None
        assert fmt.visualization.bands == ["B4", "B3", "B2"]

    def test_png_rgb_requires_3_bands(self):
        with pytest.raises(ValueError, match="exactly 3 bands"):
            Formats.png_rgb(bands=["B4", "B3"], min_val=0, max_val=3000)

    def test_jpeg_rgb_factory(self):
        fmt = Formats.jpeg_rgb(bands=["B4", "B3", "B2"], min_val=0, max_val=3000)
        assert fmt.file_format == EEFileFormat.JPEG

    def test_png_palette_factory(self):
        fmt = Formats.png_palette(band="NDVI", min_val=-1, max_val=1)
        assert fmt.file_format == EEFileFormat.PNG
        assert fmt.visualization.palette is not None

    def test_auto_rgb_factory(self):
        fmt = Formats.auto_rgb(bands=["B4", "B3", "B2"], min_val=0, max_val=3000)
        assert fmt.file_format == EEFileFormat.AUTO_JPEG_PNG


class TestVisPresets:
    """Tests for VisPresets class."""

    def test_s2_truecolor(self):
        vis = VisPresets.s2_truecolor()
        assert vis.bands == ["B4", "B3", "B2"]
        assert vis.min == 0
        assert vis.max == 3000

    def test_s2_truecolor_custom_range(self):
        vis = VisPresets.s2_truecolor(min_val=100, max_val=5000)
        assert vis.min == 100
        assert vis.max == 5000

    def test_s2_falsecolor(self):
        vis = VisPresets.s2_falsecolor()
        assert vis.bands == ["B8", "B4", "B3"]

    def test_ndvi_has_palette(self):
        vis = VisPresets.ndvi()
        assert vis.palette is not None
        assert len(vis.palette) == 4
