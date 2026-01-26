"""Tests for downloader module."""

from cubexpress.download.core import _build_request, temp_workspace
from cubexpress.formats.specs import EEFileFormat, ExportFormat, VisualizationOptions


class TestTempWorkspace:
    """Tests for temp_workspace context manager."""

    def test_creates_directory(self):
        with temp_workspace() as tmp_dir:
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()

    def test_cleanup_on_exit(self):
        with temp_workspace() as tmp_dir:
            path = tmp_dir
        assert not path.exists()

    def test_custom_prefix(self):
        with temp_workspace(prefix="mytest_") as tmp_dir:
            assert "mytest_" in str(tmp_dir)


class TestBuildRequest:
    """Tests for _build_request function."""

    def test_default_format(self):
        manifest = {"assetId": "test/image", "bandIds": ["B1"]}
        request = _build_request(manifest, None)
        assert request["fileFormat"] == "GEO_TIFF"

    def test_png_format_with_visualization(self):
        manifest = {"assetId": "test/image", "bandIds": ["B1", "B2", "B3"]}
        vis = VisualizationOptions(bands=["B4", "B3", "B2"], min=0, max=3000)
        fmt = ExportFormat(file_format=EEFileFormat.PNG, visualization=vis)

        request = _build_request(manifest, fmt)

        assert request["fileFormat"] == "PNG"
        assert request["bandIds"] == ["B4", "B3", "B2"]
        assert "visualizationOptions" in request

    def test_npy_format(self):
        manifest = {"assetId": "test/image", "bandIds": ["B1"]}
        fmt = ExportFormat(file_format=EEFileFormat.NPY)

        request = _build_request(manifest, fmt)

        assert request["fileFormat"] == "NPY"

    def test_preserves_manifest_fields(self):
        manifest = {"assetId": "test/image", "bandIds": ["B1"], "grid": {"dimensions": {"width": 256, "height": 256}}}
        request = _build_request(manifest, None)

        assert "grid" in request
        assert request["grid"]["dimensions"]["width"] == 256
