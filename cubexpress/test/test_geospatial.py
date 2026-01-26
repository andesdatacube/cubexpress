"""Tests for cubexpress.geospatial module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from cubexpress.core.exceptions import MergeError
from cubexpress.download.merge import merge_tifs


class TestMergeTifs:
    """Tests for merge_tifs function."""

    @pytest.fixture
    def create_test_tif(self, tmp_path):
        """Factory to create test GeoTIFF files."""

        def _create(name: str, data: np.ndarray, transform, crs: str = "EPSG:32630"):
            path = tmp_path / name

            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=0,
            ) as dst:
                dst.write(data, 1)

            return path

        return _create

    def test_merge_two_tiles(self, create_test_tif, tmp_path):
        """Should merge two adjacent tiles."""
        # Create two 10x10 tiles side by side
        data1 = np.ones((10, 10), dtype=np.uint16) * 100
        data2 = np.ones((10, 10), dtype=np.uint16) * 200

        transform1 = from_bounds(0, 0, 100, 100, 10, 10)
        transform2 = from_bounds(100, 0, 200, 100, 10, 10)

        tile1 = create_test_tif("tile1.tif", data1, transform1)
        tile2 = create_test_tif("tile2.tif", data2, transform2)

        output = tmp_path / "merged.tif"
        merge_tifs([tile1, tile2], output)

        assert output.exists()

        with rasterio.open(output) as src:
            merged = src.read(1)
            assert merged.shape == (10, 20)  # Combined width
            assert merged[0, 0] == 100  # From tile1
            assert merged[0, 15] == 200  # From tile2

    def test_empty_input_raises(self, tmp_path):
        """Should raise MergeError for empty input."""
        output = tmp_path / "output.tif"

        with pytest.raises(MergeError, match="empty"):
            merge_tifs([], output)

    def test_creates_parent_directory(self, create_test_tif, tmp_path):
        """Should create parent directory if needed."""
        data = np.ones((10, 10), dtype=np.uint16)
        transform = from_bounds(0, 0, 100, 100, 10, 10)
        tile = create_test_tif("tile.tif", data, transform)

        output = tmp_path / "subdir" / "deep" / "merged.tif"
        merge_tifs([tile], output)

        assert output.exists()

    def test_respects_nodata(self, create_test_tif, tmp_path):
        """Should use specified nodata value."""
        data = np.ones((10, 10), dtype=np.uint16) * 100
        transform = from_bounds(0, 0, 100, 100, 10, 10)
        tile = create_test_tif("tile.tif", data, transform)

        output = tmp_path / "merged.tif"
        merge_tifs([tile], output, nodata=255)

        with rasterio.open(output) as src:
            assert src.nodata == 255


class TestSquareRoi:
    """Tests for _square_roi function (mocked GEE)."""

    def test_creates_polygon(self):
        """Should create ee.Geometry.Polygon."""
        mock_polygon = MagicMock()

        with patch("cubexpress.geospatial.ee") as mock_ee:
            mock_ee.Geometry.Polygon.return_value = mock_polygon

            from cubexpress.core.types import _square_roi

            roi = _square_roi(lon=-0.1, lat=51.5, edge_size=256, scale=10)

            mock_ee.Geometry.Polygon.assert_called_once()
            assert roi == mock_polygon

    def test_tuple_edge_size(self):
        """Should accept tuple edge_size."""
        mock_polygon = MagicMock()

        with patch("cubexpress.geospatial.ee") as mock_ee:
            mock_ee.Geometry.Polygon.return_value = mock_polygon

            from cubexpress.core.types import _square_roi

            roi = _square_roi(lon=-0.1, lat=51.5, edge_size=(256, 128), scale=10)

            mock_ee.Geometry.Polygon.assert_called_once()

    def test_coordinates_in_degrees(self):
        """Polygon coordinates should be in degrees."""
        with patch("cubexpress.geospatial.ee") as mock_ee:
            from cubexpress.core.types import _square_roi

            _square_roi(lon=0.0, lat=0.0, edge_size=100, scale=10)

            # Verify coordinates passed to Polygon
            call_args = mock_ee.Geometry.Polygon.call_args
            coords = call_args[0][0]

            # All coords should be within valid lon/lat ranges
            for coord in coords:
                assert -180 <= coord[0] <= 180
                assert -90 <= coord[1] <= 90
