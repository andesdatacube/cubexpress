"""Tests for cubexpress.geospatial module."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from cubexpress.exceptions import MergeError, TilingError
from cubexpress.geospatial import calculate_cell_size, merge_tifs, quadsplit_manifest
from unittest.mock import patch, MagicMock


class TestQuadsplitManifest:
    """Tests for quadsplit_manifest function."""

    def test_power_1_creates_4_tiles(self, sample_manifest):
        """Power=1 should create 2x2=4 tiles."""
        tiles = quadsplit_manifest(
            manifest=sample_manifest,
            cell_width=128,
            cell_height=128,
            power=1,
        )
        
        assert len(tiles) == 4

    def test_power_2_creates_16_tiles(self, sample_manifest):
        """Power=2 should create 4x4=16 tiles."""
        tiles = quadsplit_manifest(
            manifest=sample_manifest,
            cell_width=64,
            cell_height=64,
            power=2,
        )
        
        assert len(tiles) == 16

    def test_tile_dimensions_updated(self, sample_manifest):
        """Each tile should have updated dimensions."""
        tiles = quadsplit_manifest(
            manifest=sample_manifest,
            cell_width=128,
            cell_height=128,
            power=1,
        )
        
        for tile in tiles:
            assert tile["grid"]["dimensions"]["width"] == 128
            assert tile["grid"]["dimensions"]["height"] == 128

    def test_tiles_have_different_origins(self, sample_manifest):
        """Each tile should have unique translateX/Y."""
        tiles = quadsplit_manifest(
            manifest=sample_manifest,
            cell_width=128,
            cell_height=128,
            power=1,
        )
        
        origins = [
            (t["grid"]["affineTransform"]["translateX"], 
             t["grid"]["affineTransform"]["translateY"])
            for t in tiles
        ]
        
        # All origins should be unique
        assert len(set(origins)) == len(origins)

    def test_original_manifest_unchanged(self, sample_manifest):
        """Original manifest should not be modified."""
        original_copy = deepcopy(sample_manifest)
        
        _ = quadsplit_manifest(
            manifest=sample_manifest,
            cell_width=128,
            cell_height=128,
            power=1,
        )
        
        assert sample_manifest == original_copy

    def test_tiles_cover_original_extent(self, sample_manifest):
        """Tiles should cover the same area as original."""
        original_width = sample_manifest["grid"]["dimensions"]["width"]
        scale = sample_manifest["grid"]["affineTransform"]["scaleX"]
        
        tiles = quadsplit_manifest(
            manifest=sample_manifest,
            cell_width=128,
            cell_height=128,
            power=1,
        )
        
        # Check that tiles span the full width
        x_coords = [t["grid"]["affineTransform"]["translateX"] for t in tiles]
        min_x = min(x_coords)
        max_x = max(x_coords) + 128 * scale
        
        expected_extent = original_width * scale
        actual_extent = max_x - min_x
        
        assert abs(actual_extent - expected_extent) < 0.01


class TestCalculateCellSize:
    """Tests for calculate_cell_size function."""

    def test_parses_pixel_limit_error(self):
        """Should parse 'Pixel limit exceeded' error."""
        error_msg = "Total request size (67108864 bytes) exceeds limit (50331648 bytes)"
        
        cell_w, cell_h, power = calculate_cell_size(error_msg, size=256)
        
        assert power >= 1
        assert cell_w == 256 // (2 ** power)
        assert cell_h == 256 // (2 ** power)

    def test_power_increases_with_ratio(self):
        """Higher ratio should result in higher power."""
        # Small overflow
        error1 = "Total request size (60000000 bytes) exceeds limit (50000000 bytes)"
        _, _, power1 = calculate_cell_size(error1, size=256)
        
        # Large overflow
        error2 = "Total request size (200000000 bytes) exceeds limit (50000000 bytes)"
        _, _, power2 = calculate_cell_size(error2, size=256)
        
        assert power2 >= power1

    def test_invalid_error_message_raises(self):
        """Should raise TilingError for unparseable message."""
        with pytest.raises(TilingError, match="Cannot parse"):
            calculate_cell_size("Some random error", size=256)

    def test_cell_size_divides_original(self):
        """Cell size should evenly divide original size."""
        error_msg = "Total request size (100000000 bytes) exceeds limit (50000000 bytes)"
        
        cell_w, cell_h, power = calculate_cell_size(error_msg, size=512)
        
        assert 512 % cell_w == 0
        assert 512 % cell_h == 0


class TestMergeTifs:
    """Tests for merge_tifs function."""

    @pytest.fixture
    def create_test_tif(self, tmp_path):
        """Factory to create test GeoTIFF files."""
        def _create(name: str, data: np.ndarray, transform, crs: str = "EPSG:32630"):
            path = tmp_path / name
            
            with rasterio.open(
                path, 'w',
                driver='GTiff',
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
        
        with patch('cubexpress.geospatial.ee') as mock_ee:
            mock_ee.Geometry.Polygon.return_value = mock_polygon
            
            from cubexpress.geospatial import _square_roi
            roi = _square_roi(lon=-0.1, lat=51.5, edge_size=256, scale=10)
            
            mock_ee.Geometry.Polygon.assert_called_once()
            assert roi == mock_polygon

    def test_tuple_edge_size(self):
        """Should accept tuple edge_size."""
        mock_polygon = MagicMock()
        
        with patch('cubexpress.geospatial.ee') as mock_ee:
            mock_ee.Geometry.Polygon.return_value = mock_polygon
            
            from cubexpress.geospatial import _square_roi
            roi = _square_roi(lon=-0.1, lat=51.5, edge_size=(256, 128), scale=10)
            
            mock_ee.Geometry.Polygon.assert_called_once()

    def test_coordinates_in_degrees(self):
        """Polygon coordinates should be in degrees."""
        with patch('cubexpress.geospatial.ee') as mock_ee:
            from cubexpress.geospatial import _square_roi
            _square_roi(lon=0.0, lat=0.0, edge_size=100, scale=10)
            
            # Verify coordinates passed to Polygon
            call_args = mock_ee.Geometry.Polygon.call_args
            coords = call_args[0][0]  
            
            # All coords should be within valid lon/lat ranges
            for coord in coords:
                assert -180 <= coord[0] <= 180
                assert -90 <= coord[1] <= 90