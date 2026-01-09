"""Tests for cubexpress.tiling module."""

from __future__ import annotations

import pytest

from cubexpress.tiling import (
    TilingStrategy,
    calculate_tiling_from_error,
    generate_tile_manifests,
    get_manifest_group_key,
)


class TestGetManifestGroupKey:
    """Tests for get_manifest_group_key."""

    def test_same_manifest_same_key(self, sample_manifest):
        """Same manifest should produce same key."""
        key1 = get_manifest_group_key(sample_manifest)
        key2 = get_manifest_group_key(sample_manifest)
        assert key1 == key2

    def test_different_bands_different_key(self, sample_manifest):
        """Different bands should produce different key."""
        manifest2 = sample_manifest.copy()
        manifest2["bandIds"] = ["B1", "B2"]
        
        key1 = get_manifest_group_key(sample_manifest)
        key2 = get_manifest_group_key(manifest2)
        assert key1 != key2

    def test_different_dimensions_different_key(self, sample_manifest):
        """Different dimensions should produce different key."""
        manifest2 = {
            **sample_manifest,
            "grid": {
                **sample_manifest["grid"],
                "dimensions": {"width": 512, "height": 512}
            }
        }
        
        key1 = get_manifest_group_key(sample_manifest)
        key2 = get_manifest_group_key(manifest2)
        assert key1 != key2

    def test_key_contains_bands_and_dimensions(self, sample_manifest):
        """Key should be tuple of (bands, width, height)."""
        key = get_manifest_group_key(sample_manifest)
        
        assert isinstance(key, tuple)
        assert len(key) == 3
        bands, width, height = key
        assert isinstance(bands, tuple)
        assert width == 256
        assert height == 256


class TestCalculateTilingFromError:
    """Tests for calculate_tiling_from_error."""

    def test_parses_gee_error_format(self):
        """Should parse standard GEE error message."""
        error = "Total request size (75000000 bytes) must be less than or equal to 50331648 bytes"
        
        strategy = calculate_tiling_from_error(error, width=2500, height=2500)
        
        assert isinstance(strategy, TilingStrategy)
        assert strategy.total_tiles >= 2
        assert not strategy.is_single_tile

    def test_higher_ratio_more_tiles(self):
        """Higher byte ratio should produce more tiles."""
        # Small overflow
        error1 = "Total request size (60000000 bytes) must be less than or equal to 50000000 bytes"
        strategy1 = calculate_tiling_from_error(error1, width=1000, height=1000)
        
        # Large overflow (4x)
        error2 = "Total request size (200000000 bytes) must be less than or equal to 50000000 bytes"
        strategy2 = calculate_tiling_from_error(error2, width=1000, height=1000)
        
        assert strategy2.total_tiles >= strategy1.total_tiles

    def test_fallback_when_unparseable(self):
        """Should use fallback ratio when can't parse error."""
        error = "Some random error message without bytes"
        
        strategy = calculate_tiling_from_error(error, width=1000, height=1000)
        
        # Should still return valid strategy with default ratio
        assert isinstance(strategy, TilingStrategy)
        assert strategy.total_tiles >= 2

    def test_tile_dimensions_cover_original(self):
        """Tile dimensions × count should cover original."""
        error = "Total request size (100000000 bytes) must be less than or equal to 50000000 bytes"
        
        strategy = calculate_tiling_from_error(error, width=2000, height=2000)
        
        # Tiles should cover full extent
        assert strategy.tile_width * strategy.n_tiles_x >= 2000
        assert strategy.tile_height * strategy.n_tiles_y >= 2000


class TestTilingStrategy:
    """Tests for TilingStrategy dataclass."""

    def test_is_single_tile_true(self):
        """is_single_tile should be True when total_tiles=1."""
        strategy = TilingStrategy(
            n_tiles_x=1, n_tiles_y=1,
            tile_width=1000, tile_height=1000,
            total_tiles=1
        )
        assert strategy.is_single_tile

    def test_is_single_tile_false(self):
        """is_single_tile should be False when total_tiles>1."""
        strategy = TilingStrategy(
            n_tiles_x=2, n_tiles_y=2,
            tile_width=500, tile_height=500,
            total_tiles=4
        )
        assert not strategy.is_single_tile


class TestGenerateTileManifests:
    """Tests for generate_tile_manifests."""

    def test_single_tile_returns_original(self, sample_manifest):
        """Single tile strategy returns original manifest."""
        strategy = TilingStrategy(
            n_tiles_x=1, n_tiles_y=1,
            tile_width=256, tile_height=256,
            total_tiles=1
        )
        
        tiles = generate_tile_manifests(sample_manifest, strategy)
        
        assert len(tiles) == 1
        assert tiles[0] == sample_manifest

    def test_2x2_creates_4_tiles(self, sample_manifest):
        """2x2 strategy should create 4 tiles."""
        strategy = TilingStrategy(
            n_tiles_x=2, n_tiles_y=2,
            tile_width=128, tile_height=128,
            total_tiles=4
        )
        
        tiles = generate_tile_manifests(sample_manifest, strategy)
        
        assert len(tiles) == 4

    def test_tile_dimensions_updated(self, sample_manifest):
        """Tiles should have updated dimensions."""
        strategy = TilingStrategy(
            n_tiles_x=2, n_tiles_y=2,
            tile_width=128, tile_height=128,
            total_tiles=4
        )
        
        tiles = generate_tile_manifests(sample_manifest, strategy)
        
        for tile in tiles:
            assert tile["grid"]["dimensions"]["width"] <= 128
            assert tile["grid"]["dimensions"]["height"] <= 128

    def test_tile_origins_unique(self, sample_manifest):
        """Each tile should have unique origin."""
        strategy = TilingStrategy(
            n_tiles_x=2, n_tiles_y=2,
            tile_width=128, tile_height=128,
            total_tiles=4
        )
        
        tiles = generate_tile_manifests(sample_manifest, strategy)
        
        origins = [
            (t["grid"]["affineTransform"]["translateX"],
             t["grid"]["affineTransform"]["translateY"])
            for t in tiles
        ]
        assert len(set(origins)) == len(origins)

    def test_original_manifest_unchanged(self, sample_manifest):
        """Original manifest should not be modified."""
        from copy import deepcopy
        original_copy = deepcopy(sample_manifest)
        
        strategy = TilingStrategy(
            n_tiles_x=2, n_tiles_y=2,
            tile_width=128, tile_height=128,
            total_tiles=4
        )
        
        _ = generate_tile_manifests(sample_manifest, strategy)
        
        assert sample_manifest == original_copy

    def test_tiles_preserve_other_fields(self, sample_manifest):
        """Tiles should preserve assetId, bandIds, etc."""
        strategy = TilingStrategy(
            n_tiles_x=2, n_tiles_y=2,
            tile_width=128, tile_height=128,
            total_tiles=4
        )
        
        tiles = generate_tile_manifests(sample_manifest, strategy)
        
        for tile in tiles:
            assert tile["assetId"] == sample_manifest["assetId"]
            assert tile["bandIds"] == sample_manifest["bandIds"]
            assert tile["fileFormat"] == sample_manifest["fileFormat"]
            assert tile["grid"]["crsCode"] == sample_manifest["grid"]["crsCode"]