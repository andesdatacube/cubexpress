"""Tests for cubexpress.cache module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCacheKey:
    """Tests for _cache_key function."""

    def test_deterministic(self):
        """Same inputs should produce same cache key."""
        from cubexpress.cache import _cache_key
        
        with patch('cubexpress.cache.CACHE_DIR', Path('/tmp/test_cache')):
            key1 = _cache_key(
                lon=-0.1, lat=51.5, edge_size=256, 
                scale=10, collection="TEST/COLLECTION"
            )
            key2 = _cache_key(
                lon=-0.1, lat=51.5, edge_size=256, 
                scale=10, collection="TEST/COLLECTION"
            )
            
            assert key1 == key2

    def test_different_coords_different_key(self):
        """Different coordinates should produce different keys."""
        from cubexpress.cache import _cache_key
        
        with patch('cubexpress.cache.CACHE_DIR', Path('/tmp/test_cache')):
            key1 = _cache_key(
                lon=-0.1, lat=51.5, edge_size=256, 
                scale=10, collection="TEST"
            )
            key2 = _cache_key(
                lon=-0.2, lat=51.5, edge_size=256, 
                scale=10, collection="TEST"
            )
            
            assert key1 != key2

    def test_coordinate_rounding(self):
        """Coordinates should be rounded to 4 decimals."""
        from cubexpress.cache import _cache_key
        
        with patch('cubexpress.cache.CACHE_DIR', Path('/tmp/test_cache')):
            key1 = _cache_key(
                lon=-0.12341, lat=51.54321, edge_size=256, 
                scale=10, collection="TEST"
            )
            key2 = _cache_key(
                lon=-0.12344, lat=51.54324, edge_size=256, 
                scale=10, collection="TEST"
            )
            
            assert key1 == key2

    def test_tuple_edge_size(self):
        """Tuple edge_size should work."""
        from cubexpress.cache import _cache_key
        
        with patch('cubexpress.cache.CACHE_DIR', Path('/tmp/test_cache')):
            key = _cache_key(
                lon=-0.1, lat=51.5, edge_size=(256, 128), 
                scale=10, collection="TEST"
            )
            
            assert key.suffix == ".parquet"


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_empty_cache(self, temp_cache_dir):
        """Clearing empty cache should return 0."""
        from cubexpress.cache import clear_cache
        
        with patch('cubexpress.cache.CACHE_DIR', temp_cache_dir):
            count = clear_cache()
            assert count == 0

    def test_clear_with_files(self, temp_cache_dir):
        """Should delete parquet files and return count."""
        from cubexpress.cache import clear_cache
        
        # Create fake cache files
        (temp_cache_dir / "test1.parquet").touch()
        (temp_cache_dir / "test2.parquet").touch()
        (temp_cache_dir / "other.txt").touch()  # Should not be deleted
        
        with patch('cubexpress.cache.CACHE_DIR', temp_cache_dir):
            count = clear_cache()
            
        assert count == 2
        assert not (temp_cache_dir / "test1.parquet").exists()
        assert not (temp_cache_dir / "test2.parquet").exists()
        assert (temp_cache_dir / "other.txt").exists()


class TestGetCacheSize:
    """Tests for get_cache_size function."""

    def test_empty_cache(self, temp_cache_dir):
        """Empty cache should return (0, 0)."""
        from cubexpress.cache import get_cache_size
        
        with patch('cubexpress.cache.CACHE_DIR', temp_cache_dir):
            count, size = get_cache_size()
            
        assert count == 0
        assert size == 0

    def test_with_files(self, temp_cache_dir):
        """Should count files and sum sizes."""
        from cubexpress.cache import get_cache_size
        
        # Create fake cache files with content
        (temp_cache_dir / "test1.parquet").write_text("a" * 100)
        (temp_cache_dir / "test2.parquet").write_text("b" * 200)
        
        with patch('cubexpress.cache.CACHE_DIR', temp_cache_dir):
            count, size = get_cache_size()
            
        assert count == 2
        assert size == 300