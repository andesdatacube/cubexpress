"""Shared fixtures for cubexpress tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------------
# Mock Earth Engine BEFORE any cubexpress import
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def mock_ee_init():
    """
    Mock ee.Initialize() globally to avoid requiring GEE credentials.

    This runs automatically for all tests.
    """
    mock_ee = MagicMock()
    mock_ee.Initialize = MagicMock(return_value=None)
    mock_ee.Geometry.Point = MagicMock(return_value=MagicMock())
    mock_ee.Geometry.Polygon = MagicMock(return_value=MagicMock())
    mock_ee.Image = MagicMock(return_value=MagicMock())
    mock_ee.ImageCollection = MagicMock(return_value=MagicMock())
    mock_ee.Feature = MagicMock(return_value=MagicMock())
    mock_ee.Filter = MagicMock()
    mock_ee.Date = MagicMock(return_value=MagicMock())
    mock_ee.data = MagicMock()
    mock_ee.ee_exception = MagicMock()
    mock_ee.ee_exception.EEException = Exception

    with patch.dict(sys.modules, {"ee": mock_ee}):
        yield mock_ee


# -----------------------------------------------------------------------------
# Common test data fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_coords():
    """Sample coordinates for testing (London)."""
    return {"lon": -0.1276, "lat": 51.5074}


@pytest.fixture
def sample_coords_valencia():
    """Sample coordinates for testing (Valencia)."""
    return {"lon": -0.3763, "lat": 39.4699}


@pytest.fixture
def sample_edge_size():
    """Standard edge size for tests."""
    return 256


@pytest.fixture
def sample_scale():
    """Standard scale (10m for S2)."""
    return 10


@pytest.fixture
def sample_geotransform():
    """Valid geotransform dictionary."""
    return {
        "scaleX": 10,
        "shearX": 0,
        "translateX": 500000.0,
        "scaleY": -10,
        "shearY": 0,
        "translateY": 4500000.0,
    }


@pytest.fixture
def sample_manifest(sample_geotransform):
    """Sample Earth Engine manifest for testing."""
    return {
        "assetId": "COPERNICUS/S2_HARMONIZED/20230101T000000_20230101T000000_T30TYK",
        "fileFormat": "GEO_TIFF",
        "bandIds": ["B4", "B3", "B2"],
        "grid": {
            "dimensions": {"width": 256, "height": 256},
            "affineTransform": sample_geotransform,
            "crsCode": "EPSG:32630",
        },
    }


# -----------------------------------------------------------------------------
# Temporary directory fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
