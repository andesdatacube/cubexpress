"""Shared pytest fixtures for the entire test suite."""

import os
import pytest


# Try to initialize EE once at import; record whether it worked.
_EE_AVAILABLE = False
try:
    import ee
    _project = os.environ.get("GEE_PROJECT")
    if _project:
        ee.Initialize(project=_project)
        _EE_AVAILABLE = True
    else:
        try:
            ee.Initialize()
            _EE_AVAILABLE = True
        except Exception:
            _EE_AVAILABLE = False
except Exception:
    _EE_AVAILABLE = False


@pytest.fixture(scope="session", autouse=True)
def ee_initialize_session():
    """EE is initialized at import (above); this fixture is a no-op hook."""
    yield


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `needs_ee` when Earth Engine could not initialize."""
    if _EE_AVAILABLE:
        return
    skip_ee = pytest.mark.skip(reason="Earth Engine not initialized (CI without EE creds)")
    for item in items:
        if "needs_ee" in item.keywords:
            item.add_marker(skip_ee)


@pytest.fixture
def require_ee():
    """Per-test guard: skip if Earth Engine is unavailable."""
    if not _EE_AVAILABLE:
        pytest.skip("Earth Engine not available — skipping EE test")