"""Tests for cubexpress.exceptions module."""

from __future__ import annotations

import pytest

from cubexpress.core.exceptions import CubExpressError, DownloadError, MergeError, TilingError, ValidationError


class TestExceptionHierarchy:
    """Test exception inheritance."""

    def test_all_inherit_from_base(self):
        """All custom exceptions should inherit from CubExpressError."""
        assert issubclass(DownloadError, CubExpressError)
        assert issubclass(ValidationError, CubExpressError)
        assert issubclass(TilingError, CubExpressError)
        assert issubclass(MergeError, CubExpressError)

    def test_base_inherits_from_exception(self):
        """Base should inherit from Exception."""
        assert issubclass(CubExpressError, Exception)


class TestExceptionMessages:
    """Test exception message handling."""

    def test_validation_error_message(self):
        """ValidationError should preserve message."""
        msg = "Invalid input data"
        err = ValidationError(msg)
        assert str(err) == msg

    def test_download_error_message(self):
        """DownloadError should preserve message."""
        msg = "Failed to download from GEE"
        err = DownloadError(msg)
        assert str(err) == msg

    def test_can_catch_by_base_class(self):
        """Should be catchable by base class."""
        with pytest.raises(CubExpressError):
            raise ValidationError("test")

        with pytest.raises(CubExpressError):
            raise DownloadError("test")
