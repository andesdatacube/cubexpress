"""Custom exceptions for cubexpress operations."""

from __future__ import annotations


class CubExpressError(Exception):
    """Base exception for all cubexpress operations."""

    pass


class DownloadError(CubExpressError):
    """Earth Engine download operation failed."""

    pass


class ValidationError(CubExpressError):
    """Data validation failed."""

    pass


class TilingError(CubExpressError):
    """Tile calculation or splitting failed."""

    pass


class MergeError(CubExpressError):
    """GeoTIFF merging operation failed."""

    pass
