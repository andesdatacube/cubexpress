"""Logging configuration for cubexpress."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO, format_string: str | None = None) -> logging.Logger:
    """
    Configure a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def disable_external_loggers() -> None:
    """Suppress verbose logging from external libraries."""
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    logging.getLogger("fiona").setLevel(logging.ERROR)
