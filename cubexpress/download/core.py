"""Earth Engine data download utilities."""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import ee

from cubexpress.download.merge import merge_tifs
from cubexpress.formats.specs import VISUALIZATION_FORMATS, EEFileFormat


@contextmanager
def temp_workspace(prefix: str = "cubexpress_") -> Iterator[pathlib.Path]:
    """Create a temporary directory with automatic cleanup."""
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp_dir
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_request(
    manifest: dict[str, Any],
    export_format: Any | None = None,
) -> dict[str, Any]:
    """Build request dict with format and visualization options.

    Args:
        manifest: Original manifest dict
        export_format: ExportFormat instance or None

    Returns:
        Modified request dict ready for getPixels/computePixels
    """

    request = deepcopy(manifest)

    if export_format is None:
        request["fileFormat"] = "GEO_TIFF"
        return request

    request["fileFormat"] = export_format.file_format.value

    # For visualization formats, handle bands and visualization options separately
    if export_format.file_format in VISUALIZATION_FORMATS and export_format.visualization:
        vis = export_format.visualization

        # Bands go to bandIds, NOT to visualizationOptions
        if vis.bands:
            request["bandIds"] = vis.bands

        # Build visualizationOptions (without bands)
        vis_opts = vis.to_ee_dict()
        if vis_opts:  # Only add if not empty
            request["visualizationOptions"] = vis_opts

    return request


def download_manifest(
    ulist: dict[str, Any],
    full_outname: pathlib.Path,
    export_format: Any | None = None,
) -> Any:
    """Download data from Earth Engine based on a manifest dictionary.

    Args:
        ulist: Manifest dictionary with assetId or expression
        full_outname: Output file path
        export_format: ExportFormat instance or None

    Returns:
        For NUMPY_NDARRAY format: numpy structured array
        For other formats: None (writes to file)
    """

    # Build request with format options
    request = _build_request(ulist, export_format)

    # Determine file format
    file_format = EEFileFormat.GEO_TIFF
    if export_format:
        file_format = export_format.file_format

    # Execute request
    if "assetId" in request:
        if file_format == EEFileFormat.NUMPY_NDARRAY:
            request["fileFormat"] = "NUMPY_NDARRAY"
            return ee.data.getPixels(request)
        else:
            result = ee.data.getPixels(request)
    elif "expression" in request:
        ee_image = ee.deserializer.decode(json.loads(request["expression"]))
        request_copy = deepcopy(request)
        request_copy["expression"] = ee_image

        if file_format == EEFileFormat.NUMPY_NDARRAY:
            request_copy["fileFormat"] = "NUMPY_NDARRAY"
            return ee.data.computePixels(request_copy)
        else:
            result = ee.data.computePixels(request_copy)
    else:
        raise ValueError("Manifest must contain 'assetId' or 'expression'")

    # Write binary result to file
    full_outname.parent.mkdir(parents=True, exist_ok=True)
    with full_outname.open("wb") as f:
        f.write(result)
    return None


def download_manifests(
    manifests: list[dict[str, Any]],
    full_outname: pathlib.Path,
    max_workers: int = 1,
    export_format: Any | None = None,
) -> None:
    """Download multiple manifests concurrently and merge into one file.

    Only works for GeoTIFF format (other formats cannot be merged).
    """

    # Only GeoTIFF can be merged
    if export_format and export_format.file_format != EEFileFormat.GEO_TIFF:
        raise ValueError("Only GEO_TIFF format supports tiled downloads with merging")

    with temp_workspace() as tmp_dir:
        tile_dir = tmp_dir / full_outname.stem
        tile_dir.mkdir(parents=True, exist_ok=True)

        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    download_manifest,
                    ulist=manifest,
                    full_outname=tile_dir / f"{idx:06d}.tif",
                    export_format=export_format,
                ): idx
                for idx, manifest in enumerate(manifests)
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)

        if errors:
            raise errors[0]

        input_files = sorted(tile_dir.glob("*.tif"))
        if not input_files:
            raise ValueError(f"No tiles downloaded in {tile_dir}")

        merge_tifs(input_files, full_outname)
