"""GeoTIFF merging and COG conversion utilities."""

from __future__ import annotations

import pathlib

import rasterio as rio
import rasterio.shutil as rio_shutil
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge

from cubexpress.core.exceptions import MergeError

OUTPUT_PROFILES = {
    "geotiff": {},
    "cog": {
        "driver": "COG",
        "compress": "DEFLATE",
        "predictor": 2,
        "overview_resampling": "nearest",
    },
    "cog_lzw": {
        "driver": "COG",
        "compress": "LZW",
        "predictor": 2,
    },
    "cog_zstd": {
        "driver": "COG",
        "compress": "ZSTD",
        "predictor": 2,
    },
}


def merge_tifs(
    input_files: list[pathlib.Path],
    output_path: pathlib.Path,
    *,
    nodata: int | float | None = None,
    gdal_threads: int = 8,
) -> None:
    """Merge multiple GeoTIFF files into a single mosaic.

    Args:
        input_files: Paths to GeoTIFF tiles
        output_path: Destination path for merged file
        nodata: NoData value for the mosaic. If None, inferred from source.
        gdal_threads: Number of threads for GDAL operations

    Raises:
        MergeError: If merge operation fails
    """
    if not input_files:
        raise MergeError("Input files list is empty")

    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rio.Env(GDAL_NUM_THREADS=str(gdal_threads), NUM_THREADS=str(gdal_threads)):
            srcs = [rio.open(fp) for fp in input_files]

            if nodata is None:
                if srcs[0].nodata is not None:
                    merge_nodata = srcs[0].nodata
                else:
                    merge_nodata = 0
            else:
                merge_nodata = nodata

            try:
                mosaic, out_transform = rio_merge(srcs, nodata=merge_nodata, resampling=Resampling.nearest)

                meta = srcs[0].profile.copy()
                meta.update(
                    {
                        "transform": out_transform,
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "nodata": merge_nodata,
                    }
                )

                with rio.open(output_path, "w", **meta) as dst:
                    dst.write(mosaic)

            finally:
                for src in srcs:
                    src.close()

    except Exception as e:
        raise MergeError(f"Failed to merge {len(input_files)} files: {e}") from e


def convert_to_cog(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    profile_overrides: dict | None = None,
    in_place: bool = True,
) -> pathlib.Path:
    """Convert GeoTIFF to COG without loading full image into memory.

    Args:
        input_path: Input GeoTIFF path
        output_path: Output path. If None and in_place=True, uses temp file.
        profile_overrides: Override default COG profile settings
        in_place: If True, replaces original file

    Returns:
        Path to converted COG file
    """
    input_path = pathlib.Path(input_path)

    if output_path is None:
        if in_place:
            output_path = input_path.with_suffix(".cog.tif")
        else:
            raise ValueError("output_path required when in_place=False")

    output_path = pathlib.Path(output_path)

    cog_profile = {
        "driver": "COG",
        "compress": "DEFLATE",
        "predictor": 2,
    }

    if profile_overrides:
        cog_profile.update(profile_overrides)

    with rio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        rio_shutil.copy(input_path, output_path, **cog_profile)

    if in_place and output_path != input_path:
        input_path.unlink()
        output_path.rename(input_path)
        return input_path

    return output_path


def apply_output_format(
    file_path: pathlib.Path,
    output_format: str | dict | None,
) -> None:
    """Apply output format to a GeoTIFF file in-place.

    Args:
        file_path: Path to GeoTIFF
        output_format: "cog", "cog_lzw", "cog_zstd", or dict with profile options
    """
    if output_format is None or output_format == "geotiff":
        return

    if isinstance(output_format, str):
        if output_format not in OUTPUT_PROFILES:
            raise ValueError(f"Unknown format '{output_format}'. Available: {list(OUTPUT_PROFILES.keys())}")
        profile = OUTPUT_PROFILES[output_format]
    else:
        profile = output_format

    if not profile:
        return

    convert_to_cog(file_path, profile_overrides=profile, in_place=True)
