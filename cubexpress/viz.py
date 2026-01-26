"""Viz: GIF animation and QA band decoding utilities."""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import numpy as np
import rasterio as rio
from rasterio.transform import Affine

from cubexpress.download import download_manifest, temp_workspace
from cubexpress.formats import EEFileFormat, ExportFormat

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_gif(
    image_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    duration: int = 500,
    loop: int = 0,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> pathlib.Path:
    """Create animated GIF from image sequence."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow") from None

    if not image_paths:
        raise ValueError("No images provided for GIF creation")

    frames = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, background_color)
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        frames.append(img)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_p = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames_p[0].save(
        output_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=duration,
        loop=loop,
        optimize=False,
        disposal=2,
    )
    return output_path


def create_gif_from_requests(
    requests: Any,
    visualization: Any | None = None,
    output_path: pathlib.Path | str = "animation.gif",
    duration: int = 500,
    background_color: tuple[int, int, int] = (0, 0, 0),
    keep_frames: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Create animated GIF from a RequestSet.

    Args:
        requests: RequestSet or DataFrame from table_to_requestset
        visualization: VisualizationOptions for PNG rendering
        output_path: Output GIF path
        duration: Frame duration in milliseconds
        background_color: RGB tuple for background (replaces transparency)
        keep_frames: If provided, save PNG frames to this folder

    Returns:
        Path to created GIF
    """
    if visualization is None:
        raise ValueError("visualization is required. Use VisPresets.s2_truecolor() or similar.")

    output_path = pathlib.Path(output_path)
    dataframe = requests._dataframe if hasattr(requests, "_dataframe") else requests

    if dataframe.empty:
        raise ValueError("Request set is empty")

    png_format = ExportFormat(file_format=EEFileFormat.PNG, visualization=visualization)

    if keep_frames:
        frame_dir = pathlib.Path(keep_frames)
        frame_dir.mkdir(parents=True, exist_ok=True)
        ctx = None
    else:
        ctx = temp_workspace(prefix="cubexpress_gif_")
        frame_dir = ctx.__enter__()

    try:
        n = len(dataframe)
        frames = []
        print(f"⏳ Rendering {n} frames...", end="", flush=True)

        for i, (_, row) in enumerate(dataframe.iterrows()):
            frame_path = frame_dir / f"frame_{i:04d}_{row.id}.png"
            download_manifest(ulist=row.manifest, full_outname=frame_path, export_format=png_format)
            frames.append(frame_path)

        print(f"\r✅ Rendered {n} frames")
        print("⏳ Creating GIF...", end="", flush=True)

        result = create_gif(
            image_paths=frames, output_path=output_path, duration=duration, background_color=background_color
        )
        print(f"\r✅ Created GIF: {output_path} ({n} frames)")

    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    return result


# ---------------------------------------------------------------------------
# QA bit utilities
# ---------------------------------------------------------------------------


def extract_bit(array: np.ndarray, bit: int) -> np.ndarray:
    """Extract single bit from a QA array."""
    return ((array >> bit) & 1).astype(np.uint8)


def extract_bits(array: np.ndarray, start_bit: int, end_bit: int) -> np.ndarray:
    """Extract multi-bit value from a QA array."""
    mask = (1 << (end_bit - start_bit + 1)) - 1
    return ((array >> start_bit) & mask).astype(np.uint8)


# ---------------------------------------------------------------------------
# QA_PIXEL decoders
# ---------------------------------------------------------------------------


def decode_qa_pixel_mss(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode MSS QA_PIXEL band. See source for full flag list."""
    available = [
        "fill",
        "valid_data",
        "cloud",
        "no_cloud",
        "cloud_conf_none",
        "cloud_conf_low",
        "cloud_conf_medium",
        "cloud_conf_high",
        "clear",
        "cloud_any_conf",
        "high_quality",
    ]
    if flags == "all":
        flags = available

    bit0 = extract_bit(qa_array, 0)
    bit3 = extract_bit(qa_array, 3)
    conf = extract_bits(qa_array, 8, 9)
    results = {}

    for flag in flags:
        if flag == "fill":
            results[flag] = bit0
        elif flag == "valid_data":
            results[flag] = (~bit0.astype(bool)).astype(np.uint8)
        elif flag == "cloud":
            results[flag] = bit3
        elif flag == "no_cloud":
            results[flag] = (~bit3.astype(bool)).astype(np.uint8)
        elif flag == "cloud_conf_none":
            results[flag] = (conf == 0).astype(np.uint8)
        elif flag == "cloud_conf_low":
            results[flag] = (conf == 1).astype(np.uint8)
        elif flag == "cloud_conf_medium":
            results[flag] = (conf == 2).astype(np.uint8)
        elif flag == "cloud_conf_high":
            results[flag] = (conf == 3).astype(np.uint8)
        elif flag == "clear":
            results[flag] = ((~bit0.astype(bool)) & (~bit3.astype(bool))).astype(np.uint8)
        elif flag == "cloud_any_conf":
            results[flag] = (bit3 | ((conf == 1) | (conf == 3))).astype(np.uint8)
        elif flag == "high_quality":
            results[flag] = ((~bit0.astype(bool)) & (~bit3.astype(bool)) & ((conf == 0) | (conf == 1))).astype(np.uint8)
        else:
            raise ValueError(f"Unknown flag '{flag}' for MSS. Available: {available}")
    return results


def decode_qa_pixel_mss_esa(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode ESA/Amalfi BQA for Landsat MSS Collection 1. See source for full flag list."""
    available = [
        "fill",
        "valid_data",
        "dropped",
        "cloud",
        "no_cloud",
        "land_water",
        "sat_none",
        "sat_low",
        "sat_medium",
        "sat_high",
        "sat_any",
        "cloud_conf_none",
        "cloud_conf_low",
        "cloud_conf_medium",
        "cloud_conf_high",
        "shadow_conf_none",
        "shadow_conf_low",
        "shadow_conf_medium",
        "shadow_conf_high",
        "snow_conf_none",
        "snow_conf_low",
        "snow_conf_medium",
        "snow_conf_high",
        "sla_band4",
        "sla_band5",
        "sla_band6",
        "sla_band7",
        "sla_any",
        "clear",
        "high_quality",
        "cloud_or_shadow",
        "usable",
    ]
    if flags == "all":
        flags = available

    bit0 = extract_bit(qa_array, 0)
    bit1 = extract_bit(qa_array, 1)
    bit4 = extract_bit(qa_array, 4)
    bit11 = extract_bit(qa_array, 11)
    bit12 = extract_bit(qa_array, 12)
    bit13 = extract_bit(qa_array, 13)
    bit14 = extract_bit(qa_array, 14)
    bit15 = extract_bit(qa_array, 15)
    sat = extract_bits(qa_array, 2, 3)
    cloud_conf = extract_bits(qa_array, 5, 6)
    shadow_conf = extract_bits(qa_array, 7, 8)
    snow_conf = extract_bits(qa_array, 9, 10)
    results = {}

    for flag in flags:
        if flag == "fill":
            results[flag] = bit0
        elif flag == "valid_data":
            results[flag] = (~bit0.astype(bool)).astype(np.uint8)
        elif flag == "dropped":
            results[flag] = bit1
        elif flag == "cloud":
            results[flag] = bit4
        elif flag == "no_cloud":
            results[flag] = (~bit4.astype(bool)).astype(np.uint8)
        elif flag == "land_water":
            results[flag] = bit15
        elif flag == "sat_none":
            results[flag] = (sat == 0).astype(np.uint8)
        elif flag == "sat_low":
            results[flag] = (sat == 1).astype(np.uint8)
        elif flag == "sat_medium":
            results[flag] = (sat == 2).astype(np.uint8)
        elif flag == "sat_high":
            results[flag] = (sat == 3).astype(np.uint8)
        elif flag == "sat_any":
            results[flag] = (sat > 0).astype(np.uint8)
        elif flag == "cloud_conf_none":
            results[flag] = (cloud_conf == 0).astype(np.uint8)
        elif flag == "cloud_conf_low":
            results[flag] = (cloud_conf == 1).astype(np.uint8)
        elif flag == "cloud_conf_medium":
            results[flag] = (cloud_conf == 2).astype(np.uint8)
        elif flag == "cloud_conf_high":
            results[flag] = (cloud_conf == 3).astype(np.uint8)
        elif flag == "shadow_conf_none":
            results[flag] = (shadow_conf == 0).astype(np.uint8)
        elif flag == "shadow_conf_low":
            results[flag] = (shadow_conf == 1).astype(np.uint8)
        elif flag == "shadow_conf_medium":
            results[flag] = (shadow_conf == 2).astype(np.uint8)
        elif flag == "shadow_conf_high":
            results[flag] = (shadow_conf == 3).astype(np.uint8)
        elif flag == "snow_conf_none":
            results[flag] = (snow_conf == 0).astype(np.uint8)
        elif flag == "snow_conf_low":
            results[flag] = (snow_conf == 1).astype(np.uint8)
        elif flag == "snow_conf_medium":
            results[flag] = (snow_conf == 2).astype(np.uint8)
        elif flag == "snow_conf_high":
            results[flag] = (snow_conf == 3).astype(np.uint8)
        elif flag == "sla_band4":
            results[flag] = bit11
        elif flag == "sla_band5":
            results[flag] = bit12
        elif flag == "sla_band6":
            results[flag] = bit13
        elif flag == "sla_band7":
            results[flag] = bit14
        elif flag == "sla_any":
            results[flag] = (bit11 | bit12 | bit13 | bit14).astype(np.uint8)
        elif flag == "clear":
            results[flag] = ((~bit0.astype(bool)) & (~bit4.astype(bool)) & (shadow_conf != 3)).astype(np.uint8)
        elif flag == "high_quality":
            no_sla = ~(bit11 | bit12 | bit13 | bit14).astype(bool)
            results[flag] = (
                (~bit0.astype(bool)) & (~bit4.astype(bool)) & (shadow_conf < 3) & (sat == 0) & no_sla
            ).astype(np.uint8)
        elif flag == "cloud_or_shadow":
            results[flag] = (bit4 | (shadow_conf >= 2)).astype(np.uint8)
        elif flag == "usable":
            results[flag] = ((~bit0.astype(bool)) & (cloud_conf < 3) & (sat < 3)).astype(np.uint8)
        else:
            raise ValueError(f"Unknown flag '{flag}' for ESA MSS. Available: {available}")
    return results


def decode_qa_pixel_tm(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode TM/ETM+/OLI QA_PIXEL band. See source for full flag list."""
    available = [
        "fill",
        "valid_data",
        "dilated_cloud",
        "cloud",
        "no_cloud",
        "cloud_shadow",
        "no_shadow",
        "snow",
        "clear",
        "water",
        "land",
        "cloud_conf_none",
        "cloud_conf_low",
        "cloud_conf_medium",
        "cloud_conf_high",
        "shadow_conf_none",
        "shadow_conf_low",
        "shadow_conf_high",
        "snow_conf_none",
        "snow_conf_low",
        "snow_conf_high",
        "cloud_or_shadow",
        "cloud_shadow_dilated",
        "high_quality",
    ]
    if flags == "all":
        flags = available

    bit0 = extract_bit(qa_array, 0)
    bit1 = extract_bit(qa_array, 1)
    bit3 = extract_bit(qa_array, 3)
    bit4 = extract_bit(qa_array, 4)
    bit5 = extract_bit(qa_array, 5)
    bit6 = extract_bit(qa_array, 6)
    bit7 = extract_bit(qa_array, 7)
    cloud_conf = extract_bits(qa_array, 8, 9)
    shadow_conf = extract_bits(qa_array, 10, 11)
    snow_conf = extract_bits(qa_array, 12, 13)
    results = {}

    for flag in flags:
        if flag == "fill":
            results[flag] = bit0
        elif flag == "valid_data":
            results[flag] = (~bit0.astype(bool)).astype(np.uint8)
        elif flag == "dilated_cloud":
            results[flag] = bit1
        elif flag == "cloud":
            results[flag] = bit3
        elif flag == "no_cloud":
            results[flag] = (~bit3.astype(bool)).astype(np.uint8)
        elif flag == "cloud_shadow":
            results[flag] = bit4
        elif flag == "no_shadow":
            results[flag] = (~bit4.astype(bool)).astype(np.uint8)
        elif flag == "snow":
            results[flag] = bit5
        elif flag == "clear":
            results[flag] = bit6
        elif flag == "water":
            results[flag] = bit7
        elif flag == "land":
            results[flag] = (~bit7.astype(bool)).astype(np.uint8)
        elif flag == "cloud_conf_none":
            results[flag] = (cloud_conf == 0).astype(np.uint8)
        elif flag == "cloud_conf_low":
            results[flag] = (cloud_conf == 1).astype(np.uint8)
        elif flag == "cloud_conf_medium":
            results[flag] = (cloud_conf == 2).astype(np.uint8)
        elif flag == "cloud_conf_high":
            results[flag] = (cloud_conf == 3).astype(np.uint8)
        elif flag == "shadow_conf_none":
            results[flag] = (shadow_conf == 0).astype(np.uint8)
        elif flag == "shadow_conf_low":
            results[flag] = (shadow_conf == 1).astype(np.uint8)
        elif flag == "shadow_conf_high":
            results[flag] = (shadow_conf == 3).astype(np.uint8)
        elif flag == "snow_conf_none":
            results[flag] = (snow_conf == 0).astype(np.uint8)
        elif flag == "snow_conf_low":
            results[flag] = (snow_conf == 1).astype(np.uint8)
        elif flag == "snow_conf_high":
            results[flag] = (snow_conf == 3).astype(np.uint8)
        elif flag == "cloud_or_shadow":
            results[flag] = (bit3 | bit4).astype(np.uint8)
        elif flag == "cloud_shadow_dilated":
            results[flag] = (bit1 | bit3 | bit4).astype(np.uint8)
        elif flag == "high_quality":
            results[flag] = (
                (~bit0.astype(bool))
                & bit6.astype(bool)
                & ((cloud_conf == 0) | (cloud_conf == 1))
                & ((shadow_conf == 0) | (shadow_conf == 1))
            ).astype(np.uint8)
        else:
            raise ValueError(f"Unknown flag '{flag}' for TM/ETM+/OLI. Available: {available}")
    return results


# ---------------------------------------------------------------------------
# QA_RADSAT decoders
# ---------------------------------------------------------------------------


def decode_qa_radsat_mss(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode MSS QA_RADSAT band."""
    available = ["b1_sat", "b2_sat", "b3_sat", "b4_sat", "b5_sat", "b6_sat", "dropped", "any_saturated"]
    if flags == "all":
        flags = available
    bit_map = {"b1_sat": 0, "b2_sat": 1, "b3_sat": 2, "b4_sat": 3, "b5_sat": 4, "b6_sat": 5, "dropped": 9}
    results = {}
    for flag in flags:
        if flag == "any_saturated":
            sat = extract_bit(qa_array, 0)
            for b in range(1, 6):
                sat |= extract_bit(qa_array, b)
            results[flag] = sat.astype(np.uint8)
        elif flag in bit_map:
            results[flag] = extract_bit(qa_array, bit_map[flag])
        else:
            raise ValueError(f"Unknown flag '{flag}' for MSS QA_RADSAT. Available: {available}")
    return results


def decode_qa_radsat_tm(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode TM/ETM+ QA_RADSAT band."""
    available = [
        "b1_sat",
        "b2_sat",
        "b3_sat",
        "b4_sat",
        "b5_sat",
        "b6l_sat",
        "b7_sat",
        "b6h_sat",
        "dropped",
        "any_saturated",
    ]
    if flags == "all":
        flags = available
    bit_map = {
        "b1_sat": 0,
        "b2_sat": 1,
        "b3_sat": 2,
        "b4_sat": 3,
        "b5_sat": 4,
        "b6l_sat": 5,
        "b7_sat": 6,
        "b6h_sat": 8,
        "dropped": 9,
    }
    results = {}
    for flag in flags:
        if flag == "any_saturated":
            sat = extract_bit(qa_array, 0)
            for b in [1, 2, 3, 4, 5, 6, 8]:
                sat |= extract_bit(qa_array, b)
            results[flag] = sat.astype(np.uint8)
        elif flag in bit_map:
            results[flag] = extract_bit(qa_array, bit_map[flag])
        else:
            raise ValueError(f"Unknown flag '{flag}' for TM/ETM+ QA_RADSAT. Available: {available}")
    return results


def decode_qa_radsat_oli(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode OLI QA_RADSAT band."""
    available = [
        "b1_sat",
        "b2_sat",
        "b3_sat",
        "b4_sat",
        "b5_sat",
        "b6_sat",
        "b7_sat",
        "b8_sat",
        "b9_sat",
        "b10_sat",
        "b11_sat",
        "dropped",
    ]
    if flags == "all":
        flags = available
    bit_map = {f"b{i}_sat": i - 1 for i in range(1, 12)}
    bit_map["dropped"] = 11
    results = {}
    for flag in flags:
        if flag in bit_map:
            results[flag] = extract_bit(qa_array, bit_map[flag])
        else:
            raise ValueError(f"Unknown flag '{flag}' for OLI QA_RADSAT. Available: {available}")
    return results


# ---------------------------------------------------------------------------
# Combined masks
# ---------------------------------------------------------------------------


def get_cloud_shadow_mask(
    qa_pixel: np.ndarray,
    sensor: Literal["MSS", "TM", "ETM+", "OLI"],
    include_dilated: bool = True,
    confidence: Literal["any", "high"] = "high",
) -> np.ndarray:
    """Get combined cloud + shadow mask (1 = cloud/shadow, 0 = clear)."""
    if sensor == "MSS":
        masks = decode_qa_pixel_mss(qa_pixel, ["cloud_conf_high" if confidence == "high" else "cloud"])
        return masks[next(iter(masks.keys()))]

    flags = ["cloud", "cloud_shadow"] + (["dilated_cloud"] if include_dilated else [])
    if confidence == "high":
        flags += ["cloud_conf_high", "shadow_conf_high"]
    masks = decode_qa_pixel_tm(qa_pixel, flags)
    combined = masks["cloud"] | masks["cloud_shadow"]
    if include_dilated:
        combined |= masks["dilated_cloud"]
    if confidence == "high":
        combined |= masks["cloud_conf_high"] | masks["shadow_conf_high"]
    return combined.astype(np.uint8)


def get_clear_mask(
    qa_pixel: np.ndarray,
    sensor: Literal["MSS", "TM", "ETM+", "OLI"],
) -> np.ndarray:
    """Get clear pixel mask (1 = clear, 0 = not clear)."""
    if sensor == "MSS":
        masks = decode_qa_pixel_mss(qa_pixel, ["fill", "cloud"])
        return (~(masks["fill"] | masks["cloud"])).astype(np.uint8)
    return decode_qa_pixel_tm(qa_pixel, ["clear"])["clear"]


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------


def save_mask_geotiff(
    mask: np.ndarray,
    output_path: pathlib.Path,
    transform: Affine,
    crs: str,
    nodata: int = 255,
) -> None:
    """Save binary mask as GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="DEFLATE",
        predictor=2,
    ) as dst:
        dst.write(mask, 1)


def export_all_qa_masks(
    qa_pixel_path: pathlib.Path,
    output_dir: pathlib.Path,
    sensor: Literal["MSS", "TM", "ETM+", "OLI"],
    qa_radsat_path: pathlib.Path | None = None,
) -> dict[str, pathlib.Path]:
    """Export all QA masks from a Landsat scene to GeoTIFFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with rio.open(qa_pixel_path) as src:
        qa_pixel = src.read(1)
        transform = src.transform
        crs = src.crs

    pixel_masks = decode_qa_pixel_mss(qa_pixel, "all") if sensor == "MSS" else decode_qa_pixel_tm(qa_pixel, "all")

    exported = {}
    for name, mask in pixel_masks.items():
        outpath = output_dir / f"qa_pixel_{name}.tif"
        save_mask_geotiff(mask, outpath, transform, crs)
        exported[f"pixel_{name}"] = outpath

    for label, fn in [("cloud_shadow_combined", get_cloud_shadow_mask), ("clear_mask", get_clear_mask)]:
        outpath = output_dir / f"{label}.tif"
        save_mask_geotiff(fn(qa_pixel, sensor), outpath, transform, crs)
        exported[label] = outpath

    if qa_radsat_path and qa_radsat_path.exists():
        with rio.open(qa_radsat_path) as src:
            qa_radsat = src.read(1)
        radsat_fn = (
            decode_qa_radsat_mss
            if sensor == "MSS"
            else decode_qa_radsat_tm if sensor in ("TM", "ETM+") else decode_qa_radsat_oli
        )
        for name, mask in radsat_fn(qa_radsat, "all").items():
            outpath = output_dir / f"qa_radsat_{name}.tif"
            save_mask_geotiff(mask, outpath, transform, crs)
            exported[f"radsat_{name}"] = outpath

    return exported


def export_all_esa_mss_masks(
    bqa_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Export all ESA MSS BQA masks with statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with rio.open(bqa_path) as src:
        bqa = src.read(1)
        transform = src.transform
        crs = src.crs

    all_masks = decode_qa_pixel_mss_esa(bqa, flags="all")
    total_pixels = bqa.size
    effective = (~((bqa >> 0) & 1).astype(bool)).sum()

    print(f"\n{'='*70}")
    print(f"ESA MSS BQA Analysis: {bqa_path.name}")
    print(f"{'='*70}")
    print(f"Total pixels: {total_pixels:,}  |  Effective: {effective:,} ({effective/total_pixels*100:.2f}%)")
    print(f"\n{'Mask Name':<30} {'Count':>12} {'% Total':>10} {'% Effective':>12}")
    print(f"{'-'*70}")

    exported = {}
    for name, mask in all_masks.items():
        outpath = output_dir / f"esa_mss_{name}.tif"
        save_mask_geotiff(mask, outpath, transform, crs)
        exported[name] = outpath
        count = mask.sum()
        print(
            f"{name:<30} {count:>12,} {count/total_pixels*100:>9.2f}% {count/effective*100 if effective else 0:>11.2f}%"
        )

    print(f"{'='*70}")
    print(f"✅ Exported {len(exported)} masks to {output_dir}\n")
    return exported


# ---------------------------------------------------------------------------
# Pixel-level cloud masks
# ---------------------------------------------------------------------------


def cloud_shadow_mask_mss_gee(qa_pixel: np.ndarray) -> np.ndarray:
    """MSS GEE Collection 2 cloud mask (0=clear, 1=cloud)."""
    mask = np.zeros_like(qa_pixel, dtype=np.uint8)
    mask[extract_bit(qa_pixel, 3).astype(bool)] = 1
    return mask


def cloud_shadow_mask_mss_esa(bqa: np.ndarray) -> np.ndarray:
    """MSS ESA Collection 1 cloud+shadow mask (0=clear, 1=cloud, 2=shadow)."""
    mask = np.zeros_like(bqa, dtype=np.uint8)
    shadow = (extract_bits(bqa, 7, 8) >= 2).astype(bool)
    cloud = extract_bit(bqa, 4).astype(bool)
    mask[shadow] = 2
    mask[cloud] = 1
    return mask


def cloud_shadow_mask_tm_gee(qa_pixel: np.ndarray) -> np.ndarray:
    """TM/ETM+/OLI GEE Collection 2 cloud+shadow mask (0=clear, 1=cloud, 2=shadow)."""
    mask = np.zeros_like(qa_pixel, dtype=np.uint8)
    shadow = extract_bit(qa_pixel, 4).astype(bool)
    cloud = extract_bit(qa_pixel, 3).astype(bool)
    mask[shadow] = 2
    mask[cloud] = 1
    return mask
