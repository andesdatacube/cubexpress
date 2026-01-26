from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import rasterio as rio
from rasterio.transform import Affine

# BIT EXTRACTION UTILITIES


def extract_bit(array: np.ndarray, bit: int) -> np.ndarray:
    """Extract single bit from array.

    Args:
        array: Input QA array (uint16)
        bit: Bit position (0-15)

    Returns:
        Binary mask (1 where bit is set, 0 otherwise)
    """
    return ((array >> bit) & 1).astype(np.uint8)


def extract_bits(array: np.ndarray, start_bit: int, end_bit: int) -> np.ndarray:
    """Extract multi-bit value from array.

    Args:
        array: Input QA array (uint16)
        start_bit: Starting bit position
        end_bit: Ending bit position (inclusive)

    Returns:
        Decoded values (0-3 for 2-bit fields, etc.)
    """
    num_bits = end_bit - start_bit + 1
    mask = (1 << num_bits) - 1
    return ((array >> start_bit) & mask).astype(np.uint8)


# QA_PIXEL DECODERS


def decode_qa_pixel_mss(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode MSS QA_PIXEL band into ALL possible masks.

    INDIVIDUAL BITS:
    - fill: Fill data (bit 0 = 1)
    - valid_data: Valid image data (bit 0 = 0) [INVERSE of fill]
    - cloud: Cloud present (bit 3 = 1)
    - no_cloud: No cloud (bit 3 = 0) [INVERSE of cloud]

    CONFIDENCE LEVELS (bits 8-9):
    - cloud_conf_none: No confidence set (bits 8-9 = 0)
    - cloud_conf_low: Low confidence (bits 8-9 = 1)
    - cloud_conf_medium: Medium confidence (bits 8-9 = 2) [exists though "unused"]
    - cloud_conf_high: High confidence (bits 8-9 = 3)

    LOGICAL COMBINATIONS:
    - clear: Valid data AND no cloud
    - cloud_any_conf: Cloud OR (conf_low OR conf_high)
    - high_quality: Valid AND no_cloud AND (conf_none OR conf_low)

    Args:
        qa_array: QA_PIXEL array (H, W) uint16
        flags: List of flags to extract, or "all" for everything

    Returns:
        Dictionary mapping flag name to binary mask (1=True, 0=False)
    """
    available = [
        # Individual bits
        "fill",
        "valid_data",
        "cloud",
        "no_cloud",
        # Cloud confidence
        "cloud_conf_none",
        "cloud_conf_low",
        "cloud_conf_medium",
        "cloud_conf_high",
        # Useful combinations
        "clear",
        "cloud_any_conf",
        "high_quality",
    ]

    if flags == "all":
        flags = available

    results = {}

    # Pre-calculate common values to avoid recalculation
    bit0 = extract_bit(qa_array, 0)
    bit3 = extract_bit(qa_array, 3)
    conf = extract_bits(qa_array, 8, 9)

    for flag in flags:
        # --- INDIVIDUAL BITS ---
        if flag == "fill":
            results[flag] = bit0  # 1 = fill

        elif flag == "valid_data":
            results[flag] = (~bit0.astype(bool)).astype(np.uint8)  # 1 = valid

        elif flag == "cloud":
            results[flag] = bit3  # 1 = cloud

        elif flag == "no_cloud":
            results[flag] = (~bit3.astype(bool)).astype(np.uint8)  # 1 = no cloud

        # --- CLOUD CONFIDENCE ---
        elif flag == "cloud_conf_none":
            results[flag] = (conf == 0).astype(np.uint8)

        elif flag == "cloud_conf_low":
            results[flag] = (conf == 1).astype(np.uint8)

        elif flag == "cloud_conf_medium":
            results[flag] = (conf == 2).astype(np.uint8)  # Exists though "unused"

        elif flag == "cloud_conf_high":
            results[flag] = (conf == 3).astype(np.uint8)

        # --- LOGICAL COMBINATIONS ---
        elif flag == "clear":
            # Valid pixel WITHOUT cloud
            valid = (~bit0.astype(bool)).astype(np.uint8)
            no_cloud = (~bit3.astype(bool)).astype(np.uint8)
            results[flag] = (valid & no_cloud).astype(np.uint8)

        elif flag == "cloud_any_conf":
            # Cloud OR low/high confidence
            conf_any = ((conf == 1) | (conf == 3)).astype(np.uint8)
            results[flag] = (bit3 | conf_any).astype(np.uint8)

        elif flag == "high_quality":
            # Valid + no cloud + none/low confidence
            valid = (~bit0.astype(bool)).astype(np.uint8)
            no_cloud = (~bit3.astype(bool)).astype(np.uint8)
            low_conf = ((conf == 0) | (conf == 1)).astype(np.uint8)
            results[flag] = (valid & no_cloud & low_conf).astype(np.uint8)

        else:
            raise ValueError(f"Unknown flag '{flag}' for MSS. Available: {available}")

    return results


def decode_qa_pixel_mss_esa(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode ESA/Amalfi BQA for Landsat MSS (Collection 1).

    INDIVIDUAL BITS:
    - fill: Designated fill (bit 0 = 1) - Outside image area
    - valid_data: Valid observations (bit 0 = 0) [INVERSE]
    - dropped: Dropped pixel (bit 1 = 1) - Lost during transmission
    - cloud: Cloud identified (bit 4 = 1)
    - no_cloud: No cloud (bit 4 = 0) [INVERSE]
    - land_water: Land/Water mask (bit 15 = 1 for water, 0 for land)

    RADIOMETRIC SATURATION (bits 2-3):
    - sat_none: No saturation (00) - Optimal quality
    - sat_low: 1-2 bands saturated (01) - Rare
    - sat_medium: 3 bands saturated (10)
    - sat_high: All 4 bands saturated (11) - Exclude from NDVI
    - sat_any: Any saturation [COMBINATION]

    CLOUD CONFIDENCE (bits 5-6):
    - cloud_conf_none: Not determined (00)
    - cloud_conf_low: <33% probability (01) - Likely clear
    - cloud_conf_medium: 34-66% probability (10) - Edge/haze
    - cloud_conf_high: >66% probability (11) - Opaque cloud

    CLOUD SHADOW CONFIDENCE (bits 7-8):
    - shadow_conf_none: Not determined (00)
    - shadow_conf_low: Unlikely shadow (01)
    - shadow_conf_medium: Possible shadow (10)
    - shadow_conf_high: High probability shadow (11) - Mask to avoid water misclassification

    SNOW/ICE CONFIDENCE (bits 9-10):
    - snow_conf_none: Not determined (00)
    - snow_conf_low: Unlikely snow (01)
    - snow_conf_medium: Possible snow (10)
    - snow_conf_high: High probability snow/ice (11) - MSS struggles to separate clouds from snow

    SCAN LINE ARTEFACTS (bits 11-14):
    - sla_band4: Band 4 (Green) anomaly (bit 11 = 1) - Stripes/noise
    - sla_band5: Band 5 (Red) anomaly (bit 12 = 1)
    - sla_band6: Band 6 (NIR1) anomaly (bit 13 = 1)
    - sla_band7: Band 7 (NIR2) anomaly (bit 14 = 1)
    - sla_any: Any SLA detected [COMBINATION]

    LOGICAL COMBINATIONS:
    - clear: Valid AND no cloud AND no shadow (high conf)
    - high_quality: Valid AND clear AND no saturation AND no SLA
    - cloud_or_shadow: Cloud OR shadow (any confidence)
    - usable: Valid AND no high-conf cloud AND no saturation

    Args:
        qa_array: ESA BQA array (H, W) uint16
        flags: List of flags to extract, or "all"

    Returns:
        Dictionary mapping flag name to binary mask (1=True, 0=False)

    Example:
        >>> bqa = rio.open("LM51940281984182FUI00_BQA.TIF").read(1)
        >>> masks = decode_qa_pixel_mss_esa(bqa, flags=["clear", "high_quality"])
        >>> clear_pct = (masks["clear"].sum() / masks["clear"].size) * 100
        >>> print(f"Clear pixels: {clear_pct:.2f}%")
    """
    available = [
        # Individual bits
        "fill",
        "valid_data",
        "dropped",
        "cloud",
        "no_cloud",
        "land_water",
        # Saturation (bits 2-3)
        "sat_none",
        "sat_low",
        "sat_medium",
        "sat_high",
        "sat_any",
        # Cloud confidence (bits 5-6)
        "cloud_conf_none",
        "cloud_conf_low",
        "cloud_conf_medium",
        "cloud_conf_high",
        # Shadow confidence (bits 7-8)
        "shadow_conf_none",
        "shadow_conf_low",
        "shadow_conf_medium",
        "shadow_conf_high",
        # Snow confidence (bits 9-10)
        "snow_conf_none",
        "snow_conf_low",
        "snow_conf_medium",
        "snow_conf_high",
        # Scan line artefacts (bits 11-14)
        "sla_band4",
        "sla_band5",
        "sla_band6",
        "sla_band7",
        "sla_any",
        # Combinations
        "clear",
        "high_quality",
        "cloud_or_shadow",
        "usable",
    ]

    if flags == "all":
        flags = available

    results = {}

    # Pre-calculate common values
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

    for flag in flags:
        # --- INDIVIDUAL BITS ---
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
            results[flag] = bit15  # 1=water, 0=land

        # --- SATURATION (bits 2-3) ---
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

        # --- CLOUD CONFIDENCE (bits 5-6) ---
        elif flag == "cloud_conf_none":
            results[flag] = (cloud_conf == 0).astype(np.uint8)

        elif flag == "cloud_conf_low":
            results[flag] = (cloud_conf == 1).astype(np.uint8)

        elif flag == "cloud_conf_medium":
            results[flag] = (cloud_conf == 2).astype(np.uint8)

        elif flag == "cloud_conf_high":
            results[flag] = (cloud_conf == 3).astype(np.uint8)

        # --- SHADOW CONFIDENCE (bits 7-8) ---
        elif flag == "shadow_conf_none":
            results[flag] = (shadow_conf == 0).astype(np.uint8)

        elif flag == "shadow_conf_low":
            results[flag] = (shadow_conf == 1).astype(np.uint8)

        elif flag == "shadow_conf_medium":
            results[flag] = (shadow_conf == 2).astype(np.uint8)

        elif flag == "shadow_conf_high":
            results[flag] = (shadow_conf == 3).astype(np.uint8)

        # --- SNOW CONFIDENCE (bits 9-10) ---
        elif flag == "snow_conf_none":
            results[flag] = (snow_conf == 0).astype(np.uint8)

        elif flag == "snow_conf_low":
            results[flag] = (snow_conf == 1).astype(np.uint8)

        elif flag == "snow_conf_medium":
            results[flag] = (snow_conf == 2).astype(np.uint8)

        elif flag == "snow_conf_high":
            results[flag] = (snow_conf == 3).astype(np.uint8)

        # --- SCAN LINE ARTEFACTS (bits 11-14) ---
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

        # --- LOGICAL COMBINATIONS ---
        elif flag == "clear":
            # Valid AND no cloud AND no high-conf shadow
            valid = (~bit0.astype(bool)).astype(np.uint8)
            no_cloud = (~bit4.astype(bool)).astype(np.uint8)
            no_high_shadow = (shadow_conf != 3).astype(np.uint8)
            results[flag] = (valid & no_cloud & no_high_shadow).astype(np.uint8)

        elif flag == "high_quality":
            # Valid AND clear AND no saturation AND no SLA
            valid = (~bit0.astype(bool)).astype(np.uint8)
            no_cloud = (~bit4.astype(bool)).astype(np.uint8)
            no_shadow = (shadow_conf < 3).astype(np.uint8)
            no_sat = (sat == 0).astype(np.uint8)
            no_sla = (~(bit11 | bit12 | bit13 | bit14).astype(bool)).astype(np.uint8)
            results[flag] = (valid & no_cloud & no_shadow & no_sat & no_sla).astype(np.uint8)

        elif flag == "cloud_or_shadow":
            # Cloud OR medium/high shadow confidence
            cloud_or_shadow = bit4 | (shadow_conf >= 2).astype(np.uint8)
            results[flag] = cloud_or_shadow.astype(np.uint8)

        elif flag == "usable":
            # Valid AND no high-conf cloud AND no high saturation
            valid = (~bit0.astype(bool)).astype(np.uint8)
            no_high_cloud = (cloud_conf < 3).astype(np.uint8)
            no_high_sat = (sat < 3).astype(np.uint8)
            results[flag] = (valid & no_high_cloud & no_high_sat).astype(np.uint8)

        else:
            raise ValueError(f"Unknown flag '{flag}' for ESA MSS. Available: {available}")

    return results


def decode_qa_pixel_tm(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode TM/ETM+/OLI QA_PIXEL band into ALL possible masks.

    INDIVIDUAL BITS:
    - fill: Fill data (bit 0 = 1)
    - valid_data: Valid image data (bit 0 = 0) [INVERSE of fill]
    - dilated_cloud: Dilated cloud (bit 1 = 1)
    - cloud: High confidence cloud (bit 3 = 1)
    - no_cloud: No cloud (bit 3 = 0) [INVERSE of cloud]
    - cloud_shadow: High confidence cloud shadow (bit 4 = 1)
    - no_shadow: No shadow (bit 4 = 0) [INVERSE of shadow]
    - snow: High confidence snow (bit 5 = 1)
    - clear: Clear (bit 6 = 1)
    - water: Water (bit 7 = 1)
    - land: Land (bit 7 = 0) [INVERSE of water]

    CONFIDENCE LEVELS:
    - cloud_conf_none, cloud_conf_low, cloud_conf_medium, cloud_conf_high (bits 8-9)
    - shadow_conf_none, shadow_conf_low, shadow_conf_high (bits 10-11)
    - snow_conf_none, snow_conf_low, snow_conf_high (bits 12-13)

    LOGICAL COMBINATIONS:
    - cloud_or_shadow: Cloud OR shadow (high conf)
    - cloud_shadow_dilated: Cloud OR shadow OR dilated
    - high_quality: Valid AND clear AND low cloud/shadow confidence

    Args:
        qa_array: QA_PIXEL array (H, W) uint16
        flags: List of flags to extract, or "all" for everything

    Returns:
        Dictionary mapping flag name to binary mask
    """
    available = [
        # Individual bits
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
        # Cloud confidence
        "cloud_conf_none",
        "cloud_conf_low",
        "cloud_conf_medium",
        "cloud_conf_high",
        # Shadow confidence
        "shadow_conf_none",
        "shadow_conf_low",
        "shadow_conf_high",
        # Snow confidence
        "snow_conf_none",
        "snow_conf_low",
        "snow_conf_high",
        # Combinations
        "cloud_or_shadow",
        "cloud_shadow_dilated",
        "high_quality",
    ]

    if flags == "all":
        flags = available

    results = {}

    # Pre-calculate common values
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

    for flag in flags:
        # --- INDIVIDUAL BITS ---
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

        # --- CLOUD CONFIDENCE ---
        elif flag == "cloud_conf_none":
            results[flag] = (cloud_conf == 0).astype(np.uint8)
        elif flag == "cloud_conf_low":
            results[flag] = (cloud_conf == 1).astype(np.uint8)
        elif flag == "cloud_conf_medium":
            results[flag] = (cloud_conf == 2).astype(np.uint8)
        elif flag == "cloud_conf_high":
            results[flag] = (cloud_conf == 3).astype(np.uint8)

        # --- SHADOW CONFIDENCE ---
        elif flag == "shadow_conf_none":
            results[flag] = (shadow_conf == 0).astype(np.uint8)
        elif flag == "shadow_conf_low":
            results[flag] = (shadow_conf == 1).astype(np.uint8)
        elif flag == "shadow_conf_high":
            results[flag] = (shadow_conf == 3).astype(np.uint8)

        # --- SNOW CONFIDENCE ---
        elif flag == "snow_conf_none":
            results[flag] = (snow_conf == 0).astype(np.uint8)
        elif flag == "snow_conf_low":
            results[flag] = (snow_conf == 1).astype(np.uint8)
        elif flag == "snow_conf_high":
            results[flag] = (snow_conf == 3).astype(np.uint8)

        # --- COMBINATIONS ---
        elif flag == "cloud_or_shadow":
            results[flag] = (bit3 | bit4).astype(np.uint8)
        elif flag == "cloud_shadow_dilated":
            results[flag] = (bit1 | bit3 | bit4).astype(np.uint8)
        elif flag == "high_quality":
            valid = (~bit0.astype(bool)).astype(np.uint8)
            low_cloud = ((cloud_conf == 0) | (cloud_conf == 1)).astype(np.uint8)
            low_shadow = ((shadow_conf == 0) | (shadow_conf == 1)).astype(np.uint8)
            results[flag] = (valid & bit6 & low_cloud & low_shadow).astype(np.uint8)

        else:
            raise ValueError(f"Unknown flag '{flag}' for TM/ETM+/OLI. Available: {available}")

    return results


# QA_RADSAT DECODERS


def decode_qa_radsat_mss(
    qa_array: np.ndarray,
    flags: list[str] | Literal["all"] = "all",
) -> dict[str, np.ndarray]:
    """Decode MSS QA_RADSAT band into saturation masks.

    Available flags for MSS:
    - b1_sat, b2_sat, b3_sat, b4_sat: Band saturation (bits 0-3)
    - b5_sat: Band 5 saturation (bit 4, MSS4/MSS5 only)
    - b6_sat: Band 6 saturation (bit 5, MSS4/MSS5 only)
    - dropped: Dropped pixel (bit 9)
    - any_saturated: Any band saturated [COMBINATION]

    Args:
        qa_array: QA_RADSAT array (H, W) uint16
        flags: List of flags to extract, or "all" for everything

    Returns:
        Dictionary mapping flag name to binary mask
    """
    available = ["b1_sat", "b2_sat", "b3_sat", "b4_sat", "b5_sat", "b6_sat", "dropped", "any_saturated"]

    if flags == "all":
        flags = available

    results = {}

    bit_map = {"b1_sat": 0, "b2_sat": 1, "b3_sat": 2, "b4_sat": 3, "b5_sat": 4, "b6_sat": 5, "dropped": 9}

    for flag in flags:
        if flag == "any_saturated":
            # Combine bits 0-5 (all band saturation)
            sat = extract_bit(qa_array, 0)
            for bit in range(1, 6):
                sat |= extract_bit(qa_array, bit)
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
    """Decode TM/ETM+ QA_RADSAT band into saturation masks.

    Available flags for TM/ETM+:
    - b1_sat through b7_sat: Band saturation
    - b6h_sat: Band 6H saturation (ETM+ bit 8)
    - dropped: Dropped pixel (bit 9)
    - any_saturated: Any band saturated [COMBINATION]

    Args:
        qa_array: QA_RADSAT array (H, W) uint16
        flags: List of flags to extract, or "all" for everything

    Returns:
        Dictionary mapping flag name to binary mask
    """
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

    results = {}

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

    for flag in flags:
        if flag == "any_saturated":
            sat = extract_bit(qa_array, 0)
            for bit in [1, 2, 3, 4, 5, 6, 8]:
                sat |= extract_bit(qa_array, bit)
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
    """Decode OLI QA_RADSAT band into saturation masks.

    Available flags for OLI:
    - b1_sat through b9_sat: Band saturation (bits 0-8)
    - b10_sat, b11_sat: Thermal band saturation (bits 9-10, OLI8/9)
    - dropped: Dropped pixel (bit 11, note: different from TM!)

    Args:
        qa_array: QA_RADSAT array (H, W) uint16
        flags: List of flags to extract, or "all" for everything

    Returns:
        Dictionary mapping flag name to binary mask
    """
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

    results = {}

    bit_map = {
        "b1_sat": 0,
        "b2_sat": 1,
        "b3_sat": 2,
        "b4_sat": 3,
        "b5_sat": 4,
        "b6_sat": 5,
        "b7_sat": 6,
        "b8_sat": 7,
        "b9_sat": 8,
        "b10_sat": 9,
        "b11_sat": 10,
        "dropped": 11,
    }

    for flag in flags:
        if flag in bit_map:
            results[flag] = extract_bit(qa_array, bit_map[flag])
        else:
            raise ValueError(f"Unknown flag '{flag}' for OLI QA_RADSAT. Available: {available}")

    return results


# COMBINED MASKS (COMMON USE CASES)


def get_cloud_shadow_mask(
    qa_pixel: np.ndarray,
    sensor: Literal["MSS", "TM", "ETM+", "OLI"],
    include_dilated: bool = True,
    confidence: Literal["any", "high"] = "high",
) -> np.ndarray:
    """Get combined cloud + shadow mask.

    Args:
        qa_pixel: QA_PIXEL array (H, W) uint16
        sensor: Sensor type
        include_dilated: Include dilated cloud (TM/ETM+/OLI only)
        confidence: "any" = cloud bit OR conf_high, "high" = conf_high only

    Returns:
        Binary mask (1 = cloud or shadow, 0 = clear)
    """
    if sensor == "MSS":
        if confidence == "high":
            masks = decode_qa_pixel_mss(qa_pixel, ["cloud_conf_high"])
            return masks["cloud_conf_high"]
        else:
            masks = decode_qa_pixel_mss(qa_pixel, ["cloud"])
            return masks["cloud"]

    else:  # TM, ETM+, OLI
        flags = ["cloud", "cloud_shadow"]
        if include_dilated:
            flags.append("dilated_cloud")
        if confidence == "high":
            flags.extend(["cloud_conf_high", "shadow_conf_high"])

        masks = decode_qa_pixel_tm(qa_pixel, flags)

        # Combine: cloud OR shadow OR dilated
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
    """Get clear pixel mask (inverse of cloud+shadow+fill).

    Args:
        qa_pixel: QA_PIXEL array (H, W) uint16
        sensor: Sensor type

    Returns:
        Binary mask (1 = clear, 0 = not clear)
    """
    if sensor == "MSS":
        masks = decode_qa_pixel_mss(qa_pixel, ["fill", "cloud"])
        return (~(masks["fill"] | masks["cloud"])).astype(np.uint8)
    else:
        # TM/ETM+/OLI have a "clear" bit
        masks = decode_qa_pixel_tm(qa_pixel, ["clear"])
        return masks["clear"]


# EXPORT UTILITIES


def save_mask_geotiff(
    mask: np.ndarray,
    output_path: pathlib.Path,
    transform: Affine,
    crs: str,
    nodata: int = 255,
) -> None:
    """Save binary mask as GeoTIFF.

    Args:
        mask: Binary mask (H, W) uint8
        output_path: Output .tif path
        transform: Rasterio affine transform
        crs: CRS string (e.g., "EPSG:32630")
        nodata: NoData value
    """
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
    """Export all QA masks from a Landsat scene.

    Args:
        qa_pixel_path: Path to QA_PIXEL.tif
        output_dir: Output directory for masks
        sensor: Sensor type
        qa_radsat_path: Optional path to QA_RADSAT.tif

    Returns:
        Dictionary mapping mask name to output path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read QA_PIXEL
    with rio.open(qa_pixel_path) as src:
        qa_pixel = src.read(1)
        transform = src.transform
        crs = src.crs

    exported = {}

    # Decode QA_PIXEL
    if sensor == "MSS":
        pixel_masks = decode_qa_pixel_mss(qa_pixel, "all")
    else:
        pixel_masks = decode_qa_pixel_tm(qa_pixel, "all")

    # Export QA_PIXEL masks
    for name, mask in pixel_masks.items():
        outpath = output_dir / f"qa_pixel_{name}.tif"
        save_mask_geotiff(mask, outpath, transform, crs)
        exported[f"pixel_{name}"] = outpath

    # Export combined masks
    cloud_shadow = get_cloud_shadow_mask(qa_pixel, sensor)
    outpath = output_dir / "cloud_shadow_combined.tif"
    save_mask_geotiff(cloud_shadow, outpath, transform, crs)
    exported["cloud_shadow_combined"] = outpath

    clear = get_clear_mask(qa_pixel, sensor)
    outpath = output_dir / "clear_mask.tif"
    save_mask_geotiff(clear, outpath, transform, crs)
    exported["clear_mask"] = outpath

    # Decode QA_RADSAT if provided
    if qa_radsat_path and qa_radsat_path.exists():
        with rio.open(qa_radsat_path) as src:
            qa_radsat = src.read(1)

        if sensor == "MSS":
            radsat_masks = decode_qa_radsat_mss(qa_radsat, "all")
        elif sensor in ["TM", "ETM+"]:
            radsat_masks = decode_qa_radsat_tm(qa_radsat, "all")
        else:  # OLI
            radsat_masks = decode_qa_radsat_oli(qa_radsat, "all")

        for name, mask in radsat_masks.items():
            outpath = output_dir / f"qa_radsat_{name}.tif"
            save_mask_geotiff(mask, outpath, transform, crs)
            exported[f"radsat_{name}"] = outpath

    return exported


def export_all_esa_mss_masks(
    bqa_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Export ALL ESA MSS BQA masks with statistics.

    Args:
        bqa_path: Path to ESA BQA.TIF file
        output_dir: Output directory for masks

    Returns:
        Dictionary mapping mask name to output path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read BQA
    with rio.open(bqa_path) as src:
        bqa = src.read(1)
        transform = src.transform
        crs = src.crs

    # Decode ALL masks
    all_masks = decode_qa_pixel_mss_esa(bqa, flags="all")

    exported = {}
    total_pixels = bqa.size
    effective_pixels = (~((bqa >> 0) & 1).astype(bool)).sum()

    print(f"\n{'='*70}")
    print(f"ESA MSS BQA Analysis: {bqa_path.name}")
    print(f"{'='*70}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Effective pixels: {effective_pixels:,} ({effective_pixels/total_pixels*100:.2f}%)")
    print(f"\n{'Mask Name':<30} {'Count':>12} {'% Total':>10} {'% Effective':>12}")
    print(f"{'-'*70}")

    # Export each mask with statistics
    for name, mask in all_masks.items():
        outpath = output_dir / f"esa_mss_{name}.tif"
        save_mask_geotiff(mask, outpath, transform, crs)
        exported[name] = outpath

        count = mask.sum()
        pct_total = (count / total_pixels) * 100
        pct_effective = (count / effective_pixels) * 100 if effective_pixels > 0 else 0

        print(f"{name:<30} {count:>12,} {pct_total:>9.2f}% {pct_effective:>11.2f}%")

    print(f"{'='*70}")
    print(f"✅ Exported {len(exported)} masks to {output_dir}\n")

    return exported


def cloud_shadow_mask_mss_gee(qa_pixel: np.ndarray) -> np.ndarray:
    """Generate non-binary cloud mask for MSS GEE/USGS Collection 2.

    Values:
        0: Clear
        1: Cloud

    Args:
        qa_pixel: QA_PIXEL array (H, W) uint16

    Returns:
        Classified mask (H, W) uint8
    """
    mask = np.zeros_like(qa_pixel, dtype=np.uint8)
    cloud = extract_bit(qa_pixel, 3).astype(bool)  # without confidence
    mask[cloud] = 1
    return mask


def cloud_shadow_mask_mss_esa(bqa: np.ndarray) -> np.ndarray:
    """Generate non-binary cloud+shadow mask for MSS ESA Collection 1.

    Values:
        0: Clear
        1: Cloud (overwrites shadow)
        2: Shadow

    Args:
        bqa: BQA array (H, W) uint16

    Returns:
        Classified mask (H, W) uint8
    """
    mask = np.zeros_like(bqa, dtype=np.uint8)
    cloud = extract_bit(bqa, 4).astype(bool)
    shadow_conf = extract_bits(bqa, 7, 8)
    shadow = (shadow_conf >= 2).astype(bool)  # Medium
    mask[shadow] = 2
    mask[cloud] = 1

    return mask


def cloud_shadow_mask_tm_gee(qa_pixel: np.ndarray) -> np.ndarray:
    """Generate non-binary cloud+shadow mask for TM/ETM+/OLI GEE Collection 2.

    Values:
        0: Clear
        1: Cloud (overwrites shadow)
        2: Shadow

    Args:
        qa_pixel: QA_PIXEL array (H, W) uint16

    Returns:
        Classified mask (H, W) uint8
    """
    mask = np.zeros_like(qa_pixel, dtype=np.uint8)
    cloud = extract_bit(qa_pixel, 3).astype(bool)
    shadow = extract_bit(qa_pixel, 4).astype(bool)
    mask[shadow] = 2  # Shadow
    mask[cloud] = 1  # Cloud overwrites shadow

    return mask
