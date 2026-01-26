"""TOA calibration and band transformations for Earth Engine imagery."""

from __future__ import annotations

import ee


def _is_mss_collection(collection: str | list[str]) -> bool:
    """Check if collection is MSS."""
    coll_str = collection[0] if isinstance(collection, list) else collection
    return "/LM0" in coll_str


def _is_toa_collection(collection: str | list[str]) -> bool:
    """Check if collection is TOA (has _TOA in name or is S2_HARMONIZED)."""
    coll_str = collection[0] if isinstance(collection, list) else collection
    return "_TOA" in coll_str or "S2_HARMONIZED" in coll_str


def _should_apply_mss_toa(asset_id: str, toa: bool | None) -> bool:
    """Determine if MSS TOA conversion should be applied."""
    if "/LM0" not in asset_id:
        return False

    if toa is None:
        return False

    return toa


def _should_apply_toa_scaling(asset_id: str, toa: bool | None) -> bool:
    """Determine if TOA scaling (0-1 -> 0-10000) should be applied."""
    if "_TOA" not in asset_id or "/LM0" in asset_id:
        return False

    if toa is None:
        return True

    return toa


def _get_spectral_bands(bands: list[str]) -> list[str]:
    """Get only spectral bands (exclude QA and thermal bands)."""
    qa_prefixes = (
        "QA_",
        "SAT_",
        "SR_QA_",
        "ST_QA_",
        "ST_ATRAN",
        "ST_CDIST",
        "ST_DRAD",
        "ST_EMIS",
        "ST_EMSD",
        "ST_TRAD",
        "ST_URAD",
        "ST_B10",
    )
    return [b for b in bands if not b.startswith(qa_prefixes)]


def _scale_toa_bands(image_source: str | ee.Image, bands: list[str]) -> ee.Image:
    """Scale TOA reflectance bands to UINT16 (0-10000), preserve QA bands."""
    if isinstance(image_source, str):
        image_source = ee.Image(image_source)

    spectral_bands = _get_spectral_bands(bands)
    qa_bands = [b for b in bands if b not in spectral_bands]

    if not spectral_bands:
        return image_source

    scaled = image_source.select(spectral_bands).multiply(10000).toUint16()

    if qa_bands:
        qa_img = image_source.select(qa_bands)
        return ee.ImageCollection([scaled, qa_img]).toBands().rename(bands)

    return scaled


def _apply_toa_to_single(image_source: str | ee.Image, bands: list[str]) -> ee.Image:
    """Apply TOA calibration to MSS images and scale to UINT16."""
    if isinstance(image_source, str):
        image_source = ee.Image(image_source)

    spectral_bands = _get_spectral_bands(bands)
    qa_bands = [b for b in bands if b not in spectral_bands]

    if not spectral_bands:
        return image_source

    if spectral_bands:
        toa_img = ee.Algorithms.Landsat.TOA(image_source.select(spectral_bands))
        scaled = toa_img.multiply(10000).toUint16()

    if qa_bands:
        qa_img = image_source.select(qa_bands)
        return ee.ImageCollection([scaled, qa_img]).toBands().rename(bands)

    return scaled
