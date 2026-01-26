"""ESA download utilities."""

from __future__ import annotations

import asyncio
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import rasterio as rio

try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def _is_in_async_context():
    """Check if running in async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


if _is_in_async_context():
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass


async def _download_esa_file_async(
    url: str,
    username: str,
    password: str,
    output_dir: Path,
    headless: bool,
) -> Path:
    """Internal async download function."""

    os.environ.setdefault("NO_AT_BRIDGE", "1")
    os.environ.setdefault("GTK_MODULES", "")

    filename = url.split("/")[-1]
    expected_path = output_dir / filename

    if expected_path.exists():
        return expected_path

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless, args=["--disable-gpu", "--no-sandbox"])
        ctx = await browser.new_context(accept_downloads=True)
        page = await ctx.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)

            await page.fill("input#username", username)
            await page.fill("input#password", password)
            await page.click("button[type='submit']")
            await page.wait_for_timeout(5000)

            async with page.expect_download(timeout=900000) as download_info:
                await page.evaluate(
                    f"""
                    () => {{
                        const a = document.createElement('a');
                        a.href = '{url}';
                        a.download = '{filename}';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }}
                """
                )

            download = await download_info.value
            await download.save_as(expected_path)

            return expected_path

        finally:
            await browser.close()


async def _download_esa_files_async(
    urls: list[str],
    username: str,
    password: str,
    output_dir: Path,
    headless: bool,
) -> list[Path]:
    """Internal async batch download function."""

    os.environ.setdefault("NO_AT_BRIDGE", "1")
    os.environ.setdefault("GTK_MODULES", "")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless, args=["--disable-gpu", "--no-sandbox"])
        ctx = await browser.new_context(accept_downloads=True)
        page = await ctx.new_page()

        try:
            await page.goto(urls[0], wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
            await page.fill("input#username", username)
            await page.fill("input#password", password)
            await page.click("button[type='submit']")
            await page.wait_for_timeout(5000)

            downloaded = []

            for url in urls:
                filename = url.split("/")[-1]
                expected_path = output_dir / filename

                if expected_path.exists():
                    downloaded.append(expected_path)
                    continue

                async with page.expect_download(timeout=900000) as download_info:
                    await page.evaluate(
                        f"""
                        () => {{
                            const a = document.createElement('a');
                            a.href = '{url}';
                            a.download = '{filename}';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                        }}
                    """
                    )

                download = await download_info.value
                await download.save_as(expected_path)
                downloaded.append(expected_path)

            return downloaded

        finally:
            await browser.close()


def download_esa_file(
    url: str,
    username: str,
    password: str,
    output_dir: str | Path = "esa_raw",
    headless: bool = True,
) -> Path:
    """Download ESA file with authentication."""

    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError(
            "ESA downloads require Playwright:\n" "  pip install cubexpress[esa]\n" "  playwright install chromium"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_in_async_context():
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_download_esa_file_async(url, username, password, output_dir, headless))
    else:
        return asyncio.run(_download_esa_file_async(url, username, password, output_dir, headless))


def download_esa_files(
    urls: list[str],
    username: str,
    password: str,
    output_dir: str | Path = "esa_raw",
    headless: bool = True,
) -> list[Path]:
    """Download multiple ESA files."""

    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("pip install cubexpress[esa] && playwright install chromium")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_in_async_context():
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_download_esa_files_async(urls, username, password, output_dir, headless))
    else:
        return asyncio.run(_download_esa_files_async(urls, username, password, output_dir, headless))


def _parse_mtl(mtl_path: Path) -> dict:
    """Parse MTL file to dictionary."""

    metadata = {}

    with mtl_path.open() as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("GROUP") and not line.startswith("END"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"')

                # Try to convert to number (avoid dates with dashes)
                if "-" not in value and ":" not in value:
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass

                metadata[key] = value

    return metadata


def extract_esa_bands(
    zip_path: Path | str,
    output_dir: Path | str,
    return_array: bool = True,
) -> tuple[np.ndarray | Path, dict]:
    """Extract ESA ZIP to 5-band GeoTIFF and parse MTL.

    Args:
        zip_path: Path to ESA .SIP.ZIP file
        output_dir: Output directory
        return_array: If True, return array. If False, return TIF path.

    Returns:
        Tuple of (array or path, metadata_dict)
        - array: (5, height, width) with bands [B1, B2, B3, B4, BQA]
        - metadata: Parsed MTL as dictionary
    """

    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp_extract"
    temp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    inner_zip = next(iter(temp_dir.glob("*.ZIP")))
    inner_temp = temp_dir / "inner"
    inner_temp.mkdir(exist_ok=True)

    with zipfile.ZipFile(inner_zip, "r") as zf:
        zf.extractall(inner_temp)

    tiff_dir = next(iter(inner_temp.rglob("*.TIFF")))

    b1 = next(iter(tiff_dir.glob("*_B1.TIF")))
    b2 = next(iter(tiff_dir.glob("*_B2.TIF")))
    b3 = next(iter(tiff_dir.glob("*_B3.TIF")))
    b4 = next(iter(tiff_dir.glob("*_B4.TIF")))
    bqa = next(iter(tiff_dir.glob("*_BQA.TIF")))
    mtl = next(iter(tiff_dir.glob("*_MTL.txt")))

    temp_name = zip_path.stem.replace(".SIP", "")

    # Create clean profile
    with rio.open(b1) as src:
        profile = {
            "driver": "GTiff",
            "dtype": src.dtypes[0],
            "width": src.width,
            "height": src.height,
            "count": 5,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "DEFLATE",
        }

    output_tif = output_dir / f"{temp_name}.tif"

    with rio.open(output_tif, "w", **profile) as dst:
        for i, band_path in enumerate([b1, b2, b3, b4, bqa], start=1):
            with rio.open(band_path) as src:
                dst.write(src.read(1), i)

    # Copy MTL to output directory
    output_mtl = output_dir / f"{temp_name}_MTL.txt"
    shutil.copy(mtl, output_mtl)

    # Parse MTL
    metadata = _parse_mtl(mtl)

    # Cleanup
    shutil.rmtree(temp_dir)

    # Return array or path
    if return_array:
        with rio.open(output_tif) as src:
            array = src.read()
        return array, metadata
    else:
        return output_tif, metadata
