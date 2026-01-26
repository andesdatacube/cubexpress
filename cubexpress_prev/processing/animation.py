"""Animation utilities for creating GIFs from image sequences."""

from __future__ import annotations

import pathlib
from typing import Any

from cubexpress.download.core import download_manifest, temp_workspace
from cubexpress.formats.specs import EEFileFormat, ExportFormat


def create_gif(
    image_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    duration: int = 500,
    loop: int = 0,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> pathlib.Path:
    """Create animated GIF from image sequence.

    Args:
        image_paths: List of PNG/JPEG image paths in order
        output_path: Output GIF path
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
        background_color: RGB tuple for background (replaces transparency)

    Returns:
        Path to created GIF
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required for GIF creation: pip install Pillow") from None

    if not image_paths:
        raise ValueError("No images provided for GIF creation")

    # Load all frames and convert to RGB (no transparency issues)
    frames = []
    for path in image_paths:
        img = Image.open(path)

        # Handle transparency by compositing onto background
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, background_color)
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        frames.append(img)

    # Save as animated GIF
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to palette mode for GIF
    frames_p = []
    for frame in frames:
        frame_p = frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames_p.append(frame_p)

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
    requests: Any,  # RequestSet or DataFrame
    visualization: Any | None = None,
    output_path: pathlib.Path | str = "animation.gif",
    duration: int = 500,
    background_color: tuple[int, int, int] = (0, 0, 0),
    keep_frames: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Create animated GIF from RequestSet with FIXED geometry.

    Uses getPixels/computePixels directly to ensure correct geometry
    without distortion.

    Args:
        requests: RequestSet or DataFrame from table_to_requestset
        visualization: VisualizationOptions (e.g., VisPresets.s2_truecolor())
        output_path: Output GIF path
        duration: Frame duration in ms
        background_color: RGB tuple for background (replaces transparency)
        keep_frames: If provided, save PNG frames to this folder

    Returns:
        Path to created GIF

    Example:
        >>> df = sensor_table("S2_TOA", lon=-0.3763, lat=39.4699, edge_size=256,
        ...                   start="2024-01-01", end="2024-03-01", max_cloud=20)
        >>> requests = table_to_requestset(df, mosaic=True)
        >>> create_gif_from_requests(
        ...     requests,
        ...     visualization=VisPresets.s2_truecolor(),
        ...     output_path="valencia_timelapse.gif",
        ...     duration=300,
        ... )
    """
    output_path = pathlib.Path(output_path)

    # Get dataframe
    if hasattr(requests, "_dataframe"):
        dataframe = requests._dataframe
    else:
        dataframe = requests

    if dataframe.empty:
        raise ValueError("Request set is empty")

    n_frames = len(dataframe)

    if visualization is None:
        raise ValueError("visualization is required for GIF creation. " "Use VisPresets.s2_truecolor() or similar.")

    png_format = ExportFormat(
        file_format=EEFileFormat.PNG,
        visualization=visualization,
    )

    # Determine frame folder
    if keep_frames:
        frame_dir = pathlib.Path(keep_frames)
        frame_dir.mkdir(parents=True, exist_ok=True)
        use_temp = False
        tmp_context = None
    else:
        use_temp = True
        tmp_context = temp_workspace(prefix="cubexpress_gif_")

    try:
        if use_temp:
            frame_dir = tmp_context.__enter__()

        frame_paths = []

        print(f"⏳ Rendering {n_frames} frames...", end="", flush=True)

        for i, (_, row) in enumerate(dataframe.iterrows()):
            frame_path = frame_dir / f"frame_{i:04d}_{row.id}.png"

            download_manifest(
                ulist=row.manifest,
                full_outname=frame_path,
                export_format=png_format,
            )
            frame_paths.append(frame_path)

        print(f"\r✅ Rendered {n_frames} frames")
        print("⏳ Creating GIF...", end="", flush=True)

        result = create_gif(
            image_paths=frame_paths,
            output_path=output_path,
            duration=duration,
            background_color=background_color,
        )

        print(f"\r✅ Created GIF: {output_path} ({n_frames} frames)")

    finally:
        if use_temp and tmp_context:
            tmp_context.__exit__(None, None, None)

    return result
