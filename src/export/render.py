"""Streaming export: a single pass over rendered frames feeds MP4, one or
more GIF variants, and PNG snapshot outputs simultaneously.

Matplotlib rendering, not encoding, dominates the cost of producing
these animations, so this module is deliberately structured to consume
a frame generator exactly once, in `export_stream`: every output
format (MP4, each GIF variant, PNG snapshots) is written from the same
single pass, streaming straight into the video/GIF encoders as frames
are produced, keeping only the current frame in memory at a time.

`export_stream` is generic over *any* RGBA frame generator — this
project's own animation, the genetic algorithm project's 2D/3D
animations, and future series entries all share this exact module
unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio
import numpy as np
from PIL import Image

DEFAULT_FPS = 24


@dataclass(frozen=True)
class GifVariant:
    """One GIF output configuration (e.g. "repo" vs "blog" weight class).

    Attributes:
        name: Suffix used in the output filename, e.g. "repo" ->
            "{basename}_repo.gif".
        stride: Keep every Nth rendered frame. Playback duration is
            preserved by scaling per-frame GIF duration by the same
            stride, so every variant runs for the same real-world
            length regardless of how many unique frames it contains.
        width: Output width/height in pixels (square).
        max_colors: Palette size cap.
    """

    name: str
    stride: int
    width: int
    max_colors: int


DEFAULT_GIF_VARIANTS = (
    GifVariant(name="repo", stride=2, width=440, max_colors=110),
    GifVariant(name="blog", stride=4, width=300, max_colors=64),
)


def export_stream(
    frames: Iterable[np.ndarray],
    out_dir: str | Path,
    basename: str,
    total_frames: int,
    fps: int = DEFAULT_FPS,
    gif_variants: tuple[GifVariant, ...] = DEFAULT_GIF_VARIANTS,
    mid_index: int | None = None,
) -> dict[str, Path]:
    """Stream any RGBA frame sequence once into MP4 + GIF variant(s) + PNGs.

    Args:
        frames: Iterable/generator of (H, W, 4) uint8 RGBA frames, in
            playback order. Consumed exactly once.
        out_dir: Directory to write outputs into (created if missing).
        basename: Filename stem shared by all outputs.
        total_frames: Total frame count, known ahead of render time
            (e.g. via a cheap timing-only pass), used to place the
            "mid" PNG snapshot without buffering the whole sequence.
        fps: Playback frame rate for the MP4 and (pre-stride) GIFs.
        gif_variants: One or more `GifVariant` configs, all produced in
            the same streaming pass.
        mid_index: Frame index to save as the "mid" PNG snapshot.
            Defaults to `total_frames // 2`.

    Returns:
        Dict mapping output kind ("mp4", "gif_<variant name>",
        "png_initial", "png_mid", "png_final", "png_hero") to path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_index = total_frames // 2 if mid_index is None else mid_index

    paths: dict[str, Path] = {
        "mp4": out_dir / f"{basename}.mp4",
        "png_initial": out_dir / f"{basename}_frame_initial.png",
        "png_mid": out_dir / f"{basename}_frame_mid.png",
        "png_final": out_dir / f"{basename}_frame_final.png",
        "png_hero": out_dir / f"{basename}_hero.png",
    }
    for variant in gif_variants:
        paths[f"gif_{variant.name}"] = out_dir / f"{basename}_{variant.name}.gif"

    mp4_writer = imageio.get_writer(
        str(paths["mp4"]), fps=fps, codec="libx264", quality=8
    )
    gif_writers = {
        variant.name: imageio.get_writer(
            str(paths[f"gif_{variant.name}"]),
            duration=variant.stride / fps,
            loop=0,
            mode="I",
            palettesize=variant.max_colors,
        )
        for variant in gif_variants
    }

    last_frame: np.ndarray | None = None
    try:
        for idx, frame in enumerate(frames):
            rgb = frame[..., :3]
            mp4_writer.append_data(rgb)

            resize_cache: dict[int, np.ndarray] = {}
            for variant in gif_variants:
                if idx % variant.stride != 0:
                    continue
                if variant.width not in resize_cache:
                    resize_cache[variant.width] = np.asarray(
                        Image.fromarray(rgb).resize(
                            (variant.width, variant.width), Image.LANCZOS
                        )
                    )
                gif_writers[variant.name].append_data(resize_cache[variant.width])

            if idx == 0:
                imageio.imwrite(str(paths["png_initial"]), frame)
            if idx == mid_index:
                imageio.imwrite(str(paths["png_mid"]), frame)

            last_frame = frame
    finally:
        mp4_writer.close()
        for writer in gif_writers.values():
            writer.close()

    if last_frame is not None:
        imageio.imwrite(str(paths["png_final"]), last_frame)
        imageio.imwrite(str(paths["png_hero"]), last_frame)  # RGBA preserved

    return paths
