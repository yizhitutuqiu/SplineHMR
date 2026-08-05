from __future__ import annotations

"""Minimal video helpers adapted from GVHMR's hmr4d/utils/video_io_utils.py."""

from pathlib import Path
from typing import Iterator

import imageio.v3 as iio
import numpy as np
import torch


def get_video_lwh(video_path: str | Path) -> tuple[int, int, int]:
    """Return (length, width, height) for an RGB video."""
    length, height, width, _ = iio.improps(str(video_path), plugin="pyav").shape
    return int(length), int(width), int(height)


def get_video_reader(video_path: str | Path) -> Iterator[np.ndarray]:
    return iio.imiter(str(video_path), plugin="pyav")


def get_writer(video_path: str | Path, fps: float = 30, crf: int = 23):
    """Return an imageio/pyav writer. Remember to call close()."""
    writer = iio.imopen(str(video_path), "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=float(fps))
    writer._video_stream.options = {"crf": str(int(crf))}
    return writer


def save_video(images, video_path: str | Path, fps: float = 30, crf: int = 23) -> None:
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.asarray(images).astype(np.uint8)
    with get_writer(video_path, fps=fps, crf=crf) as writer:
        writer.write(images)
