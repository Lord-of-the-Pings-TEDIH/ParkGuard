"""
app/pipeline/frame_extractor.py

Utilities for extracting sampled frames from a video file using OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Generator


def extract_frames(
    video_path: str,
    fps_target: int,
) -> Generator[tuple[int, int, np.ndarray], None, None]:
    """Yield sampled frames from *video_path* at approximately *fps_target* frames/sec.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the video file.
    fps_target:
        Desired output frame-rate.  Actual extraction rate is the closest
        achievable given the source FPS (``max(1, round(video_fps / fps_target))``
        frames are skipped between each yielded frame).

    Yields
    ------
    frame_index : int
        Zero-based index of the frame **within the source video** (not within
        the yielded subset).
    pts_ms : int
        Presentation timestamp in milliseconds, computed as
        ``round(frame_index / video_fps * 1000)``.
    frame : np.ndarray
        BGR image array with shape ``(H, W, 3)``.

    Raises
    ------
    ValueError
        If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path!r}")

    try:
        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0  # default to 25 if unset
        frame_interval: int = max(1, int(video_fps / fps_target))

        frame_index: int = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                pts_ms: int = round(frame_index / video_fps * 1000)
                yield frame_index, pts_ms, frame

            frame_index += 1

    finally:
        cap.release()


def get_video_info(path: str) -> dict:
    """Return basic metadata about a video file.

    Parameters
    ----------
    path:
        Path to the video file.

    Returns
    -------
    dict with keys:
        * ``fps``          – frames per second (float)
        * ``total_frames`` – total frame count (int)
        * ``duration_s``   – duration in seconds (float)
        * ``width``        – frame width in pixels (int)
        * ``height``       – frame height in pixels (int)

    Raises
    ------
    ValueError
        If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path!r}")

    try:
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s: float = (total_frames / fps) if fps else 0.0
    finally:
        cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "width": width,
        "height": height,
    }
