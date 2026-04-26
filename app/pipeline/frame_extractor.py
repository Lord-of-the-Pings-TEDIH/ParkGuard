"""
app/pipeline/frame_extractor.py

Utilities for extracting sampled frames from a video file using OpenCV.
"""

from __future__ import annotations

import math
import queue
import threading
from typing import Generator

import cv2
import numpy as np

DEFAULT_VIDEO_FPS = 25.0

# Sentinel placed in the frame queue by the reader thread when it finishes.
_SENTINEL = object()


def _validate_fps_target(fps_target: int) -> int:
    if not isinstance(fps_target, int):
        raise TypeError("fps_target must be an integer")
    if fps_target <= 0:
        raise ValueError("fps_target must be > 0")
    return fps_target


def _normalize_video_fps(raw_fps: float) -> float:
    fps = float(raw_fps) if raw_fps is not None else 0.0
    if not math.isfinite(fps) or fps <= 0.0:
        return DEFAULT_VIDEO_FPS
    return fps


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure returned frames are 3-channel BGR arrays."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def extract_frames(
    video_path: str,
    fps_target: int,
    *,
    prefetch: int = 8,
) -> Generator[tuple[int, int, np.ndarray], None, None]:
    """Yield sampled frames from *video_path* at approximately *fps_target* frames/sec.

    A background thread reads and decodes frames while the consumer (YOLO +
    OCR) processes the previous batch.  The ``prefetch`` queue depth controls
    how many decoded frames are buffered ahead of the consumer; 8 frames is
    enough to hide normal disk-seek latency without significant memory cost.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the video file.
    fps_target:
        Desired output frame-rate.
    prefetch:
        Maximum number of decoded frames to buffer in the reader thread.

    Yields
    ------
    frame_index : int
        Zero-based index of the frame within the source video.
    pts_ms : int
        Presentation timestamp in milliseconds.
    frame : np.ndarray
        BGR image array with shape ``(H, W, 3)``.

    Raises
    ------
    ValueError
        If *fps_target* is invalid, the video file cannot be opened, or a
        read error occurs in the background thread.
    """
    _validate_fps_target(fps_target)
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("video_path must be a non-empty string")

    frame_queue: queue.Queue = queue.Queue(maxsize=prefetch)
    stop_event = threading.Event()

    def _reader() -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            frame_queue.put(ValueError(f"Cannot open video: {video_path!r}"))
            return
        try:
            video_fps = _normalize_video_fps(cap.get(cv2.CAP_PROP_FPS))
            frame_interval: int = max(1, round(video_fps / fps_target))
            frame_index: int = 0
            while not stop_event.is_set():
                if not cap.grab():
                    break
                if frame_index % frame_interval == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    pts_ms: int = round(frame_index / video_fps * 1000)
                    item = (frame_index, pts_ms, _normalize_frame(frame))
                    # Non-blocking put with stop check so the thread exits
                    # promptly when the consumer raises or breaks early.
                    while not stop_event.is_set():
                        try:
                            frame_queue.put(item, timeout=0.05)
                            break
                        except queue.Full:
                            pass
                frame_index += 1
        finally:
            cap.release()
            try:
                frame_queue.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        while True:
            item = frame_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # Signal the reader to stop (e.g. if the consumer exited early via
        # exception) and drain the queue so the thread can unblock from put().
        stop_event.set()
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=5.0)


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
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path!r}")

    try:
        fps = _normalize_video_fps(cap.get(cv2.CAP_PROP_FPS))
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        duration_s = total_frames / fps
    finally:
        cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "width": width,
        "height": height,
    }
