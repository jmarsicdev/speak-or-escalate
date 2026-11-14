"""Frame sampling and sliding windows over a video file or frame directory.

Em-Garde's matcher setting: 2-second windows, 2 fps, stepped every 0.5 s.
Both are config knobs here (`matcher.window_s`, `matcher.fps`, `matcher.stride_s`).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .types import Window


def iter_frames(path: str | Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, frame) sampled at `fps` from a video file.

    Uses OpenCV; imported lazily so the rest of the package works without it.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx / src_fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def sliding_windows(
    frames: Iterator[tuple[float, np.ndarray]],
    window_s: float,
    stride_s: float,
    fps: float,
) -> Iterator[Window]:
    """Group a timestamped frame stream into overlapping windows."""
    n_keep = max(int(round(window_s * fps)), 1)
    buf: deque[tuple[float, np.ndarray]] = deque(maxlen=n_keep)
    next_emit = window_s
    for t, frame in frames:
        buf.append((t, frame))
        if t + 1e-6 >= next_emit and len(buf) == n_keep:
            yield Window(frames=[f for _, f in buf], t_start=buf[0][0], t_end=t)
            next_emit += stride_s
