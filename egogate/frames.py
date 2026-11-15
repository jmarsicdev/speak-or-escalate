"""Causal frame windows at chosen decision times, decoded in one forward pass over the video.

For a decision at time t the window holds the frames sampled at t - (n-1)/fps, ..., t  (n = window_s * fps,
so 4 frames at 2 s and 2 fps). Nothing after t is ever used, which is rule 1 of docs/RATIONALE.md D12.
Frames are resized either so the long side is at most `max_side` (decider), or, if `max_pixels` is given, with Qwen's
smart_resize rule to an area of at most `max_pixels` (cue matcher; Em-Garde uses 280 x 280 = 78,400).

Timestamps come from frame index / source fps. For constant-frame-rate video (the benchmark videos) that
is exact to within one source frame; for variable-frame-rate video it can drift slightly.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .types import Window


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 84 * 84, max_pixels: int = 280 * 280) -> tuple[int, int]:
    """Qwen-VL's rule: round each side to a multiple of `factor`, keep the area within [min_pixels, max_pixels].
    Same logic as `smart_resize` in Qwen's vision utils and in Em-Garde's frame loader (which uses max_pixels = 280*280)."""
    import math  # noqa: PLC0415

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def window_sample_times(t: float, window_s: float = 2.0, fps: float = 2.0) -> list[float]:
    n = max(int(round(window_s * fps)), 1)
    return [round(t - (n - 1 - k) / fps, 6) for k in range(n)]


def stream_windows(video_path, ticks: Iterable[float], window_s: float = 2.0, fps: float = 2.0,
                   max_side: int | None = 448, max_pixels: int | None = None) -> Iterator[Window]:
    """Yield one Window per tick, in time order, skipping ticks the video is too short for."""
    import cv2  # noqa: PLC0415

    ticks = sorted(ticks)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9

    def idx_of(sec: float) -> int:
        return max(int(round(sec * src_fps)), 0)

    need = {t: window_sample_times(t, window_s, fps) for t in ticks}
    # a tick needs a full window; ticks too early in the video (frames before time 0) are skipped, not padded
    ticks = [t for t in ticks if min(need[t]) >= -1e-9]
    needed_idx = sorted({idx_of(s) for ss in need.values() for s in ss if s >= 0})
    keep: dict[int, np.ndarray] = {}
    pos = -1          # index of the last frame grabbed
    ti = 0
    for target in needed_idx:
        while pos < target and pos + 1 < n_total:
            ok = cap.grab()
            if not ok:
                n_total = pos + 1
                break
            pos += 1
        if pos != target:
            break      # ran out of video
        ok, frame = cap.retrieve()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if max_pixels:
            h, w = frame.shape[:2]
            nh, nw = smart_resize(h, w, max_pixels=max_pixels)
            if (nh, nw) != (h, w):
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        elif max_side:
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        keep[target] = frame
        # emit every tick whose frames are all available
        while ti < len(ticks):
            idxs = [idx_of(s) for s in need[ticks[ti]]]
            if any(i > target for i in idxs):
                break
            t = ticks[ti]
            yield Window(frames=[keep[i] for i in idxs], t_start=need[t][0], t_end=t)
            ti += 1
            floor = min(idx_of(s) for s in need[ticks[ti]]) if ti < len(ticks) else target + 1
            for k in [k for k in keep if k < floor]:
                del keep[k]
    cap.release()
