"""Confidence intervals that respect the data's structure (docs/RATIONALE.md section 5).

Several questions can share one video, so we resample *videos*, not questions: a video is drawn with all its questions.
`paired_diff` resamples the same videos for two systems and gives the interval of the difference, which is the honest way to say
"A beats B": the gain counts only if the interval excludes zero.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np


def _groups(ids: Sequence[str], video_of: dict[str, str]) -> list[list[int]]:
    g: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(ids):
        g[video_of.get(q, q)].append(i)
    return list(g.values())


def bootstrap(items: Sequence, stat: Callable[[list], float], groups: list[list[int]], n: int = 2000, seed: int = 0) -> dict:
    """Resample groups of items with replacement; `stat` maps a list of items to a number."""
    rng = np.random.default_rng(seed)
    point = stat(list(items))
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = [i for k in pick for i in groups[k]]
        vals.append(stat([items[i] for i in idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_groups": len(groups)}


def paired_diff(items_a: Sequence, items_b: Sequence, stat: Callable[[list], float], ids: Sequence[str],
                video_of: dict[str, str], n: int = 2000, seed: int = 0) -> dict:
    """Interval for stat(A) - stat(B), resampling the same videos for both. items_a[i] and items_b[i] are the same question."""
    assert len(items_a) == len(items_b) == len(ids)
    groups = _groups(ids, video_of)
    rng = np.random.default_rng(seed)
    point = stat(list(items_a)) - stat(list(items_b))
    diffs = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = [i for k in pick for i in groups[k]]
        diffs.append(stat([items_a[i] for i in idx]) - stat([items_b[i] for i in idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0), "n_groups": len(groups)}
