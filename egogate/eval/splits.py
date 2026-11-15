"""Seeded dev / val / test split, grouped by video.

Why grouped: ten FAR videos carry several questions (one has nine). If the same video landed in
both dev and test, a model or threshold tuned on dev would have seen test footage. So whole
videos move together. Within that constraint we stratify by task so each split has a similar
REC / SSR / CRR mix. Same seed -> same split, on any machine.

Roles (decision D6): dev = build and debug; val = fit temperature scaling and pick thresholds;
test = touched once at the end.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .datasets import Sample

DEFAULT_FRACTIONS = {"dev": 0.4, "val": 0.3, "test": 0.3}


def split_samples(samples: list[Sample], seed: int = 0, fractions: dict[str, float] | None = None) -> dict[str, list[Sample]]:
    fractions = fractions or DEFAULT_FRACTIONS
    assert abs(sum(fractions.values()) - 1.0) < 1e-9
    names = list(fractions)
    by_video: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_video[str(s.video)].append(s)

    # group videos by the task mix of their questions so every split sees all tasks
    by_task: dict[str, list[str]] = defaultdict(list)
    for v, ss in by_video.items():
        by_task["+".join(sorted({s.task for s in ss}))].append(v)

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for key in sorted(by_task):
        vids = sorted(by_task[key])
        rng.shuffle(vids)
        n = len(vids)
        cuts, acc = [], 0.0
        for nm in names[:-1]:
            acc += fractions[nm] * n
            cuts.append(round(acc))
        for i, v in enumerate(vids):
            idx = sum(i >= c for c in cuts)
            assign[v] = names[idx]

    out = {nm: [] for nm in names}
    for s in samples:
        out[assign[str(s.video)]].append(s)
    return out
