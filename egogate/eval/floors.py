"""No-skill reference systems (experiment row 0). They need no model and no video.

They exist so every later number has a floor and a ceiling to be read against:

  always_silent   never responds                       -> F1 must be 0
  always_respond  responds at every decision point     -> shows what "yes to everything" scores here
  random(rate)    responds with probability `rate`     -> a coin flip at the same cadence
  oracle          responds exactly at ground-truth time -> must score 1.0 (wiring check, not a baseline)
"""
from __future__ import annotations

import random

from .datasets import Sample
from .metrics import DEFAULT_TOL_S, aggregate, score_question
from .ovo import decision_times


def detections_for(policy: str, sample: Sample, stride_s: float, rng: random.Random, rate: float = 0.5) -> list[float]:
    grid = decision_times(sample, stride_s)
    if policy == "always_silent":
        return []
    if policy == "always_respond":
        return grid
    if policy == "random":
        return [t for t in grid if rng.random() < rate]
    if policy == "oracle":
        return [e.t for e in sample.events]
    raise ValueError(policy)


def run_floor(policy: str, samples: list[Sample], stride_s: float = 2.0, tol: float = DEFAULT_TOL_S,
              seed: int = 0, rate: float = 0.5) -> dict:
    rng = random.Random(seed)
    rows = []
    for s in samples:
        det = detections_for(policy, s, stride_s, rng, rate)
        q = score_question(det, [e.t for e in s.events], tol)
        rows.append({"id": s.id, "task": s.task, "recall": q.recall, "precision": q.precision,
                     "n_det": q.n_det, "n_gt": q.n_gt, "n_correct": q.n_correct})
    res = aggregate(rows)
    res.update({"policy": policy, "stride_s": stride_s, "tol_s": tol, "rate": rate if policy == "random" else None,
                "n_questions": len(rows), "total_detections": sum(r["n_det"] for r in rows)})
    return res
