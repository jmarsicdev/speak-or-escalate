"""Stage 2 for the cue matcher: stored cue similarities -> trigger times, under any rule.

Two trigger rules, both reproducing Em-Garde (docs/RATIONALE.md D19):

  "paper"  the rule written in the paper (section 3.4): trigger when any cue's similarity rose by more than theta
           since the previous step.
  "code"   the rule in their released code (`src/trigger_detection.py`, scheme "surge"): trigger when the current
           similarity is more than theta above its value `fps` steps earlier AND above the average of the older history
           (their history is trimmed to the latest 500-600 steps).

`agg` decides how the cues are combined before the rule is applied (Gate 2): "any" applies the rule to each cue and
fires if any cue fires; "mean" applies it to the cues' average similarity.

Duplicate suppression: "cooldown" drops a trigger within N seconds of the previous kept one (needs only the
similarities); "emgarde" copies their rule: skip a trigger if it is within 5 s of the last one AND the window embedding is
at least 0.8 cosine-similar to the embedding at that last trigger (needs the stored window vectors).
"""
from __future__ import annotations

import numpy as np

from .eval.datasets import Sample
from .eval.metrics import DEFAULT_TOL_S, aggregate, score_question


def _combine(sims: np.ndarray, agg: str) -> np.ndarray:
    return sims if agg == "any" else sims.mean(axis=1, keepdims=True)


def surge_paper(sims: np.ndarray, theta: float, agg: str = "any") -> np.ndarray:
    x = _combine(sims, agg)
    trig = np.zeros(len(x), dtype=bool)
    if len(x) > 1:
        trig[1:] = (x[1:] - x[:-1]).max(axis=1) > theta
    return trig


def surge_code(sims: np.ndarray, theta: float, agg: str = "any", fps: int = 2) -> np.ndarray:
    """Em-Garde's released rule, step by step as their streaming loop applies it (history trimmed at 600 -> 500)."""
    x = _combine(sims, agg)
    trig = np.zeros(len(x), dtype=bool)
    start = 0   # index of the oldest step still in their history
    for t in range(len(x)):
        if t + 1 - start > 600:
            start = t + 1 - 500
        h = x[start:t + 1]
        if len(h) < fps + 1:
            continue
        cond1 = ((h[-1] - h[-fps - 1]) > theta)
        cond2 = ((h[-1] - h[:-fps].mean(axis=0)) > 0)
        trig[t] = bool((cond1 & cond2).any())
    return trig


def dedup(times: np.ndarray, trig: np.ndarray, how: str = "none", cooldown_s: float = 0.0,
          vecs: np.ndarray | None = None) -> list[float]:
    """Turn a boolean trigger series into detection times, dropping duplicates."""
    out: list[float] = []
    last_i = None
    for i in np.flatnonzero(trig):
        t = float(times[i])
        if how == "cooldown":
            if out and t - out[-1] < max(cooldown_s, 1e-9):
                continue
        elif how == "emgarde":
            if last_i is not None and t - float(times[last_i]) <= 5.0 and \
                    float(vecs[i].astype(np.float32) @ vecs[last_i].astype(np.float32)) >= 0.8:
                continue
        out.append(t)
        last_i = i
    return out


def matcher_detections(times: np.ndarray, sims: np.ndarray, rule: str, theta: float, agg: str = "any",
                       dedup_how: str = "none", cooldown_s: float = 0.0, vecs: np.ndarray | None = None) -> list[float]:
    if len(times) == 0:
        return []
    trig = surge_paper(sims, theta, agg) if rule == "paper" else surge_code(sims, theta, agg)
    return dedup(times, trig, dedup_how, cooldown_s, vecs)


def score_matcher(samples: list[Sample], data: dict, rule: str, theta: float, agg: str = "any", dedup_how: str = "none",
                  cooldown_s: float = 0.0, tol: float = DEFAULT_TOL_S) -> dict:
    """data: id -> {"t": array, "sims": array, "vecs": optional array}."""
    per_q = []
    for s in samples:
        d = data.get(s.id)
        det = matcher_detections(d["t"], d["sims"], rule, theta, agg, dedup_how, cooldown_s, d.get("vecs")) if d else []
        q = score_question(det, [e.t for e in s.events], tol)
        per_q.append({"id": s.id, "task": s.task, "recall": q.recall, "precision": q.precision,
                      "n_det": q.n_det, "n_gt": q.n_gt, "n_correct": q.n_correct})
    res = aggregate(per_q)
    res["per_question"] = per_q
    res["total_detections"] = sum(r["n_det"] for r in per_q)
    return res


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Chance a random positive scores above a random negative (0.5 = no skill). Ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def similarity_diagnostics(samples: list[Sample], data: dict, tol: float = DEFAULT_TOL_S) -> dict:
    """Gate 2 diagnostic: does the cue similarity (or its rise) separate steps near an event from the rest?
    A step is 'near' if it is within +/- tol of a true event time. 0.5 = no separation."""
    feats = {"max sim": [], "mean sim": [], "max rise (paper)": []}
    labels = []
    for s in samples:
        d = data.get(s.id)
        if not d or len(d["t"]) < 2:
            continue
        sims, t = d["sims"], d["t"]
        rise = np.concatenate([[0.0], (sims[1:] - sims[:-1]).max(axis=1)])
        g = np.array([e.t for e in s.events])
        near = (np.abs(t[:, None] - g[None, :]) <= tol).any(axis=1)
        labels.append(near)
        feats["max sim"].append(sims.max(axis=1))
        feats["mean sim"].append(sims.mean(axis=1))
        feats["max rise (paper)"].append(rise)
    lab = np.concatenate(labels) if labels else np.array([], dtype=bool)
    out = {"n_steps": int(len(lab)), "frac_near": float(lab.mean()) if len(lab) else float("nan")}
    for k, v in feats.items():
        x = np.concatenate(v) if v else np.array([])
        out[k] = auc(x[lab], x[~lab]) if len(x) else float("nan")
    return out
