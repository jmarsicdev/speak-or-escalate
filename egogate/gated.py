"""Row 4: the matcher decides WHEN to ask, the decider decides whether it is the moment (decision D20, D13).

Candidates are the matcher's trigger times (no duplicate removal). The decider is run at exactly those times, on demand,
and every answer is remembered, so a later setting that needs the same moment reuses it. The stored decider answers
at the regular 2 s ticks from row 1 are reused too when a trigger happens to land on one.

Why one set of answers serves many thresholds: for the released-code rule, a trigger at a higher threshold is also a
trigger at any lower one, so the decider only has to be run at the lowest threshold's triggers.

    ondemand store:  <decider cache dir>/ondemand/<id>.jsonl      rows like the row 1 cache: {"t", "lp_yes", "lp_no", "ms", "n_frames"}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .decider_cache import read_rows
from .eval.datasets import Sample
from .eval.metrics import DEFAULT_TOL_S, aggregate, score_question
from .frames import stream_windows
from .matcher_replay import surge_code, surge_paper
from .replay import detections


def trigger_times(t: np.ndarray, sims: np.ndarray, rule: str, theta: float, agg: str = "any") -> list[float]:
    if len(t) == 0:
        return []
    trig = surge_paper(sims, theta, agg) if rule == "paper" else surge_code(sims, theta, agg)
    return [round(float(x), 6) for x in t[trig]]


def _known_rows(dec_dir: Path, sample_id: str) -> dict[float, dict]:
    rows = {}
    for f in (dec_dir / f"{sample_id}.jsonl", dec_dir / "ondemand" / f"{sample_id}.jsonl"):
        for r in read_rows(f):
            rows[round(r["t"], 6)] = r
    return rows


def run_ondemand(sample: Sample, edge, cfg: dict, times: list[float], dec_dir: Path, video_root: Path) -> dict[float, dict]:
    """Decider answers at `times`, computing only the ones not already stored. Returns {t: row} for the times that exist."""
    d = cfg["decider"]
    known = _known_rows(dec_dir, sample.id)
    missing = sorted({round(t, 6) for t in times} - set(known))
    out_dir = dec_dir / "ondemand"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    if missing:
        with open(out_dir / f"{sample.id}.jsonl", "a") as f:
            for w in stream_windows(video_root / sample.video, missing, d["window_s"], d["fps"], d["max_side"]):
                t0 = time.perf_counter()
                o = edge.speak_now(sample.query, w.frames, None, None, window_s=d["window_s"])
                row = {"t": round(w.t_end, 6), "lp_yes": o.raw["lp_yes"], "lp_no": o.raw["lp_no"],
                       "ms": round((time.perf_counter() - t0) * 1000, 1), "n_frames": len(w.frames)}
                f.write(json.dumps(row) + "\n")
                f.flush()
                known[row["t"]] = row
                n_new += 1
    return {"rows": {t: known[t] for t in {round(x, 6) for x in times} if t in known}, "n_new": n_new, "n_missing_video": len(missing) - n_new}


def score_gated(samples: list[Sample], matcher: dict, rows: dict[str, dict[float, dict]], rule: str, theta: float, tau: float,
                agg: str = "any", cooldown_s: float = 0.0, temperature: float = 1.0, tol: float = DEFAULT_TOL_S) -> dict:
    """A response = the matcher fires (threshold `theta`) AND the decider says Yes (p >= tau); cooldown applied to the responses."""
    per_q = []
    for s in samples:
        m = matcher.get(s.id)
        det: list[float] = []
        if m is not None:
            trig = trigger_times(m["t"], m["sims"], rule, theta, agg)
            cand = [rows[s.id][t] for t in trig if t in rows.get(s.id, {})]
            det = detections(cand, tau, temperature, cooldown_s)
        q = score_question(det, [e.t for e in s.events], tol)
        per_q.append({"id": s.id, "task": s.task, "recall": q.recall, "precision": q.precision,
                      "n_det": q.n_det, "n_gt": q.n_gt, "n_correct": q.n_correct})
    res = aggregate(per_q)
    res["per_question"] = per_q
    res["total_detections"] = sum(r["n_det"] for r in per_q)
    return res
