"""Stage 2: turn stored decider scores into detections, on CPU, with any setting.

    p(Yes) = sigmoid((lp_yes - lp_no) / T)

T = 1 is the raw model. Temperature scaling (Guo et al. 2017) fits T on the validation split. A detection is a
tick where p >= tau. `cooldown_s` drops a detection that comes within that many seconds of the previous kept one,
because one event usually looks "Yes" on two consecutive ticks and the second would count as a false positive.
"""
from __future__ import annotations

import math
from pathlib import Path

from .decider_cache import read_rows
from .eval.datasets import Sample
from .eval.metrics import DEFAULT_TOL_S, aggregate, score_question


def p_yes(row: dict, temperature: float = 1.0) -> float:
    z = (row["lp_yes"] - row["lp_no"]) / temperature
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def detections(rows: list[dict], tau: float, temperature: float = 1.0, cooldown_s: float = 0.0) -> list[float]:
    out: list[float] = []
    for r in sorted(rows, key=lambda r: r["t"]):
        if p_yes(r, temperature) >= tau and (not out or r["t"] - out[-1] >= max(cooldown_s, 1e-9)):
            out.append(r["t"])
    return out


def load_split(cache_dir: Path, samples: list[Sample]) -> dict[str, list[dict]]:
    return {s.id: read_rows(cache_dir / f"{s.id}.jsonl") for s in samples}


def score_setting(samples: list[Sample], rows_by_id: dict[str, list[dict]], tau: float, temperature: float = 1.0,
                  cooldown_s: float = 0.0, tol: float = DEFAULT_TOL_S) -> dict:
    per_q = []
    for s in samples:
        det = detections(rows_by_id.get(s.id, []), tau, temperature, cooldown_s)
        q = score_question(det, [e.t for e in s.events], tol)
        per_q.append({"id": s.id, "task": s.task, "recall": q.recall, "precision": q.precision,
                      "n_det": q.n_det, "n_gt": q.n_gt, "n_correct": q.n_correct})
    res = aggregate(per_q)
    res["per_question"] = per_q
    res["total_detections"] = sum(r["n_det"] for r in per_q)
    return res
