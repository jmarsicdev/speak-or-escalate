"""Scoring without an LLM judge: Em-Garde's online recall / precision / F1 on OVO-Bench.

This reproduces the scoring loop in Em-Garde's `eval/ovo_bench/ovo_bench_eval_online.py`
(MIT licence, https://github.com/air-embodied-brain/Em-Garde), so our numbers are comparable
to theirs. The rules, in plain terms:

  * A detection is correct if it is within +/- `tol` seconds of a ground-truth time
    (symmetric: a response slightly BEFORE the event also counts). Em-Garde uses tol = 2.
  * Matching is one-to-one. For each ground-truth time, in order, the first unused
    detection within tol is consumed. Extra detections near the same event are false positives.
  * Per question: recall = correct / n_gt;  precision = correct / n_detections
    (0 when the model made no detection at all).
  * Per task: average recall and precision over questions, THEN F1 of those two averages
    (not the average of per-question F1s).
  * Overall: the unweighted mean of the per-task F1s.

See docs/RATIONALE.md, decisions D3 and D4, for why we copy this exactly.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..types import Action, Decision, Event

DEFAULT_TOL_S = 2.0


def match_detections(det_times: list[float], gt_times: list[float], tol: float = DEFAULT_TOL_S) -> list[bool]:
    """Which detections are correct. Same greedy order as Em-Garde: loop over GT, take the
    first still-unused detection within tol. Returns one flag per detection, in input order."""
    used = [False] * len(det_times)
    for gt in gt_times:
        for i, t in enumerate(det_times):
            if not used[i] and abs(t - gt) <= tol:
                used[i] = True
                break
    return used


@dataclass
class QuestionScore:
    recall: float
    precision: float
    n_correct: int
    n_det: int
    n_gt: int


def score_question(det_times: list[float], gt_times: list[float], tol: float = DEFAULT_TOL_S) -> QuestionScore:
    n_correct = sum(match_detections(det_times, gt_times, tol))
    n_det, n_gt = len(det_times), len(gt_times)
    recall = n_correct / n_gt if n_gt else 0.0
    precision = n_correct / n_det if n_det else 0.0
    return QuestionScore(recall, precision, n_correct, n_det, n_gt)


def f1_of(recall: float, precision: float) -> float:
    return 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0


def aggregate(per_question: list[dict]) -> dict:
    """per_question rows need keys: task, recall, precision. Returns per-task and overall scores."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in per_question:
        by_task[r["task"]].append(r)
    out: dict = {"tasks": {}}
    for task, rows in sorted(by_task.items()):
        rec = float(np.mean([r["recall"] for r in rows]))
        pre = float(np.mean([r["precision"] for r in rows]))
        out["tasks"][task] = {"recall": rec, "precision": pre, "f1": f1_of(rec, pre), "n_questions": len(rows)}
    f1s = [t["f1"] for t in out["tasks"].values()]
    out["avg_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return out


def detection_times(decisions: list[Decision]) -> list[float]:
    """Times at which the system chose to respond (edge answer or cloud answer)."""
    return [d.t for d in decisions
            if d.action in (Action.EDGE_ANSWER, Action.ESCALATE) and d.answer is not None]


def responded(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.action in (Action.EDGE_ANSWER, Action.ESCALATE) and d.answer is not None]


def calibration_table(decisions: list[Decision], events: list[Event], tol: float = DEFAULT_TOL_S, bins: int = 10) -> list[dict]:
    """For every *triggered* decision (silent or not), was 'now' a correct response time?
    Groups by p_speak bin: does high confidence mean correct?  (Gate 1)."""
    trig = sorted((d for d in decisions if d.meta.get("triggered")), key=lambda d: d.t)
    if not trig:
        return []
    lab = np.array(match_detections([d.t for d in trig], [e.t for e in events], tol))
    ps = np.array([d.p_speak for d in trig])
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (ps >= edges[i]) & (ps < edges[i + 1] if i < bins - 1 else ps <= edges[i + 1])
        if m.sum():
            rows.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                         "n": int(m.sum()), "mean_p": float(ps[m].mean()), "frac_correct": float(lab[m].mean())})
    return rows


def summarize_run(decisions: list[Decision], events: list[Event], tol: float = DEFAULT_TOL_S) -> dict:
    resp = responded(decisions)
    qs = score_question([d.t for d in resp], [e.t for e in events], tol)
    n_trig = sum(1 for d in decisions if d.meta.get("triggered"))
    return {
        "recall": qs.recall, "precision": qs.precision, "f1": f1_of(qs.recall, qs.precision),
        "n_gt": qs.n_gt, "n_responses": qs.n_det, "n_correct": qs.n_correct,
        "n_false_positives": qs.n_det - qs.n_correct,
        "n_triggers": n_trig, "n_cloud_calls": sum(1 for d in decisions if d.cost > 0),
        "n_edge_answers": sum(1 for d in resp if d.source == "edge"),
        "n_cloud_answers": sum(1 for d in resp if d.source == "cloud"),
    }
