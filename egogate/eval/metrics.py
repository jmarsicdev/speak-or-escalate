"""Scoring without an LLM judge.

online_pr: Em-Garde's online recall/precision for response *timing*.
  A predicted response time is correct if it falls within [event_t, event_t + tol]
  of a ground-truth event; each event can be matched at most once.
  (OVO-Bench's original offline accuracy can be gamed by always answering "yes";
   this metric cannot.)

calibration_table: bins p_speak and reports how often a response in each bin was
  actually correct, for the "is the model's confidence accurate?" question and the
  F1/recall heatmap over (lo, hi) thresholds.

summarize_run: one row per (video, query, policy) with timing F1, cloud calls, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Action, Decision, Event


@dataclass
class PR:
    recall: float
    precision: float
    f1: float
    n_correct: int
    n_pred: int
    n_gt: int


def match_responses(pred_times: list[float], events: list[Event], tol: float = 4.0) -> list[bool]:
    """Greedy one-to-one matching in time order. Returns correctness per prediction."""
    unmatched = sorted(e.t for e in events)
    correct = []
    for t in sorted(pred_times):
        hit = None
        for i, et in enumerate(unmatched):
            if et <= t <= et + tol:
                hit = i
                break
        if hit is not None:
            unmatched.pop(hit)
        correct.append(hit is not None)
    return correct


def online_pr(pred_times: list[float], events: list[Event], tol: float = 4.0) -> PR:
    correct = match_responses(pred_times, events, tol)
    n_c, n_p, n_g = sum(correct), len(pred_times), len(events)
    rec = n_c / n_g if n_g else 0.0
    pre = n_c / n_p if n_p else 0.0
    f1 = 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0
    return PR(rec, pre, f1, n_c, n_p, n_g)


def responded(decisions: list[Decision]) -> list[Decision]:
    return [d for d in decisions if d.action in (Action.EDGE_ANSWER, Action.ESCALATE) and d.answer is not None]


def calibration_table(decisions: list[Decision], events: list[Event], tol: float = 4.0, bins: int = 10) -> list[dict]:
    """For every *triggered* decision (silent or not), was 'now' actually a response time?
    Groups by p_speak bin so we can see whether high confidence means correct."""
    trig = [d for d in decisions if d.meta.get("triggered")]
    if not trig:
        return []
    labels = match_responses([d.t for d in trig], events, tol)
    # match_responses sorts by time; re-align to trig order
    order = np.argsort([d.t for d in trig])
    lab = np.empty(len(trig), dtype=bool)
    lab[order] = labels
    ps = np.array([d.p_speak for d in trig])
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (ps >= edges[i]) & (ps < edges[i + 1] if i < bins - 1 else ps <= edges[i + 1])
        if m.sum():
            rows.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                         "n": int(m.sum()), "mean_p": float(ps[m].mean()), "frac_correct": float(lab[m].mean())})
    return rows


def summarize_run(decisions: list[Decision], events: list[Event], tol: float = 4.0) -> dict:
    resp = responded(decisions)
    pr = online_pr([d.t for d in resp], events, tol)
    n_trig = sum(1 for d in decisions if d.meta.get("triggered"))
    n_cloud = sum(1 for d in decisions if d.cost > 0)
    return {
        "recall": pr.recall, "precision": pr.precision, "f1": pr.f1,
        "n_gt": pr.n_gt, "n_responses": pr.n_pred, "n_correct": pr.n_correct,
        "n_false_positives": pr.n_pred - pr.n_correct,
        "n_triggers": n_trig, "n_cloud_calls": n_cloud,
        "n_edge_answers": sum(1 for d in resp if d.source == "edge"),
        "n_cloud_answers": sum(1 for d in resp if d.source == "cloud"),
    }
