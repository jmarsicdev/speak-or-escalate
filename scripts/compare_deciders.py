"""Compare two stored deciders (for example row 1, question only, against row 2, with cues) on the same questions.

    python scripts/compare_deciders.py --a results/cache/decider/<row1 dir> --b results/cache/decider/<row2 dir> \
        --samples data/processed/ovo_far_val.jsonl [--matcher results/cache/matcher/<dir>]

For each question, the AUC of p(Yes) for telling moments near an event (within 2 s) from the others; then the average over questions,
overall and by group, and the paired difference B - A with a 95% interval from resampling whole videos. No setting is chosen, so there is
no selection to flatter the result. With --matcher, it also does the comparison at the matcher's trigger moments (needs both caches to hold answers there).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.bootstrap import paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.ovo import decision_times  # noqa: E402
from egogate.gated import _known_rows, trigger_times  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402


def p_yes(r):
    return 1 / (1 + math.exp(-(r["lp_yes"] - r["lp_no"])))


def per_question(samples, rows_by_id, times_by_id=None):
    out = {}
    for s in samples:
        g = np.array([e.t for e in s.events])
        rows = rows_by_id.get(s.id, {})
        ts = sorted(times_by_id[s.id]) if times_by_id else sorted(rows)
        ts = [t for t in ts if t in rows]
        if len(ts) < 2:
            continue
        p = np.array([p_yes(rows[t]) for t in ts])
        lab = np.array([(np.abs(g - t) <= 2).any() for t in ts])
        if lab.any() and (~lab).any():
            out[s.id] = (auc(p[lab], p[~lab]), float(p.mean()), float((p >= 0.5).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="decider cache dir, the reference (e.g. question only)")
    ap.add_argument("--b", required=True, help="decider cache dir to compare (e.g. with cues)")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--matcher", default=None)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    A = {s.id: _known_rows(Path(a.a), s.id) for s in samples}
    B = {s.id: _known_rows(Path(a.b), s.id) for s in samples}
    views = [("the regular 2 s ticks", {s.id: decision_times(s, 2.0) for s in samples})]
    if a.matcher:
        times = {}
        for s in samples:
            f = Path(a.matcher) / f"{s.id}.json"
            if f.exists():
                r = json.load(open(f))
                times[s.id] = trigger_times(np.array(r["t"]), np.array(r["sims"]).reshape(len(r["t"]), -1), "code", 0.02)
        views.append(("matcher trigger moments (theta 0.02)", times))
    for title, times in views:
        qa, qb = per_question(samples, A, times), per_question(samples, B, times)
        common = [s for s in samples if s.id in qa and s.id in qb]
        print(f"\n{title}: {len(common)} questions with both kinds of moment in both caches")
        if not common:
            continue
        print(f"{'group':30s} {'n':>3s} {'AUC A':>7s} {'AUC B':>7s} {'B - A':>7s} {'95% interval':>20s}  {'Yes-rate A':>10s} {'B':>5s}")
        groups = [("all", lambda s: True)] + [(f"{t} / {src}", (lambda t=t, src=src: (lambda s: s.task == t and s.meta.get("source") == src))())
                                                for t, src in (("REC", "thumos"), ("REC", "perception_test"), ("SSR", "COIN"), ("CRR", "MovieNet"))]
        for name, flt in groups:
            ss = [s for s in common if flt(s)]
            if not ss:
                continue
            ids = [s.id for s in ss]
            ia, ib = [qa[i][0] for i in ids], [qb[i][0] for i in ids]
            d = paired_diff(ia, ib, lambda v: float(np.mean(v)), ids, video_of)
            ya, yb = np.mean([qa[i][2] for i in ids]), np.mean([qb[i][2] for i in ids])
            flag = "  <- excludes 0" if d["excludes_zero"] else ""
            print(f"{name:30s} {len(ss):3d} {np.mean(ia):7.3f} {np.mean(ib):7.3f} {-d['diff']:7.3f} [{-d['hi']:6.3f}, {-d['lo']:6.3f}]  {ya:10.2f} {yb:5.2f}{flag}")
    print("\nAUC 0.5 = no skill. 'B - A' is the change from adding what B has; the interval resamples whole videos (several questions share one).")


if __name__ == "__main__":
    main()
