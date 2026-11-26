"""95% intervals for the threshold rescoring (CPU): model F1 minus the per-question timer, resampling whole videos.

For each model and threshold share (crossing rule, counting + tutorial questions, repeats not punished, as the viewer's default), the gap is
recomputed on bootstrap resamples of the videos: the model's per-question recall and precision against the timer's (the timer's averaged over
20 phases per question, so the timer noise is not mixed into the interval). Also the paired difference between two models' gaps.

    python scripts/bootstrap_rescore.py --samples data/processed/ovo_far_val.jsonl [--set common]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogate.eval.bootstrap import _groups  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.metrics import f1_of  # noqa: E402
from rescore_thresholds import MODELS, RATES, detect, lenient, load  # noqa: E402


def per_question(ss, dd, tau, seeds=20):
    """{qid: (task, model recall, model precision, timer recall, timer precision)} for counting and tutorial questions."""
    rng = random.Random(0)
    out = {}
    for s in ss:
        if s.task == "CRR":
            continue
        t, p = dd[s.id]
        gt = [e.t for e in s.events]
        det = detect(t, p, tau, s.task, "edge")
        mr, mp = lenient(det, gt)
        n, tr, tp = len(det), [], []
        for _ in range(seeds):
            if n:
                step = len(t) / n
                ph = rng.random() * step
                td = [float(t[min(int(ph + j * step), len(t) - 1)]) for j in range(n)]
            else:
                td = []
            r_, p_ = lenient(td, gt)
            tr.append(r_)
            tp.append(p_)
        out[s.id] = (s.task, mr, mp, float(np.mean(tr)), float(np.mean(tp)))
    return out


def perceptual_gap(rows):
    """Mean over the two tasks of F1(avg recall, avg precision), model minus timer, as in rescore_thresholds.aggregate."""
    gm, gt = [], []
    for task in ("REC", "SSR"):
        r = [v for v in rows if v[0] == task]
        if r:
            gm.append(f1_of(np.mean([v[1] for v in r]), np.mean([v[2] for v in r])))
            gt.append(f1_of(np.mean([v[3] for v in r]), np.mean([v[4] for v in r])))
    return 100 * (np.mean(gm) - np.mean(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--set", choices=["common", "all"], default="common")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    ids = [s.id for s in samples]
    video_of = {s.id: str(s.video) for s in samples}
    data = {k: v for k, v in ((n, load(d, ids)) for n, d in MODELS.items()) if len(v) >= 30}
    common = set.intersection(*[set(v) for v in data.values()])
    pq = {}
    for name, dd in data.items():
        ss = [s for s in samples if s.id in dd and (a.set == "all" or s.id in common)]
        allp = np.concatenate([dd[s.id][1] for s in ss])
        pq[name] = {r: per_question(ss, dd, float(np.quantile(allp, 1 - r))) for r in RATES}
    qids = sorted(set.intersection(*[set(pq[m][RATES[0]]) for m in pq]))
    groups = _groups(qids, video_of)
    rng = np.random.default_rng(0)
    boots = [[i for k in rng.integers(0, len(groups), len(groups)) for i in groups[k]] for _ in range(a.n)]
    res = {"set": a.set, "n_questions": len(qids), "rates": RATES, "gap": {}}
    print(f"{len(qids)} counting + tutorial questions ({a.set} set), crossing rule, repeats not punished; F1 gap over the per-question timer, 95% interval")
    print(f"{'model':38s} " + "  ".join(f"{int(r * 100):>3d}%{'':16s}" for r in RATES))
    for name in pq:
        cells = []
        res["gap"][name] = {}
        for r in RATES:
            rows = [pq[name][r][q] for q in qids]
            point = perceptual_gap(rows)
            bs = [perceptual_gap([rows[i] for i in b]) for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res["gap"][name][str(r)] = [point, float(lo), float(hi)]
            cells.append(f"{point:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{name:38s} " + " ".join(cells))
    print("* interval excludes 0")
    # paired differences between selected pairs, at each share
    pairs = [("235B", "8B"), ("change gate + 8B", "8B"), ("change gate + 2B", "2B"), ("2B", "8B"), ("change gate + 235B", "235B")]
    res["pairs"] = {}
    print("\npaired difference of gaps (first minus second), 95% interval")
    for x, y in pairs:
        if x not in pq or y not in pq:
            continue
        cells = []
        for r in RATES:
            rx, ry = [pq[x][r][q] for q in qids], [pq[y][r][q] for q in qids]
            point = perceptual_gap(rx) - perceptual_gap(ry)
            bs = [perceptual_gap([rx[i] for i in b]) - perceptual_gap([ry[i] for i in b]) for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res["pairs"][f"{x} - {y}"] = res["pairs"].get(f"{x} - {y}", {}) | {str(r): [point, float(lo), float(hi)]}
            cells.append(f"{point:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{x + ' - ' + y:38s} " + " ".join(cells))
    out = Path(f"results/cache/rescore/bootstrap_{a.set}_val.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
