"""Score stored decider runs on StreamingBench Proactive Output and ProactiveVideoQA, timing only (no LLM judge). CPU.

Same conventions as the OVO rescoring (D24): a model-specific threshold (p(Yes) >= tau on 5, 15 or 30% of the model's own moments), responses
at upward crossings, and a matched baseline, with a 95% interval from resampling videos.
  StreamingBench (task SBPO): one answer per question: the first moment p(Yes) >= tau; correct if within 2 s of the ground-truth time
      (the paper's criterion). Baseline: one response at a uniformly random tick (exact expectation). Score: accuracy.
  ProactiveVideoQA (tasks PVQA-*): every reply has an interval; a response is right if it falls inside any interval, and several
      responses in one interval are not punished (recall = intervals with a response / intervals; precision = responses inside an
      interval / responses; F1 of the per-question averages). Baseline: a per-question timer with the same number of responses (20 phases).
  Also the within-question AUC of p(Yes) (a tick is positive if inside an interval, or within 2 s of the StreamingBench answer).

    python scripts/score_intervals.py --samples data/processed/pvqa_ego_n60.jsonl --caches results/cache/decider/<dir> [...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogate.eval.bootstrap import _groups  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.metrics import f1_of  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402
from rescore_thresholds import detect  # noqa: E402

SHARES, TOL = (0.05, 0.15, 0.30), 2.0


def load(d, s):
    f = Path(d) / f"{s.id}.jsonl"
    if not f.exists():
        return None
    rows = sorted((json.loads(x) for x in open(f) if x.strip()), key=lambda r: r["t"])
    t = np.array([r["t"] for r in rows])
    p = 1 / (1 + np.exp(-np.clip(np.array([r["lp_yes"] - r["lp_no"] for r in rows]), -50, 50)))
    return t, p


def positive(s, t):
    if s.task == "SBPO":
        return np.abs(t - s.events[0].t) <= TOL
    iv = np.array(s.meta["intervals"])
    return ((t[:, None] >= iv[None, :, 0]) & (t[:, None] <= iv[None, :, 1])).any(axis=1)


def rp(s, det):
    """Interval recall and precision (repeats in one interval not punished)."""
    iv = s.meta["intervals"]
    if not det:
        return 0.0, 0.0
    rec = float(np.mean([any(a <= d <= b for d in det) for a, b in iv]))
    pre = float(np.mean([any(a <= d <= b for a, b in iv) for d in det]))
    return rec, pre


def per_question(ss, data, tau, seeds=20):
    """{id: (model score..., baseline score...)}: SBPO (acc, expected random acc); PVQA (rec, pre, timer rec, timer pre)."""
    rng = random.Random(0)
    out = {}
    for s in ss:
        t, p = data[s.id]
        det = detect(t, p, tau, "CRR" if s.task == "SBPO" else s.task, "edge")   # "CRR" = answer once, the first time
        if s.task == "SBPO":
            pos = positive(s, t)
            out[s.id] = (float(bool(det) and abs(det[0] - s.events[0].t) <= TOL), float(pos.mean()))
            continue
        mr, mp = rp(s, det)
        n, tr, tp = len(det), [], []
        for _ in range(seeds):
            if n:
                step = len(t) / n
                ph = rng.random() * step
                td = [float(t[min(int(ph + j * step), len(t) - 1)]) for j in range(n)]
            else:
                td = []
            a, b = rp(s, td)
            tr.append(a)
            tp.append(b)
        out[s.id] = (mr, mp, float(np.mean(tr)), float(np.mean(tp)))
    return out


def stat(rows, sbpo):
    if sbpo:
        return 100 * (np.mean([r[0] for r in rows]) - np.mean([r[1] for r in rows])), 100 * np.mean([r[0] for r in rows])
    m = f1_of(np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows]))
    b = f1_of(np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]))
    return 100 * (m - b), 100 * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    sbpo = samples[0].task == "SBPO"
    res = {"samples": a.samples, "models": {}}
    if sbpo:   # every answer comes 4 to 24 s after the question: a no-video rule "answer d seconds after the question" is a telling floor
        res["fixed_delay"] = {d: 100 * float(np.mean([abs(s.start_s + d - s.events[0].t) <= TOL for s in samples])) for d in (4, 6, 8, 10, 12)}
        print("  no-video floor, answer d s after the question: " + ", ".join(f"{d} s {v:.1f}%" for d, v in res["fixed_delay"].items()))
    print(f"{a.samples}: {len(samples)} questions; " + ("accuracy (first response within 2 s) minus one random response" if sbpo else
          "interval F1 (repeats not punished) minus a per-question timer") + ", 95% interval over videos")
    for d in a.caches:
        data = {s.id: x for s in samples if (x := load(d, s)) is not None}
        ss = [s for s in samples if s.id in data]
        if not ss:
            print(f"  {Path(d).name}: no results yet")
            continue
        aucs = [auc(data[s.id][1][pos], data[s.id][1][~pos]) for s in ss if (pos := positive(s, data[s.id][0])).any() and (~pos).any()]
        allp = np.concatenate([data[s.id][1] for s in ss])
        groups = _groups([s.id for s in ss], video_of)
        rng = np.random.default_rng(0)
        boots = [[i for k in rng.integers(0, len(groups), len(groups)) for i in groups[k]] for _ in range(a.n)]
        cells, res["models"][Path(d).name] = [], {"n": len(ss), "auc": float(np.mean(aucs)) if aucs else None, "shares": {}}
        for r in SHARES:
            pq = per_question(ss, data, float(np.quantile(allp, 1 - r)))
            rows = [pq[s.id] for s in ss]
            gap, score = stat(rows, sbpo)
            bs = [stat([rows[i] for i in b], sbpo)[0] for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res["models"][Path(d).name]["shares"][str(r)] = {"score": score, "gap": gap, "ci": [lo, hi]}
            cells.append(f"{int(r * 100):>2d}%: {score:5.1f} ({gap:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '})")
        print(f"  {Path(d).name:58s} n={len(ss):3d} AUC {res['models'][Path(d).name]['auc']:.3f}  " + "  ".join(cells))
    out = Path("results/cache/rescore") / f"intervals_{Path(a.samples).stem}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
