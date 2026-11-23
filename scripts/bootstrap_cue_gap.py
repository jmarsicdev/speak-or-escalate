"""Does better cue wording help the matcher beyond what a timer would get? Paired bootstrap, CPU, about a minute.

    python scripts/bootstrap_cue_gap.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl

For each cue source (edge model's cues, oracle) and each threshold of Em-Garde's released rule (4 s cooldown), the matcher is scored with
the benchmark's averaged F1 (mean over the three tasks, as in the earlier tables), and so is the timer whose number of responses is closest
(per-question scores averaged over 10 random phases). "Gap" = matcher F1 minus that timer's F1, so a setting that simply answers more
often gets no credit. The comparison is the oracle's gap minus the edge cues' gap, with a 95% interval from resampling whole videos.
Thresholds are paired by threshold (the two sources then make different numbers of responses, which the timer matching absorbs) and the
last row averages over all thresholds, which avoids picking one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.bootstrap import paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.floors import detections_for  # noqa: E402
from egogate.eval.metrics import aggregate, score_question  # noqa: E402
from egogate.matcher_replay import matcher_detections  # noqa: E402

THETAS = (0.01, 0.02, 0.03, 0.04, 0.05)
PERIODS = (2, 4, 6, 8, 10, 12, 16, 20, 30, 45, 60, 90)


def rows_for(samples, dets):
    out = []
    for s in samples:
        q = score_question(dets[s.id], [e.t for e in s.events])
        out.append({"task": s.task, "recall": q.recall, "precision": q.precision, "n_det": q.n_det})
    return out


def timer_rows(samples, seeds=10):
    """period -> (per-question rows averaged over phases, mean total responses)."""
    res = {}
    for p in PERIODS:
        acc = [rows_for(samples, {s.id: detections_for("periodic", s, 2.0, random.Random(sd), period_s=float(p)) for s in samples}) for sd in range(seeds)]
        rows = [{"task": acc[0][i]["task"], **{k: float(np.mean([a[i][k] for a in acc])) for k in ("recall", "precision", "n_det")}}
                for i in range(len(samples))]
        res[p] = (rows, sum(r["n_det"] for r in rows))
    return res


def load(cache, sims_from, samples):
    out = {}
    for s in samples:
        r = json.load(open(Path(sims_from or cache) / f"{s.id}.json"))
        t = np.array(r["t"])
        out[s.id] = (t, np.array(r["sims"], dtype=float).reshape(len(t), -1))
    return out


def gap_items(samples, data, timers, th):
    dets = {s.id: matcher_detections(*data[s.id], "code", th, "any", "cooldown", 4.0) for s in samples}
    rows = rows_for(samples, dets)
    n = sum(r["n_det"] for r in rows)
    per = min(PERIODS, key=lambda p: abs(timers[p][1] - n))
    return list(zip(rows, timers[per][0])), n, per


def gap_stat(items):
    return aggregate([a for a, _ in items])["avg_f1"] * 100 - aggregate([b for _, b in items])["avg_f1"] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = [s for s in load_jsonl(a.samples) if (Path(a.cache) / f"{s.id}.json").exists()]
    ids = [s.id for s in samples]
    video_of = {s.id: str(s.video) for s in samples}
    timers = timer_rows(samples)
    src = {"edge": load(a.cache, None, samples), "oracle": load(a.cache, str(a.cache) + ".cues-oracle", samples)}
    items, info = {}, {}
    for name, data in src.items():
        for th in THETAS:
            items[name, th], n, per = gap_items(samples, data, timers, th)
            info[name, th] = (n, per)
    print(f"{len(samples)} questions; gap = matcher avg F1 - matched timer avg F1 (points)\n")
    print(f"{'theta':>6s} | {'edge: resp, gap':>16s} | {'oracle: resp, gap':>18s} | {'oracle - edge':>14s} {'95% interval':>18s}")
    for th in THETAS:
        d = paired_diff(items["oracle", th], items["edge", th], gap_stat, ids, video_of)
        print(f"{th:6.2f} | {info['edge', th][0]:5d} {gap_stat(items['edge', th]):+8.1f} | {info['oracle', th][0]:6d} {gap_stat(items['oracle', th]):+9.1f} |"
              f" {d['diff']:+14.1f} [{d['lo']:+6.1f}, {d['hi']:+6.1f}]{'  <- excludes 0' if d['excludes_zero'] else ''}")
    mean_stat = lambda its: float(np.mean([gap_stat([x[k] for x in its]) for k in range(len(THETAS))]))   # noqa: E731
    pack = lambda name: [tuple(items[name, th][i] for th in THETAS) for i in range(len(samples))]       # noqa: E731
    d = paired_diff(pack("oracle"), pack("edge"), mean_stat, ids, video_of)
    print(f"{'mean':>6s} | {'':16s} | {'':18s} | {d['diff']:+14.1f} [{d['lo']:+6.1f}, {d['hi']:+6.1f}]{'  <- excludes 0' if d['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
