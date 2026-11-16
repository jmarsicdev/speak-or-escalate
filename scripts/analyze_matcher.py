"""Break the cue matcher's result down by task and by video source, with matched-timer comparisons. CPU only.

    python scripts/analyze_matcher.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl

Answers: where does the matcher carry signal (which task, which kind of video), and is that more than a timer making the same
number of responses would get?
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.floors import detections_for  # noqa: E402
from egogate.eval.metrics import aggregate, score_question  # noqa: E402
from egogate.matcher_replay import matcher_detections, similarity_diagnostics  # noqa: E402
import random  # noqa: E402


def load(cache: Path, samples):
    data = {}
    for s in samples:
        f = cache / f"{s.id}.json"
        if f.exists():
            r = json.load(open(f))
            data[s.id] = {"t": np.array(r["t"]), "sims": np.array(r["sims"]).reshape(len(r["t"]), -1)}
    return data


def f1_group(samples, dets):
    rows = []
    for s in samples:
        q = score_question(dets.get(s.id, []), [e.t for e in s.events])
        rows.append({"task": "all", "recall": q.recall, "precision": q.precision, "n_det": q.n_det})
    a = aggregate(rows)["tasks"]["all"] if rows else {"f1": 0, "recall": 0, "precision": 0}
    return a["f1"] * 100, a["recall"] * 100, a["precision"] * 100, sum(r["n_det"] for r in rows)


def timer_matching(samples, n_target, seeds=30):
    """F1 of a timer (respond every k s, random phase) whose number of responses is closest to n_target; averaged over phases."""
    best = None
    for period in (2, 4, 6, 8, 10, 12, 16, 20, 30, 45, 60, 90):
        vals, counts = [], []
        for sd in range(seeds):
            rng = random.Random(sd)
            dets = {s.id: detections_for("periodic", s, 2.0, rng, period_s=float(period)) for s in samples}
            f, _, _, n = f1_group(samples, dets)
            vals.append(f); counts.append(n)
        n_mean = float(np.mean(counts))
        if best is None or abs(n_mean - n_target) < abs(best[1] - n_target):
            best = (period, n_mean, float(np.mean(vals)), float(np.std(vals)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--rule", default="code")
    ap.add_argument("--theta", type=float, default=0.03)
    ap.add_argument("--cooldown", type=float, default=4.0)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    data = load(Path(a.cache), samples)
    samples = [s for s in samples if s.id in data]
    dets = {s.id: matcher_detections(data[s.id]["t"], data[s.id]["sims"], a.rule, a.theta, "any", "cooldown", a.cooldown) for s in samples}
    print(f"setting: rule={a.rule}, any cue, theta={a.theta:g}, cooldown {a.cooldown:g}s; {len(samples)} questions\n")

    groups = defaultdict(list)
    for s in samples:
        groups[("task", s.task)].append(s)
        groups[("source", s.meta.get("source", "?"))].append(s)
        groups[("task+source", f"{s.task} / {s.meta.get('source', '?')}")].append(s)
    print(f"{'group':28s} {'n':>3s} {'F1':>6s} {'recall':>7s} {'prec':>6s} {'resp':>5s} | {"timer F1":>9s} (closest timer) | {'AUC max sim':>11s} {'AUC rise':>9s}")
    for kind in ("task", "source", "task+source"):
        for (k, name), ss in sorted(groups.items()):
            if k != kind:
                continue
            f, r, p, n = f1_group(ss, dets)
            tm = timer_matching(ss, n) if n else None
            d = similarity_diagnostics(ss, {s.id: data[s.id] for s in ss})
            tstr = f"{tm[2]:5.1f}±{tm[3]:.1f} (every {tm[0]} s, {tm[1]:.0f} resp)" if tm else "      n/a"
            print(f"{kind[:4]:4s} {name:23s} {len(ss):3d} {f:6.1f} {r:7.1f} {p:6.1f} {n:5d} | {tstr:>34s} | {d['max sim']:11.3f} {d['max rise (paper)']:9.3f}")
    print()


if __name__ == "__main__":
    main()
