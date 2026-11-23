"""Compare two matcher caches question by question (CPU, seconds): within-question AUC of the best-cue similarity for steps within 2 s of
an event against other steps (10 s of history required, as in pilot_context_windows.py), paired difference B - A with a video-level 95% interval.

    python scripts/compare_embedders.py --a results/cache/matcher/<ops dir> --b results/cache/matcher/<alt dir> --samples data/processed/ovo_far_val.jsonl
Run it once for the edge cues (dirs as they are) and once for oracle cues (add .cues-oracle to both dirs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.bootstrap import paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402

H = 20


def per_q(d: Path, s):
    f = d / f"{s.id}.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    t = np.array(r["t"]); S = np.array(r["sims"], dtype=float).reshape(len(t), -1).max(axis=1)
    return t, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    ids, A, B, grp = [], [], [], []
    for s in samples:
        ra, rb = per_q(Path(a.a), s), per_q(Path(a.b), s)
        if ra is None or rb is None:
            continue
        n = min(len(ra[0]), len(rb[0])); t = ra[0][:n]
        g = np.array([e.t for e in s.events]); lab = np.array([(np.abs(g - x) <= 2).any() for x in t]); ok = np.arange(n) >= H
        if lab[ok].any() and (~lab[ok]).any():
            ids.append(s.id); grp.append(f"{s.task} / {s.meta.get('source')}")
            A.append(auc(ra[1][:n][ok & lab], ra[1][:n][ok & ~lab])); B.append(auc(rb[1][:n][ok & lab], rb[1][:n][ok & ~lab]))
    print(f"{len(ids)} questions with both kinds of step in both caches")
    print(f"{'group':26s} {'n':>3s} {'AUC A':>7s} {'AUC B':>7s} {'B - A':>7s} {'95% interval':>18s}")
    for name in ["all"] + sorted(set(grp)):
        idx = [i for i, g in enumerate(grp) if name == "all" or g == name]
        if not idx:
            continue
        ii = [ids[i] for i in idx]
        d = paired_diff([B[i] for i in idx], [A[i] for i in idx], lambda v: float(np.mean(v)), ii, video_of)
        print(f"{name:26s} {len(idx):3d} {np.mean([A[i] for i in idx]):7.3f} {np.mean([B[i] for i in idx]):7.3f} {d['diff']:+7.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{'  <- excludes 0' if d['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
