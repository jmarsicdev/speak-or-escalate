"""Short test (CPU, seconds): does using more of the stream than the current 2 s window separate event moments?

Uses the stored matcher cache of one val run: window vectors (one per 0.5 s step, each from a 2 s window) and the cue similarities.
No video is decoded and no model is loaded.

    python scripts/pilot_context_windows.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl [--sims-from <dir.cues-oracle>]

Scores per step (all use only past steps):
  base         the matcher's cue similarity, best cue (what the matcher uses now)
  roll_k       base averaged over the last k steps (k=4: 2 s, 8: 4 s, 16: 8 s): a longer view of the same signal
  contrast_k   base minus its average over the previous k steps: "is it more cue-like than just before"
  change_k     1 - cosine(window vector now, window vector k steps ago), k=4 (2 s), 8 (4 s): visual change, no cue at all
Label: a step is positive if a true event is within 2 s. Within-question AUC (never pooled), averaged over questions that have both
kinds of step; paired difference from base with a 95% interval from resampling whole videos. Only steps with 20 steps (10 s) of history are
used, for every score, so they are compared on the same steps. Several scores are shown, so read the intervals as unadjusted.
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

H = 20  # steps of history required


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full_like(x, np.nan)
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out


def scores(V: np.ndarray, S: np.ndarray) -> dict[str, np.ndarray]:
    base = S.max(axis=1)
    sc = {"base": base}
    for k in (4, 8, 16):
        sc[f"roll_{k}"] = rolling_mean(S, k).max(axis=1)
    for k in (8, 20):
        prev = np.full_like(S, np.nan)
        rm = rolling_mean(S, k)
        prev[k:] = rm[k - 1:-1]                      # mean of the k steps BEFORE this one
        sc[f"contrast_{k}"] = (S - prev).max(axis=1)
    for k in (4, 8):
        ch = np.full(len(V), np.nan)
        ch[k:] = 1 - np.sum(V[k:] * V[:-k], axis=1)
        sc[f"change_{k}"] = ch
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sims-from", default=None, help="a .cues-oracle style dir: take cue similarities from there (vectors still from --cache)")
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    per: dict[str, dict[str, float]] = {}
    for s in samples:
        f = Path(a.cache) / f"{s.id}.json"
        if not f.exists() or not (Path(a.cache) / f"{s.id}.npy").exists():
            continue
        V = np.load(Path(a.cache) / f"{s.id}.npy")
        r = json.load(open(Path(a.sims_from) / f"{s.id}.json" if a.sims_from else f))
        t = np.array(r["t"])
        S = np.array(r["sims"], dtype=float).reshape(len(t), -1)
        g = np.array([e.t for e in s.events])
        lab = np.array([(np.abs(g - x) <= 2).any() for x in t])
        ok = np.arange(len(t)) >= H
        if lab[ok].any() and (~lab[ok]).any():
            per[s.id] = {k: auc(v[ok & lab], v[ok & ~lab]) for k, v in scores(V, S).items()}
    ids = list(per)
    print(f"{len(ids)} questions with both kinds of step (of {len(samples)})")
    base = [per[i]["base"] for i in ids]
    print(f"{'score':14s} {'mean AUC':>9s} {'change from base':>17s} {'95% interval':>18s}")
    for k in per[ids[0]]:
        v = [per[i][k] for i in ids]
        if k == "base":
            print(f"{k:14s} {np.mean(v):9.3f}")
            continue
        d = paired_diff(v, base, lambda x: float(np.mean(x)), ids, video_of)
        print(f"{k:14s} {np.mean(v):9.3f} {d['diff']:+17.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{'  <- excludes 0' if d['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
