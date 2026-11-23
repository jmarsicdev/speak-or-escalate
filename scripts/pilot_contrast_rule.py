"""Short test (CPU, about a minute): turn the "contrast against recent past" score into triggers and score it against a matched timer.

    python scripts/pilot_contrast_rule.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl [--sims-from <dir.cues-oracle>]

Rules, all with a 4 s cooldown, each swept over a threshold grid (the whole grid is printed, no cell is picked):
  code      Em-Garde's released rule on the cue similarities (what the matcher uses now)
  contrast  fire when a cue's similarity is more than theta above its average over the previous 20 steps (10 s)
  change    same, but on the visual change signal 1 - cos(window now, window 4 steps ago), no cue involved
For every row the timer is the one whose number of responses is closest, so read the "gap" column (F1 minus that timer's F1), not the F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_matcher import f1_group, timer_matching  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import dedup, matcher_detections  # noqa: E402

K = 20


def contrast_trig(x: np.ndarray, theta: float) -> np.ndarray:
    x = x.reshape(len(x), -1)
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    trig = np.zeros(len(x), dtype=bool)
    for i in range(K, len(x)):
        prev = (c[i] - c[i - K]) / K            # mean of steps i-K .. i-1
        trig[i] = bool(((x[i] - prev) > theta).any())
    return trig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sims-from", default=None)
    a = ap.parse_args()
    samples = [s for s in load_jsonl(a.samples) if (Path(a.cache) / f"{s.id}.npy").exists()]
    D = {}
    for s in samples:
        V = np.load(Path(a.cache) / f"{s.id}.npy").astype(np.float32)
        r = json.load(open(Path(a.sims_from or a.cache) / f"{s.id}.json"))
        t = np.array(r["t"])
        ch = np.zeros(len(t)); ch[4:] = 1 - np.sum(V[4:] * V[:-4], axis=1)
        D[s.id] = (t, np.array(r["sims"], dtype=float).reshape(len(t), -1), ch)
    print(f"{len(samples)} questions\n{'rule':9s} {'theta':>6s} {'F1':>6s} {'resp':>5s} | {'timer F1':>9s} {'(every s, resp)':>16s} | {'gap':>6s}")
    for name, grid in (("code", (0.01, 0.02, 0.03, 0.04, 0.05)), ("contrast", (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)),
                       ("change", (0.02, 0.04, 0.06, 0.08, 0.10, 0.15))):
        for th in grid:
            dets = {}
            for s in samples:
                t, S, ch = D[s.id]
                if name == "code":
                    dets[s.id] = matcher_detections(t, S, "code", th, "any", "cooldown", 4.0)
                else:
                    trig = contrast_trig(S if name == "contrast" else ch, th)
                    dets[s.id] = dedup(t, trig, "cooldown", 4.0)
            f1, rc, pr, n = f1_group(samples, dets)
            per, n_t, f_t, sd = timer_matching(samples, n, seeds=10)
            print(f"{name:9s} {th:6.2f} {f1:6.1f} {n:5d} | {f_t:5.1f}±{sd:3.1f} {per:>8d}s {n_t:6.0f} | {f1 - f_t:+6.1f}", flush=True)
        print()


if __name__ == "__main__":
    main()
