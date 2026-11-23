"""Short test (CPU, seconds): is the matcher better at "an event is around now" than at "an event is exactly now"?

For each stored matcher cache, the within-question AUC of the best-cue similarity for steps within R seconds of a true event against the
steps further away, for several R. R = 2 is what the benchmark scores (F1 uses +/- 2 s). A score that rises with R is a build-up detector more than an
exact-moment detector. Steps with less than 10 s of history are left out, and so are steps inside the event band when the AUC is taken for
a larger R's "far" side only if they are within R (the far side is everything beyond R).

    python scripts/pilot_label_radius.py --samples data/processed/ovo_far_val.jsonl --caches <dir> [<dir> ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402

H, RADII = 20, (2, 4, 6, 10, 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    print(f"{'cache':58s} " + " ".join(f"R={r:<3d}" for r in RADII) + "   (mean within-question AUC; n questions per R in brackets)")
    for c in a.caches:
        row = []
        for R in RADII:
            v = []
            for s in samples:
                f = Path(c) / f"{s.id}.json"
                if not f.exists():
                    continue
                r = json.load(open(f)); t = np.array(r["t"]); x = np.array(r["sims"], dtype=float).reshape(len(t), -1).max(axis=1)
                d = np.min(np.abs(t[:, None] - np.array([e.t for e in s.events])[None, :]), axis=1)
                ok = np.arange(len(t)) >= H
                near, far = ok & (d <= R), ok & (d > R)
                if near.any() and far.any():
                    v.append(auc(x[near], x[far]))
            row.append((np.mean(v), len(v)))
        print(f"{Path(c).name:58s} " + " ".join(f"{m:.3f}" for m, _ in row) + f"   [{row[0][1]}..{row[-1][1]}]")


if __name__ == "__main__":
    main()
