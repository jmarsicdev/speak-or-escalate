"""Short test (CPU, seconds): is the 2 s window too large for the matcher? Needs the per-frame file written by run_embed_alt.py.

A step's score with a k-frame window is the mean over its last k frames (0.5 s apart) of each frame's similarity to a cue, best cue taken last.
k=1 is a single frame (the newest), k=4 is the current 2 s window (up to the small difference between averaging vectors and averaging similarities),
k=8 is 4 s. Within-question AUC (never pooled) for steps within R s of an event against steps beyond R, R = 2 (what is scored) and 4; steps with
less than 10 s of history are dropped for every k so all sizes see the same steps. Paired difference from k=4 with a video-level 95% interval.

    python scripts/pilot_window_size.py --dir results/cache/matcher/<siglip dir> --samples data/processed/ovo_far_val.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.bootstrap import paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402

KS, H = (1, 2, 4, 8), 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    for cues in ("edge", "oracle"):
        for R in (2, 4):
            per, ids = {}, []
            for s in samples:
                f = Path(a.dir) / f"{s.id}.frames.npz"
                if not f.exists():
                    continue
                z = np.load(f); t = z["times"]; S = z["sims_" + cues]
                ev = np.array([e.t for e in s.events]); d = np.min(np.abs(t[:, None] - ev[None, :]), axis=1)
                ok = np.arange(len(t)) >= H
                near, far = ok & (d <= R), ok & (d > R)
                if not (near.any() and far.any()):
                    continue
                per[s.id] = {}
                for k in KS:
                    c = np.cumsum(np.vstack([np.zeros((1, S.shape[1])), S]), axis=0)
                    m = np.full_like(S, np.nan); m[k - 1:] = (c[k:] - c[:-k]) / k
                    x = m.max(axis=1)
                    per[s.id][k] = auc(x[near], x[far])
                ids.append(s.id)
            print(f"\ncues={cues}, R={R} s: {len(ids)} questions")
            base = [per[i][4] for i in ids]
            for k in KS:
                v = [per[i][k] for i in ids]
                if k == 4:
                    print(f"  k={k} ({k * 0.5:g} s) AUC {np.mean(v):.3f}   (reference)")
                    continue
                dd = paired_diff(v, base, lambda x: float(np.mean(x)), ids, video_of)
                print(f"  k={k} ({k * 0.5:g} s) AUC {np.mean(v):.3f}   change {dd['diff']:+.3f} [{dd['lo']:+.3f}, {dd['hi']:+.3f}]{'  <- excludes 0' if dd['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
