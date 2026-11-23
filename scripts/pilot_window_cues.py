"""Matcher on SigLIP2 frame vectors: shorter windows (1, 2 frames) and scene-level ("general") oracle cues, one table. CPU, no video.

Needs the per-frame vectors written by run_embed_alt.py (<dir>/<id>.frames.npz with "vecs") and the cues from write_general_cues.py.
A step's window vector is the normalised mean of its last k frame vectors (0.5 s apart; k=4 is the 2 s window the matcher uses), its score the
best cosine over the cues. Within-question AUC (never pooled) of steps within 2 s of an event against the rest, 10 s of history required so
every row sees the same steps; paired differences with a video-level 95% interval. Also the mean similarity itself (does a cue "match" better
in absolute terms) near and away from events, since a higher similarity is not the same as a better separation.

    python scripts/pilot_window_cues.py --dir results/cache/matcher/siglip2-so400m-patch14-384__alt \
        --general results/cache/cues/general_oracle__qwen3-vl-8b-instruct.jsonl --samples data/processed/ovo_far_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogate.cue_sources import oracle_cues  # noqa: E402
from egogate.eval.bootstrap import paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402

KS, H, R = (1, 2, 4), 20, 2.0


def windows(F, k):
    c = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F]), axis=0)
    W = np.full_like(F, np.nan)
    W[k - 1:] = (c[k:] - c[:-k]) / k
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--general", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from run_embed_alt import Siglip  # noqa: PLC0415

    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    general = {r["id"]: r["cues"] for r in map(json.loads, open(a.general, encoding="utf-8"))}
    emb = Siglip("google/siglip2-so400m-patch14-384", device=a.device)
    res, ids = {}, []
    for s in samples:
        f = Path(a.dir) / f"{s.id}.frames.npz"
        if not f.exists() or s.id not in general:
            continue
        z = np.load(f)
        if "vecs" not in z:
            continue
        t, F = z["times"], z["vecs"].astype(np.float32)
        ev = np.array([e.t for e in s.events])
        d = np.min(np.abs(t[:, None] - ev[None, :]), axis=1)
        ok = np.arange(len(t)) >= H
        near, far = ok & (d <= R), ok & (d > R)
        if not (near.any() and far.any()):
            continue
        edge_cues = json.load(open(Path(a.dir) / f"{s.id}.json"))["cues"]
        texts = {"edge": edge_cues, "oracle": oracle_cues(s), "general": general[s.id]}
        res[s.id] = {}
        for name, cues in texts.items():
            T = emb.text(cues)
            for k in KS:
                x = (windows(F, k) @ T.T).max(axis=1)
                res[s.id][(name, k)] = (auc(x[near], x[far]), float(x[near].mean()), float(x[far].mean()))
        ids.append(s.id)
    print(f"{len(ids)} questions (within 2 s of an event vs the rest, 10 s of history required)\n")
    mean = lambda v: float(np.mean(v))  # noqa: E731
    print(f"{'cues':8s} {'window':>8s}  {'AUC':>6s}  {'sim near':>8s} {'sim far':>8s}   vs edge 2 s               vs oracle 2 s")
    for name in ("edge", "oracle", "general"):
        for k in KS:
            v = [res[i][(name, k)][0] for i in ids]
            cols = []
            for ref in (("edge", 4), ("oracle", 4)):
                if (name, k) == ref:
                    cols.append(f"{'(reference)':24s}")
                    continue
                dd = paired_diff(v, [res[i][ref][0] for i in ids], mean, ids, video_of)
                cols.append(f"{dd['diff']:+.3f} [{dd['lo']:+.3f}, {dd['hi']:+.3f}]{'*' if dd['excludes_zero'] else ' '}")
            sn = mean([res[i][(name, k)][1] for i in ids])
            sf = mean([res[i][(name, k)][2] for i in ids])
            print(f"{name:8s} {k * 0.5:>6g} s  {mean(v):6.3f}  {sn:8.3f} {sf:8.3f}   {cols[0]}  {cols[1]}")
    print("\n* interval excludes 0. AUC 0.5 = no separation. 'sim' = mean best-cue cosine near / away from events.")


if __name__ == "__main__":
    main()
