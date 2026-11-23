"""Short test: does scoring what is NEW in the scene-graph triplets (against the previous tick) separate event ticks?

    python scripts/pilot_triplet_novelty.py --graphs results/cache/scene_graph/<dir> --samples data/processed/ovo_far_val.jsonl

Text against text with the Ops-MM embedder (run on its own, not with the decider). Per tick t (a 5 s extraction tick covering [t-5, t]):
  rel      max cosine between any triplet and any cue
  new      mean over triplets of 1 - (max cosine to the previous tick's triplets): how much changed
  new_rel  max over triplets of cue-cosine * novelty: relevant AND new
  rise     rel(t) - rel(t-1), the paper's rise idea applied to text
A tick is "near an event" if an event lies in [t-5, t+2]. Within-question AUC (never pooled) of each score for near vs not near,
averaged over questions that have both kinds of tick, with cues = oracle (upper bound) and = the question text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.cue_sources import oracle_cues, question_cues  # noqa: E402
from egogate.embed import OpsMMEmbedder  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    emb = OpsMMEmbedder()
    res = {(src, sc): [] for src in ("oracle", "query") for sc in ("rel", "new", "new_rel", "rise")}
    for s in load_jsonl(a.samples):
        f = Path(a.graphs) / f"{s.id}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
        vecs = [emb.embed_text(r["triplets"]) if r["triplets"] else None for r in rows]
        ev = [e.t for e in s.events]
        near = np.array([any(r["t"] - 5 <= t <= r["t"] + 2 for t in ev) for r in rows])
        for src, cues in (("oracle", oracle_cues(s)), ("query", question_cues(s))):
            cv = emb.embed_text(cues)
            rel, new, newrel = [], [], []
            for i, v in enumerate(vecs):
                if v is None or i == 0 or vecs[i - 1] is None:
                    rel.append(np.nan); new.append(np.nan); newrel.append(np.nan)
                    continue
                c = (v @ cv.T).max(axis=1)                       # per triplet: best cue cosine
                nv = 1 - (v @ vecs[i - 1].T).max(axis=1)         # per triplet: how unlike anything in the previous tick
                rel.append(float(c.max())); new.append(float(nv.mean())); newrel.append(float((c * nv).max()))
            rel, new, newrel = map(np.array, (rel, new, newrel))
            rise = np.concatenate([[np.nan], rel[1:] - rel[:-1]])
            ok_all = ~np.isnan(rel)
            for name, sc in (("rel", rel), ("new", new), ("new_rel", newrel), ("rise", rise)):
                ok = ~np.isnan(sc)
                p, n = sc[ok & near], sc[ok & ~near]
                if len(p) and len(n):
                    res[(src, name)].append((s.id, auc(p, n)))
    for src in ("oracle", "query"):
        for name in ("rel", "new", "new_rel", "rise"):
            v = [x[1] for x in res[(src, name)]]
            print(f"cues={src:6s} score={name:8s} within-question AUC {np.mean(v):.3f} over {len(v)} questions")


if __name__ == "__main__":
    main()
