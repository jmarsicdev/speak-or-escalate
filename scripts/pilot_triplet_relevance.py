"""Step B follow-up: do triplets from event ticks look more like the cue text than triplets from other ticks?

Text against text with the Ops-MM embedder (separate process from the decider). For each tick: cosine between the
tick's triplets (one string per triplet) and each cue; keep the max over triplets and cues. Then compare ticks within
5 s of an event with the rest, as a within-question AUC (never pooled).

    python scripts/pilot_triplet_relevance.py --dir results/cache/scene_graph_pilot_fixed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.embed import OpsMMEmbedder  # noqa: E402


def auc(pos, neg):
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    emb = OpsMMEmbedder()
    out = []
    for f in sorted(Path(args.dir).glob("ovo_*.json")):
        j = json.loads(f.read_text(encoding="utf-8-sig"))
        refs = {"oracle": j["oracle"], "query": [j["query"]]}
        rv = {k: emb.embed_text(v) for k, v in refs.items()}
        row = {"id": j["id"]}
        for k in refs:
            sc, near = [], []
            for t in j["ticks"]:
                if not t["triplets"]:
                    continue
                tv = emb.embed_text(t["triplets"])
                sc.append(float((tv @ rv[k].T).max()))
                near.append(t["near_event"])
            row[k] = (auc([s for s, n in zip(sc, near) if n], [s for s, n in zip(sc, near) if not n]),
                      float(np.mean(sc)) if sc else None, len(sc), sum(near))
        out.append(row)
        print(row, flush=True)
    for k in ("oracle", "query"):
        a = [r[k][0] for r in out if r[k][0] is not None]
        print(f"{k}: within-question AUC mean {np.mean(a):.3f} over {len(a)} questions that have both kinds of tick")


if __name__ == "__main__":
    main()
