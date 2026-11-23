"""Try a different set of cues on the stored window embeddings: no video, no re-embedding of frames. Takes seconds to minutes.

    python scripts/recompute_matcher_sims.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl --source oracle
    python scripts/recompute_matcher_sims.py --cache ... --source question
    python scripts/recompute_matcher_sims.py --cache ... --source file --cues path/to/cues.jsonl --name gpt

Reads the window vectors (<id>.npy) saved by run_embed_cache.py, embeds the new cue texts with the same embedding model, and writes
<cache>.cues-<name>/<id>.json in the same format, so replay_matcher.py and analyze_matcher.py work on it unchanged
(Em-Garde's embedding-based duplicate rule is not available there; use the cooldown rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_cache import load_cues  # noqa: E402
from egogate.cue_sources import oracle_cues, question_cues  # noqa: E402
from egogate.embed import OpsMMEmbedder, RandomEmbedder  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402


def recompute(cache: Path, samples, cue_fn, embedder, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in samples:
        jf, nf = cache / f"{s.id}.json", cache / f"{s.id}.npy"
        if not (jf.exists() and nf.exists()):
            continue
        cues = cue_fn(s)
        if not cues:
            continue
        r = json.load(open(jf))
        V = np.load(nf).astype(np.float32)
        sims = V @ embedder.embed_text(cues).T
        json.dump({"t": r["t"], "sims": np.round(sims, 5).tolist(), "cues": cues}, open(out / f"{s.id}.json", "w"))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--source", required=True, choices=["question", "oracle", "file"])
    ap.add_argument("--cues", default=None, help="cues.jsonl, for --source file")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
    over.setdefault("matcher.embedder", "opsmm")
    cfg = load_config(None, over)
    m = cfg["matcher"]
    samples = load_jsonl(a.samples)
    if a.source == "file":
        table = load_cues(a.cues)
        cue_fn = lambda s: table.get(s.id, [])  # noqa: E731
    else:
        cue_fn = question_cues if a.source == "question" else oracle_cues
    emb = OpsMMEmbedder(m["model_id"], m["device"], m["attn"], m.get("min_pixels")) if m["embedder"] == "opsmm" else RandomEmbedder(seed=cfg["seed"])
    name = a.name or a.source
    out = Path(str(Path(a.cache)) + f".cues-{name}")
    n = recompute(Path(a.cache), samples, cue_fn, emb, out)
    print(f"{n} questions -> {out}")
    print("score it:  python scripts/replay_matcher.py --cache", out, "--samples", a.samples)
    print("by group:  python scripts/analyze_matcher.py --cache", out, "--samples", a.samples)


if __name__ == "__main__":
    main()
