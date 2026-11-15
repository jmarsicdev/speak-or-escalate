"""Stage 1a: embed every 2 s window (every 0.5 s) and store cue similarities. Needs the cues from run_cues.py.

    python scripts/run_embed_cache.py --samples data/processed/ovo_far_val.jsonl --cues results/cache/cues/<dir>/cues.jsonl \
        --set matcher.embedder=opsmm

Writes results/cache/matcher/<model>__<key>/<id>.json (committable) and <id>.npy (window vectors, local only).
Resumable: finished questions are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_cache import load_cues  # noqa: E402
from egogate.embed import OpsMMEmbedder, RandomEmbedder  # noqa: E402
from egogate.embed_cache import cache_dir_for, make_meta, run_sample  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--cues", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shortest", action="store_true")
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
over.setdefault("matcher.embedder", "opsmm")
cfg = load_config(None, over)
m = cfg["matcher"]

samples = load_jsonl(a.samples)
if a.limit:
    if a.shortest:
        samples = sorted(samples, key=lambda s: (s.end_s or 0) - s.start_s)
    samples = samples[: a.limit]
cues = load_cues(a.cues)
samples = [s for s in samples if s.id in cues]
meta = make_meta(cfg)
out_dir = cache_dir_for(cfg, meta)
out_dir.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(out_dir / "_meta.json", "w"), indent=1)
print(f"cache: {out_dir}  ({len(samples)} questions with cues)")

embedder = OpsMMEmbedder(m["model_id"], m["device"], m["attn"]) if m["embedder"] == "opsmm" else RandomEmbedder(seed=cfg["seed"])
video_root = Path(cfg["data"]["video_root"])
t0 = time.time()
for i, s in enumerate(samples, 1):
    st = run_sample(s, embedder, cues[s.id], cfg, out_dir, video_root)
    print(f"[{i}/{len(samples)}] {s.id} {s.task}: {st['status']}, {st['n']} steps"
          + (f", median {st['ms_median']:.0f} ms/step" if st.get("ms_median") else ""), flush=True)
print(f"done in {time.time() - t0:.0f} s")
