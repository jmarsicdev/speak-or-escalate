"""Stage 0: parse every question into visual cues with the edge model (text only) and store them.

    python scripts/run_cues.py --samples data/processed/ovo_far_val.jsonl

Writes results/cache/cues/<model>__<key>/cues.jsonl (resumable). Run once for all splits you will use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_cache import cue_dir, cue_meta, run_cues  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True, help="one or more split files")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
over.setdefault("edge.backend", "hf")
cfg = load_config(None, over)
samples = [s for f in a.samples for s in load_jsonl(f)]
if a.limit:
    samples = samples[: a.limit]
meta = cue_meta(cfg)
d = cue_dir(cfg, meta)
d.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(d / "_meta.json", "w"), indent=1)
print(f"cues -> {d / 'cues.jsonl'}  ({len(samples)} questions)")
run_cues(samples, build_edge_model(cfg["edge"]), cfg, d / "cues.jsonl")
