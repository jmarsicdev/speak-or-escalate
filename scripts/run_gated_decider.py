"""Row 4, stage 1b-on-demand: run the decider at the moments the cue matcher fires. Needs row 3's stored similarities.

    python scripts/run_gated_decider.py --samples data/processed/ovo_far_val.jsonl --matcher results/cache/matcher/<dir>

Triggers come from the released-code rule with "any cue" at the lowest threshold we will consider (0.02, no duplicate removal),
so the stored answers serve every higher threshold too. Resumable and memoised: re-running never repeats a decider call.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_cache import load_cues  # noqa: E402
from egogate.decider_cache import cache_dir_for, cues_key_for, make_meta  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.gated import run_ondemand, trigger_times  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--matcher", required=True, help="results/cache/matcher/<dir> from run_embed_cache.py")
ap.add_argument("--theta-min", type=float, default=0.02)
ap.add_argument("--rule", default="code", choices=["code", "paper"])
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--cues", default=None, help="cues.jsonl; needed when decider.variant is query_cues (row 2 decider at the matcher moments)")
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
over.setdefault("edge.backend", "hf")
cfg = load_config(None, over)
samples = load_jsonl(a.samples)
if a.limit:
    samples = samples[: a.limit]
mdir = Path(a.matcher)
cues = None
if cfg["decider"]["variant"] != "query_only":
    if not a.cues:
        raise SystemExit("decider.variant is not query_only: pass --cues <cues.jsonl>")
    cues = load_cues(a.cues)
meta = make_meta(cfg, cfg["edge"]["model_id"], cues_key_for(a.cues) if cues else None)
dec_dir = cache_dir_for(cfg, meta)
dec_dir.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(dec_dir / "_meta.json", "w"), indent=1)

jobs, total = [], 0
for s in samples:
    f = mdir / f"{s.id}.json"
    if not f.exists():
        continue
    r = json.load(open(f))
    t, sims = np.array(r["t"]), np.array(r["sims"]).reshape(len(r["t"]), -1)
    times = trigger_times(t, sims, a.rule, a.theta_min, "any")
    jobs.append((s, times))
    total += len(times)
print(f"decider cache: {dec_dir}\n{len(jobs)} questions, {total} trigger moments to ask about (rule {a.rule}, any cue, theta >= {a.theta_min})")
edge = build_edge_model(cfg["edge"])
video_root = Path(cfg["data"]["video_root"])
t0, new = time.time(), 0
for i, (s, times) in enumerate(jobs, 1):
    r = run_ondemand(s, edge, cfg, times, dec_dir, video_root, cues[s.id] if cues else None)
    new += r["n_new"]
    print(f"[{i}/{len(jobs)}] {s.id} {s.task}: {len(times)} moments, {r['n_new']} new decider calls", flush=True)
print(f"done: {new} decider calls in {time.time() - t0:.0f} s")
