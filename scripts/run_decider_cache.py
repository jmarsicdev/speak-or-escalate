"""Row 1, stage 1b: run the edge model alone over a split and store its Yes/No scores.

    # quick smoke test on 2 short videos
    python scripts/run_decider_cache.py --samples data/processed/ovo_far_val.jsonl --limit 2

    # a full split (resumable: re-run the same command after an interruption)
    python scripts/run_decider_cache.py --samples data/processed/ovo_far_val.jsonl \
        --set edge.model_id=Qwen/Qwen3-VL-2B-Instruct --set edge.backend=hf

Writes results/cache/decider/<model>__<variant>__<key>/<id>.jsonl (see egogate/decider_cache.py).
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
from egogate.decider_cache import cache_dir_for, cues_key_for, make_meta, run_sample  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--shortest", action="store_true", help="with --limit: take the shortest questions first")
    ap.add_argument("--cues", default=None, help="cues.jsonl from run_cues.py; needed when decider.variant is query_cues (row 2)")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value config overrides")
    args = ap.parse_args()

    over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in args.set)}
    over.setdefault("edge.backend", "hf")
    cfg = load_config(None, over)
    samples = load_jsonl(args.samples)
    if args.limit:
        if args.shortest:
            samples = sorted(samples, key=lambda s: (s.end_s or 0) - s.start_s)
        samples = samples[: args.limit]

    cues = None
    if cfg["decider"]["variant"] != "query_only":
        if not args.cues:
            raise SystemExit("decider.variant is not query_only: pass --cues <cues.jsonl>")
        cues = load_cues(args.cues)
        samples = [s for s in samples if s.id in cues]
    meta = make_meta(cfg, cfg["edge"]["model_id"], cues_key_for(args.cues) if cues else None)
    out_dir = cache_dir_for(cfg, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(out_dir / "_meta.json", "w"), indent=1)
    print(f"cache: {out_dir}")

    edge = build_edge_model(cfg["edge"])
    video_root = Path(cfg["data"]["video_root"])
    t0, n_calls = time.time(), 0
    for i, s in enumerate(samples, 1):
        st = run_sample(s, edge, cfg, out_dir, video_root, cues=cues[s.id] if cues else None)
        n_calls += st.get("new", 0)
        print(f"[{i}/{len(samples)}] {s.id} {s.task}: {st['status']}, {st['n']} ticks"
              + (f", {st['skipped_ticks']} skipped (video too short)" if st.get("skipped_ticks") else ""), flush=True)
    print(f"done: {n_calls} model calls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
