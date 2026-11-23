"""Row 5, stage 2: the decider with recent scene-graph triplets in its prompt (question + recent evidence, no cues).

    python scripts/run_decider_evidence.py --samples data/processed/ovo_far_val.jsonl --graphs results/cache/scene_graph/<dir>

At a decision at time t the prompt gets the triplets of the last --ticks extraction ticks that END at or before t
(never the future), up to --max-lines lines, newest last. Before the first extraction tick (the first 5 s of a question)
there is no evidence and the prompt is the row 1 prompt. Writes results/cache/decider/<model>__query_evidence__<key>/
in the same format as row 1, so replay_decider.py and compare_deciders.py work on it unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.decider_cache import cache_dir_for, make_meta, run_sample  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--graphs", required=True, help="results/cache/scene_graph/<dir> from run_scene_graph_cache.py")
    ap.add_argument("--ticks", type=int, default=2, help="how many recent extraction ticks go into the prompt")
    ap.add_argument("--max-lines", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()

    over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
    over.setdefault("edge.backend", "hf")
    over["decider.variant"] = "query_evidence"
    cfg = load_config(None, over)
    gdir = Path(a.graphs)
    samples = [s for s in load_jsonl(a.samples)[: a.limit] if (gdir / f"{s.id}.jsonl").exists()]
    print(f"{len(samples)} questions with stored graphs")

    rows: dict[str, list[dict]] = {s.id: [json.loads(x) for x in open(gdir / f"{s.id}.jsonl", encoding="utf-8") if x.strip()] for s in samples}

    def evidence_fn(sample, w):
        past = [r for r in rows[sample.id] if r["t"] <= w.t_end + 1e-6][-a.ticks:]
        lines = [t for r in past for t in r["triplets"]]
        return lines[-a.max_lines:] or None

    key = hashlib.sha1((json.dumps(sorted(p.name for p in gdir.glob("*.jsonl"))) + gdir.name + f"{a.ticks}/{a.max_lines}").encode()).hexdigest()[:8]
    meta = make_meta(cfg, cfg["edge"]["model_id"], f"sg:{gdir.name}:{a.ticks}x{a.max_lines}:{key}")
    out = cache_dir_for(cfg, meta)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(out / "_meta.json", "w"), indent=1)
    print("cache:", out)

    edge = build_edge_model(cfg["edge"])
    root = Path(cfg["data"]["video_root"])
    t0, n = time.time(), 0
    for i, s in enumerate(samples, 1):
        st = run_sample(s, edge, cfg, out, root, cues=None, evidence_fn=evidence_fn)
        n += st.get("new", 0)
        print(f"[{i}/{len(samples)}] {s.id} {s.task}: {st['status']}, {st['n']} ticks", flush=True)
    print(f"done: {n} model calls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
