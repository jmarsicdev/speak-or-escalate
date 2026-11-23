"""Step B (D22): scene-graph pilot. The edge model writes (subject, relation, object) triplets every N s from the
frames alone (the question is NOT in the prompt). We store them, time the generation and compute two cheap proxies.

    python scripts/pilot_scene_graph.py --samples data/processed/ovo_far_val.jsonl --per-source 1 --max-span 90

Output: results/cache/scene_graph_pilot/<id>.json  (triplets per tick) and a printed summary.
Proxies (all descriptive, a handful of questions, no claims):
  parse rate      share of ticks with at least one parsable triplet
  novelty         share of a tick's triplets (lowercased text) not seen in the previous tick; compared near events vs elsewhere
  cue-word hit    share of ticks whose triplet text contains a content word of the oracle cue; near events vs elsewhere
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_sources import oracle_cues  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.evidence import TRIPLET_PROMPT, parse_triplets  # noqa: E402
from egogate.frames import stream_windows  # noqa: E402

STOP = set("the a an of to in on and or is are with at for from by it this that step then".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--per-source", type=int, default=1)
    ap.add_argument("--max-span", type=float, default=90.0, help="seconds of video per question")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--per-clip", type=int, default=8)
    ap.add_argument("--out", default="results/cache/scene_graph_pilot")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in args.set)}
    over.setdefault("edge.backend", "hf")
    cfg = load_config(None, over)
    samples = load_jsonl(args.samples)
    by_src: dict[str, list] = {}
    for s in sorted(samples, key=lambda s: (s.end_s or 0) - s.start_s):
        by_src.setdefault(s.meta.get("source", s.task), []).append(s)
    chosen = [s for v in by_src.values() for s in v[: args.per_source]]
    print("questions:", [(s.id, s.meta.get("source")) for s in chosen])

    edge = build_edge_model(cfg["edge"])
    video_root = Path(cfg["data"]["video_root"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = TRIPLET_PROMPT.format(k=args.per_clip)
    rows = []   # per tick: dict
    gen_ms = []
    for s in chosen:
        end = min(s.end_s if s.end_s is not None else s.start_s + args.max_span, s.start_s + args.max_span)
        ticks = [round(s.start_s + args.every * k, 3) for k in range(1, int((end - s.start_s) / args.every) + 1)]
        cue_words = words(" ".join(oracle_cues(s)))
        prev: set[str] = set()
        rec = []
        for w in stream_windows(video_root / s.video, ticks, args.every, 1.0, cfg["decider"]["max_side"]):
            t0 = time.perf_counter()
            text = edge.complete(prompt, images=w.frames, max_new_tokens=160)
            ms = (time.perf_counter() - t0) * 1000
            gen_ms.append(ms)
            trips = parse_triplets(text, w.t_end)
            cur = {f"{t.subject} {t.relation} {t.obj}".lower() for t in trips}
            nov = len(cur - prev) / len(cur) if cur else None
            near = any(abs(w.t_end - e.t) <= args.every for e in s.events)
            hit = bool(words(" ".join(cur)) & cue_words)
            rows.append({"id": s.id, "task": s.task, "near_event": near, "n": len(cur), "novelty": nov, "hit": hit})
            rec.append({"t": w.t_end, "ms": round(ms), "raw": text, "triplets": [str(t) for t in trips], "near_event": near})
            prev = cur
        json.dump({"id": s.id, "query": s.query, "oracle": oracle_cues(s), "events": [e.t for e in s.events], "ticks": rec},
                  open(out_dir / f"{s.id}.json", "w"), indent=1)
        print(f"{s.id} ({s.task}): {len(rec)} ticks", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nticks {len(rows)}; generation {mean(gen_ms):.0f} ms per tick (median {sorted(gen_ms)[len(gen_ms)//2]:.0f})")
    print(f"parse rate {mean([r['n'] > 0 for r in rows]):.2f}; triplets per tick {mean([r['n'] for r in rows]):.1f}")
    for lab, sel in (("near event", True), ("elsewhere", False)):
        g = [r for r in rows if r["near_event"] == sel]
        print(f"{lab:11s} n={len(g):3d}  novelty {mean([r['novelty'] for r in g]):.2f}  cue-word hit {mean([r['hit'] for r in g]):.2f}")


if __name__ == "__main__":
    main()
