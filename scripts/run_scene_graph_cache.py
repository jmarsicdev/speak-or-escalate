"""Row 5, stage 1: the edge model writes scene-graph triplets every 5 s for every question in a split (no question in the prompt).

    python scripts/run_scene_graph_cache.py --samples data/processed/ovo_far_val.jsonl

Settings are the ones from the step B pilot (docs/RATIONALE.md 9a): fixed prompt, previous tick's triplets as text,
repetition penalty 1.15, 80 new tokens. Writes results/cache/scene_graph/<model>__<key>/<id>.jsonl, one row per tick:
{"t", "ms", "triplets": ["subject, relation, object", ...], "raw"}. Resumable: finished questions are skipped, a part file continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.frames import stream_windows  # noqa: E402

PROMPT = """You are watching a first-person video clip of the last {secs} seconds.
Write at most {k} scene-graph triplets, one per line, as (subject, relation, object), in English only.
Describe what the people DO (the actions and what changed), not what they wear or what the room contains.
Each line must be different. Do not repeat a line. Do not write the word "relation".
{prev}"""
SETTINGS = {"every_s": 5.0, "per_clip": 8, "rep_penalty": 1.15, "max_new": 80, "fps": 1.0, "version": "sg-v2"}

_LABEL = re.compile(r"\b(subject|action|object|relation)\s*:\s*", re.I)


def parse(text: str) -> list[str]:
    """One cleaned string per triplet. Labelled lines "(Subject: a) (Action: b)" are joined into one; a line holding several
    plain bracket groups is split into one triplet per closed group; a group cut off by the token limit is dropped."""
    out, seen = [], set()
    for ln in (x.strip() for x in text.splitlines() if x.strip()):
        if _LABEL.search(ln):
            if ln.count("(") != ln.count(")"):
                continue
            items = [re.sub(r"\)\s*\(", ", ", _LABEL.sub("", ln))]
        else:
            items = re.findall(r"\(([^()]*)\)", ln) or ([ln] if "(" not in ln else [])
        for it in items:
            it = re.sub(r"\s+", " ", it.replace("(", "").replace(")", "")).strip(" ,.")
            if len(it) >= 5 and it.lower() not in seen:
                seen.add(it.lower())
                out.append(it)
    return out

def cache_dir(cfg) -> Path:
    key = hashlib.sha1((json.dumps(SETTINGS, sort_keys=True) + PROMPT + cfg["edge"]["model_id"] + str(cfg["decider"]["max_side"])
                        + str(cfg["edge"].get("dtype"))).encode()).hexdigest()[:8]
    return Path(cfg["decider"]["cache_dir"]) / "scene_graph" / f"{cfg['edge']['model_id'].split('/')[-1]}__{key}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-source", type=int, default=None, help="short pilot: only the N shortest questions of each source")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
    over.setdefault("edge.backend", "hf")
    cfg = load_config(None, over)
    samples = load_jsonl(a.samples)[: a.limit]
    if a.per_source:
        by: dict = {}
        for s in sorted(samples, key=lambda s: (s.end_s or 0) - s.start_s):
            by.setdefault(s.meta.get("source", s.task), []).append(s)
        samples = [s for v in by.values() for s in v[: a.per_source]]
        print("pilot subset:", [s.id for s in samples])
    out = cache_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**SETTINGS, "prompt": PROMPT, "model_id": cfg["edge"]["model_id"]}, open(out / "_meta.json", "w"), indent=1)
    print("cache:", out)
    edge = build_edge_model(cfg["edge"])
    root = Path(cfg["data"]["video_root"])
    t_all, n = time.time(), 0
    for i, s in enumerate(samples, 1):
        final, part = out / f"{s.id}.jsonl", out / f"{s.id}.jsonl.part"
        if final.exists():
            continue
        end = s.end_s if s.end_s is not None else s.start_s
        ticks = [round(s.start_s + SETTINGS["every_s"] * k, 3) for k in range(1, int((end - s.start_s) / SETTINGS["every_s"] + 1e-9) + 1)]
        rows = [json.loads(x) for x in open(part)] if part.exists() else []
        done = {round(r["t"], 3) for r in rows}
        prev = rows[-1]["triplets"][: SETTINGS["per_clip"]] if rows else []
        with open(part, "a") as f:
            for w in stream_windows(root / s.video, [t for t in ticks if t not in done], SETTINGS["every_s"], SETTINGS["fps"], cfg["decider"]["max_side"]):
                prv = ("Previous clip: " + " ".join(f"({x})" for x in prev) + "\nNow say what is happening that is new or different.") if prev else ""
                pr = PROMPT.format(secs=int(SETTINGS["every_s"]), k=SETTINGS["per_clip"], prev=prv)
                t0 = time.perf_counter()
                inp = edge._inputs(pr, w.frames)
                with edge.torch.no_grad():
                    o = edge.model.generate(**inp, max_new_tokens=SETTINGS["max_new"], do_sample=False, repetition_penalty=SETTINGS["rep_penalty"])
                text = edge.processor.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                ms = (time.perf_counter() - t0) * 1000
                trips = parse(text)
                f.write(json.dumps({"t": round(w.t_end, 3), "ms": round(ms), "triplets": trips, "raw": text}) + "\n")
                f.flush()
                prev = trips[: SETTINGS["per_clip"]]
                n += 1
        part.rename(final)
        print(f"[{i}/{len(samples)}] {s.id}: {len(ticks)} ticks, {n} ticks so far, {time.time() - t_all:.0f} s", flush=True)
    print(f"done: {n} ticks in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
