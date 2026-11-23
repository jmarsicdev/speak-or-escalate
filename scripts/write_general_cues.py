"""Experiment: oracle cues made more general (scene-level rather than one instant), written by a model from the benchmark's own descriptions.

The oracle cues (egogate/cue_sources.py) describe the exact moment: a tutorial step, the counted activity, the clue's answer. A 2 s window is
compared against them as a whole scene, so a scene-level rewording might match the window vectors better. Each oracle cue is rewritten by a
model (text only, temperature 0) into one line describing the scene a camera would show while it is going on. One rewritten cue per oracle cue.

    python scripts/write_general_cues.py --samples data/processed/ovo_far_val.jsonl --model qwen/qwen3-vl-8b-instruct

Writes results/cache/cues/general_oracle__<model tag>.jsonl: {"id", "cues", "oracle"} per question. Resumable.
Like the oracle cues, these use the benchmark's annotations, which no deployed system has: an upper-bound probe, not a method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.cue_sources import oracle_cues  # noqa: E402
from egogate.edge import APIEdgeVLM  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402

PROMPT = """A video question and the description of the moment that answers it:
Question: "{query}"
Moment: "{cue}"

Rewrite the moment as a short description of the whole scene a camera would show while this is going on: the setting, the people and
objects, and the ongoing activity, rather than the single instant. Keep it visual and concrete. One line, at most 15 words, no quotes."""


def clean(text: str) -> str:
    line = next((x for x in text.strip().splitlines() if x.strip()), "")
    return line.strip().strip('"').strip("'").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    a = ap.parse_args()
    out = Path("results/cache/cues") / f"general_oracle__{a.model.split('/')[-1]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["id"] for x in open(out, encoding="utf-8")} if out.exists() else set()
    edge = APIEdgeVLM(a.model)
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(load_jsonl(a.samples), 1):
            if s.id in done:
                continue
            orc = oracle_cues(s)
            gen = [clean(edge.complete(PROMPT.format(query=s.query, cue=c), None, max_new_tokens=40)) or c for c in orc]
            f.write(json.dumps({"id": s.id, "cues": gen, "oracle": orc}) + "\n")
            f.flush()
            print(f"[{i}] {s.id}: {orc[0]!r} -> {gen[0]!r}" + (f" (+{len(gen) - 1} more)" if len(gen) > 1 else ""), flush=True)
    print(f"{edge.n_calls} calls -> {out}")


if __name__ == "__main__":
    main()
