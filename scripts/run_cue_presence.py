"""Stage 1, signal C (D24): does each cue appear in the frames? One separate 8B call per cue per decision tick.

The decider treats text in its prompt as a checklist to approve (rows 2 and 5, "what changed"). Here the model never sees the question:
for each cue it is shown the same last 4 frames (2 s, 448 px) as the decider and asked only whether that one description is visible.
p(cue visible) is read from the Yes/No log-probabilities, per cue; the row's lp_yes/lp_no hold the highest cue probability, so the cache
reads like a decider cache (compare_deciders.py, rescore_thresholds.py), and "cues" keeps every cue's pair for later combinations.

    python scripts/run_cue_presence.py --samples data/processed/ovo_far_val.jsonl --cues results/cache/cues/qwen3-vl-8b-instruct__text/cues.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_cache import load_cues  # noqa: E402
from egogate.decider_cache import read_rows  # noqa: E402
from egogate.edge import APIEdgeVLM  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.ovo import decision_times  # noqa: E402
from egogate.frames import stream_windows  # noqa: E402

PROMPT = ("You are shown the 4 most recent frames of a video, covering the last 2 seconds.\n"
          'Is the following visible in these frames: "{cue}"?\nAnswer with a single word: Yes or No.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    a = ap.parse_args()
    cfg = load_config(None, {})
    cues = load_cues(a.cues)
    name = Path(a.cues).parent.name
    out = Path("results/cache/decider") / f"{a.model.split('/')[-1]}__cuepresence__{name}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model_id": a.model, "cues": a.cues, "prompt": PROMPT, "window_s": 2.0, "fps": 2.0, "max_side": 448,
               "row": "lp_yes/lp_no = the cue with the highest p(visible); cues = [lp_yes, lp_no] per cue"}, open(out / "_meta.json", "w"), indent=1)
    print(f"cache: {out}", flush=True)
    edge = APIEdgeVLM(a.model)
    root = Path(cfg["data"]["video_root"])
    t0 = time.time()
    samples = [s for s in load_jsonl(a.samples) if s.id in cues]
    for i, s in enumerate(samples, 1):
        final, part = out / f"{s.id}.jsonl", out / f"{s.id}.jsonl.part"
        if final.exists():
            continue
        done = {round(r["t"], 6) for r in read_rows(part)}
        with open(part, "a") as f:
            for w in stream_windows(root / s.video, decision_times(s, 2.0), 2.0, 2.0, 448):
                if round(w.t_end, 6) in done:
                    continue
                pairs = [list(edge._yes_no_logprobs(PROMPT.format(cue=c), w.frames)) for c in cues[s.id]]
                best = max(pairs, key=lambda p: p[0] - p[1])
                f.write(json.dumps({"t": round(w.t_end, 6), "lp_yes": best[0], "lp_no": best[1], "cues": pairs}) + "\n")
                f.flush()
        part.rename(final)
        print(f"[{i}/{len(samples)}] {s.id}: {len(read_rows(final))} ticks, {edge.n_calls} calls, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {edge.n_calls} calls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
