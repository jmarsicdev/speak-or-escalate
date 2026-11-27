"""D27: the cloud model's verdict and answer at every 2 s tick (any OpenRouter vision model; Gemini 2.5 Flash by default).

The cloud model sees what the edge decider sees (the question and the last 2 s, 4 frames at 448 px) and replies "Yes: <answer>" or "No".
Rows: {"t", "yes" (bool), "text", "in_tokens", "out_tokens"}. Stored for every tick, so routing at any budget is simulated later without
further calls. Reasoning is switched off. Resumable per tick.

    python scripts/run_cloud.py --samples data/processed/pvqa_ego_n60.jsonl --set data.video_root=data/raw/proactivevideoqa/EGO/videos
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.decider_cache import read_rows  # noqa: E402
from egogate.edge import APIEdgeVLM  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.ovo import decision_times  # noqa: E402
from egogate.frames import stream_windows  # noqa: E402

PROMPT = ("You are an assistant watching a first-person video stream. You are shown the 4 most recent frames (the last 2 seconds).\n"
          'The user asked earlier: "{query}"\n'
          "Is RIGHT NOW a moment when something that answers the question is happening or has just become visible? "
          'Reply on one line: "Yes: <one-sentence answer>" or "No".')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(None, dict(p.split("=", 1) for p in a.set))
    edge = APIEdgeVLM(a.model)
    edge.extra = {"extra_body": {"reasoning": {"enabled": False}}}   # no log-probabilities needed here, so no provider restriction; thinking off
    out = Path("results/cache/cloud") / a.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "prompt": PROMPT, "frames": "last 2 s, 4 frames, 448 px", "reasoning": "disabled"}, open(out / "_meta.json", "w"), indent=1)
    root = Path(cfg["data"]["video_root"])
    t0, tin, tout = time.time(), 0, 0
    samples = load_jsonl(a.samples)
    for i, s in enumerate(samples, 1):
        final, part = out / f"{s.id}.jsonl", out / f"{s.id}.jsonl.part"
        if final.exists():
            continue
        done = {round(r["t"], 6) for r in read_rows(part)}
        with open(part, "a", encoding="utf-8") as f:
            for w in stream_windows(root / s.video, decision_times(s, 2.0), 2.0, 2.0, 448):
                if round(w.t_end, 6) in done:
                    continue
                r = edge._call_ok(PROMPT.format(query=s.query), list(w.frames), max_tokens=80)
                text = (r.choices[0].message.content or "").strip()
                u = r.usage
                tin += u.prompt_tokens
                tout += u.completion_tokens
                f.write(json.dumps({"t": round(w.t_end, 6), "yes": text.lower().startswith("yes"), "text": text[:300],
                                    "in_tokens": u.prompt_tokens, "out_tokens": u.completion_tokens}) + "\n")
                f.flush()
        part.rename(final)
        print(f"[{i}/{len(samples)}] {s.id}: {len(read_rows(final))} ticks, {edge.n_calls} calls, tokens in/out this run {tin}/{tout}, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {edge.n_calls} calls")


if __name__ == "__main__":
    main()
