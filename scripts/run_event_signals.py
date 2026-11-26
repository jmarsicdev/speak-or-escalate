"""D25 C: three signals aimed at finding more events (8B through the API, every 2 s tick, settings fixed in RATIONALE D25 C).

  --kind goal         once per question the 8B writes the visible goal state; each tick, without the question: "Is the following visible
                      in these frames: <goal>? Yes or No" on the decider's last 4 frames (2 s, 448 px)
  --kind beforeafter  4 frames from 6.5 to 5 s before the tick, then the last 4 frames, with the question: "Did something happen between
                      the first clip and the second clip that answers the user's question? Yes or No"
  --kind anticipate   the last 4 frames with the question: "Is the moment that answers the user's question happening now or about to
                      happen in the next few seconds? Yes or No"
Rows have the decider format (t, lp_yes, lp_no), so compare_deciders.py and the D25 analysis read them directly.

    python scripts/run_event_signals.py --samples data/processed/ovo_far_val.jsonl --kind goal
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

GOAL_WRITE = ('A user watching a video asked: "{query}"\n'
              "In one short sentence (under 20 words), describe what the camera would SEE at the moment the answer becomes available: "
              "the visible state of the people and objects, not an intention. If the question asks to count a repeated action, describe one "
              "repetition being completed. Reply with the sentence only.")
GOAL_CHECK = ("You are shown the 4 most recent frames of a video, covering the last 2 seconds.\n"
              'Is the following visible in these frames: "{goal}"?\nAnswer with a single word: Yes or No.')
BEFORE_AFTER = ("You are shown two short clips from a video stream, in time order: the first 4 frames are from about 5 seconds ago, "
                "the last 4 frames cover the last 2 seconds.\n"
                'The user asked earlier: "{query}"\n'
                "Did something happen between the first clip and the second clip that answers the user's question?\n"
                "Answer with a single word: Yes or No.")
ANTICIPATE = ("You are shown the 4 most recent frames of a video stream, covering the last 2 seconds.\n"
              'The user asked earlier: "{query}"\n'
              "Is the moment that answers the user's question happening now or about to happen in the next few seconds?\n"
              "Answer with a single word: Yes or No.")


def goals(samples, edge, path: Path):
    have = {json.loads(x)["id"]: json.loads(x)["cues"][0] for x in open(path, encoding="utf-8")} if path.exists() else {}
    with open(path, "a", encoding="utf-8") as f:
        for s in samples:
            if s.id in have:
                continue
            g = (edge.complete(GOAL_WRITE.format(query=s.query), None, max_new_tokens=60).strip().splitlines() or [s.query])[0].strip().strip('"')
            have[s.id] = g
            f.write(json.dumps({"id": s.id, "query": s.query, "cues": [g]}) + "\n")
            f.flush()
            print(f"goal {s.id}: {g}", flush=True)
    return have


def windows(video, ticks, kind):
    if kind != "beforeafter":
        for w in stream_windows(video, ticks, 2.0, 2.0, 448):
            yield w.t_end, list(w.frames)
        return
    full = [t for t in ticks if t - 6.5 >= 0]
    for t in [t for t in ticks if t - 6.5 < 0]:
        span = (int(t * 2) + 1) / 2
        for w in stream_windows(video, [t], span, 2.0, 448):
            fr = list(w.frames)
            yield t, fr[:4] + fr[-4:] if len(fr) >= 8 else fr[:1] * 4 + fr[-4:]
    for w in stream_windows(video, full, 7.0, 2.0, 448):   # 14 frames from t - 6.5 to t: the first 4 and the last 4
        fr = list(w.frames)
        yield w.t_end, fr[:4] + fr[-4:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--kind", choices=["goal", "beforeafter", "anticipate"], required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    ap.add_argument("--set", action="append", default=[], help="data.video_root=... for other benchmarks")
    a = ap.parse_args()
    over = dict(p.split("=", 1) for p in a.set)
    cfg = load_config(None, over)
    tag = a.model.split("/")[-1]
    edge = APIEdgeVLM(a.model)
    samples = load_jsonl(a.samples)
    gdir = Path(f"results/cache/cues/{tag}__goal")
    if a.kind == "goal":
        gdir.mkdir(parents=True, exist_ok=True)
    goal = goals(samples, edge, gdir / "cues.jsonl") if a.kind == "goal" else None
    out = Path("results/cache/decider") / f"{tag}__signal_{a.kind}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model_id": a.model, "kind": a.kind, "prompt": {"goal": GOAL_CHECK, "beforeafter": BEFORE_AFTER, "anticipate": ANTICIPATE}[a.kind],
               "goal_write": GOAL_WRITE if a.kind == "goal" else None}, open(out / "_meta.json", "w"), indent=1)
    root = Path(cfg["data"]["video_root"])
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        final, part = out / f"{s.id}.jsonl", out / f"{s.id}.jsonl.part"
        if final.exists():
            continue
        done = {round(r["t"], 6) for r in read_rows(part)}
        with open(part, "a") as f:
            for t, fr in windows(root / s.video, decision_times(s, 2.0), a.kind):
                if round(t, 6) in done:
                    continue
                prompt = (GOAL_CHECK.format(goal=goal[s.id]) if a.kind == "goal" else
                          BEFORE_AFTER.format(query=s.query) if a.kind == "beforeafter" else ANTICIPATE.format(query=s.query))
                lp_yes, lp_no = edge._yes_no_logprobs(prompt, fr)
                f.write(json.dumps({"t": round(t, 6), "lp_yes": lp_yes, "lp_no": lp_no, "n_frames": len(fr)}) + "\n")
                f.flush()
        part.rename(final)
        print(f"[{i}/{len(samples)}] {s.id}: {len(read_rows(final))} ticks, {edge.n_calls} calls, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {edge.n_calls} calls")


if __name__ == "__main__":
    main()
