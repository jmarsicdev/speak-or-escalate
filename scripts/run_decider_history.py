"""Experiment: the decider given history, and asked what changed (val; any API model, 8B by default).

Row 1 shows the decider only the last 2 s (4 frames). Events such as "a step has finished" or "the action happened again" are defined against
what came before, so here the decider also sees two earlier frames, from about 10 s and 5 s before the decision (causal: nothing after t).

  --variant frames   one call: the 2 earlier frames + the last 4, the row 1 question, plus "compare now with the earlier frames"
  --variant change   two calls: (1) the same 6 frames, "in one sentence, what changed between the earlier frames and now?" (generated,
                     neutral: the user's question is not shown, so the description is not steered toward "yes"); (2) the row 1 question
                     with that sentence added as "what changed recently", same 6 frames, Yes/No log-probabilities

Near the start of a video, where 10 s of history do not exist, the earliest available frame is used (the first 1.5 s have no decision).
Rows have the same format as the row 1 cache, so scripts/compare_deciders.py compares them directly; "change" rows also store the sentence.

    python scripts/run_decider_history.py --samples data/processed/ovo_far_val.jsonl --variant change
"""
from __future__ import annotations

import argparse
import hashlib
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

HIST_S, FPS, MAX_SIDE = 10.0, 2.0, 448

HEAD = ("You are an assistant watching a first-person video stream.\n"
        "You are shown 6 frames in time order: the first two are from about {h1:g} and {h2:g} seconds ago, the last four cover the last 2 seconds.\n")
CHANGE_Q = HEAD + ("In one sentence, what changed between the earlier frames and the last two seconds (actions, objects, people)? "
                   "If nothing visible changed, answer: Nothing changed.")
DECIDE = (HEAD + 'The user asked earlier: "{query}"\n{extra}'
          "Respond only at the moment the answer becomes available: when what the user asked about has just happened or has just become visible. "
          "Compare now with the earlier frames. Is RIGHT NOW such a moment?\nAnswer with a single word: Yes or No.")


def windows(video, ticks):
    """Yield (t, frames sampled at 2 fps over the last up-to-10 s). Ticks with a full 10 s come from one pass; earlier ticks get what exists."""
    full = [t for t in ticks if t - (HIST_S - 1 / FPS) >= 0]
    early = [t for t in ticks if t not in set(full)]
    for t in early:   # a handful per question: the shortest window that starts at or after 0
        span = (int(t * FPS) + 1) / FPS
        for w in stream_windows(video, [t], span, FPS, MAX_SIDE):
            yield t, w.frames
    for w in stream_windows(video, full, HIST_S, FPS, MAX_SIDE):
        yield w.t_end, w.frames


def pick(frames):
    """The 2 earlier frames (about 10 s and 5 s back, or the earliest available) and the last 4; returns (images, h1, h2) or None."""
    n = len(frames)
    if n < 4:
        return None
    recent = frames[-4:]
    i1 = 0
    i2 = max(n - 1 - int(5 * FPS), 0)
    h1, h2 = (n - 1 - i1) / FPS, (n - 1 - i2) / FPS
    return [frames[i1], frames[i2]] + recent, h1, h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--variant", choices=["frames", "change"], required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    a = ap.parse_args()
    cfg = load_config(None, {})
    meta = {"model_id": a.model, "variant": f"history_{a.variant}", "hist_s": HIST_S, "fps": FPS, "max_side": MAX_SIDE, "stride_s": 2.0,
            "prompt_hash": hashlib.sha1((CHANGE_Q + DECIDE).encode()).hexdigest()[:10]}
    key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:8]
    out = Path("results/cache/decider") / f"{a.model.split('/')[-1]}__history_{a.variant}__{key}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(out / "_meta.json", "w"), indent=1)
    print(f"cache: {out}", flush=True)
    edge = APIEdgeVLM(a.model)
    root = Path(cfg["data"]["video_root"])
    t0 = time.time()
    samples = load_jsonl(a.samples)
    for i, s in enumerate(samples, 1):
        final, part = out / f"{s.id}.jsonl", out / f"{s.id}.jsonl.part"
        if final.exists():
            continue
        done = {round(r["t"], 6) for r in read_rows(part)}
        ticks = [t for t in decision_times(s, 2.0) if t >= 1.5 - 1e-9]
        with open(part, "a") as f:
            for t, frames in windows(root / s.video, ticks):
                if round(t, 6) in done:
                    continue
                p = pick(frames)
                if p is None:
                    continue
                imgs, h1, h2 = p
                ts = time.perf_counter()
                change, extra = None, ""
                if a.variant == "change":
                    change = (edge.complete(CHANGE_Q.format(h1=h1, h2=h2), imgs, max_new_tokens=48).strip().splitlines() or [""])[0][:300]
                    extra = f"What changed recently, described from these frames: {change}\n"
                lp_yes, lp_no = edge._yes_no_logprobs(DECIDE.format(h1=h1, h2=h2, query=s.query, extra=extra), imgs)
                row = {"t": round(t, 6), "lp_yes": lp_yes, "lp_no": lp_no, "ms": round((time.perf_counter() - ts) * 1000, 1), "n_frames": len(imgs)}
                if change is not None:
                    row["change"] = change
                f.write(json.dumps(row) + "\n")
                f.flush()
        part.rename(final)
        print(f"[{i}/{len(samples)}] {s.id} {s.task}: {len(read_rows(final))} ticks, {edge.n_calls} calls, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {edge.n_calls} calls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
