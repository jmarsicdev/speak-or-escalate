"""D25 C4: a rolling goal with progress steps, rewritten from recent frames, checked by separate calls (8B through the API).

Writer: at the first tick, then whenever 10 s have passed since the last write or the goal was judged reached (p >= 0.5) at the previous
tick, the 8B sees the question, the last 5 s at 1 fps and the previous goal (reached or not), and writes the next visible state that would
answer the question and is not yet visible, plus 2 or 3 intermediate states before it. Checks every tick on the last 4 frames, without
the question: the goal, and the next unmet step (done once p >= 0.5). Rows: decider format (lp_yes/lp_no = the goal check) plus the goal,
its id, the checked step and its log-probabilities, and progress. Settings fixed in RATIONALE D25 C4.

    python scripts/run_rolling_goal.py --samples data/processed/ovo_far_val.jsonl
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
from egogate.cues import parse_cue_json  # noqa: E402
from egogate.decider_cache import read_rows  # noqa: E402
from egogate.edge import APIEdgeVLM  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.ovo import decision_times  # noqa: E402
from egogate.frames import stream_windows  # noqa: E402

REFRESH_S, REACHED = 10.0, 0.5
WRITE = ('You are helping a system that watches a video stream. The user asked: "{query}"\n'
         "You are shown the last 5 seconds of the stream (5 frames, one per second).{prev}\n"
         "Write the NEXT visible state that would answer the user's question and that is NOT yet visible in these frames, and 2 or 3 "
         "shorter intermediate states that would come before it, in order. Describe only what a camera would see (people, hands, objects), "
         "under 15 words each.\n"
         'Return JSON only: {{"next_state": "...", "steps": ["...", "..."]}}')
CHECK = ("You are shown the 4 most recent frames of a video, covering the last 2 seconds.\n"
         'Is the following visible in these frames: "{state}"?\nAnswer with a single word: Yes or No.')


def p_of(lp):
    return 1 / (1 + math.exp(-max(-50, min(50, lp[0] - lp[1]))))


def parse(raw, query):
    try:
        s = raw[raw.index("{"): raw.rindex("}") + 1]
        d = json.loads(s)
        goal = str(d.get("next_state") or "").strip()
        steps = [str(x).strip() for x in d.get("steps", []) if str(x).strip()][:3]
        if goal:
            return goal, steps
    except (ValueError, json.JSONDecodeError):
        pass
    lines = parse_cue_json(raw)
    return (lines[-1] if lines else query), lines[:-1][:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    ap.add_argument("--set", action="append", default=[], help="data.video_root=... for other benchmarks")
    a = ap.parse_args()
    cfg = load_config(None, dict(p.split("=", 1) for p in a.set))
    tag = a.model.split("/")[-1]
    out = Path("results/cache/decider") / f"{tag}__signal_rollinggoal"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model_id": a.model, "write": WRITE, "check": CHECK, "refresh_s": REFRESH_S, "reached": REACHED}, open(out / "_meta.json", "w"), indent=1)
    edge = APIEdgeVLM(a.model)
    root = Path(cfg["data"]["video_root"])
    t0 = time.time()
    samples = load_jsonl(a.samples)
    for i, s in enumerate(samples, 1):
        final, part = out / f"{s.id}.jsonl", out / f"{s.id}.jsonl.part"
        if final.exists():
            continue
        if part.exists():
            part.unlink()   # the state (goal, steps) is sequential: an interrupted question restarts from its first tick
        goal, steps, done, gid, t_write, prev_p, last_reached = None, [], 0, -1, -1e9, None, False
        with open(part, "w") as f:
            for w in stream_windows(root / s.video, decision_times(s, 2.0), 5.0, 2.0, 448):
                t, fr = w.t_end, list(w.frames)   # 10 frames over the last 5 s at 2 fps
                if goal is None or t - t_write >= REFRESH_S or last_reached:
                    prev = "" if goal is None else f'\nThe previously expected state was: "{goal}" ({"now reached" if last_reached else "not reached yet"}).'
                    raw = edge.complete(WRITE.format(query=s.query, prev=prev), fr[1::2], max_new_tokens=120)
                    goal, steps = parse(raw, s.query)
                    done, gid, t_write, prev_p = 0, gid + 1, t, None
                recent = fr[-4:]
                lp_g = edge._yes_no_logprobs(CHECK.format(state=goal), recent)
                pg = p_of(lp_g)
                step_lp = None
                if done < len(steps):
                    step_lp = edge._yes_no_logprobs(CHECK.format(state=steps[done]), recent)
                    if p_of(step_lp) >= REACHED:
                        done += 1
                rise = 0.0 if prev_p is None else pg - prev_p
                f.write(json.dumps({"t": round(t, 6), "lp_yes": lp_g[0], "lp_no": lp_g[1], "goal": goal, "goal_id": gid, "rise": rise,
                                    "step": steps[done - 1] if step_lp and done and p_of(step_lp) >= REACHED else (steps[done] if done < len(steps) else None),
                                    "step_lp": list(step_lp) if step_lp else None, "progress": done / len(steps) if steps else 0.0}) + "\n")
                f.flush()
                prev_p, last_reached = pg, pg >= REACHED
        part.rename(final)
        print(f"[{i}/{len(samples)}] {s.id}: {len(read_rows(final))} ticks, {gid + 1} goals, {edge.n_calls} calls, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {edge.n_calls} calls")


if __name__ == "__main__":
    main()
