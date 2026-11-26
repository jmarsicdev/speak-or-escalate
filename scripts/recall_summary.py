"""Recall at equal false-alarm rates, for every stored decider and signal (CPU, no model calls).

A model that responds more always catches more events, so recall is compared at a fixed false-alarm budget: for each model, the threshold
is swept (crossing rule) and the share of events caught (any response within 2 s; repeats not punished) is read at 1 and at 2 false
alarms per minute (a false alarm = a response with no event within 2 s), interpolated between thresholds. A timer is swept the same way.
OVO val counting + tutorial questions (36; the 235B has 29 of them), all ticks, and ProactiveVideoQA EGO (an event = a reply interval,
caught if a response falls inside it; false alarm = a response outside every interval).

    python scripts/recall_summary.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from rescore_thresholds import detect  # noqa: E402

D = Path("results/cache/decider")
OVO = {
    "2B decider": "Qwen3-VL-2B-Instruct__query_only__1a7df950",
    "2B + cues in prompt": "Qwen3-VL-2B-Instruct__query_cues__b1de186d",
    "8B decider": "qwen3-vl-8b-instruct__query_only__d0bdbc25",
    "8B, 1 frame": "qwen3-vl-8b-instruct__query_only__05293dc6",
    "8B + history frames": "qwen3-vl-8b-instruct__history_frames__9f7a5597",
    "8B + 'what changed' in prompt": "qwen3-vl-8b-instruct__history_change__d9dc4ad4",
    "8B + question-aware 'what changed'": "qwen3-vl-8b-instruct__history_change_q__a0c6603e",
    "G, graded change gate": "qwen3-vl-8b-instruct__history_gate_q__*",
    "C, cue presence": "qwen3-vl-8b-instruct__cuepresence__qwen3-vl-8b-instruct__text",
    "C1, fixed goal": "qwen3-vl-8b-instruct__signal_goal",
    "C2, before/after": "qwen3-vl-8b-instruct__signal_beforeafter",
    "C3, anticipation": "qwen3-vl-8b-instruct__signal_anticipate",
    "C4, rolling goal (p)": "qwen3-vl-8b-instruct__signal_rollinggoal",
    "235B decider (29 questions)": "qwen3-vl-235b-a22b-instruct__query_only__b1bb72f9",
}
TOL, FA = 2.0, (1.0, 2.0)


def series(pat, s):
    hits = sorted(D.glob(pat))
    f = hits[-1] / f"{s.id}.jsonl" if hits else None
    if not f or not f.exists():
        return None
    rows = sorted((json.loads(x) for x in open(f) if x.strip()), key=lambda r: r["t"])
    t = np.array([r["t"] for r in rows])
    return t, 1 / (1 + np.exp(-np.clip(np.array([r["lp_yes"] - r["lp_no"] for r in rows]), -50, 50)))


def counts(samples, dets, mins, intervals=False):
    ev = caught = fa = 0
    for s in samples:
        d = dets[s.id]
        if intervals:
            iv = s.meta["intervals"]
            ev += len(iv)
            caught += sum(any(a <= x <= b for x in d) for a, b in iv)
            fa += sum(not any(a <= x <= b for a, b in iv) for x in d)
        else:
            g = [e.t for e in s.events]
            ev += len(g)
            caught += sum(any(abs(x - e) <= TOL for x in d) for e in g)
            fa += sum(not any(abs(x - e) <= TOL for e in g) for x in d)
    return caught / ev, fa / mins


def curve(samples, data, intervals=False):
    mins = sum((data[s.id][0][-1] - data[s.id][0][0]) / 60 for s in samples)
    allp = np.concatenate([data[s.id][1] for s in samples])
    pts = []
    for r in np.linspace(0.01, 0.7, 70):
        tau = float(np.quantile(allp, 1 - r))
        dets = {s.id: detect(data[s.id][0], data[s.id][1], tau, s.task, "edge") for s in samples}
        pts.append(counts(samples, dets, mins, intervals)[::-1])   # (fa per min, recall)
    return pts


def timer_curve(samples, data, intervals=False):
    mins = sum((data[s.id][0][-1] - data[s.id][0][0]) / 60 for s in samples)
    pts = []
    for k in range(1, 40):
        dets = {s.id: [float(x) for x in data[s.id][0][(j % k)::k]] for j, s in enumerate(samples)}
        pts.append(counts(samples, dets, mins, intervals)[::-1])
    return pts


def at(pts, fa):
    """Highest recall reachable with at most `fa` false alarms per minute (the best threshold under that budget)."""
    ok = [r for f, r in pts if f <= fa]
    return 100 * max(ok) if ok else float("nan")


def table(title, samples, items, intervals=False):
    print(f"\n{title}")
    print(f"  {'':38s} recall at <=1 FA/min   <=2 FA/min")
    base = None
    out = {}
    for name, pat in items.items():
        data = {s.id: x for s in samples if (x := series(pat, s)) is not None}
        ss = [s for s in samples if s.id in data]
        if len(ss) < 10:
            continue
        if base is None:
            tp = timer_curve(ss, data, intervals)
            base = (at(tp, FA[0]), at(tp, FA[1]))
            print(f"  {'timer (no model)':38s} {base[0]:8.1f}%        {base[1]:8.1f}%")
            out["timer"] = base
        pts = curve(ss, data, intervals)
        r1, r2 = at(pts, FA[0]), at(pts, FA[1])
        out[name] = (r1, r2, len(ss))
        print(f"  {name:38s} {r1:8.1f}%        {r2:8.1f}%")
    return out


def main():
    ovo = [s for s in load_jsonl("data/processed/ovo_far_val.jsonl") if s.task != "CRR"]
    res = {"ovo": table("OVO val, counting + tutorials (36 questions): share of events caught", ovo, OVO)}
    tut = [s for s in ovo if s.task == "SSR"]
    res["ovo_tutorials"] = table("OVO val, tutorials only (12 questions)", tut, {k: v for k, v in OVO.items() if "235B" not in k})
    ego = load_jsonl("data/processed/pvqa_ego_n60.jsonl")
    res["ego"] = table("ProactiveVideoQA EGO (60 questions): share of reply intervals caught", ego, {"8B decider": OVO["8B decider"]}, intervals=True)
    json.dump(res, open("results/cache/rescore/recall_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
