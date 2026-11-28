"""Build viewer/explorer2.html: every stored run as a toggle, on OVO val and ProactiveVideoQA EGO, with response rules computed live.

    python viewer/dump_frames.py viewer/frames                                   # OVO thumbnails (gitignored)
    python viewer/dump_frames.py viewer/frames_ego data/processed/pvqa_ego_n60.jsonl data/raw/proactivevideoqa/EGO/videos
    python viewer/build_viewer2.py                                                # writes viewer/explorer2.html

The page embeds each run's p(Yes) per tick; thresholds (a share of each model's own moments), the response rules (crossing, re-arm,
every 6 or 10 s while above, first time for one-answer questions) and the scores (events caught, false alarms per minute, a timer with
the same number of responses per question) are computed in the browser, so every toggle updates the numbers. viewer/rescore.html is kept.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

from egogate.cue_sources import oracle_cues  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402

D = Path("results/cache/decider")
OVO_MODELS = [
    ("8B decider", "qwen3-vl-8b-instruct__query_only__d0bdbc25", "decider"),
    ("2B decider", "Qwen3-VL-2B-Instruct__query_only__1a7df950", "decider"),
    ("235B decider", "qwen3-vl-235b-a22b-instruct__query_only__b1bb72f9", "decider"),
    ("8B, 1 frame", "qwen3-vl-8b-instruct__query_only__05293dc6", "decider"),
    ("8B, 2 frames", "qwen3-vl-8b-instruct__query_only__2ee12114", "decider"),
    ("2B + cues in prompt", "Qwen3-VL-2B-Instruct__query_cues__b1de186d", "prompt"),
    ("8B + history frames", "qwen3-vl-8b-instruct__history_frames__9f7a5597", "prompt"),
    ("8B + 'what changed' in prompt", "qwen3-vl-8b-instruct__history_change__d9dc4ad4", "prompt"),
    ("8B + question-aware 'what changed'", "qwen3-vl-8b-instruct__history_change_q__a0c6603e", "prompt"),
    ("Change gate (graded)", "qwen3-vl-8b-instruct__history_gate_q__*", "change"),
    ("Before/after", "qwen3-vl-8b-instruct__signal_beforeafter", "change"),
    ("Anticipation", "qwen3-vl-8b-instruct__signal_anticipate", "change"),
    ("Cue presence", "qwen3-vl-8b-instruct__cuepresence__qwen3-vl-8b-instruct__text", "signal"),
    ("Cue presence (frame-grounded cues)", "qwen3-vl-8b-instruct__cuepresence__qwen3-vl-8b-instruct__frames5", "signal"),
    ("Fixed goal", "qwen3-vl-8b-instruct__signal_goal", "signal"),
    ("Rolling goal", "qwen3-vl-8b-instruct__signal_rollinggoal", "signal"),
    ("Change gate (sentence) + 8B", "gate_changeq__8B", "combo"),
    ("Change gate (sentence) + 2B", "gate_changeq__2B", "combo"),
]
EGO_MODELS = [("8B decider", "qwen3-vl-8b-instruct__query_only__d0bdbc25", "decider"),
              ("Change gate (graded)", "qwen3-vl-8b-instruct__history_gate_q__*", "change"),
              ("Before/after", "qwen3-vl-8b-instruct__signal_beforeafter", "change"),
              ("Gemini 2.5 Flash (cloud verdict)", "CLOUD:gemini-2.5-flash", "cloud")]
TEST_MODELS = [("8B decider", "qwen3-vl-8b-instruct__query_only__d0bdbc25", "decider"),
               ("Change gate (graded)", "qwen3-vl-8b-instruct__history_gate_q__*", "change"),
               ("Before/after", "qwen3-vl-8b-instruct__signal_beforeafter", "change")]


def p_of(r):
    return round(1 / (1 + math.exp(-max(-50, min(50, r["lp_yes"] - r["lp_no"])))), 5)


def rows(pat, sid):
    if pat.startswith("CLOUD:"):   # the cloud model gives a verdict, not a probability: Yes -> 0.9, No -> 0.1 (blocked counts as No)
        f = Path("results/cache/cloud") / pat.split(":", 1)[1] / f"{sid}.jsonl"
        if not f.exists():
            return None
        out = []
        for x in open(f, encoding="utf-8"):
            if x.strip():
                r = json.loads(x)
                out.append({"t": r["t"], "lp_yes": math.log(0.9 if r["yes"] else 0.1), "lp_no": math.log(0.1 if r["yes"] else 0.9), "change": r["text"]})
        return sorted(out, key=lambda r: r["t"])
    hits = sorted(D.glob(pat))
    f = hits[-1] / f"{sid}.jsonl" if hits else None
    if not f or not f.exists():
        return None
    return sorted((json.loads(x) for x in open(f) if x.strip()), key=lambda r: r["t"])


def dataset(name, samples_path, models, frames_dir, goals=None):
    qs = []
    for s in load_jsonl(samples_path):
        q = {"id": s.id, "task": s.task, "source": s.meta.get("source", ""), "query": s.query, "events": [e.t for e in s.events],
             "intervals": s.meta.get("intervals"), "p": {}, "txt": {}}
        if s.task in ("REC", "SSR", "CRR"):
            q["oracle"] = oracle_cues(s)
        for label, pat, _ in models:
            r = rows(pat, s.id)
            if not r:
                continue
            q["p"][label] = [[x["t"], p_of(x)] for x in r]
            if "change" in r[0]:
                q["txt"][label] = {str(x["t"]): x["change"] for x in r}
            if "goal" in r[0]:
                q["txt"][label] = {str(x["t"]): "goal: " + x["goal"] for x in r}
        if goals and s.id in goals:
            q["goal"] = goals[s.id]
        m = Path("results/cache/matcher/siglip2-so400m-patch14-384__alt.cues-8b-text") / f"{s.id}.json"
        if m.exists():
            mm = json.load(open(m))
            q["cues"] = mm["cues"]
            q["sim"] = [[t, round(max(v), 4)] for t, v in zip(mm["t"], mm["sims"])]
        qs.append(q)
    return {"name": name, "frames": frames_dir, "models": [{"label": l, "group": g} for l, _, g in models], "questions": qs}


def main():
    goals = {json.loads(x)["id"]: json.loads(x)["cues"][0] for x in open("results/cache/cues/qwen3-vl-8b-instruct__goal/cues.jsonl", encoding="utf-8")}
    data = [dataset("OVO-Bench val", "data/processed/ovo_far_val.jsonl", OVO_MODELS, "frames", goals),
            dataset("OVO-Bench test (used once, D26)", "data/processed/ovo_far_test.jsonl", TEST_MODELS, "frames"),
            dataset("ProactiveVideoQA EGO (first 60)", "data/processed/pvqa_ego_n60.jsonl", EGO_MODELS, "frames_ego"),
            dataset("ProactiveVideoQA EGO (fresh 100, D26; partial while running)", "data/processed/pvqa_ego_n100_s1.jsonl", TEST_MODELS, "frames_ego")]
    for d in data:   # keep only questions that have at least one run (the fresh sample is still running)
        d["questions"] = [q for q in d["questions"] if q["p"]]
    html = open("viewer/explorer2_template.html", encoding="utf-8").read().replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
    open("viewer/explorer2.html", "w", encoding="utf-8").write(html)
    print("viewer/explorer2.html", round(len(html) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
