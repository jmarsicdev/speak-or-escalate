"""OVO-Bench "Forward Active Responding" (REC, SSR, CRR) -> normalised Samples.

Source: https://github.com/JoeLeelyf/OVO-Bench, file data/ovo_bench_new.json (CC BY-NC-SA 4.0).
Field handling copies Em-Garde's `eval/ovo_bench/ovo_bench_eval_online.py` so the ground truth
we score against is the one their numbers use (see docs/RATIONALE.md, decision D3).

For each question we derive three things:

  query      what the user "asks" (REC and SSR have no natural question, so Em-Garde's wording is used)
  events     ground-truth response times
               CRR: [clue_time]
               REC: [t + 2 for t in start_times]
               SSR: [t + 2 for t in start_time]
             (the +2 is Em-Garde's: the response is due once the 2 s window has seen the event start)
  start_s    when the stream starts being judged:  REC, SSR -> 0;  CRR -> ask_time
  end_s      stop at the last test point:  test_info[-1]["realtime"]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..types import Event
from .datasets import Sample

FAR_TASKS = ("REC", "SSR", "CRR")


def _query(r: dict) -> str:
    if r["task"] == "REC":
        return "How many times does the event: {} happen?".format(r["activity"])
    if r["task"] == "SSR":
        # CamelCase tutorial name -> "put on hair extensions"
        return "Describe the steps of {}".format(re.sub(r"(?<!^)(?=[A-Z])", " ", r["tutorial"]).lower())
    return r["question"]


def _gt_times(r: dict) -> list[float]:
    if r["task"] == "CRR":
        return [float(r["clue_time"])]
    key = "start_times" if r["task"] == "REC" else "start_time"
    return [float(t) + 2.0 for t in r[key]]


def parse_ovo(path: str | Path, tasks: tuple[str, ...] = FAR_TASKS) -> list[Sample]:
    rows = json.load(open(path))
    out = []
    for r in rows:
        if r["task"] not in tasks:
            continue
        keep = {k: v for k, v in r.items() if k in ("activity", "tutorial", "all_steps", "answer", "ask_time", "clue_time")}
        keep["source"] = r["video"].split("/")[0]
        out.append(Sample(
            id=f"ovo_{r['task']}_{r['id']}", video=Path(r["video"]), query=_query(r),
            events=[Event(t=t) for t in _gt_times(r)], task=r["task"], meta=keep,
            start_s=0.0 if r["task"] in ("REC", "SSR") else float(r["ask_time"]),
            end_s=float(r["test_info"][-1]["realtime"]),
        ))
    return out


def decision_times(sample: Sample, stride_s: float) -> list[float]:
    """The moments the system is asked "respond now?": start_s + k * stride_s, up to end_s.

    The first decision is one stride after the start so a full window of video exists. Baseline
    stride is 2 s (decision D5). Floors and the real decider both use this grid so their
    numbers are directly comparable.
    """
    end = sample.end_s if sample.end_s is not None else sample.start_s
    n = int((end - sample.start_s) / stride_s + 1e-9)
    return [round(sample.start_s + k * stride_s, 6) for k in range(1, n + 1)]
