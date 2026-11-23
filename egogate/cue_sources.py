"""Other sources of cues, to find out how much cue quality limits the matcher (docs/RATIONALE.md D22).

  question   the question text itself as the one and only cue (no writer at all)
  oracle     the benchmark's own description of the right moment, which no deployed system would have:
               tutorial steps -> the list of step descriptions;  counting -> the activity phrase;  clue reveal -> the answer text
  file       any cues.jsonl (for example cues written by another model, or hand-written)

`oracle` is an upper bound on what better cue wording could give, not a method.
"""
from __future__ import annotations

from .eval.datasets import Sample


def question_cues(s: Sample) -> list[str]:
    return [s.query]


def oracle_cues(s: Sample) -> list[str]:
    m = s.meta
    if s.task == "SSR" and m.get("all_steps"):
        return [str(x) for x in m["all_steps"]]
    if s.task == "REC" and m.get("activity"):
        return [str(m["activity"])]
    if s.task == "CRR" and m.get("answer"):
        return [str(m["answer"])]
    return [s.query]
