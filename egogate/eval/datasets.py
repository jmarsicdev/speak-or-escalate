"""Dataset loaders. Each returns a list of Sample(video_path, query, events).

Target benchmarks (proactive-timing tasks, so scoring needs no LLM judge):
  - OVO-Bench Forward Active Responding (CRR / SSR / REC)  https://github.com/JoeLeelyf/OVO-Bench
  - StreamingBench Proactive Output (PO)                    https://github.com/THUNLP-MT/StreamingBench

`scripts/ingest_dataset.py` converts whichever raw format we download into one
normalised JSONL under data/processed/ so the rest of the code never cares which
benchmark it came from:

  {"id": "...", "video": "data/raw/ovo/videos/x.mp4", "query": "...", "task": "SSR",
   "events": [{"t": 12.5, "answer": "..."}, ...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..types import Event


@dataclass
class Sample:
    id: str
    video: Path
    query: str
    events: list[Event]
    task: str = ""
    meta: dict = field(default_factory=dict)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[Sample]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append(Sample(
                id=r["id"], video=Path(r["video"]), query=r["query"],
                events=[Event(t=float(e["t"]), answer=e.get("answer")) for e in r["events"]],
                task=r.get("task", ""), meta=r.get("meta", {}),
            ))
            if limit and len(out) >= limit:
                break
    return out


def write_jsonl(samples: list[Sample], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "id": s.id, "video": str(s.video), "query": s.query, "task": s.task,
                "events": [{"t": e.t, "answer": e.answer} for e in s.events], "meta": s.meta,
            }) + "\n")


def toy_samples(n: int = 3) -> list[Sample]:
    """Synthetic samples with no real video, for dry runs of the whole pipeline."""
    return [
        Sample(id=f"toy{i}", video=Path("<none>"), query="Tell me when the water starts boiling",
               events=[Event(t=10.0 + 7 * i, answer="boiling now"), Event(t=25.0 + 7 * i, answer="boiling again")],
               task="SSR")
        for i in range(n)
    ]
