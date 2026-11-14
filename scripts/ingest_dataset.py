"""Convert a raw benchmark download into data/processed/<split>.jsonl.

    python scripts/ingest_dataset.py ovo  --raw data/raw/ovo --out data/processed/dev.jsonl
    python scripts/ingest_dataset.py toy  --out data/processed/toy.jsonl

TODO(data): fill in the OVO-Bench and StreamingBench parsers once the raw files are
downloaded. Both ship a JSON annotation file plus a videos/ folder; the fields we need
are video path, the query text, and the ground-truth response timestamps (OVO calls
them `realtime`/`gt_timestamp` depending on task; StreamingBench PO has `ground_truth_time`).
Verify the exact keys against the downloaded files before relying on this.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import Sample, toy_samples, write_jsonl  # noqa: E402
from egogate.types import Event  # noqa: E402


def ingest_ovo(raw: Path) -> list[Sample]:
    """OVO-Bench Forward Active Responding (CRR / SSR / REC)."""
    ann_files = list(raw.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"no annotation json under {raw}")
    samples = []
    for ann in ann_files:
        for r in json.load(open(ann)):
            task = r.get("task", ann.stem)
            if task not in ("CRR", "SSR", "REC"):
                continue
            # TODO(data): confirm field names
            events = [Event(t=float(t), answer=None) for t in r.get("gt_timestamps", [])]
            samples.append(Sample(id=f"ovo_{task}_{r['id']}", video=raw / "videos" / r["video"],
                                  query=r["question"], events=events, task=task))
    return samples


def ingest_streamingbench(raw: Path) -> list[Sample]:
    """StreamingBench Proactive Output task."""
    raise NotImplementedError("TODO(data): implement after download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["ovo", "streamingbench", "toy"])
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.source == "toy":
        samples = toy_samples(5)
    elif args.source == "ovo":
        samples = ingest_ovo(args.raw)
    else:
        samples = ingest_streamingbench(args.raw)
    write_jsonl(samples, args.out)
    print(f"wrote {len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
