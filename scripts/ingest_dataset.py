"""Turn a raw benchmark annotation file into seeded dev/val/test JSONL splits.

    # real annotations: data/ovo_bench_new.json from https://github.com/JoeLeelyf/OVO-Bench
    python scripts/ingest_dataset.py ovo --ann path/to/ovo_bench_new.json --out-dir data/processed --seed 0

    python scripts/ingest_dataset.py toy --out-dir data/processed

Writes <out-dir>/ovo_far_{all,dev,val,test}.jsonl. Only annotations are written, never video.
Videos are located later through data.video_root + each sample's relative `video` path.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import toy_samples, write_jsonl  # noqa: E402
from egogate.eval.ovo import decision_times, parse_ovo  # noqa: E402
from egogate.eval.splits import split_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["ovo", "toy"])
    ap.add_argument("--ann", type=Path, default=None, help="OVO-Bench ovo_bench_new.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "toy":
        write_jsonl(toy_samples(5), args.out_dir / "toy.jsonl")
        print(f"wrote toy samples -> {args.out_dir / 'toy.jsonl'}")
        return

    samples = parse_ovo(args.ann)
    splits = split_samples(samples, seed=args.seed)
    write_jsonl(samples, args.out_dir / "ovo_far_all.jsonl")
    for name, ss in splits.items():
        write_jsonl(ss, args.out_dir / f"ovo_far_{name}.jsonl")
        tasks = dict(collections.Counter(s.task for s in ss))
        hours = sum((s.end_s or 0) - s.start_s for s in ss) / 3600
        pts = sum(len(decision_times(s, 2.0)) for s in ss)
        print(f"{name:5s} {len(ss):4d} questions  {len({str(s.video) for s in ss}):3d} videos  {tasks}  "
              f"{hours:.2f} h judged  {pts} decision points @2s")


if __name__ == "__main__":
    main()
