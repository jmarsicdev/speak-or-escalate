"""Experiment row 0: score the no-skill reference systems on a split.

    python scripts/eval_floors.py --samples data/processed/ovo_far_val.jsonl --stride 2 --tol 2

Writes results/floors/<split>.json. Needs no model and no video.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.floors import run_floor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--stride", type=float, default=2.0, help="decision cadence in seconds (D5)")
ap.add_argument("--tol", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="results/floors")
a = ap.parse_args()

samples = load_jsonl(a.samples)
name = Path(a.samples).stem
runs = ([("always_silent", 0.0), ("always_respond", 0.0)] + [("random", r) for r in (0.1, 0.25, 0.5)]
        + [("periodic", 4.0), ("periodic", 8.0), ("periodic", 16.0)] + [("oracle", 0.0)])
out = []
print(f"{name}: {len(samples)} questions, stride {a.stride:g}s, tol {a.tol:g}s")
print(f"  {'policy':22s} {'REC':>6s} {'SSR':>6s} {'CRR':>6s} {'avg F1':>7s} {'detections':>11s}")
for pol, rate in runs:
    kw = {"period_s": rate} if pol == "periodic" else {}   # for "periodic" the second field is the period in seconds
    r = run_floor(pol, samples, a.stride, a.tol, a.seed, rate if pol != "periodic" else 0.5, **kw)
    if pol in ("random", "periodic"):  # a single coin-flip run is noisy on ~50 questions: report the mean of 10 seeds
        reps = [run_floor(pol, samples, a.stride, a.tol, a.seed + k, rate if pol != "periodic" else 0.5, **kw) for k in range(10)]
        for t in r["tasks"]:
            r["tasks"][t]["f1"] = sum(x["tasks"][t]["f1"] for x in reps) / len(reps)
        r["avg_f1"] = sum(x["avg_f1"] for x in reps) / len(reps)
        r["avg_f1_std"] = float((sum((x["avg_f1"] - r["avg_f1"]) ** 2 for x in reps) / len(reps)) ** 0.5)
        r["n_seeds"] = len(reps)
    out.append(r)
    f = {t: r["tasks"].get(t, {}).get("f1", 0) * 100 for t in ("REC", "SSR", "CRR")}
    label = pol if pol not in ("random", "periodic") else (f"random {rate:g} (10 seeds)" if pol == "random" else f"every {rate:g} s (10 seeds)")
    print(f"  {label:22s} {f['REC']:6.1f} {f['SSR']:6.1f} {f['CRR']:6.1f} {r['avg_f1']*100:7.2f} {r['total_detections']:11d}" + (f"  ±{r['avg_f1_std']*100:.1f}" if "avg_f1_std" in r else ""))
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
json.dump(out, open(Path(a.out_dir) / f"{name}.json", "w"), indent=1)
