"""Grid over (lo, hi) routing thresholds -> results/sweep_<name>.csv for the F1/recall heatmap.

    python scripts/sweep_thresholds.py configs/experiments/band_confidence.yaml --set data.toy=true
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--grid", type=int, default=5)
    args = ap.parse_args()

    base = Path(args.config).stem
    rows = []
    for lo in np.linspace(0.1, 0.5, args.grid):
        for hi in np.linspace(0.5, 0.9, args.grid):
            if hi <= lo:
                continue
            name = f"{base}_lo{lo:.2f}_hi{hi:.2f}"
            cmd = [sys.executable, str(ROOT / "scripts/run_experiment.py"), args.config,
                   "--set", f"name={name}", "--set", f"policy.lo={lo:.3f}", "--set", f"policy.hi={hi:.3f}"]
            for s in args.set:
                cmd += ["--set", s]
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            agg = json.load(open(ROOT / "results" / name / "summary.json"))["aggregate"]
            rows.append({"lo": lo, "hi": hi, **{k: agg[k] for k in ("recall", "precision", "f1", "n_cloud_calls")}})
            print(f"lo={lo:.2f} hi={hi:.2f} f1={agg['f1']:.3f} cloud={agg['n_cloud_calls']:.1f}")

    out = ROOT / "results" / f"sweep_{base}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
