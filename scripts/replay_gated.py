"""Row 4, stage 2: score "matcher fires AND decider says Yes" over thresholds, Yes-thresholds and cooldowns. CPU only.

    python scripts/replay_gated.py --matcher results/cache/matcher/<dir> --decider results/cache/decider/<dir> \
        --samples data/processed/ovo_far_val.jsonl

Also prints the matching rows 1 and 3 numbers, so the three systems sit side by side. Choose the setting on val only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.gated import _known_rows, score_gated  # noqa: E402

THETAS = [0.02, 0.03, 0.04, 0.05]
TAUS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
COOLDOWNS = [0.0, 4.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True)
    ap.add_argument("--decider", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--rule", default="code")
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    mdir, ddir = Path(a.matcher), Path(a.decider)
    matcher, rows = {}, {}
    for s in samples:
        f = mdir / f"{s.id}.json"
        if f.exists():
            r = json.load(open(f))
            matcher[s.id] = {"t": np.array(r["t"]), "sims": np.array(r["sims"]).reshape(len(r["t"]), -1)}
            rows[s.id] = _known_rows(ddir, s.id)
    print(f"{Path(a.samples).stem}: {len(matcher)}/{len(samples)} questions with matcher similarities; "
          f"{sum(len(v) for v in rows.values())} stored decider answers")
    print("tau = 0.00 means the matcher alone (every trigger counts); the decider only removes responses.\n")
    results = []
    print(f"{'theta':>6s} {'cooldown':>8s} " + " ".join(f"tau={t:<4.2f}" + " " * 4 for t in TAUS) + "  avg F1 (responses)")
    for th in THETAS:
        for cd in COOLDOWNS:
            cells = []
            for tau in TAUS:
                r = score_gated(samples, matcher, rows, a.rule, th, tau, "any", cd, 1.0, a.tol)
                results.append({"theta": th, "tau": tau, "cooldown_s": cd, "avg_f1": r["avg_f1"], "tasks": r["tasks"], "total_detections": r["total_detections"]})
                cells.append(f"{r['avg_f1']*100:5.1f}({r['total_detections']:4d})")
            print(f"{th:6.2f} {cd:8.1f} " + " ".join(f"{c:>13s}" for c in cells))
    best = max(results, key=lambda r: r["avg_f1"])
    print(f"\nbest on {Path(a.samples).stem}: theta={best['theta']:g} tau={best['tau']:g} cooldown={best['cooldown_s']:g}s -> avg F1 {best['avg_f1']*100:.2f} "
          f"({best['total_detections']} responses); per task " + ", ".join(f"{t} {v['f1']*100:.1f}" for t, v in best["tasks"].items()))
    json.dump({"samples": a.samples, "settings": results, "best": best}, open(ddir / f"replay_gated_{Path(a.samples).stem}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
