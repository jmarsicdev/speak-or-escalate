"""Row 1, stage 2: score a stored decider cache under many thresholds. CPU only, takes seconds.

    python scripts/replay_decider.py --cache results/cache/decider/<dir> --samples data/processed/ovo_far_val.jsonl

Prints average F1 over a grid of thresholds (tau) and cooldowns, marks the best setting, and writes
<cache>/replay_<split>.json plus a detections file for the best setting. Choose the setting on the val split,
then apply it once to test (decision D6). Questions with no stored scores are scored as "no detections".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.replay import detections, load_split, score_setting  # noqa: E402

TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
COOLDOWNS = [0.0, 4.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=None, help="score one fixed tau instead of sweeping (use for test)")
    ap.add_argument("--cooldown", type=float, default=None)
    a = ap.parse_args()

    cache = Path(a.cache)
    samples = load_jsonl(a.samples)
    rows = load_split(cache, samples)
    have = sum(1 for r in rows.values() if r)
    name = Path(a.samples).stem
    print(f"{name}: scores stored for {have}/{len(samples)} questions, T={a.temperature:g}, tol={a.tol:g}s")

    taus = [a.tau] if a.tau is not None else TAUS
    cds = [a.cooldown] if a.cooldown is not None else COOLDOWNS
    results = []
    print(f"  {'tau':>5s} {'cooldown':>8s} {'REC':>6s} {'SSR':>6s} {'CRR':>6s} {'avg F1':>7s} {'detections':>11s}")
    for cd in cds:
        for tau in taus:
            r = score_setting(samples, rows, tau, a.temperature, cd, a.tol)
            f = {t: r["tasks"].get(t, {}).get("f1", 0) * 100 for t in ("REC", "SSR", "CRR")}
            print(f"  {tau:5.2f} {cd:8.1f} {f['REC']:6.1f} {f['SSR']:6.1f} {f['CRR']:6.1f} {r['avg_f1']*100:7.2f} {r['total_detections']:11d}")
            results.append({"tau": tau, "cooldown_s": cd, "avg_f1": r["avg_f1"], "tasks": r["tasks"],
                            "total_detections": r["total_detections"]})
    best = max(results, key=lambda r: r["avg_f1"])
    print(f"best on {name}: tau={best['tau']:g}, cooldown={best['cooldown_s']:g}s -> avg F1 {best['avg_f1']*100:.2f}")

    json.dump({"samples": a.samples, "temperature": a.temperature, "tol": a.tol, "settings": results, "best": best},
              open(cache / f"replay_{name}.json", "w"), indent=1)
    with open(cache / f"detections_{name}.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps({"id": s.id, "detections": detections(rows[s.id], best["tau"], a.temperature, best["cooldown_s"])}) + "\n")


if __name__ == "__main__":
    main()
