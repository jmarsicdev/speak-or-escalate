"""Row 3, stage 2: score the cue matcher alone (every trigger is a response) under many rules. CPU only.

    python scripts/replay_matcher.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl

Prints (1) the Gate 2 diagnostic: how well each similarity signal separates steps near a true event from the rest,
then (2) average F1 for each combination of trigger rule, cue combination, threshold and duplicate handling.
Writes <cache>/replay_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import score_matcher, similarity_diagnostics  # noqa: E402

THETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]


def load_data(cache: Path, samples, want_vecs: bool) -> dict:
    data = {}
    for s in samples:
        f = cache / f"{s.id}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        d = {"t": np.array(r["t"]), "sims": np.array(r["sims"]).reshape(len(r["t"]), -1)}
        npy = cache / f"{s.id}.npy"
        if want_vecs and npy.exists():
            d["vecs"] = np.load(npy)
        data[s.id] = d
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()
    cache, samples = Path(a.cache), load_jsonl(a.samples)
    name = Path(a.samples).stem
    data = load_data(cache, samples, want_vecs=True)
    have_vecs = bool(data) and all("vecs" in d for d in data.values())
    print(f"{name}: similarities stored for {len(data)}/{len(samples)} questions; window vectors {'present' if have_vecs else 'absent'}")

    diag = similarity_diagnostics(samples, data, a.tol)
    print(f"\nGate 2 diagnostic ({diag['n_steps']} steps, {diag['frac_near']*100:.1f}% within +/-{a.tol:g}s of an event); AUC 0.5 = no separation")
    for k in ("max sim", "mean sim", "max rise (paper)"):
        print(f"  {k:18s} AUC {diag[k]:.3f}")

    dedups = [("cooldown", 0.0), ("cooldown", 4.0), ("cooldown", 8.0)] + ([("emgarde", 0.0)] if have_vecs else [])
    results = []
    print(f"\n{'rule':6s} {'cues':5s} {'dedup':14s} " + " ".join(f"{t:>11.2f}" for t in THETAS) + "   (avg F1 (n responses) per threshold)")
    for rule in ("paper", "code"):
        for agg in ("any", "mean"):
            for how, cd in dedups:
                cells = []
                for th in THETAS:
                    r = score_matcher(samples, data, rule, th, agg, how, cd, a.tol)
                    results.append({"rule": rule, "agg": agg, "dedup": how, "cooldown_s": cd, "theta": th,
                                    "avg_f1": r["avg_f1"], "tasks": r["tasks"], "total_detections": r["total_detections"]})
                    cells.append(f"{r['avg_f1']*100:5.1f}({r['total_detections']:4d})")
                label = f"{how}{cd:g}s" if how == "cooldown" else "emgarde"
                print(f"{rule:6s} {agg:5s} {label:14s} " + " ".join(f"{c:>11s}" for c in cells))
    best = max(results, key=lambda r: r["avg_f1"])
    print(f"\nbest on {name}: rule={best['rule']} cues={best['agg']} dedup={best['dedup']}{best['cooldown_s']:g} theta={best['theta']:g}"
          f" -> avg F1 {best['avg_f1']*100:.2f} ({best['total_detections']} responses)")
    json.dump({"samples": a.samples, "diagnostic": diag, "settings": results, "best": best}, open(cache / f"replay_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
