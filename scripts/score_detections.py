"""Score any system's detections against a split, the Em-Garde way (no LLM judge).

    python scripts/score_detections.py --samples data/processed/ovo_far_val.jsonl \
        --detections results/edge_alone/val_detections.jsonl --tol 2

A detections file has one JSON object per line: {"id": "ovo_CRR_1468", "detections": [310.0, 312.0]}.
A question with no line is scored as "no detections". Every experiment arm ends by writing this
file, so every arm is scored by the same code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.metrics import DEFAULT_TOL_S, aggregate, score_question  # noqa: E402


def score_file(samples_path, detections_path, tol=DEFAULT_TOL_S) -> dict:
    samples = load_jsonl(samples_path)
    det = {}
    with open(detections_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                det[r["id"]] = sorted(float(t) for t in r["detections"])
    rows = []
    for s in samples:
        q = score_question(det.get(s.id, []), [e.t for e in s.events], tol)
        rows.append({"id": s.id, "task": s.task, "recall": q.recall, "precision": q.precision,
                     "n_det": q.n_det, "n_gt": q.n_gt, "n_correct": q.n_correct})
    res = aggregate(rows)
    res["tol_s"] = tol
    res["per_question"] = rows
    return res


def print_table(res: dict, title: str = ""):
    print(title)
    print(f"  {'task':5s} {'recall':>7s} {'precision':>9s} {'F1':>6s} {'n':>4s}")
    for t, v in res["tasks"].items():
        print(f"  {t:5s} {v['recall']*100:7.1f} {v['precision']*100:9.1f} {v['f1']*100:6.1f} {v['n_questions']:4d}")
    print(f"  avg F1 @ {res['tol_s']:g}s: {res['avg_f1']*100:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_S)
    ap.add_argument("--out", default=None, help="optional JSON path for the full result")
    a = ap.parse_args()
    r = score_file(a.samples, a.detections, a.tol)
    print_table(r, f"{a.detections}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1)
