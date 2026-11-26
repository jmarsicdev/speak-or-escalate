"""Re-score stored decider runs with model-specific thresholds and response rules (CPU only, no model calls).

The within-question AUC used so far compares every moment near an event with every other moment; it never asks "did the model respond at
the right time once it was confident enough?". This script does, from the stored p(Yes) values:

  thresholds  model-specific: for each model, tau is set so that p(Yes) >= tau on a fixed share of that model's own moments
              (3, 5, 10, 15, 20, 30%), because the models say Yes at very different base rates (6% to 80%) and one fixed tau is unfair
  rules       counting and tutorial questions (several events): respond when p(Yes) crosses tau upward ("edge"), or at every tick at or above
              tau with a 4 s cooldown ("cooldown", the earlier replay rule); clue reveal (one answer): respond the first time p(Yes) >= tau,
              then stop ("first time")
  lenient     also a version where repeat hits are not punished (see lenient()), with the same per-question timer scored the same way
  scores      the benchmark F1 (Em-Garde metric, +/- 2 s) per task, against a timer making the same number of responses in each question
              (counting and tutorials; 20 random phases; a timer matched only on the total is also stored, "base_total") and a single random response (clue reveal; exact expectation); and the event hit rate: the share of
              events with p(Yes) >= tau at some tick within 2 s, against its exact expectation if the same number of ticks were above tau at random

Question sets: "all" (every question a model has; 235B has 36) and "common" (the 36 all models share). No intervals: read the gaps, not the
cells, and only patterns across budgets. Writes results/cache/rescore/rescore_thresholds_val.json (also the detections, for the viewer).

    python scripts/rescore_thresholds.py --samples data/processed/ovo_far_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.floors import apply_cooldown  # noqa: E402
from egogate.eval.metrics import aggregate, score_question  # noqa: E402

D = Path("results/cache/decider")
MODELS = {
    "2B": "Qwen3-VL-2B-Instruct__query_only__1a7df950",
    "2B + cues": "Qwen3-VL-2B-Instruct__query_cues__b1de186d",
    "8B": "qwen3-vl-8b-instruct__query_only__d0bdbc25",
    "8B 1 frame": "qwen3-vl-8b-instruct__query_only__05293dc6",
    "8B 2 frames": "qwen3-vl-8b-instruct__query_only__2ee12114",
    "8B + history": "qwen3-vl-8b-instruct__history_frames__9f7a5597",
    "8B + what changed": "qwen3-vl-8b-instruct__history_change__d9dc4ad4",
    "8B + what changed (question-aware)": "qwen3-vl-8b-instruct__history_change_q__*",
    "235B": "qwen3-vl-235b-a22b-instruct__query_only__b1bb72f9",
    "change gate only": "gate_changeq__gate_only",
    "change gate + 2B": "gate_changeq__2B",
    "change gate + 8B": "gate_changeq__8B",
    "change gate + 235B": "gate_changeq__235B",
}
RATES = (0.03, 0.05, 0.10, 0.15, 0.20, 0.30)
REARM = 1.0   # log-odds dip and recovery that re-arms a response inside one stretch above tau (D24)
TOL = 2.0


def load(dirname, ids):
    hits = sorted(D.glob(dirname))
    if not hits:
        return {}
    out = {}
    for i in ids:
        f = hits[-1] / f"{i}.jsonl"
        if f.exists():
            rows = [json.loads(x) for x in open(f) if x.strip()]
            t = np.array([r["t"] for r in rows])
            p = 1 / (1 + np.exp(-np.clip(np.array([r["lp_yes"] - r["lp_no"] for r in rows]), -50, 50)))
            o = np.argsort(t)
            out[i] = (t[o], p[o])
    return out


def detect(t, p, tau, task, rule):
    above = p >= tau
    if task == "CRR":
        k = np.flatnonzero(above)
        return [float(t[k[0]])] if len(k) else []
    if rule == "edge":
        prev = np.concatenate([[False], above[:-1]])
        return [float(x) for x in t[above & ~prev]]
    if rule == "rearm":
        # D24: respond on the upward crossing, and again inside the same stretch above tau when the log-odds have dipped and risen back
        # by at least REARM (1.0) since the last response: a repeated event (another rep) inside one long clump above the threshold
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        lo = np.log(pc / (1 - pc))
        out, peak, low, armed = [], 0.0, 0.0, False
        for i in range(len(p)):
            if not above[i]:
                continue
            if i == 0 or not above[i - 1]:          # upward crossing: respond, start tracking this stretch
                out.append(float(t[i]))
                peak = low = lo[i]
                armed = False
                continue
            if not armed and lo[i] > peak:          # still climbing after the last response
                peak = low = lo[i]
            low = min(low, lo[i])
            if peak - low >= REARM:                 # dipped far enough below the peak
                armed = True
            if armed and lo[i] - low >= REARM:      # and came back up: a new event inside the same stretch
                out.append(float(t[i]))
                peak = low = lo[i]
                armed = False
        return out
    return apply_cooldown([float(x) for x in t[above]], 4.0)


def timer_rows(samples, data, n_target, seeds=20):
    """Counting/tutorial questions: respond every k ticks (random phase per question), k chosen to match n_target responses in total."""
    qs = [s for s in samples if s.task != "CRR"]
    best = min(range(1, 80), key=lambda k: abs(sum(math.ceil(len(data[s.id][0]) / k) for s in qs) - n_target))
    rng = random.Random(0)
    agg = []
    for _ in range(seeds):
        rows = []
        for s in qs:
            t = data[s.id][0]
            ph = rng.randrange(best)
            q = score_question([float(x) for x in t[ph::best]], [e.t for e in s.events], TOL)
            rows.append({"task": s.task, "recall": q.recall, "precision": q.precision})
        agg.append(aggregate(rows))
    return {task: float(np.mean([a["tasks"][task]["f1"] for a in agg])) for task in agg[0]["tasks"]}, best


def lenient(det, gt):
    """Repeat hits are not punished: precision = share of responses within 2 s of ANY event (several responses may share one event);
    recall = share of events with at least one response within 2 s. The benchmark metric instead matches each event to one response."""
    det, gt = np.array(det, dtype=float), np.array(gt, dtype=float)
    if not len(gt):
        return 0.0, 0.0
    rec = float(np.mean([(np.abs(det - g) <= TOL).any() for g in gt])) if len(det) else 0.0
    pre = float(np.mean([(np.abs(gt - d) <= TOL).any() for d in det])) if len(det) else 0.0
    return rec, pre


def timer_per_question(samples, data, dets, seeds=20):
    """Counting/tutorial questions: in each question, the same number of responses as the model made there, evenly spaced over its ticks
    with a random phase. Unlike a timer matched on the total, this cannot be beaten by putting more responses into questions with more events."""
    rng = random.Random(0)
    agg = []
    for _ in range(seeds):
        rows = []
        for s in samples:
            if s.task == "CRR":
                continue
            t, n = data[s.id][0], len(dets[s.id])
            if n:
                step = len(t) / n
                ph = rng.random() * step
                det = [float(t[min(int(ph + j * step), len(t) - 1)]) for j in range(n)]
            else:
                det = []
            q = score_question(det, [e.t for e in s.events], TOL)
            lr, lp = lenient(det, [e.t for e in s.events])
            rows.append({"task": s.task, "recall": q.recall, "precision": q.precision, "lrec": lr, "lpre": lp})
        agg.append((aggregate(rows), aggregate([{"task": r["task"], "recall": r["lrec"], "precision": r["lpre"]} for r in rows])))
    TIMER_RP.clear()   # per-task recall and precision of the timer (lenient), averaged over the phases, for the diagnosis
    TIMER_RP.update({task: {k: float(np.mean([a[1]["tasks"][task][k] for a in agg])) for k in ("recall", "precision")} for task in agg[0][1]["tasks"]})
    return ({task: float(np.mean([a[0]["tasks"][task]["f1"] for a in agg])) for task in agg[0][0]["tasks"]},
            {task: float(np.mean([a[1]["tasks"][task]["f1"] for a in agg])) for task in agg[0][1]["tasks"]})


TIMER_RP: dict = {}


def crr_random(samples, data):
    """Clue reveal: respond once at a uniformly random tick; recall = precision = P(that tick is within 2 s of the clue)."""
    rows = []
    for s in samples:
        if s.task != "CRR":
            continue
        t = data[s.id][0]
        ev = np.array([e.t for e in s.events])
        ph = float(np.mean(np.min(np.abs(t[:, None] - ev[None]), axis=1) <= TOL))
        rows.append({"task": "CRR", "recall": ph, "precision": ph})
    return aggregate(rows)["tasks"].get("CRR", {}).get("f1", 0.0) if rows else None


def hit_rate(samples, data, tau):
    """Share of events with p >= tau at some tick within 2 s, and its expectation if the same number of ticks per question were above tau at random."""
    got, exp, n = 0, 0.0, 0
    for s in samples:
        t, p = data[s.id]
        k_above = int((p >= tau).sum())
        for e in s.events:
            near = np.abs(t - e.t) <= TOL
            m = int(near.sum())
            if m == 0:
                continue
            n += 1
            got += bool((p[near] >= tau).any())
            exp += 1 - (math.comb(len(t) - m, k_above) / math.comb(len(t), k_above) if k_above <= len(t) - m else 0.0)
    return got / n, exp / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    ids = [s.id for s in samples]
    data = {name: load(d, ids) for name, d in MODELS.items()}
    data = {k: v for k, v in data.items() if len(v) >= 30}
    common = set.intersection(*[set(v) for v in data.values()])
    result = {"rates": RATES, "models": list(data), "sets": {}, "detections": {}}
    for set_name in ("all", "common"):
        res = {}
        for name, dd in data.items():
            ss = [s for s in samples if s.id in dd and (set_name == "all" or s.id in common)]
            allp = np.concatenate([dd[s.id][1] for s in ss])
            crr_base = crr_random(ss, dd)
            res[name] = {"n_questions": len(ss), "rows": []}
            for r in RATES:
                tau = float(np.quantile(allp, 1 - r))
                hr, hr0 = hit_rate(ss, dd, tau)
                for rule in ("edge", "cooldown", "rearm"):
                    rows, lrows, dets, n_multi = [], [], {}, 0
                    for s in ss:
                        t, p = dd[s.id]
                        det = detect(t, p, tau, s.task, rule)
                        dets[s.id] = det
                        n_multi += len(det) if s.task != "CRR" else 0
                        q = score_question(det, [e.t for e in s.events], TOL)
                        rows.append({"task": s.task, "recall": q.recall, "precision": q.precision})
                        lr, lp = lenient(det, [e.t for e in s.events])
                        lrows.append({"task": s.task, "recall": lr, "precision": lp})
                    agg = aggregate(rows)
                    lagg = aggregate(lrows)
                    tm, k = timer_rows(ss, dd, n_multi)
                    tq, tql = timer_per_question(ss, dd, dets)
                    f = {task: v["f1"] * 100 for task, v in agg["tasks"].items()}
                    base = {**{task: v * 100 for task, v in tq.items()}, "CRR": (crr_base or 0) * 100}
                    base_total = {**{task: v * 100 for task, v in tm.items()}, "CRR": (crr_base or 0) * 100}
                    perc = [x for x in ("REC", "SSR") if x in f]
                    fl = {task: v["f1"] * 100 for task, v in lagg["tasks"].items()}
                    basel = {**{task: v * 100 for task, v in tql.items()}, "CRR": (crr_base or 0) * 100}
                    row = {"rate": r, "tau": tau, "rule": rule, "responses": sum(len(v) for v in dets.values()), "timer_every_s": 2 * k,
                           "f1": f, "base": base, "base_total": base_total,
                           "perceptual": float(np.mean([f[x] for x in perc])), "perceptual_base": float(np.mean([base[x] for x in perc])),
                           "perceptual_base_total": float(np.mean([base_total[x] for x in perc])),
                           "hit": hr * 100, "hit_base": hr0 * 100,
                           "f1_len": fl, "base_len": basel,
                           "rp_len": {task: {"recall": v["recall"] * 100, "precision": v["precision"] * 100} for task, v in lagg["tasks"].items()},
                           "rp_len_timer": {task: {k: v * 100 for k, v in d.items()} for task, d in TIMER_RP.items()},
                           "perceptual_len": float(np.mean([fl[x] for x in perc])), "perceptual_len_base": float(np.mean([basel[x] for x in perc]))}
                    res[name]["rows"].append(row)
                    if set_name == "all":
                        result["detections"].setdefault(name, {}).setdefault(rule, {})[str(r)] = {"tau": tau, "det": dets}
        result["sets"][set_name] = res
    out = Path("results/cache/rescore/rescore_thresholds_val.json"); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=0)
    # console summary: counting + tutorials (perceptual) F1 gap over a matched timer, and clue-reveal first-time F1 gap, common questions
    print(f"common question set: {len(common)} questions. Gap = model F1 minus a matched baseline (timer / single random response).")
    for rule in ("edge", "cooldown"):
        print(f"\nrule for counting + tutorials: {rule}")
        print(f"{'model':38s} " + " ".join(f"{int(r * 100):>4d}%" for r in RATES) + "   | clue reveal, first time: " + " ".join(f"{int(r * 100):>4d}%" for r in RATES))
        for name, v in result["sets"]["common"].items():
            rr = [x for x in v["rows"] if x["rule"] == rule]
            print(f"{name:38s} " + " ".join(f"{x['perceptual'] - x['perceptual_base']:+5.1f}" for x in rr) + "   |                          "
                  + " ".join(f"{x['f1'].get('CRR', 0) - x['base']['CRR']:+5.1f}" for x in rr))
    print("\nevent hit rate (any tick >= tau within 2 s), model / chance, common set:")
    for name, v in result["sets"]["common"].items():
        rr = [x for x in v["rows"] if x["rule"] == "edge"]
        print(f"{name:38s} " + " ".join(f"{x['hit']:4.0f}/{x['hit_base']:<3.0f}" for x in rr))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
