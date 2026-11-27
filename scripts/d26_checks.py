"""D26 CPU checks on stored runs (no model calls): recall intervals, precision/F1 of the re-firing rules, target type, calibration.

  --check recall   recall at <= 1 and <= 2 false alarms per minute with 95% intervals (2,000 video resamples; the threshold is re-chosen
                   inside each resample), for each run and response rule, plus paired differences (a rule minus crossing; a run minus the timer)
  --check f1       interval F1 (repeats not punished) and precision for each response rule at the 5, 15, 30% shares, against the per-question timer
  --check target   ProactiveVideoQA EGO scored with point targets (each reply interval's start, +/- 2 s) and OVO with window targets
                   (+/- 4 s and +/- 8 s around each event), to separate "forgiving targets" from "content"
  --check calib    calibration of p(Yes): expected calibration error (10 bins) raw, after temperature scaling, and after Platt scaling
                   (a * logit + b), both fitted leave-one-video-out
Writes results/cache/rescore/d26_<check>_<name>.json.

    python scripts/d26_checks.py --check recall --samples data/processed/pvqa_ego_n60.jsonl
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogate.eval.bootstrap import _groups  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.metrics import f1_of  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402

D = Path("results/cache/decider")
RUNS = {"8B decider": "qwen3-vl-8b-instruct__query_only__d0bdbc25", "Change gate": "qwen3-vl-8b-instruct__history_gate_q__*",
        "Before/after": "qwen3-vl-8b-instruct__signal_beforeafter", "2B decider": "Qwen3-VL-2B-Instruct__query_only__1a7df950",
        "Anticipation": "qwen3-vl-8b-instruct__signal_anticipate"}
RULES = ("edge", "rearm", "every6", "every10")
GRID = np.linspace(0.01, 0.7, 70)


def series(pat, s):
    hits = sorted(D.glob(pat))
    f = hits[-1] / f"{s.id}.jsonl" if hits else None
    if not f or not f.exists():
        return None
    rows = sorted((json.loads(x) for x in open(f) if x.strip()), key=lambda r: r["t"])
    z = np.array([r["lp_yes"] - r["lp_no"] for r in rows])
    return np.array([r["t"] for r in rows]), 1 / (1 + np.exp(-np.clip(z, -50, 50))), z


def respond(t, p, tau, rule, one_shot):
    out, peak, low, armed, last = [], 0.0, 0.0, False, -1e9
    lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    for i in range(len(t)):
        ab, pab = p[i] >= tau, i > 0 and p[i - 1] >= tau
        if one_shot:
            if ab:
                return [float(t[i])]
            continue
        if not ab:
            continue
        if not pab:
            out.append(float(t[i]))
            last, peak, low, armed = t[i], lo[i], lo[i], False
            continue
        if rule == "rearm":
            if not armed and lo[i] > peak:
                peak = low = lo[i]
            low = min(low, lo[i])
            armed = armed or peak - low >= 1.0
            if armed and lo[i] - low >= 1.0:
                out.append(float(t[i]))
                peak = low = lo[i]
                armed = False
        elif rule in ("every6", "every10") and t[i] - last >= (6 if rule == "every6" else 10) - 1e-6:
            out.append(float(t[i]))
            last = t[i]
    return out


def windows(s, mode):
    """Target windows per question: EGO intervals (or their starts +/- 2 s in 'point' mode); OVO events +/- w s."""
    if s.meta.get("intervals"):
        iv = s.meta["intervals"]
        return [(a - 2, a + 2) for a, _ in iv] if mode == "point" else [tuple(x) for x in iv]
    w = {"default": 2.0, "w4": 4.0, "w8": 8.0}.get(mode, 2.0)
    return [(e.t - w, e.t + w) for e in s.events]


def counts(R, win):
    caught = sum(any(a <= r <= b for r in R) for a, b in win)
    fa = sum(not any(a <= r <= b for a, b in win) for r in R)
    return caught, len(win), fa, len(R)


def load(samples, names):
    data = {}
    for n in names:
        d = {s.id: x for s in samples if (x := series(RUNS[n], s)) is not None}
        if len(d) >= 0.8 * len(samples):
            data[n] = d
    return data


def check_recall(samples, video_of, data, n_boot, mode="default"):
    """Per question and threshold: (caught, events, false alarms, minutes); then resample videos and read recall at the budgets."""
    qs = [s for s in samples if all(s.id in d for d in data.values())]
    groups = _groups([s.id for s in qs], video_of)
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]
    mins = np.array([(data[next(iter(data))][s.id][0][-1] - data[next(iter(data))][s.id][0][0]) / 60 for s in qs])
    tables = {}
    for n, d in data.items():
        allp = np.concatenate([d[s.id][1] for s in qs])
        taus = [float(np.quantile(allp, 1 - r)) for r in GRID]
        for rule in RULES:
            arr = np.zeros((len(qs), len(taus), 3))
            for i, s in enumerate(qs):
                t, p, _ = d[s.id]
                win = windows(s, mode)
                for j, tau in enumerate(taus):
                    c, e, fa, _ = counts(respond(t, p, tau, rule, s.task in ("CRR", "SBPO")), win)
                    arr[i, j] = (c, e, fa)
            tables[(n, rule)] = arr
    # timer: k = respond every k ticks, phase j % k per question
    t0 = data[next(iter(data))]
    tarr = np.zeros((len(qs), 39, 3))
    for i, s in enumerate(qs):
        t = t0[s.id][0]
        win = windows(s, mode)
        for k in range(1, 40):
            c, e, fa, _ = counts([float(x) for x in t[(i % k)::k]], win)
            tarr[i, k - 1] = (c, e, fa)
    tables[("timer", "-")] = tarr

    def best(arr, idx, budget):
        a = arr[idx].sum(axis=0)
        rec, fpm = a[:, 0] / a[:, 1], a[:, 2] / mins[idx].sum()
        ok = rec[fpm <= budget]
        return 100 * ok.max() if len(ok) else 0.0

    allidx = np.arange(len(qs))
    res = {"n_questions": len(qs), "mode": mode, "cells": {}, "pairs": {}}
    for key, arr in tables.items():
        for budget in (1.0, 2.0):
            pt = best(arr, allidx, budget)
            bs = [best(arr, b, budget) for b in boots]
            res["cells"][f"{key[0]} | {key[1]} | {budget}"] = [pt, *np.percentile(bs, [2.5, 97.5]).tolist()]
    for n in data:
        for rule in ("rearm", "every6", "every10"):
            for budget in (1.0, 2.0):
                a, b = tables[(n, rule)], tables[(n, "edge")]
                pt = best(a, allidx, budget) - best(b, allidx, budget)
                bs = [best(a, x, budget) - best(b, x, budget) for x in boots]
                res["pairs"][f"{n} | {rule} minus crossing | {budget}"] = [pt, *np.percentile(bs, [2.5, 97.5]).tolist()]
            for budget in (1.0, 2.0):
                a, b = tables[(n, rule)], tables[("timer", "-")]
                pt = best(a, allidx, budget) - best(b, allidx, budget)
                bs = [best(a, x, budget) - best(b, x, budget) for x in boots]
                res["pairs"][f"{n} | {rule} minus timer | {budget}"] = [pt, *np.percentile(bs, [2.5, 97.5]).tolist()]
    return res


def check_f1(samples, video_of, data, n_boot, shares=(0.05, 0.15, 0.30), mode="default"):
    qs = [s for s in samples if all(s.id in d for d in data.values()) and s.task != "CRR"]
    groups = _groups([s.id for s in qs], video_of)
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]
    res = {"n_questions": len(qs), "mode": mode, "cells": {}}

    def score(rows, idx):
        r = rows[idx]
        m = f1_of(r[:, 0].mean(), r[:, 1].mean())
        tm = f1_of(r[:, 2].mean(), r[:, 3].mean())
        return 100 * m, 100 * tm, 100 * r[:, 1].mean()

    for n, d in data.items():
        allp = np.concatenate([d[s.id][1] for s in qs])
        for share in shares:
            tau = float(np.quantile(allp, 1 - share))
            for rule in RULES:
                rows, rng2 = [], random.Random(0)
                for s in qs:
                    t, p, _ = d[s.id]
                    win = windows(s, mode)
                    R = respond(t, p, tau, rule, False)
                    c, e, fa, nr = counts(R, win)
                    rec, pre = c / e if e else 0.0, (nr - fa) / nr if nr else 0.0
                    trs, tps = [], []
                    for _ in range(20):
                        if nr:
                            step = len(t) / nr
                            ph = rng2.random() * step
                            tr = [float(t[min(int(ph + j * step), len(t) - 1)]) for j in range(nr)]
                        else:
                            tr = []
                        c2, e2, fa2, n2 = counts(tr, win)
                        trs.append(c2 / e2 if e2 else 0.0)
                        tps.append((n2 - fa2) / n2 if n2 else 0.0)
                    rows.append((rec, pre, np.mean(trs), np.mean(tps), nr))
                rows = np.array(rows)
                m, tm, prec = score(rows, np.arange(len(qs)))
                bs = [score(rows, b) for b in boots]
                gap = [x[0] - x[1] for x in bs]
                res["cells"][f"{n} | {share} | {rule}"] = {"f1": m, "timer_f1": tm, "gap": m - tm, "gap_ci": np.percentile(gap, [2.5, 97.5]).tolist(),
                                                          "precision": prec, "responses": int(rows[:, 4].sum())}
    return res


def check_calib(samples, video_of, data, mode="default"):
    """Leave-one-video-out temperature (z / T) and Platt (a z + b) scaling; ECE with 10 equal-width bins; per-moment label = inside a window."""
    res = {}
    for n, d in data.items():
        qs = [s for s in samples if s.id in d]
        z = {s.id: d[s.id][2] for s in qs}
        y = {s.id: np.array([any(a <= x <= b for a, b in windows(s, mode)) for x in d[s.id][0]], float) for s in qs}
        vids = sorted({video_of[s.id] for s in qs})

        def fit(Z, Y, platt):
            a, b = 1.0, 0.0
            for _ in range(300):   # gradient steps on the log-loss
                pz = 1 / (1 + np.exp(-np.clip(a * Z + b, -50, 50)))
                g = pz - Y
                a -= 0.5 * float(np.mean(g * Z)) / (1 + float(np.mean(Z ** 2)))
                if platt:
                    b -= 0.5 * float(np.mean(g))
            return a, b

        def ece(P, Y):
            bins = np.minimum((P * 10).astype(int), 9)
            return float(sum(abs(P[bins == k].mean() - Y[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any()))

        out = {"raw": [], "temp": [], "platt": []}
        allP, allY = {k: [] for k in out}, []
        for v in vids:
            tr = [s for s in qs if video_of[s.id] != v]
            te = [s for s in qs if video_of[s.id] == v]
            Z = np.concatenate([z[s.id] for s in tr])
            Y = np.concatenate([y[s.id] for s in tr])
            at, _ = fit(Z, Y, False)
            ap, bp = fit(Z, Y, True)
            for s in te:
                allP["raw"].append(1 / (1 + np.exp(-np.clip(z[s.id], -50, 50))))
                allP["temp"].append(1 / (1 + np.exp(-np.clip(at * z[s.id], -50, 50))))
                allP["platt"].append(1 / (1 + np.exp(-np.clip(ap * z[s.id] + bp, -50, 50))))
                allY.append(y[s.id])
        Yc = np.concatenate(allY)
        res[n] = {"base_rate": float(Yc.mean()), **{k: {"ece": ece(np.concatenate(v), Yc), "mean_p": float(np.concatenate(v).mean())} for k, v in allP.items()}}
        # reliability table for the Platt-scaled scores
        P = np.concatenate(allP["platt"])
        res[n]["reliability_platt"] = [[k / 10, float(P[(P >= k / 10) & (P < (k + 1) / 10)].mean()) if ((P >= k / 10) & (P < (k + 1) / 10)).any() else None,
                                        float(Yc[(P >= k / 10) & (P < (k + 1) / 10)].mean()) if ((P >= k / 10) & (P < (k + 1) / 10)).any() else None,
                                        int(((P >= k / 10) & (P < (k + 1) / 10)).sum())] for k in range(10)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["recall", "f1", "target", "calib"], required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--runs", nargs="+", default=["8B decider", "Change gate", "Before/after"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tasks", default=None, help="e.g. REC,SSR to keep only those OVO tasks")
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    if a.tasks:
        samples = [s for s in samples if s.task in a.tasks.split(",")]
    video_of = {s.id: str(s.video) for s in samples}
    data = load(samples, a.runs)
    name = Path(a.samples).stem + (f"_{a.tasks.replace(',', '')}" if a.tasks else "")
    if a.check == "recall":
        res = check_recall(samples, video_of, data, a.n)
        print(f"{name}: {res['n_questions']} questions; recall at <= 1 / <= 2 false alarms per minute [95%]")
        for k, v in res["cells"].items():
            print(f"  {k:45s} {v[0]:5.1f}% [{v[1]:5.1f}, {v[2]:5.1f}]")
        print("  paired differences:")
        for k, v in res["pairs"].items():
            print(f"  {k:55s} {v[0]:+5.1f} [{v[1]:+5.1f}, {v[2]:+5.1f}]{'*' if v[1] > 0 or v[2] < 0 else ''}")
    elif a.check == "f1":
        res = check_f1(samples, video_of, data, a.n)
        print(f"{name}: {res['n_questions']} questions; F1 (repeats not punished) and its gap over the per-question timer [95%], precision")
        for k, v in res["cells"].items():
            print(f"  {k:36s} F1 {v['f1']:5.1f} timer {v['timer_f1']:5.1f} gap {v['gap']:+5.1f} [{v['gap_ci'][0]:+5.1f},{v['gap_ci'][1]:+5.1f}]"
                  f"{'*' if v['gap_ci'][0] > 0 or v['gap_ci'][1] < 0 else ' '} precision {v['precision']:5.1f}%  {v['responses']} responses")
    elif a.check == "target":
        res = {}
        modes = ["point"] if samples[0].meta.get("intervals") else ["w4", "w8"]
        for mode in modes:
            r = check_f1(samples, video_of, data, a.n, shares=(0.15,), mode=mode)
            aucs = {}
            for n, d in data.items():
                v = []
                for s in samples:
                    if s.id in d and s.task != "CRR":
                        t, p, _ = d[s.id]
                        lab = np.array([any(x <= tt <= y for x, y in windows(s, mode)) for tt in t])
                        if lab.any() and (~lab).any():
                            v.append(auc(p[lab], p[~lab]))
                aucs[n] = float(np.mean(v))
            res[mode] = {"f1": r, "auc": aucs}
            print(f"{name}, targets '{mode}': AUC " + ", ".join(f"{n} {x:.3f}" for n, x in aucs.items()))
            for k, v in r["cells"].items():
                if k.endswith("edge") or k.endswith("every6"):
                    print(f"  {k:36s} F1 {v['f1']:5.1f} timer {v['timer_f1']:5.1f} gap {v['gap']:+5.1f} [{v['gap_ci'][0]:+5.1f},{v['gap_ci'][1]:+5.1f}] precision {v['precision']:5.1f}%")
    else:
        res = check_calib(samples, video_of, data)
        for n, v in res.items():
            print(f"{name} {n}: base rate {100 * v['base_rate']:.1f}%; ECE raw {v['raw']['ece']:.3f} (mean p {v['raw']['mean_p']:.3f}), "
                  f"temperature {v['temp']['ece']:.3f}, Platt {v['platt']['ece']:.3f} (mean p {v['platt']['mean_p']:.3f})")
            print("   Platt reliability (bin, mean predicted, observed, n): " + "; ".join(f"{b:.1f}: {p:.2f}/{o:.2f}/{c}" for b, p, o, c in v["reliability_platt"] if p is not None))
    out = Path(f"results/cache/rescore/d26_{a.check}_{name}.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
