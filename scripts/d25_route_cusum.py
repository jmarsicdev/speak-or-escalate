"""D25 A and B on stored signals (CPU): routing uncertain moments from the 8B to the 235B, and change-point (CUSUM) responses.

A. Routing (Gate 3). Each model's p(Yes) becomes its percentile among its own moments (one scale for both). A budget B of moments goes to
   the 235B (they take its percentile, the rest keep the 8B's). Routers: "uncertain" (8B moments nearest its 15% threshold), "gate" (G high
   while the 8B is below its threshold, ranked by G), "random" (20 seeds). Ends: B = 0 (8B) and B = 1 (235B). Counting + tutorial questions
   both models have; within-question AUC and the F1 gap over the per-question timer at the 15% share; router minus random, 95% interval.
B. CUSUM on x(t) (mean within-question percentile of D, G, S; or of D alone): s = max(0, s + x - 0.5), respond when s >= h, then s = 0;
   h chosen so the response count matches the 8B at 5, 15 and 30%; compared with D alone at that count (gap over the per-question timer).

    python scripts/d25_route_cusum.py --samples data/processed/ovo_far_val.jsonl
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

from bootstrap_rescore import perceptual_gap  # noqa: E402
from d24_combine import labels, load_signals, pct  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402
from rescore_thresholds import detect, lenient  # noqa: E402

BIG = "results/cache/decider/qwen3-vl-235b-a22b-instruct__query_only__b1bb72f9"
BUDGETS, SHARE = (0.05, 0.10, 0.20, 0.30, 0.50), 0.15


def p_of(r):
    return 1 / (1 + math.exp(-max(-50, min(50, r["lp_yes"] - r["lp_no"]))))


def global_pct(values, ref):
    ref = np.sort(ref)
    return np.searchsorted(ref, values, side="right") / len(ref)


def timer_rows(ss, sig, dets, seeds=20):
    rng, rows = random.Random(0), {}
    for s in ss:
        t, n, gt = sig[s.id][0], len(dets[s.id]), [e.t for e in s.events]
        tr = []
        for _ in range(seeds):
            if n:
                step = len(t) / n
                ph = rng.random() * step
                tr.append(lenient([float(t[min(int(ph + j * step), len(t) - 1)]) for j in range(n)], gt))
            else:
                tr.append((0.0, 0.0))
        rows[s.id] = (s.task, *lenient(dets[s.id], gt), float(np.mean([x[0] for x in tr])), float(np.mean([x[1] for x in tr])))
    return rows


def boot(ss, video_of, n=1000):
    groups = {}
    for s in ss:
        groups.setdefault(video_of[s.id], []).append(s.id)
    groups = list(groups.values())
    rng = np.random.default_rng(0)
    return [[q for k in rng.integers(0, len(groups), len(groups)) for q in groups[k]] for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    sig = load_signals(samples)
    res = {}

    # ---------------- A. routing
    big = {}
    for s in samples:
        f = Path(BIG) / f"{s.id}.jsonl"
        if s.id in sig and f.exists():
            rows = {round(r["t"], 3): p_of(r) for r in (json.loads(x) for x in open(f) if x.strip())}
            t = sig[s.id][0]
            if all(round(x, 3) in rows for x in t):
                big[s.id] = np.array([rows[round(x, 3)] for x in t])
    ss = [s for s in samples if s.id in big and s.task != "CRR"]
    allD = np.concatenate([sig[s.id][1]["D"] for s in ss])
    allB = np.concatenate([big[s.id] for s in ss])
    allG = np.concatenate([sig[s.id][1]["G"] for s in ss])
    d_pct = {s.id: global_pct(sig[s.id][1]["D"], allD) for s in ss}
    b_pct = {s.id: global_pct(big[s.id], allB) for s in ss}
    g_pct = {s.id: global_pct(sig[s.id][1]["G"], allG) for s in ss}
    keys = [(s.id, i) for s in ss for i in range(len(sig[s.id][0]))]
    thr = 1 - SHARE
    unc = np.array([-abs(d_pct[q][i] - thr) for q, i in keys])                                      # nearest the threshold first
    gat = np.array([g_pct[q][i] if d_pct[q][i] < thr else -1.0 for q, i in keys])                   # G high, 8B below its bar
    boots = boot(ss, video_of)
    print(f"A. ROUTING: {len(ss)} counting + tutorial questions with both models, {len(keys)} moments")

    def evaluate(routed_mask):
        score, k = {}, 0
        for s in ss:
            n = len(sig[s.id][0])
            m = routed_mask[k:k + n]
            score[s.id] = np.where(m, b_pct[s.id], d_pct[s.id])
            k += n
        aucs = {s.id: auc(score[s.id][lab], score[s.id][~lab]) for s in ss if (lab := labels(s, sig[s.id][0])).any() and (~lab).any()}
        allS = np.concatenate([score[s.id] for s in ss])
        tau = float(np.quantile(allS, 1 - SHARE))
        dets = {s.id: detect(sig[s.id][0], score[s.id], tau, s.task, "edge") for s in ss}
        return aucs, timer_rows(ss, sig, dets)

    def summary(aucs, rows):
        return float(np.mean(list(aucs.values()))), perceptual_gap(list(rows.values()))

    a0, r0 = evaluate(np.zeros(len(keys), bool))
    a1, r1 = evaluate(np.ones(len(keys), bool))
    res["routing"] = {"8B_everywhere": summary(a0, r0), "235B_everywhere": summary(a1, r1), "budgets": {}}
    print(f"  8B everywhere:   AUC {res['routing']['8B_everywhere'][0]:.3f}   F1 gap over timer {res['routing']['8B_everywhere'][1]:+.1f}")
    print(f"  235B everywhere: AUC {res['routing']['235B_everywhere'][0]:.3f}   F1 gap over timer {res['routing']['235B_everywhere'][1]:+.1f}")
    for B in BUDGETS:
        n = int(round(B * len(keys)))
        out = {}
        rand = []
        for seed in range(20):
            m = np.zeros(len(keys), bool)
            m[np.random.default_rng(seed).choice(len(keys), n, replace=False)] = True
            rand.append(evaluate(m))
        ra = {q: float(np.mean([r[0][q] for r in rand])) for q in rand[0][0]}
        rr = {q: tuple([rand[0][1][q][0]] + [float(np.mean([r[1][q][j] for r in rand])) for j in range(1, 5)]) for q in rand[0][1]}
        out["random"] = summary(ra, rr)
        line = f"  budget {int(B * 100):>2d}%: random AUC {out['random'][0]:.3f} gap {out['random'][1]:+5.1f}"
        for name, pri in (("uncertain", unc), ("gate", gat)):
            m = np.zeros(len(keys), bool)
            m[np.argsort(-pri, kind="stable")[:n]] = True
            aa, rw = evaluate(m)
            s_ = summary(aa, rw)
            ids = sorted(set(aa) & set(ra))
            dA = float(np.mean([aa[q] - ra[q] for q in ids]))
            bs = [perceptual_gap([rw[q] for q in b]) - perceptual_gap([rr[q] for q in b]) for b in boots]
            dF = s_[1] - out["random"][1]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[name] = {"auc": s_[0], "gap": s_[1], "auc_minus_random": dA, "gap_minus_random": dF, "ci": [float(lo), float(hi)]}
            line += f" | {name}: AUC {s_[0]:.3f} ({dA:+.3f}) gap {s_[1]:+5.1f} ({dF:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ''})"
        res["routing"]["budgets"][str(B)] = out
        print(line)

    # ---------------- B. CUSUM
    ssB = [s for s in samples if s.id in sig and s.task != "CRR"]
    boots = boot(ssB, video_of)
    allD = np.concatenate([sig[s.id][1]["D"] for s in ssB])

    def cusum(x, h):
        out, acc = [], 0.0
        for i, v in enumerate(x):
            acc = max(0.0, acc + v - 0.5)
            if acc >= h:
                out.append(i)
                acc = 0.0
        return out

    print(f"\nB. CUSUM: {len(ssB)} counting + tutorial questions; gap over the per-question timer, and minus D alone at the same response count")
    res["cusum"] = {}
    for share in (0.05, 0.15, 0.30):
        tauD = float(np.quantile(allD, 1 - share))
        detD = {s.id: detect(sig[s.id][0], sig[s.id][1]["D"], tauD, s.task, "edge") for s in ssB}
        nD = sum(len(v) for v in detD.values())
        rowsD = timer_rows(ssB, sig, detD)
        for name, xf in (("D, G, S", lambda g: np.mean([pct(g["D"]), pct(g["G"]), pct(g["S"])], axis=0)), ("D alone", lambda g: pct(g["D"]))):
            best = None
            for h in np.linspace(0.05, 6.0, 120):
                n = sum(len(cusum(xf(sig[s.id][1]), h)) for s in ssB)
                if best is None or abs(n - nD) < abs(best[1] - nD):
                    best = (h, n)
            h = best[0]
            det = {s.id: [float(sig[s.id][0][i]) for i in cusum(xf(sig[s.id][1]), h)] for s in ssB}
            rows = timer_rows(ssB, sig, det)
            g, gD = perceptual_gap(list(rows.values())), perceptual_gap(list(rowsD.values()))
            bs = [perceptual_gap([rows[q] for q in b]) - perceptual_gap([rowsD[q] for q in b]) for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res["cusum"][f"{name} @{share}"] = {"h": float(h), "responses": best[1], "responses_D": nD, "gap": g, "gap_D": gD, "minus_D": g - gD, "ci": [float(lo), float(hi)]}
            print(f"  share {int(share * 100):>2d}% ({nD} responses for D): CUSUM on {name:8s} h={h:.2f}, {best[1]} responses: gap {g:+5.1f} vs D {gD:+5.1f}"
                  f"  ({g - gD:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ''})")
    out = Path("results/cache/rescore/d25_val.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
