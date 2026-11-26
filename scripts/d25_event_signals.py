"""D25 C analysis (CPU): do the event-finding signals help the 8B decider catch more events? Settings as fixed in RATIONALE D25 C.

Signals (8B through the API, every 2 s, aligned on common ticks, moments from 5 s):
  D        the 8B decider (question only, last 2 s), the reference
  C1, C1r  fixed goal: p(goal visible), and its rise over the tick 4 s earlier
  C2       before/after clip comparison
  C3       anticipation ("now or in the next few seconds")
  C4, C4l, C4p  rolling goal: rise of p(goal) under the same goal (primary), p(goal) itself, progress (share of steps done)
Scores: within-question AUC (difference from D, and C4 against C1); rescue rate; F1 gap over the per-question timer at the 15% share; the
union rule (respond when D crosses its threshold at share s/2 or the signal crosses its own at s/2, s = 10, 20, 30%) against D alone at the
same number of responses, with events caught and false alarms per minute. Counting + tutorials, tutorials apart, clue reveal apart.

    python scripts/d25_event_signals.py --samples data/processed/ovo_far_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_rescore import perceptual_gap  # noqa: E402
from d24_combine import auc_table, caught_fa, f1_table, rescue  # noqa: E402
from d25_route_cusum import boot, timer_rows  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from rescore_thresholds import detect  # noqa: E402

DEC = Path("results/cache/decider")
SRC = {"D": "qwen3-vl-8b-instruct__query_only__d0bdbc25", "C1": "qwen3-vl-8b-instruct__signal_goal",
       "C2": "qwen3-vl-8b-instruct__signal_beforeafter", "C3": "qwen3-vl-8b-instruct__signal_anticipate",
       "C4": "qwen3-vl-8b-instruct__signal_rollinggoal"}
T_MIN = 5.0


def load(samples, need):
    out = {}
    for s in samples:
        rows = {}
        for k in need:
            f = DEC / SRC[k] / f"{s.id}.jsonl"
            if not f.exists():
                break
            rows[k] = {round(r["t"], 3): r for r in (json.loads(x) for x in open(f) if x.strip())}
        else:
            ts = sorted(t for t in set.intersection(*[set(v) for v in rows.values()]) if t >= T_MIN)
            if len(ts) < 5:
                continue
            p = lambda r: 1 / (1 + math.exp(-max(-50, min(50, r["lp_yes"] - r["lp_no"]))))  # noqa: E731
            g = {k: np.array([p(rows[k][t]) for t in ts]) for k in rows}
            if "C1" in g:   # rise over the tick 4 s earlier (2 ticks)
                g["C1r"] = g["C1"] - np.concatenate([g["C1"][:2], g["C1"][:-2]])
            if "C4" in rows:
                g["C4l"] = g.pop("C4")
                g["C4"] = np.array([rows["C4"][t]["rise"] for t in ts])
                g["C4p"] = np.array([rows["C4"][t]["progress"] for t in ts])
            out[s.id] = (np.array(ts), g)
    return out


def union_rule(ss, sig, name, share):
    allD = np.concatenate([sig[s.id][1]["D"] for s in ss])
    allX = np.concatenate([sig[s.id][1][name] for s in ss])
    tD, tX = float(np.quantile(allD, 1 - share / 2)), float(np.quantile(allX, 1 - share / 2))
    det = {}
    for s in ss:
        t, g = sig[s.id]
        a = detect(t, g["D"], tD, s.task, "edge")
        b = detect(t, g[name], tX, s.task, "edge")
        det[s.id] = sorted(set(a) | set(b)) if s.task != "CRR" else sorted(set(a) | set(b))[:1]
    return det


def matched_D(ss, sig, n_target):
    allD = np.concatenate([sig[s.id][1]["D"] for s in ss])
    best = None
    for r in np.linspace(0.02, 0.8, 79):
        tau = float(np.quantile(allD, 1 - r))
        det = {s.id: detect(sig[s.id][0], sig[s.id][1]["D"], tau, s.task, "edge") for s in ss}
        n = sum(len(v) for v in det.values())
        if best is None or abs(n - n_target) < abs(best[1] - n_target):
            best = (det, n, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    have = [k for k in SRC if k == "D" or (DEC / SRC[k]).exists() and len(list((DEC / SRC[k]).glob("*.jsonl"))) >= len(samples)]
    sig = load(samples, have)
    names = ["D"] + [n for n in ("C1", "C1r", "C2", "C3", "C4", "C4l", "C4p") if n[:2] in have]
    print(f"{len(sig)} questions with signals {have} (moments from {T_MIN:g} s)\n")
    scores = {n: (lambda n: lambda t, g: g[n])(n) for n in names}
    res = {"signals": names, "auc": auc_table(samples, sig, scores, video_of), "rescue": rescue(samples, sig, scores)}
    tut = [s for s in samples if s.task == "SSR" and s.id in sig]
    res["auc_tutorials"] = auc_table(tut, sig, scores, video_of)["counting + tutorials"]
    print("Within-question AUC (difference from D), rescue = events D misses at 15% caught by the signal at its own 15% / chance")
    for n in names:
        a_, t_ = res["auc"]["counting + tutorials"][n], res["auc_tutorials"][n]
        d = a_["vs_D"]
        dt = t_["vs_D"]
        rs = res["rescue"][n]
        print(f"  {n:4s} count+tut {a_['auc']:.3f}" + (f" ({d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['excludes_zero'] else ''})" if d else " (reference)          ")
              + f"   tutorials {t_['auc']:.3f}" + (f" ({dt['diff']:+.3f}{'*' if dt['excludes_zero'] else ''})" if dt else "")
              + f"   clue reveal {res['auc']['clue reveal'][n]['auc']:.3f}   rescue {100 * rs['rescued']:3.0f}% / {100 * rs['chance']:3.0f}% of {rs['missed_by_D']}")
    if "C1" in names and "C4" in names:
        from egogate.eval.bootstrap import paired_diff  # noqa: PLC0415
        from egogate.matcher_replay import auc  # noqa: PLC0415
        from d24_combine import labels  # noqa: PLC0415
        per = {}
        for s in samples:
            if s.id in sig and s.task != "CRR":
                lab = labels(s, sig[s.id][0])
                if lab.any() and (~lab).any():
                    per[s.id] = {k: auc(sig[s.id][1][k][lab], sig[s.id][1][k][~lab]) for k in ("C1", "C1r", "C4", "C4l")}
        ids = list(per)
        for x, y in (("C4", "C1r"), ("C4l", "C1")):
            d = paired_diff([per[i][x] for i in ids], [per[i][y] for i in ids], lambda v: float(np.mean(v)), ids, video_of)
            res[f"{x}_vs_{y}"] = d
            print(f"  rolling vs fixed goal: {x} - {y} = {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{'*' if d['excludes_zero'] else ''}")

    res["f1"] = f1_table(samples, sig, scores, video_of)
    print("\nF1 gap over the per-question timer at 15% (counting + tutorials), and vs D at the same share:")
    for n in names:
        x = res["f1"][n]["0.15"]
        print(f"  {n:4s} {x['gap']:+5.1f} [{x['ci'][0]:+5.1f},{x['ci'][1]:+5.1f}]   vs D {x['vs_D']:+5.1f} [{x['vs_D_ci'][0]:+5.1f},{x['vs_D_ci'][1]:+5.1f}]")

    ss = [s for s in samples if s.id in sig and s.task != "CRR"]
    boots = boot(ss, video_of)
    res["union"] = {}
    print("\nUnion rule (D at share s/2 OR the signal at s/2) against D alone at the same number of responses (counting + tutorials):")
    for share in (0.10, 0.20, 0.30):
        for n in names[1:]:
            det = union_rule(ss, sig, n, share)
            nu = sum(len(v) for v in det.values())
            detD, nD, rD = matched_D(ss, sig, nu)
            ru, rd = timer_rows(ss, sig, det), timer_rows(ss, sig, detD)
            gu, gd = perceptual_gap(list(ru.values())), perceptual_gap(list(rd.values()))
            bs = [perceptual_gap([ru[q] for q in b]) - perceptual_gap([rd[q] for q in b]) for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cu, cd = caught_fa(ss, sig, det), caught_fa(ss, sig, detD)
            res["union"][f"{n}@{share}"] = {"responses": nu, "gap": gu, "gap_D": gd, "diff": gu - gd, "ci": [float(lo), float(hi)], "union": cu, "D": cd}
            print(f"  s={int(share * 100):>2d}% {n:4s} {nu:4d} resp: caught {100 * cu['events_caught']:4.1f}% FA {cu['false_alarms_per_min']:.2f}/min"
                  f" | D alone ({100 * rD:.0f}%): caught {100 * cd['events_caught']:4.1f}% FA {cd['false_alarms_per_min']:.2f}/min"
                  f" | F1 gap {gu:+5.1f} vs {gd:+5.1f} ({gu - gd:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ''})")
    out = Path("results/cache/rescore/d25_event_signals_val.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
