"""D24 Stages 1 to 3 on stored signals (CPU, no model calls): each signal alone, fixed combinations, learned weights with leave-one-video-out.

Signals per 2 s tick (val), aligned on the ticks every source has, moments before 5 s dropped (the frame-grounded cues saw the first 5 s):
  D   8B decider p(Yes), question only, last 2 s                   (the reference everything is compared with)
  H   8B decider with two history frames
  C   8B cue presence, highest p(cue visible) over the cues        (C_text: question-only cues; C_frames: frame-grounded cues)
  G   8B graded question-aware change gate p(Yes)
  S   SigLIP2 best-cue cosine with the 8B's question-only cues; S_rise = S minus its mean over the previous 10 s
  Dr  D minus its mean over the previous 10 s (the decider's own rise)

Scores (as fixed in D24): within-question AUC (paired difference from D, video-level 95% interval); thresholded F1 on counting + tutorial
questions, crossing rule, repeats not punished, gap over a per-question timer (bootstrap_rescore.py helpers), at the shares 5, 15 and 30%;
rescue rate (events D misses at its 15% threshold that a signal catches at its own 15% threshold, against chance); and for Stage 2 (ii)
events caught and false alarms per minute. Clue reveal is reported apart. Output: results/cache/rescore/d24_val.json.

    python scripts/d24_combine.py --samples data/processed/ovo_far_val.jsonl
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

from bootstrap_rescore import per_question, perceptual_gap  # noqa: E402
from egogate.eval.bootstrap import _groups, paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.matcher_replay import auc  # noqa: E402
from rescore_thresholds import detect, lenient  # noqa: E402

DEC = Path("results/cache/decider")
SOURCES = {"D": "qwen3-vl-8b-instruct__query_only__d0bdbc25", "H": "qwen3-vl-8b-instruct__history_frames__9f7a5597",
           "C_text": "qwen3-vl-8b-instruct__cuepresence__qwen3-vl-8b-instruct__text",
           "C_frames": "qwen3-vl-8b-instruct__cuepresence__qwen3-vl-8b-instruct__frames5",
           "G": "qwen3-vl-8b-instruct__history_gate_q__*"}
SIMS = Path("results/cache/matcher/siglip2-so400m-patch14-384__alt.cues-8b-text")
TOL, T_MIN, SHARES = 2.0, 5.0, (0.05, 0.15, 0.30)


def p_of(r):
    return 1 / (1 + math.exp(-max(-50, min(50, r["lp_yes"] - r["lp_no"]))))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_signals(samples):
    out = {}
    for s in samples:
        rows = {}
        for k, pat in SOURCES.items():
            hits = sorted(DEC.glob(pat))
            f = hits[-1] / f"{s.id}.jsonl" if hits else None
            if not f or not f.exists():
                break
            rows[k] = {round(r["t"], 3): p_of(r) for r in (json.loads(x) for x in open(f) if x.strip())}
        else:
            m = json.load(open(SIMS / f"{s.id}.json"))
            st, sv = np.array(m["t"]), np.array(m["sims"]).max(axis=1)
            ts = sorted(set.intersection(*[set(v) for v in rows.values()]))
            ts = [t for t in ts if t >= T_MIN]
            if len(ts) < 5:
                continue
            t = np.array(ts)
            sig = {k: np.array([rows[k][x] for x in ts]) for k in rows}
            sig["S"] = np.interp(t, st, sv)
            prev = lambda x: np.array([x[max(0, i - 5):i].mean() if i else x[0] for i in range(len(x))])  # noqa: E731
            sig["S_rise"] = sig["S"] - prev(sig["S"])
            sig["Dr"] = sig["D"] - prev(sig["D"])
            out[s.id] = (t, sig)
    return out


def pct(x):
    """Within-question percentile rank in [0, 1]."""
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def leaky(x, t, half=5.0):
    out, acc, last = np.zeros_like(x), x[0], t[0]
    for i, (v, tt) in enumerate(zip(x, t)):
        a = 0.5 ** ((tt - last) / half)
        acc = a * acc + (1 - a) * v
        out[i], last = acc, tt
    return out


def labels(s, t):
    ev = np.array([e.t for e in s.events])
    return np.min(np.abs(t[:, None] - ev[None]), axis=1) <= TOL


def auc_table(samples, sig, scores, video_of):
    """Within-question AUC of each score; paired difference from D."""
    per = {}
    for name, fn in scores.items():
        per[name] = {}
        for s in samples:
            if s.id not in sig:
                continue
            t, g = sig[s.id]
            lab = labels(s, t)
            if lab.any() and (~lab).any():
                x = fn(t, g)
                per[name][s.id] = auc(x[lab], x[~lab])
    res = {}
    for grp, flt in (("counting + tutorials", lambda s: s.task != "CRR"), ("clue reveal", lambda s: s.task == "CRR")):
        ids = [s.id for s in samples if flt(s) and s.id in per["D"]]
        res[grp] = {}
        for name in scores:
            v = [per[name][i] for i in ids]
            d = paired_diff(v, [per["D"][i] for i in ids], lambda a: float(np.mean(a)), ids, video_of) if name != "D" else None
            res[grp][name] = {"auc": float(np.mean(v)), "n": len(ids), "vs_D": d}
    return res


def f1_table(samples, sig, scores, video_of, n_boot=1000):
    """Counting + tutorials: lenient F1 gap over the per-question timer at each share, 95% interval; paired difference from D."""
    ss = [s for s in samples if s.task != "CRR" and s.id in sig]
    qids = [s.id for s in ss]
    groups = _groups(qids, video_of)
    rng = np.random.default_rng(0)
    boots = [[i for k in rng.integers(0, len(groups), len(groups)) for i in groups[k]] for _ in range(n_boot)]
    rows = {}
    for name, fn in scores.items():
        dd = {s.id: (sig[s.id][0], fn(*sig[s.id])) for s in ss}
        allp = np.concatenate([v[1] for v in dd.values()])
        rows[name] = {r: per_question(ss, dd, float(np.quantile(allp, 1 - r))) for r in SHARES}
    res = {}
    for name in scores:
        res[name] = {}
        for r in SHARES:
            a = [rows[name][r][q] for q in qids]
            b = [rows["D"][r][q] for q in qids]
            pt, ptd = perceptual_gap(a), perceptual_gap(a) - perceptual_gap(b)
            bs = [perceptual_gap([a[i] for i in k]) for k in boots]
            bd = [perceptual_gap([a[i] for i in k]) - perceptual_gap([b[i] for i in k]) for k in boots]
            res[name][str(r)] = {"gap": pt, "ci": list(np.percentile(bs, [2.5, 97.5])), "vs_D": ptd, "vs_D_ci": list(np.percentile(bd, [2.5, 97.5]))}
    return res


def rescue(samples, sig, scores, share=0.15):
    """Of the events D misses at its own threshold, the share each signal catches at its own threshold (any tick within 2 s above it)."""
    ss = [s for s in samples if s.id in sig]
    thr = {n: float(np.quantile(np.concatenate([fn(*sig[s.id]) for s in ss]), 1 - share)) for n, fn in scores.items()}
    out = {}
    for name, fn in scores.items():
        got = exp = n = 0
        for s in ss:
            t, g = sig[s.id]
            d, x = g["D"] >= thr["D"], fn(t, g) >= thr[name]
            k = int(x.sum())
            for e in s.events:
                near = np.abs(t - e.t) <= TOL
                m = int(near.sum())
                if m == 0 or d[near].any():
                    continue
                n += 1
                got += bool(x[near].any())
                exp += 1 - (math.comb(len(t) - m, k) / math.comb(len(t), k) if k <= len(t) - m else 0.0)
        out[name] = {"missed_by_D": n, "rescued": got / max(n, 1), "chance": exp / max(n, 1)}
    return out


def evidence_rule(samples, sig, alpha, tau0):
    """Stage 2 (ii): respond when D crosses tau0 - alpha * E(t) upward (first time for clue reveal). Returns detections per question."""
    det = {}
    for s in samples:
        if s.id not in sig:
            continue
        t, g = sig[s.id]
        e = leaky(np.mean([pct(g["C_text"]), pct(g["G"]), pct(g["S"])], axis=0), t)
        det[s.id] = detect(t, g["D"] - (tau0 - alpha * e) + 0.5, 0.5, s.task, "edge")
    return det


def two_threshold(samples, sig, thr):
    """Stage 2 (iv): respond when D reaches its high threshold (its 5% share) alone, or its low threshold (15% share) while another signal
    agrees (C_text or G at its own 15% threshold). Upward crossings of that combined condition; first time for clue reveal."""
    det = {}
    for s in samples:
        if s.id not in sig:
            continue
        t, g = sig[s.id]
        ok = (g["D"] >= thr["D_hi"]) | ((g["D"] >= thr["D_lo"]) & ((g["C_text"] >= thr["C_text"]) | (g["G"] >= thr["G"])))
        det[s.id] = detect(t, ok.astype(float), 0.5, s.task, "edge")
    return det


def caught_fa(samples, sig, det):
    ev = fa = dur = c = 0
    for s in samples:
        if s.id not in sig or s.task == "CRR":
            continue
        t = sig[s.id][0]
        gt = [e.t for e in s.events]
        r, _ = lenient(det[s.id], gt)
        c += r * len(gt)
        ev += len(gt)
        fa += sum(1 for d in det[s.id] if not any(abs(d - g) <= TOL for g in gt))
        dur += (t[-1] - t[0]) / 60
    return {"events_caught": c / ev, "false_alarms_per_min": fa / dur}


def fit_logreg(X, y, l2=1.0, iters=50):
    """L2 logistic regression by Newton's method (intercept not penalised)."""
    X1 = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X1.shape[1])
    R = np.eye(len(w)) * l2
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + R
        w -= np.linalg.solve(H, X1.T @ (p - y) + R @ w)
    return w


def features(t, g):
    return np.column_stack([logit(g["D"]), logit(g["C_text"]), logit(g["G"]), zs(g["S"]), g["Dr"]])


def learned(samples, sig, video_of):
    """Stage 3: leave-one-video-out; weights and the response threshold (share of moments, chosen on the training videos) fit without
    the held-out video. Returns held-out scores per question (for AUC) and held-out detections (for F1)."""
    ss = [s for s in samples if s.id in sig]
    vids = sorted({video_of[s.id] for s in ss})
    held_score, held_det, weights = {}, {}, []
    for v in vids:
        train = [s for s in ss if video_of[s.id] != v]
        test = [s for s in ss if video_of[s.id] == v]
        X = np.vstack([features(*sig[s.id]) for s in train])
        y = np.concatenate([labels(s, sig[s.id][0]) for s in train]).astype(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        w = fit_logreg((X - mu) / sd, y)
        weights.append(w.tolist())
        sc = lambda s: 1 / (1 + np.exp(-(w[0] + ((features(*sig[s.id]) - mu) / sd) @ w[1:])))  # noqa: E731
        tr_all = np.concatenate([sc(s) for s in train])
        best = None
        for r in SHARES:   # threshold chosen on the training videos only
            tau = float(np.quantile(tr_all, 1 - r))
            rows = []
            for s in train:
                if s.task == "CRR":
                    continue
                d = detect(sig[s.id][0], sc(s), tau, s.task, "edge")
                rec, pre = lenient(d, [e.t for e in s.events])
                rows.append(2 * rec * pre / (rec + pre) if rec + pre else 0.0)
            f = float(np.mean(rows)) if rows else 0.0
            if best is None or f > best[0]:
                best = (f, tau)
        for s in test:
            x = sc(s)
            held_score[s.id] = x
            held_det[s.id] = detect(sig[s.id][0], x, best[1], s.task, "edge")
    return held_score, held_det, np.array(weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    sig = load_signals(samples)
    print(f"{len(sig)} questions with every signal (moments from {T_MIN:g} s)\n")
    single = {"D": lambda t, g: g["D"], "H": lambda t, g: g["H"], "C_text": lambda t, g: g["C_text"], "C_frames": lambda t, g: g["C_frames"],
              "G": lambda t, g: g["G"], "S": lambda t, g: g["S"], "S_rise": lambda t, g: g["S_rise"], "Dr": lambda t, g: g["Dr"]}
    combos = {}
    for w in (0.25, 0.5, 1.0):
        combos[f"vote w={w}"] = (lambda w: lambda t, g: logit(g["D"]) + w * (logit(g["C_text"]) + logit(g["G"]) + zs(g["S"])))(w)
        combos[f"vote(frames cues) w={w}"] = (lambda w: lambda t, g: logit(g["D"]) + w * (logit(g["C_frames"]) + logit(g["G"]) + zs(g["S"])))(w)
    for w in (0.25, 0.5):
        combos[f"gate nudge w={w}"] = (lambda w: lambda t, g: w * g["G"] + (1 - w) * g["D"])(w)
    res = {"n_questions": len(sig)}

    res["stage1_auc"] = auc_table(samples, sig, single, video_of)
    res["stage1_rescue"] = rescue(samples, sig, single)
    corr = {n: float(np.mean([np.corrcoef(pct(fn(*sig[q])), pct(sig[q][1]["D"]))[0, 1] for q in sig])) for n, fn in single.items() if n != "D"}
    res["stage1_corr_with_D"] = corr
    print("STAGE 1: each signal alone. Within-question AUC (difference from D, 95% interval); rescue = events D misses at 15% that the signal catches / chance")
    for grp, rows in res["stage1_auc"].items():
        print(f"  {grp} ({rows['D']['n']} questions)")
        for n, v in rows.items():
            d = v["vs_D"]
            extra = f"  {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{'*' if d['excludes_zero'] else ''}" if d else "  (reference)"
            rs = res["stage1_rescue"][n]
            print(f"    {n:10s} AUC {v['auc']:.3f}{extra:32s}  rescue {100 * rs['rescued']:4.0f}% / {100 * rs['chance']:3.0f}% of {rs['missed_by_D']}"
                  + (f"   corr with D {corr[n]:+.2f}" if n in corr else ""))

    allf = {**single, **combos}
    res["f1"] = f1_table(samples, sig, allf, video_of)
    print("\nSTAGES 1-2: counting + tutorials, crossing rule, repeats not punished: F1 gap over the per-question timer [95%]; vs D [95%]")
    for n, v in res["f1"].items():
        print(f"  {n:26s} " + "  ".join(f"{int(float(r) * 100):>2d}%: {x['gap']:+5.1f} [{x['ci'][0]:+5.1f},{x['ci'][1]:+5.1f}] vs D {x['vs_D']:+5.1f} [{x['vs_D_ci'][0]:+5.1f},{x['vs_D_ci'][1]:+5.1f}]{'*' if x['vs_D_ci'][0] > 0 or x['vs_D_ci'][1] < 0 else ' '}" for r, x in v.items()))
    res["auc_combos"] = auc_table(samples, sig, {"D": single["D"], **combos}, video_of)
    print("\n  combination AUC, counting + tutorials:", ", ".join(f"{n} {v['auc']:.3f} ({v['vs_D']['diff']:+.3f}{'*' if v['vs_D']['excludes_zero'] else ''})" for n, v in res["auc_combos"]["counting + tutorials"].items() if v["vs_D"]))

    ss = [s for s in samples if s.id in sig]
    tau0 = float(np.quantile(np.concatenate([sig[s.id][1]["D"] for s in ss]), 1 - 0.15))
    base_det = {s.id: detect(sig[s.id][0], sig[s.id][1]["D"], tau0, s.task, "edge") for s in ss}
    res["stage2_evidence"] = {"tau0": tau0, "D alone": caught_fa(samples, sig, base_det)}
    print(f"\nSTAGE 2 (ii) evidence lowers the bar (tau0 = {tau0:.3f}, the 8B's 15% threshold): events caught / false alarms per minute, counting + tutorials")
    print(f"  D alone            caught {100 * res['stage2_evidence']['D alone']['events_caught']:5.1f}%  false alarms {res['stage2_evidence']['D alone']['false_alarms_per_min']:.2f}/min")
    for alpha in (0.1, 0.2, 0.3):
        det = evidence_rule(samples, sig, alpha, tau0)
        cf = caught_fa(samples, sig, det)
        ids = [s.id for s in ss if s.task != "CRR"]
        gap = perceptual_gap([(s.task, *lenient(det[s.id], [e.t for e in s.events]), *lenient(base_det[s.id], [e.t for e in s.events])) for s in ss if s.task != "CRR"])
        res["stage2_evidence"][f"alpha={alpha}"] = {**cf, "f1_minus_D_alone": gap}
        print(f"  alpha={alpha:<4}         caught {100 * cf['events_caught']:5.1f}%  false alarms {cf['false_alarms_per_min']:.2f}/min   F1 minus D alone {gap:+.1f}")

    q = lambda k, r: float(np.quantile(np.concatenate([sig[s.id][1][k] for s in ss]), 1 - r))  # noqa: E731
    thr = {"D_hi": q("D", 0.05), "D_lo": q("D", 0.15), "C_text": q("C_text", 0.15), "G": q("G", 0.15)}
    det2 = two_threshold(samples, sig, thr)
    cf = caught_fa(samples, sig, det2)
    gap = perceptual_gap([(s.task, *lenient(det2[s.id], [e.t for e in s.events]), *lenient(base_det[s.id], [e.t for e in s.events])) for s in ss if s.task != "CRR"])
    hi_det = {s.id: detect(sig[s.id][0], sig[s.id][1]["D"], thr["D_hi"], s.task, "edge") for s in ss}
    res["stage2_two_threshold"] = {**thr, **cf, "f1_minus_D_alone_15pct": gap, "D_hi_alone": caught_fa(samples, sig, hi_det)}
    print(f"\nSTAGE 2 (iv) two thresholds: D >= {thr['D_hi']:.3f} alone, or D >= {thr['D_lo']:.3f} with C or G agreeing")
    print(f"  caught {100 * cf['events_caught']:5.1f}%  false alarms {cf['false_alarms_per_min']:.2f}/min   F1 minus D alone (15%) {gap:+.1f}"
          f"   (D at its high threshold alone: caught {100 * res['stage2_two_threshold']['D_hi_alone']['events_caught']:.1f}%, {res['stage2_two_threshold']['D_hi_alone']['false_alarms_per_min']:.2f}/min)")

    held_score, held_det, W = learned(samples, sig, video_of)
    per = {}
    for s in ss:
        t = sig[s.id][0]
        lab = labels(s, t)
        if lab.any() and (~lab).any():
            per[s.id] = (auc(held_score[s.id][lab], held_score[s.id][~lab]), auc(sig[s.id][1]["D"][lab], sig[s.id][1]["D"][~lab]))
    ids = [s.id for s in ss if s.task != "CRR" and s.id in per]
    d = paired_diff([per[i][0] for i in ids], [per[i][1] for i in ids], lambda v: float(np.mean(v)), ids, video_of)
    f1rows = [(s.task, *lenient(held_det[s.id], [e.t for e in s.events]), *lenient(base_det[s.id], [e.t for e in s.events])) for s in ss if s.task != "CRR"]
    res["stage3"] = {"auc_heldout": float(np.mean([per[i][0] for i in ids])), "auc_D": float(np.mean([per[i][1] for i in ids])), "vs_D": d,
                     "f1_minus_D_at_15pct": perceptual_gap(f1rows), "weights_mean": W.mean(0).tolist(), "weights_sd": W.std(0).tolist(),
                     "features": ["intercept", "logit D", "logit C_text", "logit G", "z S", "D rise"]}
    print(f"\nSTAGE 3 learned weights, leave-one-video-out ({len(W)} folds): held-out AUC {res['stage3']['auc_heldout']:.3f} vs D {res['stage3']['auc_D']:.3f}"
          f"  ({d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{'*' if d['excludes_zero'] else ''}); F1 minus D at its 15% threshold {res['stage3']['f1_minus_D_at_15pct']:+.1f}")
    print("  mean weights (standardised features): " + ", ".join(f"{n} {m:+.2f}±{s:.2f}" for n, m, s in zip(res["stage3"]["features"], W.mean(0), W.std(0))))
    out = Path("results/cache/rescore/d24_val.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
