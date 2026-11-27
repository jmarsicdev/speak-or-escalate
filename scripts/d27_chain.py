"""D27 analysis (CPU): edge only, cloud everywhere, and routed edge-plus-cloud systems on ProactiveVideoQA EGO, from stored runs.

Edge: the 8B decider's p(Yes), Platt-calibrated leave-one-video-out; decision = calibrated p >= tau (tau at the 15% share). Cloud: Gemini's
stored Yes/No at every tick. Routed (budget B): the B share of all ticks whose calibrated edge probability is nearest tau take the cloud's
verdict; random routing at the same budget (20 seeds) is the control. A binary decision series becomes responses by the crossing rule or by
"every 6 s while yes". Scores: intervals caught (recall), precision, false alarms per minute, interval F1 against the per-question timer, cloud
calls and cost; routed minus random with 95% video-resample intervals. Settings fixed in RATIONALE D27.

    python scripts/d27_chain.py --samples data/processed/pvqa_ego_n60.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d26_checks import series, windows  # noqa: E402
from egogate.eval.bootstrap import _groups  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.metrics import f1_of  # noqa: E402

EDGE = "qwen3-vl-8b-instruct__query_only__d0bdbc25"
CLOUD = Path("results/cache/cloud/gemini-2.5-flash")
PRICE_IN, PRICE_OUT = 0.30e-6, 2.50e-6
SHARE, BUDGETS = 0.15, (0.10, 0.20, 0.30, 0.50)


def platt_lovo(qs, z, y, video_of):
    out = {}
    for v in sorted({video_of[s.id] for s in qs}):
        Z = np.concatenate([z[s.id] for s in qs if video_of[s.id] != v])
        Y = np.concatenate([y[s.id] for s in qs if video_of[s.id] != v])
        a, b = 1.0, 0.0
        for _ in range(400):
            p = 1 / (1 + np.exp(-np.clip(a * Z + b, -50, 50)))
            g = p - Y
            a -= 0.5 * float(np.mean(g * Z)) / (1 + float(np.mean(Z ** 2)))
            b -= 0.5 * float(np.mean(g))
        for s in qs:
            if video_of[s.id] == v:
                out[s.id] = 1 / (1 + np.exp(-np.clip(a * z[s.id] + b, -50, 50)))
    return out


def responses(t, state, rule):
    out, last = [], -1e9
    for i in range(len(t)):
        if not state[i]:
            continue
        if i == 0 or not state[i - 1]:
            out.append(float(t[i]))
            last = t[i]
        elif rule == "every6" and t[i] - last >= 6 - 1e-6:
            out.append(float(t[i]))
            last = t[i]
    return out


def score_q(s, t, R, rng):
    win = windows(s, "default")
    c = sum(any(a <= r <= b for r in R) for a, b in win)
    fa = sum(not any(a <= r <= b for a, b in win) for r in R)
    rec, pre = c / len(win), (len(R) - fa) / len(R) if R else 0.0
    trs, tps = [], []
    for _ in range(20):
        n = len(R)
        if n:
            step = len(t) / n
            ph = rng.random() * step
            tr = [float(t[min(int(ph + j * step), len(t) - 1)]) for j in range(n)]
        else:
            tr = []
        c2 = sum(any(a <= r <= b for r in tr) for a, b in win)
        f2 = sum(not any(a <= r <= b for a, b in win) for r in tr)
        trs.append(c2 / len(win))
        tps.append((len(tr) - f2) / len(tr) if tr else 0.0)
    return np.array([rec, pre, np.mean(trs), np.mean(tps), c, len(win), fa, len(R)])


def summarize(rows, mins):
    r = np.array(rows)
    f1, tf1 = f1_of(r[:, 0].mean(), r[:, 1].mean()), f1_of(r[:, 2].mean(), r[:, 3].mean())
    return {"recall": 100 * r[:, 4].sum() / r[:, 5].sum(), "precision": 100 * (r[:, 7].sum() - r[:, 6].sum()) / max(r[:, 7].sum(), 1),
            "fa_per_min": r[:, 6].sum() / mins, "f1": 100 * f1, "timer_f1": 100 * tf1, "gap": 100 * (f1 - tf1), "responses": int(r[:, 7].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    a = ap.parse_args()
    samples = load_jsonl(a.samples)
    video_of = {s.id: str(s.video) for s in samples}
    edge, cloud = {}, {}
    for s in samples:
        e = series(EDGE, s)
        f = CLOUD / f"{s.id}.jsonl"
        if e is None or not f.exists():
            continue
        c = {round(r["t"], 3): r for r in (json.loads(x) for x in open(f, encoding="utf-8") if x.strip())}
        keep = [i for i, t in enumerate(e[0]) if round(t, 3) in c]
        edge[s.id] = (e[0][keep], e[2][keep])
        cloud[s.id] = (np.array([c[round(t, 3)]["yes"] for t in e[0][keep]]), np.array([c[round(t, 3)]["in_tokens"] for t in e[0][keep]]),
                       np.array([c[round(t, 3)]["out_tokens"] for t in e[0][keep]]))
    qs = [s for s in samples if s.id in edge]
    y = {s.id: np.array([any(lo <= t <= hi for lo, hi in windows(s, "default")) for t in edge[s.id][0]], float) for s in qs}
    pc = platt_lovo(qs, {s.id: edge[s.id][1] for s in qs}, y, video_of)
    allp = np.concatenate([pc[s.id] for s in qs])
    tau = float(np.quantile(allp, 1 - SHARE))
    keys = [(s.id, i) for s in qs for i in range(len(edge[s.id][0]))]
    dist = np.array([abs(pc[q][i] - tau) for q, i in keys])
    mins = sum((edge[s.id][0][-1] - edge[s.id][0][0]) / 60 for s in qs)
    groups = _groups([s.id for s in qs], video_of)
    rngb = np.random.default_rng(0)
    boots = [np.concatenate([groups[k] for k in rngb.integers(0, len(groups), len(groups))]) for _ in range(1000)]
    cost_all = sum(cloud[s.id][1].sum() * PRICE_IN + cloud[s.id][2].sum() * PRICE_OUT for s in qs)
    print(f"{len(qs)} questions, {len(keys)} ticks, tau = {tau:.3f} (calibrated, 15% share); Gemini says Yes on "
          f"{100 * np.mean(np.concatenate([cloud[s.id][0] for s in qs])):.1f}% of ticks; cost of cloud everywhere ${cost_all:.2f}\n")

    def system(mask, rule, seed=0):
        rows, k, cost, calls = [], 0, 0.0, 0
        rng = random.Random(seed)
        for s in qs:
            n = len(edge[s.id][0])
            m = mask[k:k + n]
            state = np.where(m, cloud[s.id][0], pc[s.id] >= tau)
            cost += float((cloud[s.id][1][m] * PRICE_IN + cloud[s.id][2][m] * PRICE_OUT).sum())
            calls += int(m.sum())
            rows.append(score_q(s, edge[s.id][0], responses(edge[s.id][0], state, rule), rng))
            k += n
        return np.array(rows), cost, calls

    res = {"n_questions": len(qs), "tau": tau, "systems": {}}
    for rule in ("edge", "every6"):
        print(f"response rule: {'crossing' if rule == 'edge' else 'every 6 s while yes'}")
        base = {}
        for name, mask in (("edge only", np.zeros(len(keys), bool)), ("cloud everywhere", np.ones(len(keys), bool))):
            rows, cost, calls = system(mask, rule)
            sm = summarize(rows, mins)
            base[name] = rows
            res["systems"][f"{rule} | {name}"] = {**sm, "cloud_calls": calls, "cost": cost}
            print(f"  {name:22s} recall {sm['recall']:5.1f}%  precision {sm['precision']:5.1f}%  FA {sm['fa_per_min']:.2f}/min  F1 {sm['f1']:5.1f} (timer {sm['timer_f1']:5.1f}, gap {sm['gap']:+5.1f})  calls {calls:6d}  ${cost:.2f}")
        for B in BUDGETS:
            n = int(round(B * len(keys)))
            mask = np.zeros(len(keys), bool)
            mask[np.argsort(dist, kind="stable")[:n]] = True
            rows, cost, calls = system(mask, rule)
            sm = summarize(rows, mins)
            rand = []
            for seed in range(20):
                mk = np.zeros(len(keys), bool)
                mk[np.random.default_rng(seed).choice(len(keys), n, replace=False)] = True
                rand.append(system(mk, rule, seed)[0])
            rr = np.mean(rand, axis=0)
            sr = summarize(rr, mins)

            def g(rw, idx):
                x = rw[idx]
                return 100 * (f1_of(x[:, 0].mean(), x[:, 1].mean()) - f1_of(x[:, 2].mean(), x[:, 3].mean())), 100 * x[:, 4].sum() / x[:, 5].sum()

            dg = [g(rows, b)[0] - g(rr, b)[0] for b in boots]
            dr = [g(rows, b)[1] - g(rr, b)[1] for b in boots]
            res["systems"][f"{rule} | routed {B}"] = {**sm, "cloud_calls": calls, "cost": cost, "random": sr,
                                                      "gap_minus_random_ci": np.percentile(dg, [2.5, 97.5]).tolist(),
                                                      "recall_minus_random_ci": np.percentile(dr, [2.5, 97.5]).tolist()}
            lo, hi = np.percentile(dg, [2.5, 97.5])
            l2, h2 = np.percentile(dr, [2.5, 97.5])
            print(f"  routed {int(B * 100):>2d}%             recall {sm['recall']:5.1f}%  precision {sm['precision']:5.1f}%  FA {sm['fa_per_min']:.2f}/min  F1 {sm['f1']:5.1f} (gap {sm['gap']:+5.1f})  calls {calls:6d}  ${cost:.2f}"
                  f"  | random: recall {sr['recall']:5.1f}% gap {sr['gap']:+5.1f}  -> gap minus random {sm['gap'] - sr['gap']:+5.1f} [{lo:+5.1f},{hi:+5.1f}], recall minus random {sm['recall'] - sr['recall']:+5.1f} [{l2:+5.1f},{h2:+5.1f}]")
    out = Path("results/cache/rescore/d27_chain_" + Path(a.samples).stem + ".json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
