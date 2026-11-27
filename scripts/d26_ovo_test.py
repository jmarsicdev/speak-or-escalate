"""D26 claims T1 to T3 on the OVO test split (56 questions, untouched until D26), scored exactly as fixed in RATIONALE D26.

  T1  the 8B decider D beats the per-question timer on counting + tutorials (lenient F1 gap at the 15% and 30% shares, crossing rule)
  T2  before/after BA's F1 gap at 15% exceeds D's (paired, video-level interval)
  T3  on tutorials, BA's and the change gate G's within-question AUC exceed D's
Also reported: the 5% share, the change gate's F1, and val numbers for comparison. Output results/cache/rescore/d26_ovo_test.json.

    python scripts/d26_ovo_test.py
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d24_combine import auc_table, f1_table  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402

D = Path("results/cache/decider")
SRC = {"D": "qwen3-vl-8b-instruct__query_only__d0bdbc25", "BA": "qwen3-vl-8b-instruct__signal_beforeafter",
       "G": Path(glob.glob(str(D / "qwen3-vl-8b-instruct__history_gate_q__*"))[0]).name}


def load(samples):
    out = {}
    for s in samples:
        rows = {}
        for k, d in SRC.items():
            f = D / d / f"{s.id}.jsonl"
            if not f.exists():
                break
            rows[k] = {round(r["t"], 3): r for r in (json.loads(x) for x in open(f) if x.strip())}
        else:
            ts = sorted(set.intersection(*[set(v) for v in rows.values()]))
            if len(ts) < 5:
                continue
            p = lambda r: 1 / (1 + math.exp(-max(-50, min(50, r["lp_yes"] - r["lp_no"]))))  # noqa: E731
            out[s.id] = (np.array(ts), {k: np.array([p(rows[k][t]) for t in ts]) for k in rows})
    return out


def main():
    res = {}
    for split in ("test", "val"):
        samples = load_jsonl(f"data/processed/ovo_far_{split}.jsonl")
        video_of = {s.id: str(s.video) for s in samples}
        sig = load(samples)
        scores = {k: (lambda k: lambda t, g: g[k])(k) for k in ("D", "BA", "G")}
        a_all = auc_table(samples, sig, scores, video_of)
        tut = [s for s in samples if s.task == "SSR" and s.id in sig]
        a_tut = auc_table(tut, sig, scores, video_of)["counting + tutorials"]
        f1 = f1_table(samples, sig, scores, video_of, n_boot=2000)
        res[split] = {"auc": a_all, "auc_tutorials": a_tut, "f1": f1, "n_questions": len(sig),
                      "n_count_tut": sum(1 for s in samples if s.id in sig and s.task != "CRR"), "n_tut": len(tut)}
        print(f"\n=== OVO {split}: {len(sig)} questions ({res[split]['n_count_tut']} counting + tutorials, {len(tut)} tutorials)")
        print("  F1 gap over the per-question timer, counting + tutorials, crossing rule, repeats not punished [95%]; vs D [95%]")
        for k in ("D", "BA", "G"):
            print(f"    {k:3s} " + "   ".join(f"{int(float(r) * 100):>2d}%: {x['gap']:+5.1f} [{x['ci'][0]:+5.1f},{x['ci'][1]:+5.1f}]"
                                         + (f" vs D {x['vs_D']:+5.1f} [{x['vs_D_ci'][0]:+5.1f},{x['vs_D_ci'][1]:+5.1f}]" if k != "D" else "") for r, x in f1[k].items()))
        print("  within-question AUC, counting + tutorials / tutorials (difference from D)")
        for k in ("D", "BA", "G"):
            a, t = a_all["counting + tutorials"][k], a_tut[k]
            fmt = lambda v: f" ({v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]{'*' if v['excludes_zero'] else ''})" if v else ""  # noqa: E731
            print(f"    {k:3s} {a['auc']:.3f}{fmt(a['vs_D'])}   tutorials {t['auc']:.3f}{fmt(t['vs_D'])}")
    t = res["test"]
    verdict = {
        "T1": {"15%": t["f1"]["D"]["0.15"]["ci"][0] > 0, "30%": t["f1"]["D"]["0.3"]["ci"][0] > 0},
        "T2": t["f1"]["BA"]["0.15"]["vs_D_ci"][0] > 0,
        "T3": {"BA": bool(t["auc_tutorials"]["BA"]["vs_D"]["excludes_zero"] and t["auc_tutorials"]["BA"]["vs_D"]["diff"] > 0),
               "G": bool(t["auc_tutorials"]["G"]["vs_D"]["excludes_zero"] and t["auc_tutorials"]["G"]["vs_D"]["diff"] > 0)},
    }
    res["verdict"] = verdict
    print(f"\nD26 verdicts on OVO test: {verdict}")
    json.dump(res, open("results/cache/rescore/d26_ovo_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
