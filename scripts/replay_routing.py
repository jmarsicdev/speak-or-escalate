"""Row 6 on stored scores: the edge decider runs at every tick and the cloud model is consulted only where a routing rule sends a tick.

    python scripts/replay_routing.py --edge results/cache/decider/<edge dir> --cloud results/cache/decider/<cloud dir> \
        --samples data/processed/ovo_far_val.jsonl

Does it matter WHICH moments go to the cloud model? Two rules are given the same number of cloud calls:

  confidence   edge p(Yes) in the top `--hi-frac` of its own moments   -> the edge responds
               edge p(Yes) in the next `--band-frac` (the unsure band) -> the cloud decides (responds if its score >= --cloud-tau)
               below that                                             -> silent
  random       the same number of moments, drawn at random from all moments, are decided by the cloud;
               every other moment follows the edge alone (respond if in the top `--hi-frac`)

References with no cloud calls: the edge alone on its top `--hi-frac`, and the edge alone responding on the whole of top + band.
Reference with a cloud call at every moment: the cloud alone. Each row is also read against chance: the same number of responses in each
question placed at random moments. Responses within `--cooldown` seconds of the previous kept one are dropped, as in replay_decider.py.

The bands are fractions of the edge model's own moments because base Yes-rates differ widely between models (RATIONALE 9a, threshold
rescoring); ties in the edge scores are broken by a fixed random order so the band sizes are exact.
Only questions and moments present in both caches are used, so a cloud cache that covers part of a split gives a result on that part.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.decider_cache import read_rows  # noqa: E402
from egogate.eval.bootstrap import paired_diff  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.metrics import aggregate, score_question  # noqa: E402
from egogate.replay import p_yes  # noqa: E402


def load_pairs(edge_dir: Path, cloud_dir: Path, samples):
    """Per question: times, edge p, cloud p, on the moments both caches have."""
    out = {}
    for s in samples:
        e = {round(r["t"], 6): p_yes(r) for r in read_rows(edge_dir / f"{s.id}.jsonl")}
        c = {round(r["t"], 6): p_yes(r) for r in read_rows(cloud_dir / f"{s.id}.jsonl")}
        ts = sorted(set(e) & set(c))
        if ts:
            out[s.id] = (np.array(ts), np.array([e[t] for t in ts]), np.array([c[t] for t in ts]))
    return out


def keep_with_cooldown(times, cooldown):
    out = []
    for t in sorted(times):
        if not out or t - out[-1] >= max(cooldown, 1e-9):
            out.append(float(t))
    return out


def rows_for(samples, respond: dict, cooldown: float, tol: float):
    """respond[id] = boolean array over that question's moments -> one scored row per question."""
    rows = []
    for s in samples:
        t, flags = respond[s.id]
        q = score_question(keep_with_cooldown(t[flags], cooldown), [e.t for e in s.events], tol)
        rows.append({"id": s.id, "task": s.task, "recall": q.recall, "precision": q.precision, "n_det": q.n_det})
    return rows


def f1(rows):
    return 100 * aggregate(rows)["avg_f1"]


def f1_over_seeds(per_q):
    """per_q: for each question, a list (one per seed) of scored rows -> mean F1 over the seeds."""
    n = len(per_q[0])
    return float(np.mean([f1([q[k] for q in per_q]) for k in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", required=True, help="decider cache of the edge model (scored at every moment)")
    ap.add_argument("--cloud", required=True, help="decider cache of the cloud model")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--hi-frac", type=float, default=0.10, help="share of the edge model's moments it answers itself")
    ap.add_argument("--band-frac", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30], help="share of moments in the unsure band (= cloud calls)")
    ap.add_argument("--cloud-tau", type=float, default=0.5)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    a = ap.parse_args()

    samples = load_jsonl(a.samples)
    data = load_pairs(Path(a.edge), Path(a.cloud), samples)
    samples = [s for s in samples if s.id in data]
    if not samples:
        raise SystemExit("no question has scores in both caches")
    ids = [s.id for s in samples]
    video_of = {s.id: str(s.video) for s in samples}
    n_by = [len(data[i][0]) for i in ids]
    off = np.concatenate([[0], np.cumsum(n_by)])
    n_all = int(off[-1])
    edge_p = np.concatenate([data[i][1] for i in ids])
    cloud_yes = np.concatenate([data[i][2] for i in ids]) >= a.cloud_tau
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(n_all), -edge_p))          # most confident first, ties in a fixed random order
    rank = np.empty(n_all, int)
    rank[order] = np.arange(n_all)

    def split(flags):
        return {i: (data[i][0], flags[off[k]:off[k + 1]]) for k, i in enumerate(ids)}

    def scored(flags):
        return rows_for(samples, split(flags), a.cooldown, a.tol)

    def chance(rows):
        """Same number of responses in each question, at random moments."""
        per_seed = []
        for k in range(a.seeds):
            r = np.random.default_rng(1000 + k)
            flags = np.zeros(n_all, bool)
            for j, row in enumerate(rows):
                pick = r.choice(n_by[j], size=min(row["n_det"], n_by[j]), replace=False)
                flags[off[j] + pick] = True
            per_seed.append(f1(rows_for(samples, split(flags), 0.0, a.tol)))
        return float(np.mean(per_seed))

    n_hi = int(round(a.hi_frac * n_all))
    hi = rank < n_hi
    print(f"{len(samples)} questions, {n_all} moments; edge answers its top {a.hi_frac:.0%} ({n_hi} moments); "
          f"cloud says Yes on {cloud_yes.mean():.0%} of all moments; cooldown {a.cooldown:g} s")
    print(f"{'system':<34}{'cloud calls':>12}{'responses':>11}{'F1':>7}{'chance':>8}{'F1-chance':>11}")

    table = []

    def show(name, calls, rows, f1_value=None):
        v = f1(rows) if f1_value is None else f1_value
        c = chance(rows)
        n_resp = sum(r["n_det"] for r in rows)
        print(f"{name:<34}{calls:>12}{n_resp:>11}{v:>7.1f}{c:>8.1f}{v - c:>+11.1f}")
        table.append({"system": name, "cloud_calls": int(calls), "responses": int(n_resp), "f1": v, "chance_f1": c})

    show("edge alone, top band", 0, scored(hi))
    show("cloud alone, every moment", n_all, scored(cloud_yes))
    for bf in a.band_frac:
        n_band = int(round(bf * n_all))
        band = (rank >= n_hi) & (rank < n_hi + n_band)
        print(f"-- unsure band = next {bf:.0%} of moments ({n_band} cloud calls)")
        show("  edge alone, top + band", 0, scored(hi | band))
        conf_rows = scored(hi | (band & cloud_yes))
        show("  confidence routing", n_band, conf_rows)
        rand_rows = []                                   # one list of question rows per seed
        for k in range(a.seeds):
            pick = np.zeros(n_all, bool)
            pick[np.random.default_rng(k).choice(n_all, size=n_band, replace=False)] = True
            rand_rows.append(scored(np.where(pick, cloud_yes, hi)))
        rand_f1 = float(np.mean([f1(r) for r in rand_rows]))
        show(f"  random routing (mean of {a.seeds})", n_band, rand_rows[0], f1_value=rand_f1)
        per_q_rand = [[rand_rows[k][j] for k in range(a.seeds)] for j in range(len(samples))]
        per_q_conf = [[conf_rows[j]] for j in range(len(samples))]
        d = paired_diff(per_q_conf, per_q_rand, f1_over_seeds, ids, video_of, n=500)
        mark = "  <- excludes 0" if d["excludes_zero"] else ""
        print(f"  confidence - random: {d['diff']:+.1f} F1, 95% interval [{d['lo']:+.1f}, {d['hi']:+.1f}] over {d['n_groups']} videos{mark}")
        table[-1]["confidence_minus_random"] = d

    print("\nchance = the same number of responses in each question at random moments (mean of the seeds). An interval that includes 0 means "
          "the routing rule has not been shown to matter.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"args": vars(a), "n_questions": len(samples), "n_moments": n_all, "rows": table}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
