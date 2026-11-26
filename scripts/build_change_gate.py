"""The question-aware "what changed" sentence as a gate in front of a decider, built from stored runs (CPU, no model calls).

The 8B's question-aware change sentence (`run_decider_history.py --variant change_q`) says "Nothing relevant changed" on about half the
moments, and those are near an event far less often (8% against 24%), but used as prompt text it only switches the decider on or off.
Here it is used as a separate signal instead: a moment the sentence calls a relevant change ranks above every moment it does not, and within
each group the decider's own p(Yes) (from its stored run at the same tick) orders them. Score = (gate + p(Yes)) / 2, stored as a decider
cache (lp_yes = log score, lp_no = log(1 - score)), so compare_deciders.py, rescore_thresholds.py and the viewer read it like any decider.
"gate only" uses the sentence alone (score 0.25 or 0.75, all ties within a group).

    python scripts/build_change_gate.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

D = Path("results/cache/decider")
GATE = "qwen3-vl-8b-instruct__history_change_q__a0c6603e"
DECIDERS = {"2B": "Qwen3-VL-2B-Instruct__query_only__1a7df950", "8B": "qwen3-vl-8b-instruct__query_only__d0bdbc25",
            "235B": "qwen3-vl-235b-a22b-instruct__query_only__b1bb72f9", "gate only": None}


def rows(path):
    return {round(r["t"], 3): r for r in (json.loads(x) for x in open(path) if x.strip())}


def main():
    for name, dec in DECIDERS.items():
        out = D / f"gate_changeq__{name.replace(' ', '_')}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump({"derived": True, "gate": GATE, "decider": dec, "score": "(gate + p_yes) / 2" if dec else "0.25 + 0.5 * gate"},
                  open(out / "_meta.json", "w"), indent=1)
        n_q = n = 0
        for f in sorted((D / GATE).glob("ovo_*.jsonl")):
            if dec and not (D / dec / f.name).exists():
                continue
            g = rows(f)
            d = rows(D / dec / f.name) if dec else None
            with open(out / f.name, "w") as h:
                for t, r in sorted(g.items()):
                    gate = 0.0 if r["change"].lower().startswith("nothing") else 1.0
                    if dec:
                        if t not in d:
                            continue
                        p = 1 / (1 + math.exp(-max(-50, min(50, d[t]["lp_yes"] - d[t]["lp_no"]))))
                        sc = (gate + p) / 2
                    else:
                        sc = 0.25 + 0.5 * gate
                    sc = min(max(sc, 1e-9), 1 - 1e-9)
                    h.write(json.dumps({"t": t, "lp_yes": math.log(sc), "lp_no": math.log(1 - sc), "gate": int(gate)}) + "\n")
                    n += 1
            n_q += 1
        print(f"{out.name}: {n_q} questions, {n} moments")


if __name__ == "__main__":
    main()
