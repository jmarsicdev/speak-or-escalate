"""Build viewer/rescore.html: stored decider runs, model-specific thresholds, responses and the matcher, with the video frames of each moment.

    python scripts/rescore_thresholds.py --samples data/processed/ovo_far_val.jsonl     # the scores the page shows
    python viewer/dump_frames.py viewer/frames                                         # frame thumbnails (gitignored, about 50 MB)
    python viewer/build_viewer.py                                                      # writes viewer/rescore.html

Open viewer/rescore.html in a browser; it reads the thumbnails from viewer/frames/ next to it (without them the page works, minus the frames).
Val split only; nothing here calls a model.
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from egogate.eval.datasets import load_jsonl
from egogate.cue_sources import oracle_cues
from rescore_thresholds import MODELS

R = Path("results/cache")
res = json.load(open("results/cache/rescore/rescore_thresholds_val.json"))
samples = load_jsonl("data/processed/ovo_far_val.jsonl")
general = {r["id"]: r["cues"] for r in map(json.loads, open(R / "cues/general_oracle__qwen3-vl-8b-instruct.jsonl", encoding="utf-8"))}
qs = []
for s in samples:
    q = {"id": s.id, "task": s.task, "source": s.meta.get("source", ""), "query": s.query, "events": [e.t for e in s.events],
         "oracle": oracle_cues(s), "general": general.get(s.id, []), "p": {}, "chg": {}}
    for name in res["models"]:
        hits = sorted((R / "decider").glob(MODELS[name]))
        f = hits[-1] / f"{s.id}.jsonl" if hits else None
        if f and f.exists():
            rows = sorted((json.loads(x) for x in open(f) if x.strip()), key=lambda r: r["t"])
            q["p"][name] = [[r["t"], round(1 / (1 + math.exp(-max(-50, min(50, r["lp_yes"] - r["lp_no"])))), 4)] for r in rows]
            if "change" in rows[0]:
                q["chg"][name] = {str(r["t"]): r["change"] for r in rows}
    m = R / "matcher/Ops-MM-embedding-v1-2B__875248e9" / f"{s.id}.json"
    if m.exists():
        mm = json.load(open(m))
        q["edge_cues"] = mm["cues"]
        q["mat"] = {"t": mm["t"], "s": [[round(x, 3) for x in r] for r in mm["sims"]]}
    qs.append(q)
html = open(sys.argv[1] if len(sys.argv) > 1 else "viewer/template.html", encoding="utf-8").read().replace("/*RES*/", json.dumps(res)).replace("/*QS*/", json.dumps(qs))
open(sys.argv[2] if len(sys.argv) > 2 else "viewer/rescore.html", "w", encoding="utf-8").write(html)
print("bytes", len(html))
