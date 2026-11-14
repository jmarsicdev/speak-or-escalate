"""Run the baseline pipeline over a dataset and write results/<name>/.

    python scripts/run_baseline.py                                   # configs/default.yaml
    python scripts/run_baseline.py --set data.toy=true               # dry run, mocks only
    python scripts/run_baseline.py --set edge.backend=hf --set matcher.embedder=opsmm --set cloud.backend=openai

Outputs:
    results/<name>/summary.json     aggregate + per-sample timing metrics, full resolved config
    results/<name>/<id>.jsonl       one row per window (sims, deltas, p_speak, action, answer)
    results/<name>/<id>.calib.json  p_speak bins vs. fraction of triggers that were real response times
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.cloud import build_cloud  # noqa: E402
from egogate.config import load_config  # noqa: E402
from egogate.cues import CueParser  # noqa: E402
from egogate.edge import build_edge_model  # noqa: E402
from egogate.embed import Matcher, MatcherConfig, OpsMMEmbedder, RandomEmbedder  # noqa: E402
from egogate.eval.datasets import load_jsonl, toy_samples  # noqa: E402
from egogate.eval.metrics import calibration_table, summarize_run  # noqa: E402
from egogate.evidence import EvidenceMemory  # noqa: E402
from egogate.pipeline import Components, run_stream  # noqa: E402
from egogate.router import build_policy  # noqa: E402
from egogate.types import Window  # noqa: E402
from egogate.video import iter_frames, sliding_windows  # noqa: E402


def parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def toy_windows(duration_s: float, cfg: dict):
    """Fake 1x1 frames so the mock pipeline has something to iterate over."""
    fps, win, stride = cfg["video"]["fps"], cfg["video"]["window_s"], cfg["video"]["stride_s"]
    t = win
    while t <= duration_s:
        yield Window(frames=[np.zeros((1, 1, 3), dtype=np.uint8)] * int(win * fps), t_start=t - win, t_end=t)
        t += stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None, help="optional YAML overlay on configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_set(args.set))
    name = cfg["name"]
    out_dir = Path(cfg["output"]["dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = toy_samples() if cfg["data"]["toy"] else load_jsonl(cfg["data"]["jsonl"], cfg["data"]["limit"])

    # Models are loaded once and shared across samples.
    m = cfg["matcher"]
    embedder = RandomEmbedder(seed=cfg["seed"]) if m["embedder"] == "random" else OpsMMEmbedder(m["model_id"])
    edge = build_edge_model(cfg["edge"])
    cloud = build_cloud(cfg["cloud"])
    policy = build_policy(cfg["policy"])
    cue_parser = CueParser("edge", cfg["cues"]["n_cues"], model=edge)
    ev = cfg["evidence"]

    per_sample, t0 = [], time.time()
    for s in samples:
        cues = cue_parser.parse(s.query)
        comps = Components(
            matcher=Matcher(embedder, cues, s.query, MatcherConfig(m["threshold"], m["cooldown_s"])),
            edge=edge, cloud=cloud, policy=policy,
            evidence=EvidenceMemory(edge, embedder, ev["every_s"], per_clip=ev["per_clip"]),
            top_k_evidence=ev["top_k"],
        )
        if cfg["data"]["toy"]:
            windows = toy_windows(max(e.t for e in s.events) + 10, cfg)
        else:
            v = cfg["video"]
            windows = sliding_windows(iter_frames(s.video, v["fps"]), v["window_s"], v["stride_s"], v["fps"])

        decisions = run_stream(windows, s.query, cues, comps, log_all=True)
        summ = summarize_run(decisions, s.events, cfg["eval"]["tol_s"])
        summ.update({"id": s.id, "task": s.task, "cues": cues})
        per_sample.append(summ)

        with open(out_dir / f"{s.id}.jsonl", "w") as f:
            for d in decisions:
                f.write(json.dumps({"t": d.t, "action": d.action.value, "p_speak": d.p_speak, "answer": d.answer,
                                    "source": d.source, "cost": d.cost, **d.meta}) + "\n")
        with open(out_dir / f"{s.id}.calib.json", "w") as f:
            json.dump(calibration_table(decisions, s.events, cfg["eval"]["tol_s"], cfg["eval"]["calib_bins"]), f, indent=1)
        print(f"[{name}] {s.id}: f1={summ['f1']:.3f} rec={summ['recall']:.2f} pre={summ['precision']:.2f} "
              f"triggers={summ['n_triggers']} cloud={summ['n_cloud_calls']}")

    keys = ["recall", "precision", "f1", "n_triggers", "n_cloud_calls", "n_responses", "n_false_positives"]
    agg = {k: float(np.mean([r[k] for r in per_sample])) for k in keys} if per_sample else {}
    agg.update({"name": name, "n_samples": len(per_sample), "total_cloud_calls": cloud.n_calls,
                "wall_s": time.time() - t0, "config": cfg})
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"aggregate": agg, "per_sample": per_sample}, f, indent=1, default=str)
    print(f"[{name}] mean f1={agg.get('f1', 0):.3f}  mean cloud calls/sample={agg.get('n_cloud_calls', 0):.1f}  -> {out_dir}")


if __name__ == "__main__":
    main()
