"""Run one experiment config over a dataset and write results/<name>/summary.json.

    python scripts/run_experiment.py configs/experiments/band_confidence.yaml
    python scripts/run_experiment.py configs/experiments/random_trigger.yaml --set data.toy=true
    python scripts/run_experiment.py configs/experiments/band_confidence.yaml --set policy.hi=0.8 --set matcher.threshold=0.06

Sweeps are just loops over --set values; see scripts/sweep_thresholds.py.
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
        n = int(win * fps)
        yield Window(frames=[np.zeros((1, 1, 3), dtype=np.uint8)] * n, t_start=t - win, t_end=t)
        t += stride


def build(cfg: dict, cues: list[str]) -> Components:
    m = cfg["matcher"]
    embedder = RandomEmbedder(seed=cfg["seed"]) if m["embedder"] == "random" else OpsMMEmbedder(m["model_id"])
    matcher = Matcher(embedder, cues, MatcherConfig(m["threshold"], m["mode"], m["aggregate"], m["cooldown_s"]))
    edge = build_edge_model(cfg["edge"])
    cloud = build_cloud(cfg["cloud"])
    ev = cfg["evidence"]
    evidence = EvidenceMemory(edge, embedder, ev["every_s"], per_clip=ev["per_clip"]) if ev["enabled"] else None
    return Components(matcher=matcher, edge=edge, cloud=cloud, policy=build_policy(cfg["policy"]),
                      evidence=evidence, cloud_payload=cfg["cloud"]["payload"], top_k_evidence=ev["top_k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_set(args.set))
    name = cfg.get("name", Path(args.config).stem if args.config else "default")
    out_dir = Path(cfg["output"]["dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = toy_samples() if cfg["data"]["toy"] else load_jsonl(cfg["data"]["jsonl"], cfg["data"]["limit"])
    cloud_for_cues = build_cloud(cfg["cloud"])

    per_sample, t0 = [], time.time()
    for s in samples:
        cue_cfg = cfg["cues"]
        if cue_cfg["backend"] == "manual":
            cues = list(cue_cfg["manual"]) or [s.query]
        else:
            model = cloud_for_cues if cue_cfg["backend"] == "cloud" else build_edge_model(cfg["edge"])
            cues = CueParser(cue_cfg["backend"], cue_cfg["n_cues"], model=model).parse(s.query)

        comps = build(cfg, cues)
        if cfg["data"]["toy"]:
            duration = max(e.t for e in s.events) + 10
            windows = toy_windows(duration, cfg)
        else:
            v = cfg["video"]
            windows = sliding_windows(iter_frames(s.video, v["fps"]), v["window_s"], v["stride_s"], v["fps"])

        decisions = run_stream(windows, s.query, comps, log_all=True)
        summ = summarize_run(decisions, s.events, cfg["eval"]["tol_s"])
        summ.update({"id": s.id, "task": s.task, "cues": cues})
        per_sample.append(summ)
        with open(out_dir / f"{s.id}.jsonl", "w") as f:
            for d in decisions:
                f.write(json.dumps({"t": d.t, "action": d.action.value, "p_speak": d.p_speak,
                                    "answer": d.answer, "source": d.source, "cost": d.cost, **d.meta}) + "\n")
        calib = calibration_table(decisions, s.events, cfg["eval"]["tol_s"], cfg["eval"]["calib_bins"])
        with open(out_dir / f"{s.id}.calib.json", "w") as f:
            json.dump(calib, f, indent=1)
        print(f"[{name}] {s.id}: f1={summ['f1']:.3f} rec={summ['recall']:.2f} pre={summ['precision']:.2f} "
              f"triggers={summ['n_triggers']} cloud={summ['n_cloud_calls']}")

    keys = ["recall", "precision", "f1", "n_triggers", "n_cloud_calls", "n_responses"]
    agg = {k: float(np.mean([r[k] for r in per_sample])) for k in keys} if per_sample else {}
    agg.update({"name": name, "n_samples": len(per_sample), "wall_s": time.time() - t0, "config": cfg})
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"aggregate": agg, "per_sample": per_sample}, f, indent=1, default=str)
    print(f"[{name}] mean f1={agg.get('f1', 0):.3f}  mean cloud calls={agg.get('n_cloud_calls', 0):.1f}  -> {out_dir}")


if __name__ == "__main__":
    main()
