"""Find out why decisions are slow. Prints where the model lives, how much GPU memory it uses, and splits the time
per decision into input preparation (frames -> tokens) and the model forward pass, at three frame sizes.

    python scripts/diagnose_speed.py --set edge.model_id=Qwen/Qwen2.5-VL-3B-Instruct
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.edge import build_edge_model, build_speak_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--repeats", type=int, default=4)
    a = ap.parse_args()
    over = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.set)}
    over.setdefault("edge.backend", "hf")
    cfg = load_config(None, over)

    import torch
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"gpu: {p.name}, {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free before loading")

    t0 = time.time()
    edge = build_edge_model(cfg["edge"])
    print(f"loaded in {time.time() - t0:.0f} s; model device: {edge.model.device}; dtype: {next(edge.model.parameters()).dtype}")
    devs = {str(p.device) for p in edge.model.parameters()}
    print(f"parameters live on: {sorted(devs)}   <- should be only one cuda device")
    if torch.cuda.is_available():
        print(f"gpu memory after load: allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompt = build_speak_prompt("Tell me when the person finishes cutting the vegetables.", n_frames=4, window_s=2.0)
    rng = np.random.default_rng(0)
    print(f"\n{'long side':>9s} {'tokens':>7s} {'prep ms':>8s} {'forward ms':>11s} {'total ms':>9s} {'peak GB':>8s}")
    for side in (224, 336, 448):
        h = int(side * 9 / 16)
        frames = [rng.integers(0, 255, (h, side, 3), dtype=np.uint8) for _ in range(4)]
        rows = []
        for i in range(a.repeats + 1):               # first repeat is warm-up, not reported
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            inputs = edge._inputs(prompt, frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                edge.model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            if i:
                peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                rows.append(((t2 - t1) * 1000, (t3 - t2) * 1000, peak, inputs["input_ids"].shape[1]))
        prep, fwd, peak, ntok = (float(np.median([r[k] for r in rows])) for k in range(4))
        print(f"{side:9d} {int(ntok):7d} {prep:8.0f} {fwd:11.0f} {prep + fwd:9.0f} {peak:8.2f}")
    print("\nIf 'forward' is large and peak GB is near your GPU size, memory is spilling: shrink the frames.")
    print("If 'prep' is large, the image preprocessing is the bottleneck, not the model.")


if __name__ == "__main__":
    main()
