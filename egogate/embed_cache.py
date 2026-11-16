"""Stage 1a: embed every matcher window of a video and store the similarity to each cue.

For each question we store, under <cache_dir>/matcher/<model_tag>__<key>/:
    <id>.json      {"t": [...], "sims": [[cue0, cue1, cue2], ...], "cues": [...]}  small; committed so results can be read
    <id>.npy       the window embeddings (T x 1536, float16); large, local only, used to recompute similarities
                   for other cues and for Em-Garde's embedding-based duplicate suppression
The matcher steps every `stride_s` (0.5 s); each step embeds the last `window_s` (2 s) at `fps` (2) as 4 frames.
The folder name is keyed on the model, precision-relevant settings and window geometry (like the decider cache).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .embed import Embedder
from .eval.datasets import Sample
from .eval.ovo import decision_times
from .frames import stream_windows


def make_meta(cfg: dict) -> dict:
    m = cfg["matcher"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return {"embedder": m["embedder"], "model_id": m["model_id"], "attn": m.get("attn"), "stride_s": m["stride_s"],
            "window_s": m["window_s"], "fps": m["fps"], "max_pixels": m["max_pixels"], "min_pixels": m.get("min_pixels"), "window_prompt": "newline-suffix",
            "code_commit": commit}


def cache_dir_for(cfg: dict, meta: dict) -> Path:
    keyed = {k: v for k, v in meta.items() if k != "code_commit"}
    key = hashlib.sha1(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:8]
    return Path(cfg["matcher"]["cache_dir"]) / "matcher" / f"{meta['model_id'].split('/')[-1]}__{key}"


def load_sims(path: Path) -> tuple[np.ndarray, np.ndarray]:
    r = json.load(open(path))
    return np.array(r["t"]), np.array(r["sims"])


def run_sample(sample: Sample, embedder: Embedder, cues: list[str], cfg: dict, out_dir: Path, video_root: Path) -> dict:
    m = cfg["matcher"]
    final = out_dir / f"{sample.id}.json"
    if final.exists():
        return {"id": sample.id, "status": "cached", "n": len(json.load(open(final))["t"])}
    ticks = decision_times(sample, m["stride_s"])
    cue_vecs = embedder.embed_text(cues)               # (n_cues, d), normalised
    ts, vecs, ms = [], [], []
    for w in stream_windows(video_root / sample.video, ticks, m["window_s"], m["fps"], None, m["max_pixels"]):
        t0 = time.perf_counter()
        vecs.append(embedder.embed_window(w))
        ms.append((time.perf_counter() - t0) * 1000)
        ts.append(round(w.t_end, 6))
    V = np.stack(vecs) if vecs else np.zeros((0, cue_vecs.shape[1]), np.float32)
    sims = V @ cue_vecs.T if len(V) else np.zeros((0, len(cues)))
    np.save(out_dir / f"{sample.id}.npy", V.astype(np.float16))
    json.dump({"t": ts, "sims": np.round(sims, 5).tolist(), "cues": cues, "ms_median": float(np.median(ms)) if ms else None},
              open(final, "w"))
    return {"id": sample.id, "status": "ran", "n": len(ts), "skipped_ticks": len(ticks) - len(ts),
            "ms_median": float(np.median(ms)) if ms else None}
