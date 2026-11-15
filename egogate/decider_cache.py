"""Stage 1b: run the decider over a video and store its raw Yes/No scores.

For every decision tick it stores log P("Yes") and log P("No") (over the full vocabulary), so any
threshold, temperature or cooldown can be tried later without running the model again (docs/RATIONALE.md D9, D12).

Cache layout:
    <cache_dir>/decider/<model_tag>__<variant>__<key>/
        _meta.json            everything that determines the numbers (model, dtype, prompt hash, window, ...)
        <sample_id>.jsonl     one row per tick: {"t", "lp_yes", "lp_no", "ms", "n_frames"}

The folder name carries a hash of the meta, so a changed model, precision, prompt or window starts a fresh
cache instead of silently reusing old numbers. Runs resume: finished videos are skipped, and an interrupted
video continues from its last stored tick.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .edge import SPEAK_PROMPT_VERSION, EdgeVLM, prompt_fingerprint
from .eval.datasets import Sample
from .eval.ovo import decision_times
from .frames import stream_windows


def make_meta(cfg: dict, model_id: str) -> dict:
    d, e = cfg["decider"], cfg["edge"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return {
        "model_id": model_id, "dtype": e.get("dtype"), "quant": e.get("quant"), "max_pixels": e.get("max_pixels"),
        "variant": d["variant"], "prompt_version": SPEAK_PROMPT_VERSION, "prompt_hash": prompt_fingerprint(),
        "stride_s": d["stride_s"], "window_s": d["window_s"], "fps": d["fps"], "max_side": d["max_side"],
        "code_commit": commit,   # recorded for provenance; deliberately NOT part of the key
    }


def cache_dir_for(cfg: dict, meta: dict) -> Path:
    keyed = {k: v for k, v in meta.items() if k != "code_commit"}
    key = hashlib.sha1(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:8]
    tag = meta["model_id"].split("/")[-1]
    return Path(cfg["decider"]["cache_dir"]) / "decider" / f"{tag}__{meta['variant']}__{key}"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_sample(sample: Sample, edge: EdgeVLM, cfg: dict, out_dir: Path, video_root: Path,
               cues: list[str] | None = None, evidence_fn=None, log=print) -> dict:
    """Score every decision tick of one question. Returns a small status dict."""
    d = cfg["decider"]
    final, part = out_dir / f"{sample.id}.jsonl", out_dir / f"{sample.id}.jsonl.part"
    if final.exists():
        return {"id": sample.id, "status": "cached", "n": len(read_rows(final))}

    ticks = decision_times(sample, d["stride_s"])
    done = {round(r["t"], 6) for r in read_rows(part)}
    n_new = 0
    with open(part, "a") as f:
        for w in stream_windows(video_root / sample.video, ticks, d["window_s"], d["fps"], d["max_side"]):
            if round(w.t_end, 6) in done:
                continue
            use_cues = cues if d["variant"] in ("query_cues", "query_evidence") else None
            ev = evidence_fn(sample, w) if (evidence_fn and d["variant"] == "query_evidence") else None
            t0 = time.perf_counter()
            out = edge.speak_now(sample.query, w.frames, ev, use_cues, window_s=d["window_s"])
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"t": round(w.t_end, 6), "lp_yes": out.raw["lp_yes"], "lp_no": out.raw["lp_no"],
                                "ms": round(ms, 1), "n_frames": len(w.frames)}) + "\n")
            f.flush()
            n_new += 1
    rows = read_rows(part)
    part.rename(final)
    skipped = len(ticks) - len(rows)
    return {"id": sample.id, "status": "ran", "n": len(rows), "new": n_new,
            "skipped_ticks": skipped}   # ticks the video was too short for
