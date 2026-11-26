"""Stage 0a / 0a' (D24): cues written by the 8B model, from the question alone or from the question plus a few frames of the scene.

  --variant text     the existing cue prompt (egogate/cues.py, CUE_PROMPT), question only: the 8B in place of the 2B writer (D18)
  --variant frames5  the same prompt plus 5 frames at 1 fps from the 5 s before the question (clue reveal: the 5 s before ask time;
                     counting and tutorials, which start at 0 s: the first 5 s), with an added instruction to make the cues specific to
                     this scene but to describe what will look DIFFERENT when the answer is due, not the scene as it already is
                     (scene-level cues raised the similarity everywhere and did not help, 9a "Shorter windows and scene-level cues")

Causality: for counting and tutorials the frames come from the first 5 s, so moments before 5 s are dropped from every comparison that
uses these cues (both variants), to compare on the same moments. Writes results/cache/cues/qwen3-vl-8b-instruct__<variant>/cues.jsonl
in the usual format ({"id", "query", "cues", "raw"}). Resumable.

    python scripts/write_cues_8b.py --samples data/processed/ovo_far_val.jsonl --variant frames5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cues import CUE_PROMPT, parse_cue_json  # noqa: E402
from egogate.edge import APIEdgeVLM  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.frames import stream_windows  # noqa: E402

FRAMES_NOTE = """
You are also shown {k} frames from the video, one per second, from the {where}. Use them to make the cues specific to THIS scene (the
actual people, objects and setting you can see). But describe what will look DIFFERENT at the moment the user should be notified, not
the scene as it already looks in these frames."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--variant", choices=["text", "frames5"], required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct")
    a = ap.parse_args()
    cfg = load_config(None, {})
    root = Path(cfg["data"]["video_root"])
    out = Path("results/cache/cues") / f"{a.model.split('/')[-1]}__{a.variant}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"model_id": a.model, "variant": a.variant, "n_cues": 3, "prompt": CUE_PROMPT + (FRAMES_NOTE if a.variant == "frames5" else "")},
              open(out / "_meta.json", "w"), indent=1)
    f_out = out / "cues.jsonl"
    have = {json.loads(x)["id"] for x in open(f_out, encoding="utf-8")} if f_out.exists() else set()
    edge = APIEdgeVLM(a.model)
    with open(f_out, "a", encoding="utf-8") as f:
        for s in load_jsonl(a.samples):
            if s.id in have:
                continue
            prompt, frames = CUE_PROMPT.format(query=s.query, n=3), None
            if a.variant == "frames5":
                t_end = max(s.start_s, 4.0)   # clue reveal: up to the ask time; others (start 0): the first 5 s
                frames = next(stream_windows(root / s.video, [t_end], 5.0, 1.0, 448)).frames
                where = "5 seconds before the question was asked" if s.start_s >= 4.0 else "first 5 seconds of the stream"
                prompt += FRAMES_NOTE.format(k=len(frames), where=where)
            raw = edge.complete(prompt, frames, max_new_tokens=160)
            cues = (parse_cue_json(raw) or [s.query])[:3]
            f.write(json.dumps({"id": s.id, "query": s.query, "cues": cues, "raw": raw}) + "\n")
            f.flush()
            print(f"{s.id}: {cues}", flush=True)
    print(f"{edge.n_calls} calls -> {f_out}")


if __name__ == "__main__":
    main()
