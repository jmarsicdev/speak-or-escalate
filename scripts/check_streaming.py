"""Does Em-Garde's streaming shortcut work in this environment, is it equivalent to embedding each window from scratch,
and how much faster is it?

    python scripts/check_streaming.py --video data/raw/ovo/src_videos/<some>.mp4 --steps 12 [--device cuda]

The shortcut needs the library version Em-Garde pins (transformers 4.52.3); on newer versions it crashes, and the script
says so. For each 0.5 s step it embeds the same 4-frame window both ways and prints the cosine similarity between the two
vectors (1.000 means identical) and the time each way takes.
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from egogate.frames import stream_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="OpenSearch-AI/Ops-MM-embedding-v1-2B")
    ap.add_argument("--start", type=float, default=2.0, help="first window ends here (seconds)")
    a = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, device {a.device}")
    spec = importlib.util.spec_from_file_location("ops", ROOT / "third_party/em_garde/ops_mm_embedding_v1.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    m = ops.OpsMMEmbeddingV1(a.model, device=a.device, attn_implementation="sdpa")

    ticks = [a.start + 0.5 * k for k in range(a.steps)]
    wins = list(stream_windows(a.video, ticks, 2.0, 2.0, None, 78400))
    assert len(wins) == a.steps, f"video too short: got {len(wins)} windows"
    pil = lambda w: [Image.fromarray(f) for f in w.frames]  # noqa: E731

    def sync():
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()

    m.reset()
    rows = []
    for k, w in enumerate(wins):
        frames = pil(w)
        t0 = time.perf_counter()
        try:
            s = m.streaming_embed(frames if k == 0 else frames[-1:], frames_length=4)
        except Exception as e:  # noqa: BLE001
            print(f"STREAMING FAILED on transformers {transformers.__version__}: {type(e).__name__}: {e}")
            return
        sync()
        t_stream = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        std = m.embed(images=[frames])
        sync()
        t_std = (time.perf_counter() - t0) * 1000
        cos = float((s.float() @ std.float().T).item())
        rows.append((w.t_end, cos, t_stream, t_std))
        print(f"  t={w.t_end:5.1f}s  cosine(streaming, from-scratch) = {cos:.4f}   streaming {t_stream:7.0f} ms   from scratch {t_std:7.0f} ms")
    steady = rows[1:]
    print(f"\nmin cosine over steps: {min(r[1] for r in rows):.4f}; steady-state (after the first step): "
          f"streaming {statistics.median(r[2] for r in steady):.0f} ms vs from scratch {statistics.median(r[3] for r in steady):.0f} ms "
          f"= {statistics.median(r[3] for r in steady) / statistics.median(r[2] for r in steady):.2f}x")


if __name__ == "__main__":
    main()
