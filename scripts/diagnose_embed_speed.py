"""Where does the time go when the matcher embeds one 2 s window? Splits a step into
  (a) preparing the frames and prompt (CPU image processing),
  (b) the model forward pass (GPU),
and compares two ways of sizing the frames: Em-Garde's wrapper, which resizes every frame UP to at least 256 visual tokens
(a 1,085-token prompt for 4 frames), against leaving the frames at their decoded size (about 91 tokens each). Prints the time
of each and the cosine between the two vectors, so we can see what the upscaling costs and whether it matters.

    python scripts/diagnose_embed_speed.py --device cuda
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="OpenSearch-AI/Ops-MM-embedding-v1-2B")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--video", default=None, help="optional: use real frames from this video")
    a = ap.parse_args()

    import torch
    from PIL import Image
    spec = importlib.util.spec_from_file_location("ops", ROOT / "third_party/em_garde/ops_mm_embedding_v1.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    from egogate.embed import _AppendNewline

    m = ops.OpsMMEmbeddingV1(a.model, device=a.device, attn_implementation="sdpa")
    print(f"image processor: {type(m.processor.image_processor).__name__}")
    if a.device.startswith("cuda"):
        print(f"gpu memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if a.video:
        from egogate.frames import stream_windows
        w = next(iter(stream_windows(a.video, [2.0], 2.0, 2.0, None, 78400)))
        frames = [Image.fromarray(f) for f in w.frames]
    else:
        rng = np.random.default_rng(0)
        frames = [Image.fromarray(rng.integers(0, 255, (196, 364, 3), dtype=np.uint8)) for _ in range(4)]
    print(f"window: {len(frames)} frames of {frames[0].size[0]} x {frames[0].size[1]}")

    def sync():
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()

    def one(min_pixels):
        orig = m.processor
        m.processor = _AppendNewline(orig)
        sync(); t0 = time.perf_counter()
        processed = [ops.fetch_image(f, min_pixels=min_pixels) for f in frames]   # the wrapper's resize step, with a chosen minimum
        t1 = time.perf_counter()
        msg = ("<|im_start|>system\n" + m.default_instruction + "<|im_end|>\n<|im_start|>user\n"
               + "<|vision_start|><|image_pad|><|vision_end|>" * len(processed) + "<|im_end|>\n<|im_start|>assistant\n<|endoftext|>")
        inputs = m.processor(text=[msg], images=[processed], padding=True, truncation=True, max_length=m.max_length, return_tensors="pt")
        inputs = {k: v.to(m.device) for k, v in inputs.items()}
        sync(); t2 = time.perf_counter()
        with torch.inference_mode():
            vec = m.encode_input(inputs)
        sync(); t3 = time.perf_counter()
        m.processor = orig
        return vec, ((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, inputs["input_ids"].shape[1])

    results = {}
    for name, minpx in (("as Em-Garde (upscale to 256 tokens)", ops.MIN_PIXELS), ("no upscaling (min 84x84)", 84 * 84)):
        rows, vec = [], None
        for i in range(a.repeats + 1):                              # first repeat is warm-up
            vec, r = one(minpx)
            if i:
                rows.append(r)
        med = [statistics.median(r[k] for r in rows) for k in range(4)]
        results[name] = (vec, med)
        print(f"{name:38s} resize {med[0]:5.0f} ms | tokenise+pixels {med[1]:5.0f} ms | forward {med[2]:6.0f} ms | total {med[0]+med[1]+med[2]:6.0f} ms | {int(med[3])} tokens")
        if a.device.startswith("cuda"):
            print(f"{'':38s} peak gpu memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    k1, k2 = list(results)
    cos = float((results[k1][0].float() @ results[k2][0].float().T).item())
    print(f"\ncosine(vector with upscaling, vector without) = {cos:.4f}   (1.0 would mean the upscaling does not matter)")
    t1_, t2_ = sum(results[k1][1][:3]), sum(results[k2][1][:3])
    print(f"time per step: {t1_:.0f} ms with upscaling vs {t2_:.0f} ms without ({t1_ / t2_:.1f}x)")


if __name__ == "__main__":
    main()
