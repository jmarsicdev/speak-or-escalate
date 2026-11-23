"""Different embedder test: a SigLIP2 image-text model in place of Ops-MM, on the same steps as the stored Ops-MM cache.

    python scripts/run_embed_alt.py --samples data/processed/ovo_far_val.jsonl --like results/cache/matcher/<ops dir> --per-source 3

Each frame (one per 0.5 s) is embedded once; a step's window vector is the normalised mean of its 4 frame vectors (the same 2 s, 4 frames as the
Ops-MM windows). Similarity is the cosine to each cue. The cues are copied from the Ops-MM cache (the edge model's cues) and, in a second
folder, the oracle cues. The steps are exactly those of the Ops-MM cache so the two can be compared question by question.

Writes results/cache/matcher/<model tag>__alt/<id>.json (and .npy of window vectors) and <that>.cues-oracle/<id>.json.
The embedder must not share the GPU with the decider.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egogate.config import load_config  # noqa: E402
from egogate.cue_sources import oracle_cues  # noqa: E402
from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.frames import stream_windows, window_sample_times  # noqa: E402


class Siglip:
    def __init__(self, model_id, device="cuda"):
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()

    @staticmethod
    def _t(o):
        o = getattr(o, "pooler_output", o)   # newer transformers return an output object
        return o.float().cpu().numpy()

    @staticmethod
    def _norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def text(self, texts):
        inp = self.proc(text=[x.lower() for x in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self._norm(self._t(self.model.get_text_features(**inp)))

    def images(self, frames, bs=32):
        from PIL import Image  # noqa: PLC0415

        out = []
        for i in range(0, len(frames), bs):
            inp = self.proc(images=[Image.fromarray(f) for f in frames[i:i + bs]], return_tensors="pt").to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(self.torch.bfloat16)
            with self.torch.no_grad():
                out.append(self._t(self.model.get_image_features(pixel_values=inp["pixel_values"])))
        return self._norm(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--like", required=True, help="the Ops-MM matcher cache dir whose steps and edge cues are reused")
    ap.add_argument("--model-id", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--per-source", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo questions that already have a result (to add the per-frame file)")
    a = ap.parse_args()
    cfg = load_config(None, {})
    samples = [s for s in load_jsonl(a.samples) if (Path(a.like) / f"{s.id}.json").exists()]
    if a.per_source:
        by: dict = {}
        for s in sorted(samples, key=lambda s: (s.end_s or 0) - s.start_s):
            by.setdefault(s.meta.get("source", s.task), []).append(s)
        samples = [s for v in by.values() for s in v[: a.per_source]]
    tag = a.model_id.split("/")[-1] + "__alt"
    out = Path(a.like).parent / tag
    out_or = Path(str(out) + ".cues-oracle")
    out.mkdir(parents=True, exist_ok=True); out_or.mkdir(parents=True, exist_ok=True)
    print("questions:", len(samples), "->", out)
    emb = Siglip(a.model_id)
    root = Path(cfg["data"]["video_root"])
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        if (out / f"{s.id}.json").exists() and not a.force:
            continue
        ref = json.load(open(Path(a.like) / f"{s.id}.json"))
        t = np.array(ref["t"])
        cache: dict[float, np.ndarray] = {}
        vecs = []
        t0 = time.time()
        for w in stream_windows(root / s.video, list(t), 2.0, 2.0, 448):
            times = [round(x, 3) for x in window_sample_times(w.t_end, 2.0, 2.0)]
            new = [(tm, f) for tm, f in zip(times, w.frames) if tm not in cache]
            if new:
                for (tm, _), v in zip(new, emb.images([f for _, f in new])):
                    cache[tm] = v
            v = np.mean([cache[tm] for tm in times], axis=0)
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
        V = np.array(vecs)
        if len(V) != len(t):
            print(f"  {s.id}: {len(V)} windows for {len(t)} steps (video shorter?), keeping the steps we have")
            t = t[: len(V)]
        for d, cues in ((out, ref["cues"]), (out_or, oracle_cues(s))):
            S = V @ emb.text(cues).T
            json.dump({"t": [float(x) for x in t], "sims": S.tolist(), "cues": cues}, open(d / f"{s.id}.json", "w"))
        np.save(out / f"{s.id}.npy", V.astype(np.float16))
        # per-frame similarities (one row per 0.5 s frame), so shorter or longer windows can be built later without re-embedding
        ft = np.array(sorted(cache)); F = np.stack([cache[x] for x in ft])
        # the frame vectors themselves too, so any new cue set can be scored at any window length with only a text embedding
        np.savez(out / f"{s.id}.frames.npz", times=ft, vecs=F.astype(np.float16),
                 sims_edge=F @ emb.text(ref["cues"]).T, sims_oracle=F @ emb.text(oracle_cues(s)).T)
        print(f"[{i}/{len(samples)}] {s.id}: {len(V)} steps in {time.time() - t0:.0f} s, total {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
