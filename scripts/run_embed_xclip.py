"""A video-native embedder for the matcher: X-CLIP in place of Ops-MM / SigLIP2, on the same steps as the stored Ops-MM cache (GPU).

Ops-MM and SigLIP2 summarise what is in a clip; averaging or not, the summary barely moves when the event happens (docs/RATIONALE.md 9a,
"Shorter windows and scene-level cues"). X-CLIP (microsoft/xclip-*) is trained on video-text pairs and models the frames of a clip jointly
(cross-frame attention and a multi-frame integration transformer), so it can in principle represent motion. Its text embedding is also
conditioned on the clip (video-specific prompting), so the similarity is computed per window from the model's own outputs.

Each step uses the last 2 s, as the other matchers do, but at 4 fps because the model takes 8 frames per clip. Same edge-model cues as the Ops-MM
cache, and (second folder) the oracle cues. Writes results/cache/matcher/<model tag>__alt/<id>.json and <that>.cues-oracle/<id>.json
in the same format as the other matcher caches, so compare_embedders.py, pilot_label_radius.py and the viewer read them directly.

    python scripts/run_embed_xclip.py --samples data/processed/ovo_far_val.jsonl --like results/cache/matcher/Ops-MM-embedding-v1-2B__875248e9
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
from egogate.frames import stream_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--like", required=True, help="the Ops-MM matcher cache whose steps and edge cues are reused")
    ap.add_argument("--model-id", default="microsoft/xclip-large-patch14")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--text", choices=["conditioned", "plain"], default="conditioned",
                    help="conditioned: the model's video-specific text embedding (changes per clip); plain: get_text_features, the same text vector for every clip")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

    cfg = load_config(None, {})
    samples = [s for s in load_jsonl(a.samples) if (Path(a.like) / f"{s.id}.json").exists()]
    tag = a.model_id.split("/")[-1] + ("__alt" if a.text == "conditioned" else "__plain")
    out = Path(a.like).parent / tag
    out_or = Path(str(out) + ".cues-oracle")
    out.mkdir(parents=True, exist_ok=True)
    out_or.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.model_id)
    model = AutoModel.from_pretrained(a.model_id, dtype=torch.float16).cuda().eval()
    nf = model.config.vision_config.num_frames
    fps = nf / 2.0   # 8 frames over the last 2 s
    root = Path(cfg["data"]["video_root"])
    print(f"questions: {len(samples)} -> {out}; {nf} frames per 2 s window ({fps:g} fps)", flush=True)
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        if (out / f"{s.id}.json").exists() and (out_or / f"{s.id}.json").exists():
            continue
        ref = json.load(open(Path(a.like) / f"{s.id}.json"))
        edge, orc = ref["cues"], oracle_cues(s)
        texts = edge + orc
        tin = proc(text=texts, return_tensors="pt", padding=True)
        tin = {k: v.cuda() for k, v in tin.items()}
        with torch.no_grad():
            tf = model.get_text_features(**tin)
            tf = getattr(tf, "pooler_output", tf)   # newer transformers return an output object; pooler_output is the projected text vector (checked)
            tplain = torch.nn.functional.normalize(tf.float(), dim=-1)   # (n_text, d)
        t0 = time.time()
        ts, sims, buf, buf_t = [], [], [], []

        def flush():
            vin = proc(videos=buf, return_tensors="pt")["pixel_values"].cuda().half()
            with torch.no_grad():
                o = model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"], pixel_values=vin)
            v = torch.nn.functional.normalize(o.video_embeds.float(), dim=-1)            # (B, d)
            if a.text == "conditioned":
                tx = torch.nn.functional.normalize(o.text_embeds.float(), dim=-1)        # (B, n_text, d), conditioned on each clip
                sims.extend((tx @ v[:, :, None]).squeeze(-1).cpu().numpy().tolist())
            else:
                sims.extend((v @ tplain.T).cpu().numpy().tolist())
            ts.extend(buf_t)
            buf.clear()
            buf_t.clear()

        for w in stream_windows(root / s.video, list(ref["t"]), 2.0, fps, 448):
            if len(w.frames) != nf:
                continue
            buf.append(list(w.frames))
            buf_t.append(float(w.t_end))
            if len(buf) == a.batch:
                flush()
        if buf:
            flush()
        S = np.array(sims)
        for d, sl, cues in ((out, slice(0, len(edge)), edge), (out_or, slice(len(edge), len(texts)), orc)):
            json.dump({"t": ts, "sims": S[:, sl].round(5).tolist(), "cues": cues}, open(d / f"{s.id}.json", "w"))
        print(f"[{i}/{len(samples)}] {s.id}: {len(ts)} steps in {time.time() - t0:.0f} s, total {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
