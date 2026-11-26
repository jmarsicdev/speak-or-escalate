"""Stage 0c (D24): score a cue set with the SigLIP2 matcher on the stored frame vectors (no video; only the text side of the model runs).

A step's window vector is the normalised mean of its last 4 SigLIP2 frame vectors (2 s, as the matcher); similarity is the cosine to each cue.
Writes a matcher cache (results/cache/matcher/siglip2-so400m-patch14-384__alt.cues-<name>/<id>.json, same format as the others), so
compare_embedders.py compares cue sets question by question on the same steps, and the combination stage can use it as signal S.

    python scripts/score_cueset_siglip.py --cues results/cache/cues/qwen3-vl-8b-instruct__text/cues.jsonl --name 8b-text
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egogate.cue_cache import load_cues  # noqa: E402

DIR = Path("results/cache/matcher/siglip2-so400m-patch14-384__alt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", nargs="+", required=True)
    ap.add_argument("--name", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from run_embed_alt import Siglip  # noqa: PLC0415

    emb = Siglip("google/siglip2-so400m-patch14-384", device=a.device)
    for path, name in zip(a.cues, a.name):
        cues = load_cues(path)
        out = Path(str(DIR) + f".cues-{name}")
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(DIR.glob("*.frames.npz")):
            qid = f.name.split(".")[0]
            if qid not in cues:
                continue
            z = np.load(f)
            t, F = z["times"], z["vecs"].astype(np.float32)
            c = np.cumsum(np.vstack([np.zeros((1, F.shape[1])), F]), axis=0)
            W = (c[4:] - c[:-4]) / 4
            W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
            S = W @ emb.text(cues[qid]).T
            json.dump({"t": [float(x) for x in t[3:]], "sims": S.round(5).tolist(), "cues": cues[qid]}, open(out / f"{qid}.json", "w"))
            n += 1
        print(f"{name}: {n} questions -> {out}")


if __name__ == "__main__":
    main()
