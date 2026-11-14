"""Embedding matcher: cosine similarity between the current window and the cue texts.

Open questions from the project doc, each exposed as a config knob so they can be
swept in experiments:

  aggregate:  "any"  -> trigger if ANY cue crosses (Em-Garde)
              "mean" -> trigger on the mean over cues
  mode:       "surge"    -> compare sim[t] - sim[t-1] to threshold (Em-Garde, theta=0.04)
              "absolute" -> compare sim[t] itself to threshold
  threshold:  float
  cooldown_s: suppress re-triggers for this long after a trigger

The embedder itself is swappable. `OpsMMEmbedder` wraps OpenSearch-AI/Ops-MM-embedding-v1-2B;
`RandomEmbedder` lets the whole pipeline run without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MatchResult, Window


# --------------------------------------------------------------------------- embedders
class Embedder:
    def embed_text(self, texts: list[str]) -> np.ndarray:  # (n, d), L2-normalised
        raise NotImplementedError

    def embed_window(self, window: Window) -> np.ndarray:  # (d,), L2-normalised
        raise NotImplementedError


class RandomEmbedder(Embedder):
    """Deterministic fake embeddings so tests and dry runs work without models."""

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def _norm(self, v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    def embed_text(self, texts):
        return self._norm(self.rng.standard_normal((len(texts), self.dim)))

    def embed_window(self, window):
        return self._norm(self.rng.standard_normal(self.dim))


class OpsMMEmbedder(Embedder):
    """Ops-MM-embedding-v1-2B (OpenSearch-AI). Text and video land in one space.

    TODO(models): confirm the exact preprocessing in the model card
    (https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B). The sketch below
    follows the Qwen2-VL-style interface the model is built on; adjust once we've
    run it once on the dev box. Frames are passed as a list of PIL images.
    """

    def __init__(self, model_id: str = "OpenSearch-AI/Ops-MM-embedding-v1-2B", device: str = "cuda", dtype: str = "float16"):
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), trust_remote_code=True
        ).to(device).eval()

    @staticmethod
    def _norm(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    def embed_text(self, texts):
        with self.torch.no_grad():
            out = self.model.get_text_embeddings(texts=texts)  # TODO: verify method name
        return self._norm(out.float().cpu().numpy())

    def embed_window(self, window):
        from PIL import Image  # noqa: PLC0415

        imgs = [Image.fromarray(f) for f in window.frames]
        with self.torch.no_grad():
            out = self.model.get_video_embeddings(videos=[imgs])  # TODO: verify method name
        return self._norm(out.float().cpu().numpy())[0]


# --------------------------------------------------------------------------- matcher
@dataclass
class MatcherConfig:
    threshold: float = 0.04
    mode: str = "surge"  # "surge" | "absolute"
    aggregate: str = "any"  # "any" | "mean"
    cooldown_s: float = 0.0


class Matcher:
    def __init__(self, embedder: Embedder, cues: list[str], cfg: MatcherConfig):
        self.embedder = embedder
        self.cues = cues
        self.cfg = cfg
        self.cue_vecs = embedder.embed_text(cues)  # (n_cues, d)
        self.prev_sims: np.ndarray | None = None
        self.last_trigger_t: float = -1e9

    def reset(self):
        self.prev_sims = None
        self.last_trigger_t = -1e9

    def step(self, window: Window) -> MatchResult:
        v = self.embedder.embed_window(window)
        sims = self.cue_vecs @ v  # cosine, since both normalised
        deltas = np.zeros_like(sims) if self.prev_sims is None else sims - self.prev_sims
        self.prev_sims = sims

        signal = deltas if self.cfg.mode == "surge" else sims
        if self.cfg.aggregate == "mean":
            score, best = float(signal.mean()), int(np.argmax(signal))
        else:
            best = int(np.argmax(signal))
            score = float(signal[best])

        in_cooldown = (window.t_end - self.last_trigger_t) < self.cfg.cooldown_s
        triggered = score > self.cfg.threshold and not in_cooldown
        if triggered:
            self.last_trigger_t = window.t_end
        return MatchResult(t=window.t_end, sims=sims, deltas=deltas, triggered=triggered, best_cue=best)
