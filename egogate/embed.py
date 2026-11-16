"""Embedding matcher: cosine similarity between the current video window and the cue texts.

Follows Em-Garde's LPMM: a 2-second window is embedded, compared to each pre-embedded cue,
and a trigger fires when ANY cue's similarity jumps by more than `threshold` (0.04) since the
previous window. `cooldown_s` suppresses immediate re-triggers.

`OpsMMEmbedder` wraps OpenSearch-AI/Ops-MM-embedding-v1-2B (text and video in one space);
`RandomEmbedder` lets the pipeline run without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MatchResult, Window


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

    @staticmethod
    def _norm(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    def embed_text(self, texts):
        return self._norm(self.rng.standard_normal((len(texts), self.dim)))

    def embed_window(self, window):
        return self._norm(self.rng.standard_normal(self.dim))


class _AppendNewline:
    """Wrap a processor so every prompt it is given ends with a newline after the end-of-text token.

    Em-Garde's streaming path builds its window prompt that way ("...<|endoftext|>\\n"), the model's standard path does
    not, and the final token is what the embedding is read from. We measured cosine 0.91-0.93 between the two prompts'
    vectors and 0.998-1.00 once the newline is added (docs/RATIONALE.md D17), so the newline, not the frame caching,
    is what separates them."""

    def __init__(self, processor):
        self._p = processor

    def __call__(self, text=None, **kw):
        return self._p(text=[t + "\n" for t in text], **kw)

    def __getattr__(self, name):
        return getattr(self._p, name)


class OpsMMEmbedder(Embedder):
    """Ops-MM-embedding-v1-2B (OpenSearch-AI), through Em-Garde's vendored wrapper (third_party/em_garde).

    Text and video go into the same 1536-dimension space, L2-normalised, so a dot product is a cosine similarity.
    A window is embedded from scratch with the model's standard path (the last `window_s * fps` frames as images).
    We do NOT use Em-Garde's streaming shortcut (cache the earlier frames, add one per step): its patch reaches
    into model internals that changed in current transformers and fails there. Windows are embedded from scratch, with
    the same trailing newline their streaming prompt has, which reproduces their vectors; see docs/RATIONALE.md D17.
    """

    def __init__(self, model_id: str = "OpenSearch-AI/Ops-MM-embedding-v1-2B", device: str = "cuda", attn: str = "sdpa",
                 min_pixels: int | None = None):
        import importlib.util  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        path = Path(__file__).resolve().parent.parent / "third_party" / "em_garde" / "ops_mm_embedding_v1.py"
        spec = importlib.util.spec_from_file_location("ops_mm_embedding_v1", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.model = mod.OpsMMEmbeddingV1(model_id, device=device, attn_implementation=attn)
        self.model.eval()
        if min_pixels is not None:
            # The wrapper resizes every frame UP to at least MIN_PIXELS = 256*28*28 (256 visual tokens per frame) before
            # embedding. That is what Em-Garde runs, so it is our default; a smaller minimum is faster but gives different
            # vectors (cosine about 0.94 in our check). See docs/RATIONALE.md D17.
            self.model._process_images = lambda imgs: [mod.fetch_image(i, min_pixels=min_pixels) for i in imgs]

    @staticmethod
    def _norm(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    def embed_text(self, texts):
        out = self.model.get_text_embeddings(list(texts))
        return self._norm(out.float().cpu().numpy())

    def embed_window(self, window):
        from PIL import Image  # noqa: PLC0415

        imgs = [Image.fromarray(f) for f in window.frames]
        orig = self.model.processor
        self.model.processor = _AppendNewline(orig)     # window prompts only; cue text uses the standard prompt
        try:
            out = self.model.embed(images=[imgs])
        finally:
            self.model.processor = orig
        return self._norm(out.float().cpu().numpy())[0]


@dataclass
class MatcherConfig:
    threshold: float = 0.04
    cooldown_s: float = 0.0


class Matcher:
    def __init__(self, embedder: Embedder, cues: list[str], query: str, cfg: MatcherConfig):
        self.embedder = embedder
        self.cues = list(cues)
        self.cfg = cfg
        self.cue_vecs = embedder.embed_text(self.cues)  # (n_cues, d)
        self.query_vec = embedder.embed_text([query])[0]  # used for evidence retrieval
        self.prev_sims: np.ndarray | None = None
        self.last_trigger_t = -1e9

    def reset(self):
        self.prev_sims = None
        self.last_trigger_t = -1e9

    def step(self, window: Window) -> MatchResult:
        v = self.embedder.embed_window(window)
        sims = self.cue_vecs @ v  # cosine, since both normalised
        deltas = np.zeros_like(sims) if self.prev_sims is None else sims - self.prev_sims
        self.prev_sims = sims

        best = int(np.argmax(deltas))
        in_cooldown = (window.t_end - self.last_trigger_t) < self.cfg.cooldown_s
        triggered = float(deltas[best]) > self.cfg.threshold and not in_cooldown
        if triggered:
            self.last_trigger_t = window.t_end
        return MatchResult(t=window.t_end, sims=sims, deltas=deltas, triggered=triggered, best_cue=best)
