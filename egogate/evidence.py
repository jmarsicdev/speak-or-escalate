"""Evidence memory: scene-graph triplets extracted every N seconds (Response-G1 idea).

The edge VLM is asked to emit (subject, relation, object) triplets for the current
window. They go into a bounded memory; at decision time we retrieve the top-k most
relevant to the query/cues by text-embedding similarity.

The project doc's "payload vary" experiment compares:
  - frames + top-k triplets  (default)
  - triplets only            (cheaper to ship to the cloud; does accuracy survive?)
"""
from __future__ import annotations

import re
from collections import deque

import numpy as np

from .types import Triplet, Window

TRIPLET_PROMPT = """List the key objects and interactions visible in this first-person video clip
as scene-graph triplets, one per line, in the form (subject, relation, object).
Keep to the {k} most important. Only include things actually visible."""

_TRIPLET_RE = re.compile(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)")


def parse_triplets(text: str, t: float) -> list[Triplet]:
    return [Triplet(s.strip(), r.strip(), o.strip(), t) for s, r, o in _TRIPLET_RE.findall(text)]


class EvidenceMemory:
    def __init__(self, vlm, embedder, every_s: float = 5.0, max_items: int = 200, per_clip: int = 8):
        """
        vlm:      object with `.complete(prompt, images=...) -> str` (the edge model)
        embedder: Embedder with `.embed_text` for retrieval
        """
        self.vlm = vlm
        self.embedder = embedder
        self.every_s = every_s
        self.per_clip = per_clip
        self.items: deque[Triplet] = deque(maxlen=max_items)
        self.vecs: deque[np.ndarray] = deque(maxlen=max_items)
        self.last_t = -1e9

    def reset(self):
        self.items.clear()
        self.vecs.clear()
        self.last_t = -1e9

    def maybe_update(self, window: Window) -> list[Triplet]:
        """Extract triplets if `every_s` has elapsed since the last extraction."""
        if window.t_end - self.last_t < self.every_s:
            return []
        self.last_t = window.t_end
        text = self.vlm.complete(TRIPLET_PROMPT.format(k=self.per_clip), images=window.frames)
        new = parse_triplets(text, window.t_end)
        if new:
            vecs = self.embedder.embed_text([str(tr) for tr in new])
            for tr, v in zip(new, vecs):
                self.items.append(tr)
                self.vecs.append(v)
        return new

    def retrieve(self, query_vec: np.ndarray, k: int = 5) -> list[Triplet]:
        if not self.items:
            return []
        M = np.stack(self.vecs)
        order = np.argsort(-(M @ query_vec))[:k]
        return [self.items[i] for i in order]
