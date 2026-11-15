"""CueParser: turn a user query into a handful of perceptually grounded visual cues.

This is Em-Garde's IGPP, run once per query (not per frame). The cloud model is NOT
involved here: it is only called in the uncertain band of the router. Backends:

  - "edge":   the edge VLM parses its own cues (default)
  - "manual": cues given directly in the config (for controlled experiments)

Cue properties from Em-Garde we want the parser to respect:
  temporal localizability, perceptual groundability, redundancy (~3-4 cues/event).
"""
from __future__ import annotations

import json
import re

CUE_PROMPT = """You are helping a small vision model watch a first-person video stream.
The user asked: "{query}"

Write {n} short visual cues (each under 15 words) describing what the CAMERA WOULD SEE
at the exact moment the user should be notified. Rules:
- concrete, visible things only (objects, hand actions, text on screen), not intentions
- each cue should appear only when the response is due, not before
- cover different viewpoints or sub-steps of the same event

Return JSON: {{"cues": ["...", "..."]}}"""


def parse_cue_json(text: str) -> list[str]:
    """Pull a cue list out of model output, tolerating small formatting slips.

    Order: (1) valid JSON, (2) JSON missing its closing brace/bracket (small models often forget it),
    (3) the quoted strings inside a "cues" list, (4) one cue per line."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    candidates = [m.group(0)] if m else []
    start = text.find("{")
    if start >= 0:
        candidates += [text[start:] + "}", text[start:] + "]}"]
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        cues = data.get("cues", []) if isinstance(data, dict) else []
        if isinstance(cues, list) and cues:
            return [str(x).strip() for x in cues if str(x).strip()]
    if '"cues"' in text:   # JSON is too broken to load: take the quoted strings that follow the "cues" key
        tail = text[text.index('"cues"') + len('"cues"'):]
        quoted = [q.strip() for q in re.findall(r'"([^"]+)"', tail) if q.strip()]
        if quoted:
            return quoted
    # fallback: one cue per line
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


class CueParser:
    def __init__(self, backend: str = "edge", n_cues: int = 3, model=None, manual: list[str] | None = None):
        if backend not in ("edge", "manual"):
            raise ValueError(f"cue backend must be 'edge' or 'manual', got {backend!r}")
        self.backend = backend
        self.n_cues = n_cues
        self.model = model  # EdgeVLM with a `.complete(prompt, images=None)` method
        self.manual = manual or []

    def parse(self, query: str, context_frames=None) -> list[str]:
        if self.backend == "manual":
            if not self.manual:
                raise ValueError("manual cue backend selected but no cues provided")
            return list(self.manual)
        if self.model is None:
            raise ValueError("edge cue backend needs the edge model")
        prompt = CUE_PROMPT.format(query=query, n=self.n_cues)
        text = self.model.complete(prompt, images=context_frames)
        cues = parse_cue_json(text)
        return cues[: self.n_cues] if cues else [query]
