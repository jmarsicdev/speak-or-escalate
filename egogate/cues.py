"""CueParser: turn a user query into a handful of perceptually grounded visual cues.

This is Em-Garde's IGPP, run once per query (not per frame). Three backends:

  - "cloud":  GPT via the cloud client (what the project plans to use)
  - "edge":   the on-device VLM (to test whether the Nano can parse its own cues)
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
    """Robustly pull a cue list out of model output."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            cues = data.get("cues", [])
            if isinstance(cues, list) and cues:
                return [str(c).strip() for c in cues if str(c).strip()]
        except json.JSONDecodeError:
            pass
    # fallback: one cue per line
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


class CueParser:
    def __init__(self, backend: str = "manual", n_cues: int = 3, model=None, manual: list[str] | None = None):
        self.backend = backend
        self.n_cues = n_cues
        self.model = model  # CloudClient or EdgeVLM with a `.complete(prompt, images=None)` method
        self.manual = manual or []

    def parse(self, query: str, context_frames=None) -> list[str]:
        if self.backend == "manual":
            if not self.manual:
                raise ValueError("manual cue backend selected but no cues provided")
            return list(self.manual)
        if self.model is None:
            raise ValueError(f"cue backend '{self.backend}' needs a model")
        prompt = CUE_PROMPT.format(query=query, n=self.n_cues)
        text = self.model.complete(prompt, images=context_frames)
        cues = parse_cue_json(text)
        return cues[: self.n_cues] if cues else [query]
