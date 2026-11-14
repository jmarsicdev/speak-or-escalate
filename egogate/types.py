"""Shared dataclasses passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class Window:
    """A short clip of frames ending at time `t_end` (seconds into the stream)."""

    frames: list[np.ndarray]  # HxWx3 uint8, oldest first
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Triplet:
    """One scene-graph edge: (subject, relation, object)."""

    subject: str
    relation: str
    obj: str
    t: float  # timestamp it was extracted at

    def __str__(self) -> str:
        return f"({self.subject}, {self.relation}, {self.obj})"


@dataclass
class MatchResult:
    """Output of the embedding matcher for one window."""

    t: float
    sims: np.ndarray  # shape (n_cues,), cosine similarity per cue
    deltas: np.ndarray  # sims - previous sims (zeros on first window)
    triggered: bool
    best_cue: int  # index of the cue that drove the decision


class Action(str, Enum):
    SILENT = "silent"
    EDGE_ANSWER = "edge_answer"
    ESCALATE = "escalate"


@dataclass
class EdgeOutput:
    """What the edge VLM returns when asked whether/what to say."""

    text: str
    p_speak: float  # model-reported or derived probability that now is a response time
    raw: dict[str, Any] = field(default_factory=dict)  # logits, token probs, etc.


@dataclass
class Decision:
    t: float
    action: Action
    p_speak: float
    answer: str | None = None
    source: str | None = None  # "edge" or "cloud"
    cost: float = 0.0  # cloud calls, tokens, etc. for budget accounting
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """Ground-truth response time from a benchmark (OVO/StreamingBench style)."""

    t: float
    answer: str | None = None
