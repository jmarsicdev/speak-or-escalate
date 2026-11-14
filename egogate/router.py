"""The Decider: three bands on the edge model's p(speak now), after a cue trigger.

    p >= hi          -> edge answers
    lo <  p <  hi    -> uncertain: escalate to the cloud model
    p <= lo          -> stay silent

The cloud is only ever called from the uncertain band.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Action


@dataclass
class RoutingThresholds:
    lo: float = 0.3
    hi: float = 0.7


class BandPolicy:
    def __init__(self, th: RoutingThresholds):
        if not 0 <= th.lo < th.hi <= 1:
            raise ValueError(f"need 0 <= lo < hi <= 1, got lo={th.lo} hi={th.hi}")
        self.th = th

    def decide(self, p_speak: float, triggered: bool) -> Action:
        if not triggered:
            return Action.SILENT
        if p_speak >= self.th.hi:
            return Action.EDGE_ANSWER
        if p_speak <= self.th.lo:
            return Action.SILENT
        return Action.ESCALATE


def build_policy(cfg: dict) -> BandPolicy:
    return BandPolicy(RoutingThresholds(cfg.get("lo", 0.3), cfg.get("hi", 0.7)))
