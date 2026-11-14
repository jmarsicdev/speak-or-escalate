"""Decider / routing policies.

The project-doc rule (three bands on p_speak after a cue trigger):

    p >= hi          -> edge answers
    lo <  p <  hi    -> escalate to cloud
    p <= lo          -> stay silent

Baselines to compare against (project doc: "does confidence matter?"):

    RandomPolicy     -> on each trigger, call the cloud with prob `p_cloud` (e.g. 20 calls/video)
    AlwaysCloud      -> every trigger goes to the cloud (upper bound on accuracy, worst cost)
    EdgeOnly         -> never escalate (lower bound on cost)

All policies share one signature so run_experiment can swap them via config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Action, Decision


@dataclass
class RoutingThresholds:
    lo: float = 0.3
    hi: float = 0.7


class Policy:
    name = "base"

    def decide(self, t: float, p_speak: float, triggered: bool) -> Action:
        raise NotImplementedError


class BandPolicy(Policy):
    """The confidence-band router from the project diagram."""

    name = "band"

    def __init__(self, th: RoutingThresholds, require_trigger: bool = True):
        self.th = th
        self.require_trigger = require_trigger

    def decide(self, t, p_speak, triggered):
        if self.require_trigger and not triggered:
            return Action.SILENT
        if p_speak >= self.th.hi:
            return Action.EDGE_ANSWER
        if p_speak <= self.th.lo:
            return Action.SILENT
        return Action.ESCALATE


class RandomPolicy(Policy):
    """Ignore confidence; escalate a random fraction of triggers."""

    name = "random"

    def __init__(self, p_cloud: float = 0.2, seed: int = 0, require_trigger: bool = True):
        self.p_cloud = p_cloud
        self.rng = np.random.default_rng(seed)
        self.require_trigger = require_trigger

    def decide(self, t, p_speak, triggered):
        if self.require_trigger and not triggered:
            return Action.SILENT
        return Action.ESCALATE if self.rng.uniform() < self.p_cloud else Action.EDGE_ANSWER


class AlwaysCloudPolicy(Policy):
    name = "always_cloud"

    def decide(self, t, p_speak, triggered):
        return Action.ESCALATE if triggered else Action.SILENT


class EdgeOnlyPolicy(Policy):
    name = "edge_only"

    def __init__(self, hi: float = 0.5):
        self.hi = hi

    def decide(self, t, p_speak, triggered):
        return Action.EDGE_ANSWER if (triggered and p_speak >= self.hi) else Action.SILENT


def build_policy(cfg: dict) -> Policy:
    kind = cfg.get("kind", "band")
    if kind == "band":
        return BandPolicy(RoutingThresholds(cfg.get("lo", 0.3), cfg.get("hi", 0.7)), cfg.get("require_trigger", True))
    if kind == "random":
        return RandomPolicy(cfg.get("p_cloud", 0.2), cfg.get("seed", 0), cfg.get("require_trigger", True))
    if kind == "always_cloud":
        return AlwaysCloudPolicy()
    if kind == "edge_only":
        return EdgeOnlyPolicy(cfg.get("hi", 0.5))
    raise ValueError(f"unknown policy {kind}")


def make_decision(t: float, action: Action, p_speak: float, **meta) -> Decision:
    return Decision(t=t, action=action, p_speak=p_speak, meta=meta)
