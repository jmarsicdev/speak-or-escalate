"""End-to-end stream processing for one (video, query) pair.

    run_stream(windows, query, cues, components) -> list[Decision]

Every window: matcher.step -> maybe update evidence -> if triggered, ask the edge model
for p_speak -> band policy -> act (edge answers / cloud answers / silent).
All intermediate numbers (sims, deltas, p_speak) are kept in Decision.meta so results
can be analysed afterwards without re-running models.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .cloud import CloudClient
from .edge import EdgeVLM
from .embed import Matcher
from .evidence import EvidenceMemory
from .router import BandPolicy
from .types import Action, Decision, Window

CLOUD_PROMPT = """The user, wearing a first-person camera, asked: "{query}"
{evidence_block}
Is this the right moment to respond? If yes, give the answer concisely. If not, reply exactly: NO_RESPONSE"""


@dataclass
class Components:
    matcher: Matcher
    edge: EdgeVLM
    cloud: CloudClient
    policy: BandPolicy
    evidence: EvidenceMemory | None = None
    top_k_evidence: int = 5


def run_stream(windows: Iterable[Window], query: str, cues: list[str], c: Components, log_all: bool = False) -> list[Decision]:
    c.matcher.reset()
    if c.evidence is not None:
        c.evidence.reset()
    decisions: list[Decision] = []

    for w in windows:
        m = c.matcher.step(w)
        if c.evidence is not None:
            c.evidence.maybe_update(w)

        if not m.triggered:
            if log_all:
                decisions.append(Decision(t=w.t_end, action=Action.SILENT, p_speak=0.0,
                                          meta={"sims": m.sims.tolist(), "deltas": m.deltas.tolist(), "triggered": False}))
            continue

        ev_text = [str(tr) for tr in c.evidence.retrieve(c.matcher.query_vec, c.top_k_evidence)] if c.evidence else []
        eo = c.edge.speak_now(query, w.frames, ev_text, cues)
        action = c.policy.decide(eo.p_speak, m.triggered)

        d = Decision(t=w.t_end, action=action, p_speak=eo.p_speak,
                     meta={"sims": m.sims.tolist(), "deltas": m.deltas.tolist(), "best_cue": m.best_cue,
                           "triggered": True, "evidence": ev_text, **eo.raw})
        if action == Action.EDGE_ANSWER:
            d.answer, d.source = c.edge.answer(query, w.frames, ev_text), "edge"
        elif action == Action.ESCALATE:
            ev_block = ("Recent scene evidence:\n" + "\n".join(f"- {e}" for e in ev_text) + "\n") if ev_text else ""
            text = c.cloud.complete(CLOUD_PROMPT.format(query=query, evidence_block=ev_block), images=w.frames)
            d.cost = 1.0
            if text.strip() != "NO_RESPONSE":
                d.answer, d.source = text, "cloud"
            else:
                d.action = Action.SILENT
                d.meta["cloud_rejected"] = True
        decisions.append(d)
    return decisions
