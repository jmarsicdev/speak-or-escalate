"""End-to-end stream processing for one (video, query) pair.

    run_stream(windows, query, components, cfg) -> list[Decision]

Every window: matcher.step -> maybe update evidence -> if triggered, ask edge for
p_speak -> policy -> act. All intermediate numbers (sims, deltas, p_speak) are kept
in Decision.meta so experiments can plot calibration and trigger curves afterwards.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .cloud import CloudClient
from .edge import EdgeVLM
from .embed import Matcher
from .evidence import EvidenceMemory
from .router import Policy
from .types import Action, Decision, Window


@dataclass
class Components:
    matcher: Matcher
    edge: EdgeVLM
    cloud: CloudClient
    policy: Policy
    evidence: EvidenceMemory | None = None
    cloud_payload: str = "frames+evidence"  # "frames+evidence" | "frames" | "evidence"
    top_k_evidence: int = 5


def _cloud_answer(c: Components, query: str, window: Window, ev_text: list[str]) -> str:
    prompt = (
        f'The user, wearing a first-person camera, asked: "{query}"\n'
        + ("Recent scene evidence:\n" + "\n".join(f"- {e}" for e in ev_text) + "\n" if ev_text and c.cloud_payload != "frames" else "")
        + "Is this the right moment to respond? If yes, give the answer concisely. If not, reply exactly: NO_RESPONSE"
    )
    images = None if c.cloud_payload == "evidence" else window.frames
    return c.cloud.complete(prompt, images=images)


def run_stream(windows: Iterable[Window], query: str, c: Components, log_all: bool = False) -> list[Decision]:
    c.matcher.reset()
    if c.evidence is not None:
        c.evidence.reset()
    query_vec = c.matcher.cue_vecs.mean(axis=0)
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

        ev_text = [str(tr) for tr in c.evidence.retrieve(query_vec, c.top_k_evidence)] if c.evidence else []
        eo = c.edge.speak_now(query, w.frames, ev_text)
        action = c.policy.decide(w.t_end, eo.p_speak, m.triggered)

        d = Decision(t=w.t_end, action=action, p_speak=eo.p_speak,
                     meta={"sims": m.sims.tolist(), "deltas": m.deltas.tolist(), "best_cue": m.best_cue,
                           "triggered": True, "evidence": ev_text, **eo.raw})
        if action == Action.EDGE_ANSWER:
            d.answer, d.source = c.edge.answer(query, w.frames, ev_text), "edge"
        elif action == Action.ESCALATE:
            text = _cloud_answer(c, query, w, ev_text)
            d.cost = 1.0
            if text.strip() != "NO_RESPONSE":
                d.answer, d.source = text, "cloud"
            else:
                d.action = Action.SILENT
                d.meta["cloud_rejected"] = True
        decisions.append(d)
    return decisions
