import math

import numpy as np

from egogate.cues import parse_cue_json
from egogate.edge import MockEdgeVLM
from egogate.embed import Embedder, Matcher, MatcherConfig
from egogate.eval.metrics import calibration_table, online_pr, summarize_run
from egogate.evidence import parse_triplets
from egogate.router import BandPolicy, RandomPolicy, RoutingThresholds
from egogate.types import Action, Event, Window


class FixedEmbedder(Embedder):
    """Window embedding is controlled by a scripted similarity-to-cue0 schedule."""

    def __init__(self, sims):
        self.sims = list(sims)
        self.i = 0

    def embed_text(self, texts):
        n = len(texts)
        return np.eye(max(n, 2))[:n]  # cue k -> unit vector e_k

    def embed_window(self, window):
        s = self.sims[self.i]
        self.i += 1
        v = np.zeros(2)
        v[0] = s
        v[1] = math.sqrt(max(1 - s * s, 0))
        return v


def win(t):
    return Window(frames=[np.zeros((1, 1, 3), np.uint8)], t_start=t - 2, t_end=t)


def test_surge_trigger_fires_on_jump_not_level():
    m = Matcher(FixedEmbedder([0.5, 0.5, 0.6, 0.6]), ["cue"], MatcherConfig(threshold=0.04, mode="surge"))
    fired = [m.step(win(t)).triggered for t in (2, 2.5, 3, 3.5)]
    assert fired == [False, False, True, False]


def test_absolute_trigger_and_cooldown():
    m = Matcher(FixedEmbedder([0.1, 0.9, 0.9, 0.9]), ["cue"], MatcherConfig(threshold=0.5, mode="absolute", cooldown_s=1.0))
    fired = [m.step(win(t)).triggered for t in (2, 2.5, 3, 3.6)]
    assert fired == [False, True, False, True]


def test_band_policy():
    p = BandPolicy(RoutingThresholds(0.3, 0.7))
    assert p.decide(0, 0.9, True) == Action.EDGE_ANSWER
    assert p.decide(0, 0.5, True) == Action.ESCALATE
    assert p.decide(0, 0.1, True) == Action.SILENT
    assert p.decide(0, 0.9, False) == Action.SILENT


def test_random_policy_rate():
    p = RandomPolicy(p_cloud=0.2, seed=1)
    acts = [p.decide(0, 0.5, True) for _ in range(2000)]
    frac = sum(a == Action.ESCALATE for a in acts) / len(acts)
    assert 0.15 < frac < 0.25


def test_online_pr_tolerance_and_one_to_one():
    events = [Event(10.0), Event(30.0)]
    pr = online_pr([11.0, 12.0, 50.0], events, tol=4.0)  # one hit, one dup, one miss
    assert pr.n_correct == 1 and pr.n_pred == 3 and pr.n_gt == 2
    assert pr.recall == 0.5 and abs(pr.precision - 1 / 3) < 1e-9
    assert online_pr([9.0], events, tol=4.0).n_correct == 0  # before the event doesn't count


def test_parsers():
    assert parse_cue_json('blah {"cues": ["a", "b"]} blah') == ["a", "b"]
    assert parse_cue_json("- bubbling\n- steam") == ["bubbling", "steam"]
    tr = parse_triplets("(hand, holds, pot)\n(water, in, pot)", t=3.0)
    assert [str(x) for x in tr] == ["(hand, holds, pot)", "(water, in, pot)"]


def test_mock_edge_p_speak_is_probability():
    e = MockEdgeVLM(p_speak_fn=lambda imgs: 0.8)
    out = e.speak_now("q", [], [])
    assert abs(out.p_speak - 0.8) < 1e-6 and out.text == "Yes"


def test_end_to_end_mock_pipeline(tmp_path):
    from egogate.cloud import MockCloud
    from egogate.pipeline import Components, run_stream

    # similarity jumps at t=10 and t=30 -> two triggers; edge is confident at 10, unsure at 30
    sims = {10.0: 0.9, 30.0: 0.9}
    sched = []
    t = 2.0
    while t <= 40:
        sched.append(sims.get(t, 0.2))
        t += 0.5
    windows = []
    t = 2.0
    while t <= 40:
        windows.append(win(t))
        t += 0.5
    edge = MockEdgeVLM(p_speak_fn=lambda imgs: 0.9 if len(imgs) == 1 else 0.5)
    matcher = Matcher(FixedEmbedder(sched), ["cue"], MatcherConfig(threshold=0.04, mode="surge"))
    comps = Components(matcher=matcher, edge=edge, cloud=MockCloud(), policy=BandPolicy(RoutingThresholds(0.3, 0.7)))
    decisions = run_stream(windows, "q", comps)
    triggered = [d for d in decisions if d.meta.get("triggered")]
    assert [d.t for d in triggered] == [10.0, 30.0]
    summ = summarize_run(decisions, [Event(10.0), Event(30.0)], tol=4.0)
    assert summ["recall"] == 1.0 and summ["precision"] == 1.0
    assert calibration_table(decisions, [Event(10.0), Event(30.0)])
