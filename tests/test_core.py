import math

import numpy as np

from egogate.cloud import MockCloud
from egogate.cues import CueParser, parse_cue_json
from egogate.edge import MockEdgeVLM
from egogate.embed import Embedder, Matcher, MatcherConfig
from egogate.eval.metrics import calibration_table, match_detections, score_question, summarize_run
from egogate.evidence import parse_triplets
from egogate.pipeline import Components, run_stream
from egogate.router import BandPolicy, RoutingThresholds
from egogate.types import Action, Event, Window


class FixedEmbedder(Embedder):
    """Window embedding follows a scripted similarity-to-cue0 schedule."""

    def __init__(self, sims):
        self.sims = list(sims)
        self.i = 0

    def embed_text(self, texts):
        n = len(texts)
        return np.eye(max(n, 2))[:n]  # cue k -> unit vector e_k

    def embed_window(self, window):
        s = self.sims[self.i]
        self.i += 1
        return np.array([s, math.sqrt(max(1 - s * s, 0))])


def win(t):
    return Window(frames=[np.zeros((1, 1, 3), np.uint8)], t_start=t - 2, t_end=t)


def test_surge_trigger_fires_on_jump_not_level():
    m = Matcher(FixedEmbedder([0.5, 0.5, 0.6, 0.6]), ["cue"], "q", MatcherConfig(threshold=0.04))
    assert [m.step(win(t)).triggered for t in (2, 2.5, 3, 3.5)] == [False, False, True, False]


def test_cooldown_suppresses_retrigger():
    m = Matcher(FixedEmbedder([0.1, 0.5, 0.9, 0.1, 0.5]), ["cue"], "q", MatcherConfig(threshold=0.04, cooldown_s=1.0))
    assert [m.step(win(t)).triggered for t in (2, 2.5, 3, 3.5, 4)] == [False, True, False, False, True]


def test_band_policy():
    p = BandPolicy(RoutingThresholds(0.3, 0.7))
    assert p.decide(0.9, True) == Action.EDGE_ANSWER
    assert p.decide(0.5, True) == Action.ESCALATE
    assert p.decide(0.1, True) == Action.SILENT
    assert p.decide(0.9, False) == Action.SILENT


def test_metric_tolerance_is_symmetric_and_one_to_one():
    # Em-Garde counts a response within +/- tol of the event, one detection per event
    q = score_question([11.0, 12.0, 50.0], [10.0, 30.0], tol=2.0)  # one hit, one dup, one miss
    assert (q.n_correct, q.n_det, q.n_gt) == (1, 3, 2)
    assert q.recall == 0.5 and abs(q.precision - 1 / 3) < 1e-9
    assert score_question([9.0], [10.0], tol=2.0).n_correct == 1  # slightly early still counts
    assert score_question([7.9], [10.0], tol=2.0).n_correct == 0
    assert score_question([], [10.0]).precision == 0.0  # no detections -> precision 0, as in Em-Garde


def test_parsers():
    assert parse_cue_json('blah {"cues": ["a", "b"]} blah') == ["a", "b"]
    assert parse_cue_json("- bubbling\n- steam") == ["bubbling", "steam"]
    assert [str(x) for x in parse_triplets("(hand, holds, pot)\n(water, in, pot)", t=3.0)] == ["(hand, holds, pot)", "(water, in, pot)"]


def test_cues_parsed_by_edge_model():
    cues = CueParser("edge", 3, model=MockEdgeVLM()).parse("tell me when the water boils")
    assert len(cues) == 3 and all(isinstance(c, str) for c in cues)


def test_mock_edge_p_speak_is_probability():
    out = MockEdgeVLM(p_speak_fn=lambda imgs: 0.8).speak_now("q", [], [], ["cue"])
    assert abs(out.p_speak - 0.8) < 1e-6 and out.text == "Yes"


def test_end_to_end_mock_pipeline():
    # similarity jumps at t=10 (edge confident -> edge answers) and t=30 (uncertain -> cloud)
    times = [round(2.0 + 0.5 * i, 1) for i in range(77)]  # 2.0 .. 40.0
    sched = [0.9 if t in (10.0, 30.0) else 0.2 for t in times]
    edge = MockEdgeVLM(p_speak_fn=lambda imgs: 0.9 if edge.calls == 0 else 0.5)
    edge.calls = 0
    orig = edge.speak_now

    def counting(*a, **k):
        r = orig(*a, **k)
        edge.calls += 1
        return r

    edge.speak_now = counting
    cloud = MockCloud()
    comps = Components(matcher=Matcher(FixedEmbedder(sched), ["cue"], "q", MatcherConfig(0.04)),
                       edge=edge, cloud=cloud, policy=BandPolicy(RoutingThresholds(0.3, 0.7)))
    decisions = run_stream([win(t) for t in times], "q", ["cue"], comps)

    assert [d.t for d in decisions] == [10.0, 30.0]
    assert [d.source for d in decisions] == ["edge", "cloud"]
    assert cloud.n_calls == 1  # cloud only in the uncertain band
    summ = summarize_run(decisions, [Event(10.0), Event(30.0)], tol=2.0)
    assert summ["recall"] == 1.0 and summ["precision"] == 1.0 and summ["n_cloud_calls"] == 1
    assert calibration_table(decisions, [Event(10.0), Event(30.0)])
