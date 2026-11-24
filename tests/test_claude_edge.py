"""The Claude decider backend, tested offline with an injected reply function (no network, no key, no CLI)."""
import math

import numpy as np
import pytest

from egogate.decider_cache import cache_dir_for, make_meta
from egogate.edge import ClaudeEdgeVLM, build_speak_prompt, parse_score


def test_parse_score():
    assert parse_score("73") == pytest.approx(0.73)
    assert parse_score("Probability: 5%") == pytest.approx(0.05)
    assert parse_score("100") == 1.0 and parse_score("0") == 0.0
    assert parse_score("no idea") is None and parse_score("") is None
    assert parse_score("2026") is None          # a longer number is not a score


def test_speak_now_asks_for_a_score_and_stores_it_as_log_odds():
    seen = []

    def ask(prompt, jpegs):
        seen.append((prompt, jpegs))
        return "80"

    m = ClaudeEdgeVLM("claude-haiku-5-5", ask=ask)
    out = m.speak_now("did it happen?", [np.zeros((8, 8, 3), np.uint8)] * 4, window_s=2.0)
    assert out.p_speak == pytest.approx(0.8)
    assert math.exp(out.raw["lp_yes"]) == pytest.approx(0.8) and math.exp(out.raw["lp_no"]) == pytest.approx(0.2)
    prompt, jpegs = seen[0]
    assert len(jpegs) == 4 and m.n_images == 4
    assert "Yes or No" not in prompt and "0 to 100" in prompt
    assert prompt.startswith(build_speak_prompt("did it happen?", n_frames=4, window_s=2.0).rsplit("\n", 1)[0])   # the rest is the shared prompt


def test_extremes_are_clipped_and_a_reply_without_a_number_is_asked_again():
    replies = iter(["I cannot tell.", "100"])
    m = ClaudeEdgeVLM("m", ask=lambda p, j: next(replies))
    out = m.speak_now("q", None)
    assert out.p_speak == pytest.approx(0.995) and m.n_calls == 2
    with pytest.raises(RuntimeError, match="no 0 to 100 score"):
        ClaudeEdgeVLM("m", ask=lambda p, j: "?").speak_now("q", None)


def test_claude_runs_get_their_own_cache_per_transport_and_effort():
    base = {"decider": {"variant": "query_only", "stride_s": 2.0, "window_s": 2.0, "fps": 2.0, "max_side": 448, "cache_dir": "c"}}
    dirs = set()
    for edge in ({"backend": "claude", "transport": "cli", "effort": "low"}, {"backend": "claude", "transport": "api", "effort": "low"},
                 {"backend": "claude", "transport": "cli", "effort": "medium"}, {"backend": "api"}):
        cfg = {**base, "edge": edge}
        meta = make_meta(cfg, "claude-haiku-5-5")
        assert ("score_mode" in meta) == (edge["backend"] == "claude")
        dirs.add(cache_dir_for(cfg, meta))
    assert len(dirs) == 4
