"""The API edge backend, tested offline with a fake client (no network, no key)."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from egogate.edge import APIEdgeVLM, yes_no_from_top


def tok(t, lp):
    return NS(token=t, logprob=lp)


class FakeClient:
    def __init__(self, top):
        self.top, self.calls = top, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        lp = NS(content=[NS(top_logprobs=self.top)]) if self.top is not None else None
        return NS(choices=[NS(message=NS(content=" hi "), logprobs=lp)])


def test_yes_no_mass_ignores_case_and_spaces_and_sums_variants():
    top = [tok("Yes", math.log(0.5)), tok(" yes", math.log(0.2)), tok("No", math.log(0.1)), tok("The", math.log(0.1))]
    y, n = yes_no_from_top(top)
    assert math.exp(y) == pytest.approx(0.7) and math.exp(n) == pytest.approx(0.1)


def test_missing_side_gets_floor():
    y, n = yes_no_from_top([tok("Yes", -0.1)])
    assert n == -30.0 and y == pytest.approx(-0.1)


def test_speak_now_uses_logprobs_and_sends_images():
    m = APIEdgeVLM("some/model", client=FakeClient([tok("Yes", math.log(0.9)), tok("No", math.log(0.1))]))
    out = m.speak_now("did it happen?", [np.zeros((8, 8, 3), np.uint8)] * 2)
    assert out.p_speak == pytest.approx(0.9) and m.n_images == 2
    kw = m.client.calls[0]
    assert kw["logprobs"] is True and kw["max_tokens"] == 1 and kw["temperature"] == 0
    assert sum(p["type"] == "image_url" for p in kw["messages"][0]["content"]) == 2


def test_no_logprobs_gives_clear_error():
    m = APIEdgeVLM("some/model", client=FakeClient(None))
    with pytest.raises(RuntimeError, match="log-probabilities"):
        m.speak_now("q", None)


def test_generate_returns_text():
    assert APIEdgeVLM("m", client=FakeClient([])).complete("hello") == "hi"
