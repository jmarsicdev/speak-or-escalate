"""Row 4: on-demand decider at trigger times, memoisation, and the combined score."""
import numpy as np
import pytest

from egogate.config import load_config
from egogate.decider_cache import cache_dir_for, make_meta
from egogate.edge import MockEdgeVLM
from egogate.eval.datasets import Sample
from egogate.gated import run_ondemand, score_gated, trigger_times
from egogate.types import Event

cv2 = pytest.importorskip("cv2")


def make_video(path, seconds=14, fps=10, size=(64, 48)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(seconds * fps):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        f[..., 2] = int(i / fps * 15)
        vw.write(f)
    vw.release()


def frame_time(frame):
    return frame[..., 0].mean() / 15.0


def test_ondemand_runs_only_missing_moments_and_reuses_them(tmp_path):
    make_video(tmp_path / "v.avi")
    cfg = load_config(None, {"decider.cache_dir": str(tmp_path / "c"), "edge.backend": "mock"})
    s = Sample(id="g1", video="v.avi", query="q", events=[Event(8.0)], task="REC", start_s=0.0, end_s=13.0)
    calls = []
    edge = MockEdgeVLM(p_speak_fn=lambda imgs: calls.append(1) or (0.95 if abs(frame_time(imgs[-1]) - 8.0) < 0.6 else 0.05))
    dec_dir = cache_dir_for(cfg, make_meta(cfg, "mock/m"))
    dec_dir.mkdir(parents=True)
    r1 = run_ondemand(s, edge, cfg, [5.5, 8.0, 10.5], dec_dir, tmp_path)
    assert r1["n_new"] == 3 and len(calls) == 3
    r2 = run_ondemand(s, edge, cfg, [8.0, 10.5, 12.0], dec_dir, tmp_path)          # two already known, one new
    assert r2["n_new"] == 1 and len(calls) == 4 and set(r2["rows"]) == {8.0, 10.5, 12.0}


def test_gated_response_needs_both_the_trigger_and_a_yes():
    t = np.arange(2.0, 14.0, 0.5)
    sims = np.full((len(t), 1), 0.2)
    sims[t >= 8.0] = 0.5                                       # one big rise at t = 8.0 (and it stays up)
    s = Sample(id="x", video="v", query="q", events=[Event(8.0)], task="REC")
    assert trigger_times(t, sims, "paper", 0.05) == [8.0]
    yes = {"t": 8.0, "lp_yes": 2.0, "lp_no": -2.0}
    no = {"t": 8.0, "lp_yes": -2.0, "lp_no": 2.0}
    matcher = {"x": {"t": t, "sims": sims}}
    hit = score_gated([s], matcher, {"x": {8.0: yes}}, "paper", 0.05, 0.5)
    miss = score_gated([s], matcher, {"x": {8.0: no}}, "paper", 0.05, 0.5)
    assert hit["avg_f1"] == 1.0 and hit["total_detections"] == 1
    assert miss["total_detections"] == 0 and miss["avg_f1"] == 0.0     # trigger fired but the decider said No
    assert score_gated([s], matcher, {"x": {8.0: no}}, "paper", 0.05, 0.0)["total_detections"] == 1   # tau 0: matcher alone
