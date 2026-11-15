"""Matcher rules (parity with Em-Garde's released code), duplicate handling, diagnostics, and the stage 0 / 1a runners."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from egogate.config import load_config
from egogate.cue_cache import load_cues, run_cues
from egogate.edge import MockEdgeVLM
from egogate.embed import RandomEmbedder
from egogate.embed_cache import cache_dir_for, load_sims, make_meta, run_sample
from egogate.eval.datasets import Sample
from egogate.matcher_replay import auc, dedup, matcher_detections, similarity_diagnostics, surge_code, surge_paper
from egogate.types import Event

ROOT = Path(__file__).resolve().parent.parent


def test_paper_rule_is_a_rise_since_the_previous_step():
    sims = np.array([[0.5], [0.5], [0.6], [0.6], [0.55]])
    assert surge_paper(sims, 0.04).tolist() == [False, False, True, False, False]
    two = np.array([[0.5, 0.5], [0.5, 0.6]])            # only the second cue rises: "any" fires, "mean" (+0.05) too
    assert surge_paper(two, 0.04, "any").tolist() == [False, True]
    assert surge_paper(two, 0.06, "mean").tolist() == [False, False] and surge_paper(two, 0.04, "mean")[1]


def test_code_rule_matches_emgarde_trigger_detection():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("eg_trigger", ROOT / "third_party/em_garde/trigger_detection.py")
    eg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eg)
    rng = np.random.default_rng(0)
    for T in (5, 40, 700):                                # 700 > 600 exercises their history trimming
        sims = np.cumsum(rng.normal(0, 0.02, (T, 3)), axis=0) + 0.3
        ours = surge_code(sims, 0.04, "any", fps=2)
        theirs, hist = [], []
        for t in range(T):
            hist.append(torch.tensor(sims[t:t + 1]))
            if len(hist) > 600:
                hist = hist[-500:]
            r = eg.trigger_detection(torch.cat(hist, dim=0), trigger_scheme="surge", fps=2, surge_threshold=0.04)
            theirs.append(bool(r.any()) if hasattr(r, "any") else bool(r))
        assert ours.tolist() == theirs


def test_dedup_cooldown_and_embedding_rule():
    times = np.array([2.0, 2.5, 3.0, 9.0, 9.5])
    trig = np.array([True, True, True, True, True])
    assert dedup(times, trig, "cooldown", 4.0) == [2.0, 9.0]
    assert dedup(times, trig, "none") == times.tolist()
    vecs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32)
    # within 5 s AND similar (>= 0.8) -> suppressed; a different-looking window or a late one is kept
    assert dedup(times, trig, "emgarde", vecs=vecs) == [2.0, 3.0, 9.0]


def test_auc_and_diagnostic_find_a_signal_near_events():
    assert auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0 and auc(np.array([0.5]), np.array([0.5])) == 0.5
    t = np.arange(2.0, 40.0, 0.5)
    sims = np.full((len(t), 1), 0.2)
    sims[(np.abs(t - 20.0) <= 1.0)] = 0.8                 # similarity is high only around the event at t = 20
    s = Sample(id="x", video="v", query="q", events=[Event(20.0)], task="REC")
    d = similarity_diagnostics([s], {"x": {"t": t, "sims": sims}})
    assert d["max sim"] > 0.7 and 0.0 < d["frac_near"] < 0.3
    assert matcher_detections(t, sims, "paper", 0.04, "any", "cooldown", 4.0) == [pytest.approx(19.0)]


def make_video(path, seconds=12, fps=10, size=(64, 48)):
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(seconds * fps):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        f[..., 2] = int(i / fps * 20)
        vw.write(f)
    vw.release()


def test_cue_stage_stores_and_resumes(tmp_path):
    cfg = load_config(None, {"matcher.cache_dir": str(tmp_path)})
    s = Sample(id="c1", video="v", query="tell me when the water boils", events=[Event(5.0)], task="REC")
    f = tmp_path / "cues.jsonl"
    got = run_cues([s], MockEdgeVLM(), cfg, f, log=lambda *_: None)
    assert len(got["c1"]) == 3 and load_cues(f) == got
    assert run_cues([s], MockEdgeVLM(), cfg, f, log=lambda *_: None) == got and len(f.read_text().splitlines()) == 1


def test_embedding_stage_end_to_end_with_random_embedder(tmp_path):
    make_video(tmp_path / "v.avi")
    cfg = load_config(None, {"matcher.cache_dir": str(tmp_path / "c"), "matcher.embedder": "random"})
    s = Sample(id="e1", video="v.avi", query="q", events=[Event(6.0)], task="REC", start_s=0.0, end_s=10.0)
    out = cache_dir_for(cfg, make_meta(cfg))
    out.mkdir(parents=True)
    st = run_sample(s, RandomEmbedder(dim=16), ["a", "b", "c"], cfg, out, tmp_path)
    assert st["status"] == "ran" and st["n"] == 18          # steps at 1.5, 2.0, ..., 10.0: the first 1.5 s cannot fill a window
    t, sims = load_sims(out / "e1.json")
    assert sims.shape == (18, 3) and t[0] == 1.5 and t[-1] == 10.0
    assert np.load(out / "e1.npy").shape == (18, 16)
    assert run_sample(s, RandomEmbedder(dim=16), ["a", "b", "c"], cfg, out, tmp_path)["status"] == "cached"
