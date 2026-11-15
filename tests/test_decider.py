"""Frame windows, prompt builder, decider cache runner (with a mock model) and replay."""
import json

import numpy as np
import pytest

from egogate.config import load_config
from egogate.decider_cache import cache_dir_for, make_meta, read_rows, run_sample
from egogate.edge import MockEdgeVLM, build_speak_prompt, prompt_fingerprint
from egogate.eval.datasets import Sample
from egogate.frames import window_sample_times
from egogate.replay import detections, p_yes, score_setting
from egogate.types import Event

cv2 = pytest.importorskip("cv2")
from egogate.frames import stream_windows  # noqa: E402


def make_video(path, seconds=12, fps=10, size=(64, 48)):
    """Red channel encodes the time in seconds (x20), so a test can read back which moment a frame shows."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(seconds * fps):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[..., 2] = int(i / fps * 20)  # BGR: index 2 is red
        vw.write(frame)
    vw.release()


def frame_time(frame):
    return frame[..., 0].mean() / 20.0  # RGB after conversion: index 0 is red


def test_window_sample_times():
    assert window_sample_times(6.0, 2.0, 2.0) == [4.5, 5.0, 5.5, 6.0]


def test_windows_are_causal_and_aligned(tmp_path):
    make_video(tmp_path / "v.avi")
    wins = list(stream_windows(tmp_path / "v.avi", [2.0, 6.0, 10.0], 2.0, 2.0, max_side=None))
    assert [w.t_end for w in wins] == [2.0, 6.0, 10.0]
    for w in wins:
        times = [frame_time(f) for f in w.frames]
        assert len(times) == 4
        assert max(times) <= w.t_end + 0.2          # never a frame from the future
        assert abs(times[-1] - w.t_end) < 0.35 and abs(times[0] - (w.t_end - 1.5)) < 0.35


def test_ticks_beyond_the_video_are_skipped(tmp_path):
    make_video(tmp_path / "v.avi", seconds=6)
    assert [w.t_end for w in stream_windows(tmp_path / "v.avi", [2.0, 4.0, 8.0, 10.0], 2.0, 2.0, None)] == [2.0, 4.0]


def test_frames_are_downscaled(tmp_path):
    make_video(tmp_path / "v.avi", size=(640, 480))
    w = next(iter(stream_windows(tmp_path / "v.avi", [2.0], 2.0, 2.0, max_side=320)))
    assert max(w.frames[0].shape[:2]) == 320


def test_prompt_includes_cues_and_evidence_only_when_given():
    base = build_speak_prompt("when does it boil?", n_frames=4, window_s=2.0)
    assert "cues" not in base.lower() and "evidence" not in base.lower() and "4 frames" in base and "Yes or No" in base
    full = build_speak_prompt("q", ["bubbles"], ["(pot, on, stove)"])
    assert "- bubbles" in full and "- (pot, on, stove)" in full
    assert len(prompt_fingerprint()) == 10


def test_p_yes_and_cooldown():
    row = lambda t, y: {"t": t, "lp_yes": np.log(y), "lp_no": np.log(1 - y)}  # noqa: E731
    assert abs(p_yes(row(0, 0.8)) - 0.8) < 1e-9
    assert p_yes(row(0, 0.8), temperature=2.0) < 0.8       # T > 1 softens
    rows = [row(2, 0.9), row(4, 0.9), row(6, 0.1), row(8, 0.9)]
    assert detections(rows, 0.5) == [2, 4, 8]
    assert detections(rows, 0.5, cooldown_s=4.0) == [2, 8]


def test_runner_end_to_end_with_mock_model(tmp_path):
    make_video(tmp_path / "v.avi")
    sample = Sample(id="s1", video="v.avi", query="tell me when it happens", events=[Event(6.0)], task="REC",
                    start_s=0.0, end_s=10.0)
    cfg = load_config(None, {"decider.cache_dir": str(tmp_path / "cache"), "edge.backend": "mock"})
    # the "model" says Yes only when the newest frame shows ~6 s
    edge = MockEdgeVLM(p_speak_fn=lambda imgs: 0.95 if abs(frame_time(imgs[-1]) - 6.0) < 0.6 else 0.05)
    out_dir = cache_dir_for(cfg, make_meta(cfg, "mock/model"))
    out_dir.mkdir(parents=True)

    st = run_sample(sample, edge, cfg, out_dir, tmp_path)
    assert st["status"] == "ran" and st["n"] == 5            # ticks 2,4,6,8,10
    assert run_sample(sample, edge, cfg, out_dir, tmp_path)["status"] == "cached"   # second run reuses it
    rows = read_rows(out_dir / "s1.jsonl")
    assert {"t", "lp_yes", "lp_no", "ms", "n_frames"} <= set(rows[0]) and rows[0]["n_frames"] == 4

    res = score_setting([sample], {"s1": rows}, tau=0.5)
    assert res["avg_f1"] == 1.0 and res["total_detections"] == 1


def test_interrupted_run_resumes_without_recomputing(tmp_path):
    make_video(tmp_path / "v.avi")
    sample = Sample(id="s2", video="v.avi", query="q", events=[Event(6.0)], task="REC", start_s=0.0, end_s=10.0)
    cfg = load_config(None, {"decider.cache_dir": str(tmp_path / "cache"), "edge.backend": "mock"})
    calls = []
    edge = MockEdgeVLM(p_speak_fn=lambda imgs: calls.append(1) or 0.5)
    out_dir = cache_dir_for(cfg, make_meta(cfg, "mock/model"))
    out_dir.mkdir(parents=True)
    (out_dir / "s2.jsonl.part").write_text(
        "".join(json.dumps({"t": t, "lp_yes": 0.0, "lp_no": 0.0, "ms": 1, "n_frames": 4}) + "\n" for t in (2.0, 4.0)))
    st = run_sample(sample, edge, cfg, out_dir, tmp_path)
    assert st["n"] == 5 and st["new"] == 3 and len(calls) == 3   # only ticks 6, 8, 10 were computed


def test_changing_the_prompt_or_model_changes_the_cache_folder(tmp_path):
    cfg = load_config(None, {"decider.cache_dir": str(tmp_path)})
    a = cache_dir_for(cfg, make_meta(cfg, "Qwen/Qwen2.5-VL-3B-Instruct"))
    b = cache_dir_for(cfg, make_meta(cfg, "google/gemma-3-4b-it"))
    cfg2 = load_config(None, {"decider.cache_dir": str(tmp_path), "edge.dtype": "float16"})
    c = cache_dir_for(cfg2, make_meta(cfg2, "Qwen/Qwen2.5-VL-3B-Instruct"))
    assert len({a, b, c}) == 3
