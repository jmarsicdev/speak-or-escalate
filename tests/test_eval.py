"""Tests for the OVO-Bench ingestion, split, metric and floors."""
import json
import random

from egogate.eval.datasets import Sample, load_jsonl, write_jsonl
from egogate.eval.floors import run_floor
from egogate.eval.metrics import aggregate, match_detections, score_question
from egogate.eval.ovo import decision_times, parse_ovo
from egogate.eval.splits import split_samples
from egogate.types import Event

RAW = [
    {"id": 1, "task": "REC", "video": "perception_test/v1.mp4", "activity": "breaking something",
     "start_times": [6], "end_times": [17], "test_info": [{"realtime": 17, "count": 1}, {"realtime": 32.0, "count": 1}]},
    {"id": 2, "task": "SSR", "video": "COIN/v2.mp4", "tutorial": "PutOnHairExtensions", "all_steps": ["a", "b"],
     "start_time": [25, 31], "end_time": [30, 49], "test_info": [{"realtime": 27}, {"realtime": 49}]},
    {"id": 3, "task": "CRR", "video": "MovieNet/v3.mp4", "question": "What does she do?", "answer": "x",
     "ask_time": 297, "clue_time": 308, "test_info": [{"realtime": 297}, {"realtime": 338}]},
    {"id": 4, "task": "OCR", "video": "x/v4.mp4", "question": "ignored: not forward active responding"},
]


def _write(tmp_path):
    p = tmp_path / "ovo.json"
    p.write_text(json.dumps(RAW))
    return parse_ovo(p)


def test_ovo_parsing_matches_emgarde_ground_truth(tmp_path):
    rec, ssr, crr = _write(tmp_path)  # OCR row is dropped
    assert [e.t for e in rec.events] == [8.0]            # REC: start_times + 2
    assert [e.t for e in ssr.events] == [27.0, 33.0]     # SSR: start_time + 2
    assert [e.t for e in crr.events] == [308.0]          # CRR: clue_time
    assert (rec.start_s, rec.end_s) == (0.0, 32.0)       # REC/SSR judged from 0 to the last test point
    assert (crr.start_s, crr.end_s) == (297.0, 338.0)    # CRR judged from ask_time
    assert rec.query == "How many times does the event: breaking something happen?"
    assert ssr.query == "Describe the steps of put on hair extensions"


def test_decision_grid(tmp_path):
    rec, _, crr = _write(tmp_path)
    assert decision_times(rec, 2.0)[:3] == [2.0, 4.0, 6.0] and decision_times(rec, 2.0)[-1] == 32.0
    assert decision_times(crr, 2.0)[0] == 299.0 and len(decision_times(crr, 0.5)) == 82


def test_jsonl_roundtrip_keeps_window(tmp_path):
    ss = _write(tmp_path)
    write_jsonl(ss, tmp_path / "o.jsonl")
    back = load_jsonl(tmp_path / "o.jsonl")
    assert [(s.id, s.start_s, s.end_s) for s in back] == [(s.id, s.start_s, s.end_s) for s in ss]


def _emgarde_reference(det, gt_times, tol=2):
    """Verbatim logic of the scoring loop in Em-Garde's ovo_bench_eval_online.py (MIT)."""
    det_copy = det.copy()
    correct = 0
    for gt in gt_times:
        for i, t in enumerate(det_copy):
            if abs(t - gt) <= tol:
                correct += 1
                det_copy.remove(t)
                break
    recall = correct / len(gt_times)
    precision = correct / len(det) if len(det) > 0 else 0.0
    return recall, precision


def test_metric_matches_emgarde_reference_on_random_cases():
    rng = random.Random(1)
    for _ in range(500):
        gt = sorted(rng.uniform(0, 60) for _ in range(rng.randint(1, 5)))
        det = sorted(round(rng.uniform(0, 60) * 2) / 2 for _ in range(rng.randint(0, 12)))
        q = score_question(det, gt, 2.0)
        assert (q.recall, q.precision) == _emgarde_reference(det, gt, 2)


def test_aggregate_is_f1_of_averages_not_average_of_f1():
    rows = [{"task": "REC", "recall": 1.0, "precision": 0.0}, {"task": "REC", "recall": 0.0, "precision": 1.0}]
    t = aggregate(rows)["tasks"]["REC"]
    assert t["recall"] == 0.5 and t["precision"] == 0.5 and t["f1"] == 0.5  # average of per-question F1s would be 0


def _fake_samples(n_videos=60, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_videos):
        for k in range(rng.choice([1, 1, 1, 3])):
            task = rng.choice(["REC", "SSR", "CRR"])
            out.append(Sample(id=f"s{i}_{k}", video=f"{task}/v{i}.mp4", query="q", events=[Event(10.0 + k)],
                              task=task, end_s=40.0))
    return out


def test_split_is_deterministic_grouped_by_video_and_covers_everything():
    ss = _fake_samples()
    a, b = split_samples(ss, seed=0), split_samples(ss, seed=0)
    assert {k: [s.id for s in v] for k, v in a.items()} == {k: [s.id for s in v] for k, v in b.items()}
    assert sum(len(v) for v in a.values()) == len(ss)
    videos = {k: {str(s.video) for s in v} for k, v in a.items()}
    assert not (videos["dev"] & videos["val"]) and not (videos["dev"] & videos["test"]) and not (videos["val"] & videos["test"])
    assert {s.id for s in split_samples(ss, seed=1)["test"]} != {s.id for s in a["test"]}  # seed matters


def test_floors_sanity():
    ss = _fake_samples()
    assert run_floor("always_silent", ss)["avg_f1"] == 0.0
    assert run_floor("oracle", ss)["avg_f1"] == 1.0
    assert 0.0 < run_floor("always_respond", ss)["avg_f1"] < 1.0
