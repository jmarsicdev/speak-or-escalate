"""Second and third benchmarks: StreamingBench Proactive Output and ProactiveVideoQA, scored on timing only (no LLM judge).

Both are on Hugging Face as zip archives; the zips are read remotely (HTTP range requests through HfFileSystem) and only the needed videos
are extracted, as scripts/ovo_tar_remote.py does for OVO-Bench. Output: data/processed/<name>.jsonl in the usual Sample format, so
run_decider_cache.py, run_cue_presence.py and the rest run on it unchanged.

  --bench sb     StreamingBench "Proactive Output", the 50 evaluated questions (Proactive_Output_50.csv), sample_N/video.mp4.
                 One ground-truth time per question; the question says "When X, output Y". start_s = the question's time stamp,
                 end_s = start_s + 60 s (capped by the video), a fixed horizon that does not depend on the answer time (the paper's
                 own protocol polls only [-4, +4] s around the answer, which leaks when it is). task "SBPO", answered once.
  --bench pvqa   ProactiveVideoQA, one subset (EGO, WEB or VAD; TV needs subtitles and audio and is left out). Every reply has an
                 interval (reply_timespan), often several per question; they go to meta["intervals"] and events[].t is each interval's
                 start. start_s = the question's time (0), end_s = the video duration. --n picks a seeded random subset.

    python scripts/ingest_proactive_benchmarks.py --bench sb
    python scripts/ingest_proactive_benchmarks.py --bench pvqa --subset EGO --n 60
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAW = Path("data/raw")


def hms(x: str) -> float:
    h, m, s = (float(v) for v in x.split(":"))
    return h * 3600 + m * 60 + s


def duration(path: Path) -> float:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return n / fps


def extract(repo_zip: str, members: dict[str, Path]):
    """Copy zip members (name -> local path) out of a remote zip, skipping files already present."""
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    todo = {k: v for k, v in members.items() if not v.exists()}
    if not todo:
        return
    with HfFileSystem().open(repo_zip, "rb", block_size=8 << 20) as f:
        z = zipfile.ZipFile(f)
        for i, (name, dst) in enumerate(sorted(todo.items()), 1):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            with z.open(name) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 8 << 20)
            tmp.rename(dst)
            print(f"  [{i}/{len(todo)}] {name} -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)


def ingest_sb():
    """The question-to-video mapping comes from the benchmark's own src/data/questions_proactive_50.json (each entry: a video_path
    ./videos/sample_N_proactive.mp4 and its questions; 50 videos, 51 questions). The CSV's question numbers are NOT video numbers: read
    as sample_N, 6 of 50 questions fall after the end of their video. Times are taken from the JSON, which the official evaluation uses."""
    subsets = json.load(open(RAW / "streamingbench/questions_proactive_50.json", encoding="utf-8"))
    vdir = RAW / "streamingbench/videos"
    from huggingface_hub import HfFileSystem  # noqa: PLC0415

    for zp in ("Proactive Output_1-25.zip", "Proactive Output_26-50.zip"):
        path = f"datasets/mjuicem/StreamingBench/{zp}"
        with HfFileSystem().open(path, "rb", block_size=1 << 20) as f:   # most samples hold video.mp4, a few another name: take the one .mp4
            names = [i.filename for i in zipfile.ZipFile(f).infolist() if i.filename.endswith(".mp4") and not i.filename.startswith("__MACOSX")]
        extract(path, {n: vdir / f"{n.split('/')[0]}.mp4" for n in names})
    out = []
    for sub in subsets:
        sample = Path(sub["video_path"]).stem.replace("_proactive", "")   # sample_N
        v = vdir / f"{sample}.mp4"
        dur = duration(v)
        for j, q in enumerate(sub["questions"]):
            start, gt = hms(q["time_stamp"]), hms(q["ground_truth_time_stamp"])
            out.append({"id": f"sbpo_{sample.split('_')[1]}" + (f"_{j}" if j else ""), "video": f"{sample}.mp4", "query": q["question"],
                        "task": "SBPO", "events": [{"t": gt, "answer": q["ground_truth_output"]}], "start_s": start, "end_s": min(start + 60.0, dur),
                        "meta": {"source": "streamingbench", "duration": dur, "category": sub.get("video_categories"),
                                 "required_ability": q.get("required_ability")}})
    return out, vdir


def ingest_pvqa(subset: str, n: int | None, seed: int):
    anno = json.load(open(RAW / f"proactivevideoqa/{subset}/anno.json", encoding="utf-8"))
    if n:
        anno = random.Random(seed).sample(anno, min(n, len(anno)))
    vdir = RAW / f"proactivevideoqa/{subset}/videos"
    extract(f"datasets/wangyueqian/ProactiveVideoQA/{subset}/videos.zip", {f"videos/{a['video']}": vdir / a["video"] for a in anno})
    out = []
    for a in anno:
        v = vdir / a["video"]
        q = next(c for c in a["conversation"] if c["role"] == "user")
        iv = [[float(x) for x in r["reply_timespan"]] for r in a["answer"]]
        dur = float(a.get("duration") or duration(v))
        out.append({"id": f"pvqa_{subset.lower()}_{a['question_id']}", "video": a["video"], "query": q["content"], "task": f"PVQA-{subset}",
                    "events": [{"t": s, "answer": r["content"]} for (s, _), r in zip(iv, a["answer"])], "start_s": float(q.get("time", 0.0)),
                    "end_s": dur, "meta": {"source": f"pvqa-{subset}", "intervals": iv, "duration": dur}})
    return out, vdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["sb", "pvqa"], required=True)
    ap.add_argument("--subset", choices=["EGO", "WEB", "VAD"], default="EGO")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, vdir = ingest_sb() if a.bench == "sb" else ingest_pvqa(a.subset, a.n, a.seed)
    name = "sb_po" if a.bench == "sb" else f"pvqa_{a.subset.lower()}" + (f"_n{a.n}" if a.n else "")
    out = Path("data/processed") / f"{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions -> {out}; videos in {vdir} (pass --set data.video_root={vdir.as_posix()} to the runners)")


if __name__ == "__main__":
    main()
