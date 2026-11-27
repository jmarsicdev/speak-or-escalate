"""Usage: python viewer/dump_frames.py viewer/frames [samples.jsonl] [video_root]
       python viewer/dump_frames.py viewer/frames_ego data/processed/pvqa_ego_n60.jsonl data/raw/proactivevideoqa/EGO/videos

Defaults: OVO val (data/processed/ovo_far_val.jsonl, data/raw/ovo/src_videos). Save a 0.5 s grid of small JPEG frames per val question (from 10 s before its start to its end) for the explorer page."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path(sys.argv[1])


def one(args):
    import cv2
    sid, video, start, end = args
    d = OUT / sid
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    k0, k1 = max(0, int((start - 10.5) * 2)), int(end * 2) + 1
    targets = {k: int(round(k / 2 * fps)) for k in range(k0, k1 + 1)}
    first = min(targets.values())
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(first - 1, 0))
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    n = 0
    for k, idx in sorted(targets.items()):
        if (d / f"{k:05d}.jpg").exists():
            continue
        while pos < idx and pos + 1 < n_total:
            if not cap.grab():
                n_total = pos + 1
                break
            pos += 1
        if pos != idx:
            break
        ok, fr = cap.retrieve()
        if not ok:
            break
        h, w = fr.shape[:2]
        s = 200 / max(h, w)
        fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / f"{k:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        n += 1
    return sid, n


if __name__ == "__main__":
    from egogate.eval.datasets import load_jsonl
    samples = sys.argv[2] if len(sys.argv) > 2 else "data/processed/ovo_far_val.jsonl"
    root = Path(sys.argv[3] if len(sys.argv) > 3 else "data/raw/ovo/src_videos")
    jobs = [(s.id, str(root / s.video), s.start_s or 0, s.end_s) for s in load_jsonl(samples)]
    with ProcessPoolExecutor(4) as ex:
        for sid, n in ex.map(one, jobs):
            print(sid, n, flush=True)
