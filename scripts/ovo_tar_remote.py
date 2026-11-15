"""Fetch only the OVO-Bench videos we need, without downloading the 44 GB of source videos.

The source videos are one plain (uncompressed) tar archive cut into five 10 GiB parts on Hugging Face,
and the server supports HTTP range requests. So we can:

  1. `index`:  walk the tar headers (512 bytes per file, hopping over file contents with range
               requests) and record where every video starts. About 650 tiny requests; written to
               data/processed/ovo_src_index.json (small, safe to commit).
  2. `fetch`:  download just the byte ranges of the videos listed in a samples jsonl.

    python scripts/ovo_tar_remote.py index
    python scripts/ovo_tar_remote.py fetch --samples data/processed/ovo_far_all.jsonl --out data/raw/ovo/src_videos

Files already present with the right size are skipped, so an interrupted fetch can simply be re-run.
The dataset revision is pinned so the index stays valid. See docs/RATIONALE.md decision D11.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = "JoeLeelyf/OVO-Bench"
REVISION = "fec29e3385747b5642d995370143ba92d2819bd2"   # pinned; index is only valid for this revision
PARTS = [f"src_videos.tar.parta{c}" for c in "abcde"]
INDEX_PATH = Path("data/processed/ovo_src_index.json")
BLOCK = 512


def part_url(part: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{part}"


class RemoteTar:
    """Random access into the concatenation of the tar parts."""

    def __init__(self):
        self.s = requests.Session()
        self.sizes = self._sizes()
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total = sum(self.sizes)
        self.cdn: dict[str, str] = {}

    def _sizes(self) -> list[int]:
        tree = self.s.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}", timeout=60).json()
        by = {t["path"]: t["size"] for t in tree if t["type"] == "file"}
        return [by[p] for p in PARTS]

    def _get(self, part: str, a: int, b: int) -> bytes:
        """bytes [a, b] inclusive of one part, retrying and re-resolving the signed CDN link."""
        for attempt in range(5):
            try:
                url = self.cdn.get(part, part_url(part))
                r = self.s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120, allow_redirects=True)
                if r.status_code == 206:
                    self.cdn[part] = r.url
                    return r.content
                self.cdn.pop(part, None)
            except requests.RequestException:
                self.cdn.pop(part, None)
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"range {part} {a}-{b} failed")

    def read(self, off: int, n: int) -> bytes:
        out = b""
        while n > 0:
            i = max(k for k, s in enumerate(self.starts) if s <= off)
            local = off - self.starts[i]
            take = min(n, self.sizes[i] - local)
            out += self._get(PARTS[i], local, local + take - 1)
            off += take
            n -= take
        return out


def build_index() -> dict:
    import tarfile
    rt = RemoteTar()
    files, off, pending = {}, 0, {}
    while off + BLOCK <= rt.total:
        hdr = rt.read(off, BLOCK)
        if hdr == b"\0" * BLOCK:
            break
        ti = tarfile.TarInfo.frombuf(hdr, "utf-8", "surrogateescape")
        data_off = off + BLOCK
        padded = (ti.size + BLOCK - 1) // BLOCK * BLOCK
        if ti.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):   # pax extended header: holds the next file's real name
            raw = rt.read(data_off, ti.size).decode("utf-8", "surrogateescape")
            pending = {}
            for line in raw.split("\n"):
                if line and " " in line:
                    k, _, v = line.partition(" ")[2].partition("=")
                    pending[k] = v
        elif ti.isfile():
            name = pending.get("path", ti.name) if pending else ti.name
            size = int(pending.get("size", ti.size)) if pending else ti.size
            name = name.removeprefix("./").removeprefix("src_videos/")
            files[name] = {"offset": data_off, "size": size}
            pending = {}
            padded = (size + BLOCK - 1) // BLOCK * BLOCK
        else:
            pending = {}
        off = data_off + padded
        if len(files) % 100 == 0 and files:
            print(f"  indexed {len(files)} files, {off / 1e9:.1f} GB walked", flush=True)
    return {"repo": REPO, "revision": REVISION, "part_sizes": rt.sizes, "files": files}


def fetch(samples_path: str, out: Path, index: dict):
    from egogate.eval.datasets import load_jsonl
    rt = RemoteTar()
    wanted = sorted({str(s.video) for s in load_jsonl(samples_path)})
    missing = [w for w in wanted if w not in index["files"]]
    if missing:
        print(f"WARNING: {len(missing)} videos not in the index, e.g. {missing[:3]}")
    todo = [w for w in wanted if w in index["files"]]
    total = sum(index["files"][w]["size"] for w in todo)
    print(f"{len(todo)} videos, {total / 1e9:.2f} GB to fetch")
    done = 0
    for w in todo:
        ent = index["files"][w]
        dst = out / w
        if dst.exists() and dst.stat().st_size == ent["size"]:
            done += ent["size"]
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            pos, end = ent["offset"], ent["offset"] + ent["size"]
            while pos < end:
                n = min(64 << 20, end - pos)
                f.write(rt.read(pos, n))
                pos += n
        tmp.rename(dst)
        done += ent["size"]
        print(f"  {w}  ({done / 1e9:.2f}/{total / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    f = sub.add_parser("fetch")
    f.add_argument("--samples", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--index", type=Path, default=INDEX_PATH)
    a = ap.parse_args()
    if a.cmd == "index":
        idx = build_index()
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(idx, open(INDEX_PATH, "w"))
        print(f"indexed {len(idx['files'])} files -> {INDEX_PATH}")
    else:
        fetch(a.samples, a.out, json.load(open(a.index)))
