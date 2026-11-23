"""One command to run row 1 (the decider alone) with a different, larger model and compare it with the 2B model. See docs/RUN_WITH_LARGER_MODEL.md.

    # how many calls and images would it take (no key, no GPU needed)
    python scripts/run_reference_model.py --estimate --limit-questions 6

    # through an API (OpenRouter by default); the key is read from the environment, never stored
    set OPENROUTER_API_KEY=...                       (Windows)   /   export OPENROUTER_API_KEY=...   (Linux, macOS)
    python scripts/run_reference_model.py --model qwen/qwen3-vl-8b-instruct --limit-questions 6

    # on a bigger GPU, with the weights downloaded, same command shape
    python scripts/run_reference_model.py --backend hf --model Qwen/Qwen3-VL-8B-Instruct --limit-questions 6

It (1) checks that the model returns Yes/No probabilities (API only), (2) runs the decider on the question subset into results/cache/decider/,
(3) compares it with the 2B decider (row 1) on the same questions: within-question AUC and a video-level 95% interval.
Resumable: run the same command again after an interruption. Raise or drop --limit-questions only after the small run looked right.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from egogate.eval.datasets import load_jsonl  # noqa: E402
from egogate.eval.ovo import decision_times  # noqa: E402


def pick(samples, n):
    """n questions spread over the four groups (shortest first within a group), so a small run still covers every kind."""
    by: dict = {}
    for s in sorted(samples, key=lambda s: (s.end_s or 0) - s.start_s):
        by.setdefault(s.meta.get("source", s.task), []).append(s)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)][: n - len(out)]
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/processed/ovo_far_val.jsonl")
    ap.add_argument("--model", default="qwen/qwen3-vl-8b-instruct", help="API model name, or a Hugging Face id with --backend hf")
    ap.add_argument("--backend", choices=["api", "hf"], default="api")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--limit-questions", type=int, default=6, help="how many questions (spread over the four groups); 0 = all of the split")
    ap.add_argument("--estimate", action="store_true", help="only count the calls and images, then stop")
    a = ap.parse_args()

    samples = load_jsonl(a.samples)
    chosen = pick(samples, a.limit_questions) if a.limit_questions else samples
    n_calls = sum(len(decision_times(s, 2.0)) for s in chosen)
    print(f"{len(chosen)} questions, {n_calls} decisions, {n_calls * 4} images (4 frames of 2 s each); model {a.model} via {a.backend}")
    if a.estimate:
        print("Multiply the images by your provider's price per image to get the cost; one decision is one call with 4 images and 1 output token.")
        return

    # a question list file for the subset, so the existing scripts run on exactly these questions
    sub = ROOT / "results" / "cache" / "reference_subset.jsonl"
    sub.parent.mkdir(parents=True, exist_ok=True)
    keep = {s.id for s in chosen}
    with open(a.samples, encoding="utf-8") as f, open(sub, "w", encoding="utf-8") as g:
        import json  # noqa: PLC0415

        for line in f:
            if line.strip() and json.loads(line)["id"] in keep:
                g.write(line)

    sets = ["--set", f"edge.backend={a.backend}", "--set", f"edge.model_id={a.model}"]
    if a.backend == "api":
        sets += ["--set", f"edge.base_url={a.base_url}", "--set", f"edge.api_key_env={a.api_key_env}"]
        from egogate.edge import APIEdgeVLM  # noqa: PLC0415

        print("checking that the model returns Yes/No probabilities ...")
        lp = APIEdgeVLM(a.model, a.base_url, a.api_key_env).check_logprobs()
        print(f"  ok (log P(Yes)={lp[0]:.2f}, log P(No)={lp[1]:.2f})")

    run = [sys.executable, str(ROOT / "scripts" / "run_decider_cache.py"), "--samples", str(sub)] + sets
    if subprocess.call(run, cwd=ROOT) != 0:
        raise SystemExit("the decider run failed (see above); re-run the same command to resume")

    tag = a.model.split("/")[-1]
    new = sorted((ROOT / "results" / "cache" / "decider").glob(f"{tag}__query_only__*"), key=lambda p: p.stat().st_mtime)[-1]
    base = sorted((ROOT / "results" / "cache" / "decider").glob("Qwen3-VL-2B-Instruct__query_only__*"), key=lambda p: p.stat().st_mtime)
    if not base:
        print(f"no 2B row 1 cache found to compare with; the new results are in {new}")
        return
    print(f"\ncomparing {new.name} (B) with {base[-1].name} (A) on the same questions:")
    subprocess.call([sys.executable, str(ROOT / "scripts" / "compare_deciders.py"), "--a", str(base[-1]), "--b", str(new), "--samples", str(sub)], cwd=ROOT)


if __name__ == "__main__":
    main()
