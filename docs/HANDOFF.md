# Hand-off: working on this project from a session that can run code on the GPU laptop

Read this first, then `docs/RATIONALE.md` (section 0 for the current state, section 8 for every decision and why). This file holds what the rationale
doc does not: how we work, the commands, and the traps we already fell into.

## The project in three lines
An edge VLM watches egocentric video and must speak at the right moment; it escalates to GPT only when unsure. We are testing each part on OVO-Bench
Forward Active Responding (172 questions, REC / SSR / CRR) with Em-Garde's online F1 (±2 s, no LLM judge), one experiment row at a time (`docs/RATIONALE.md` section 4).
Branch: `abhi`. **The test split (`data/processed/ovo_far_test.jsonl`) is never touched until the very end, once.**

## How we work (the user's preferences, learned over the project)
- The user wants to spend their time on decisions, not on running notebooks: run the scripts yourself, read the outputs, and bring the user the interpretation and the choices.
- **Say what the evidence supports and no more.** Every score is read against a timer making the same number of responses (`docs/RATIONALE.md` section 5). Do not quote a best cell as a result; it is the best of many settings on 43 questions.
  Compare systems with the within-question AUC and the video-level bootstrap interval (`scripts/compare_deciders.py`, `egogate/eval/bootstrap.py`). A gain counts only if the interval excludes zero.
- When the user asks several questions, restate each one before answering it.
- Record every decision and every result in `docs/RATIONALE.md` (decision log in section 8, results in 9a), with the reason and the evidence, in plain language for a reader who has not followed the project.
- Do not run long jobs (more than about 15 minutes) without saying how long they will take first. Do not tune settings further on val without a reason; over 100 settings have already been tried on 43 questions.
- Keep the experiment ladder lean: the user does not want ablation sprawl. One model (Qwen3-VL-2B) until the approach shows promise; no Gemma, no quantization, no GPT or router work yet (decision D22).

## The machine
- Windows laptop, RTX 4060 Laptop GPU, **8.6 GB**. Repo root `C:\Users\abhin\Code\CV_Proj`, virtual environment `.venv` (activate with `.venv\Scripts\Activate.ps1`). `pip install -e ".[models]"` is already done.
- **The decider (Qwen3-VL-2B, 4.3 GB) and the embedder (Ops-MM-2B, 4.4 GB) cannot be loaded together.** Every stage is its own process; never import both in one session.
- Do not game or run other GPU work during a run. Timing numbers are only meaningful on an idle GPU. The laptop must stay plugged in and awake (a val matcher run takes about two hours; the step time drifted up about 25% over a run).
- Windows prints harmless warnings (symlinks in the Hugging Face cache, `torch_dtype` deprecated, `min_pixels`/`max_pixels` deprecated). Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to quiet the first. The `HF_HUB_DISABLE_XET` workaround was only needed in the cloud container.
- Videos: `data/raw/ovo/src_videos/` (val and probably more already fetched; `python scripts/ovo_tar_remote.py fetch --samples <split jsonl> --out data/raw/ovo/src_videos` fetches only what a split needs, a few GB, never the whole 44 GB archive).

## Where things are stored (all under `results/cache/`, committed except `.npy`)
| Stage | Script | Output | Measured cost on val |
|---|---|---|---|
| Cues (edge model, question only) | `scripts/run_cues.py` | `cues/<model>__<key>/cues.jsonl` | a few minutes |
| Matcher: embed each 2 s window every 0.5 s | `scripts/run_embed_cache.py --cues <cues.jsonl>` | `matcher/<model>__<key>/<id>.json` (similarities) and `<id>.npy` (window vectors, large, git-ignored) | 114 minutes, 7,903 steps |
| Decider every 2 s (question only / with cues) | `scripts/run_decider_cache.py [--cues f --set decider.variant=query_cues]` | `decider/<model>__<variant>__<key>/<id>.jsonl` | about 10 minutes, 1,983 decisions |
| Decider at the matcher's trigger moments | `scripts/run_gated_decider.py --matcher <dir> [--cues f ...]` | `<decider dir>/ondemand/<id>.jsonl` | about 8 minutes, 1,841 moments |
Everything is resumable and keyed by model, settings and cue-file hash, so reruns reuse finished work. Analysis scripts (CPU, seconds): `replay_decider.py`, `replay_matcher.py`, `replay_gated.py`, `analyze_matcher.py`, `compare_deciders.py`, `eval_floors.py`,
`recompute_matcher_sims.py` (new cues on stored vectors), `diagnose_speed.py`, `diagnose_embed_speed.py`. Tests: `python -m pytest -q` (38 pass; the video tests need `opencv-python-headless`).
Commit results with `git add results/cache && git commit ... && git pull --rebase origin abhi && git push origin abhi`.

## Traps we already fell into
- Em-Garde's embedder wrapper upscales every frame to 256 tokens and uses a prompt detail that changes the vectors; our code reproduces their vectors (D17). Do not "simplify" it.
- A pooled AUC over all questions mixes in differences between questions; always average the AUC within each question.
- The cooldown (dropping a response within N seconds of the previous one) is worth about ten F1 points on its own; compare with a timer, not with zero.
- Models above about 2B do not fit this GPU (the 3B spilled into shared memory and ran at 37 s per decision).
- The cue writer's JSON sometimes lacks its closing brace; `parse_cue_json` handles it. Re-run the cue stage if the cue file predates that fix.
- `results/cache/**/*.npy` is git-ignored on purpose (the window vectors are large and recomputable); the `.json` files are what get pushed.

## Where we are (2026-11-23) and what is next
Rows 0 to 4 done on val. Findings in one paragraph: the 2B decider ranks moments weakly (AUC about 0.61) however it is prompted; the cue matcher is modestly better than a timer on distinctive clips; the decider adds nothing on top of the matcher; cues in the decider's prompt do not help.
Decision D22: fix the base before routing. In order:
1. **A. Cue-quality ceiling.** `python scripts/recompute_matcher_sims.py --cache results/cache/matcher/<dir> --samples data/processed/ovo_far_val.jsonl --source oracle` (and `--source question`), then `replay_matcher.py` and `analyze_matcher.py` on each output (`<cache>.cues-oracle`, `<cache>.cues-question`).
   Report the Gate 2 diagnostic and the by-group table next to the baseline cues. Rule: if oracle cues clearly lift the matcher, cue wording is a main limit; if not, the embedder is.
2. **B. Scene-graph pilot** (not built): the 2B model writes (subject, relation, object) triplets every 5 s without the question in the prompt (`egogate/evidence.py` has the prompt and parser); try a few val questions, read the triplets, time the generation, measure a quality proxy.
3. **C.** If the graphs are usable: question-as-triplet retrieval, the decider with evidence (row 5), and a memory-aware cue writer that writes short subject-relation-object cues about what should appear next.
Not now: the GPT ceiling check, the router (Gate 3), Gemma, quantization, more decider prompts, dev/test embedding runs.

## To start a fresh session with this context
Paste: "Read docs/HANDOFF.md, then docs/RATIONALE.md section 0 and section 8. We are on branch abhi. Do step A in docs/HANDOFF.md, then tell me what it shows and what you would do next."
