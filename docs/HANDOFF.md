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
- `results/cache/**/*.npy` is git-ignored on purpose (the window vectors are large and recomputable); the `.json` files are what get pushed. The SigLIP2 per-frame vectors (`*.frames.npz`, 37 MB) were committed; `.npz` is not covered by that rule.
- `results/*.json` is git-ignored: anything meant to be committed goes under `results/cache/` (a `git add` that names an ignored file fails as a whole).
- API log-probabilities come from the top 20 tokens; check that Yes and No are both present (no -30 floor values) and that values are not heavily tied before trusting an AUC (both checked for the 8B and 235B).
- A plain image of the first seconds is not representative: many sports clips open with a title card (the frame-grounded cues described it).
- With the history prompt, the first 10 s of a video have no 10 s of history; the earliest frame stands in.

## Larger models through the API (since 2026-11-23)
- The 8B and 235B Qwen3-VL models run through OpenRouter (`edge.backend=api`, D23). The key is read from the gitignored `.env` (`OPENROUTER_API_KEY=...`); `pip install -e ".[api]"` installs the client. Requests are restricted to providers that return log-probabilities, and rate-limit errors (429) are retried up to 10 times.
- API runs decode video on the laptop's CPU and send frames; they do not use the GPU, so they can run alongside a GPU job. Every run stores each decision as it goes and resumes with the same command after a crash or a restart of the session.
- New scripts (all write under `results/cache/`): `run_reference_model.py` (row 1 with any API model), `run_decider_history.py` (history frames; variants frames, change, change_q, gate_q), `write_cues_8b.py`, `run_cue_presence.py`, `run_embed_alt.py` (SigLIP2, now also stores per-frame vectors), `run_embed_xclip.py`; CPU analyses `pilot_window_cues.py`, `pilot_label_radius.py`, `score_cueset_siglip.py`, `rescore_thresholds.py`, `bootstrap_rescore.py`, `build_change_gate.py`, `d24_combine.py`.
- The viewer: `viewer/rescore.html` (built by `viewer/build_viewer.py`; frame thumbnails from `viewer/dump_frames.py`, gitignored).

## Where we are (2026-11-26) and what is next
Done on val since D22 (details and numbers in RATIONALE 9a): cue-quality ceiling (step A), scene-graph pilot and row 5, several-window and contrast tests, SigLIP2 and X-CLIP embedders, label radius, 8B and 235B deciders, shorter windows, scene-level cues, decider history and "what changed" (neutral and question-aware), threshold rescoring with intervals, the change sentence as a gate. In one paragraph: every embedder tried barely moves at events (X-CLIP is worst); cue wording is not the limit; the 8B ranks moments no better than the 2B, the 235B a little better; text in a decider's prompt makes it agree more and discriminate less; scored as responses with a per-model threshold, deciders do beat a per-question timer on counting and tutorials (with intervals); history and a question-aware change check help tutorials, not counting.
Decision D24 (RATIONALE section 8): the 8B is the main model; other evidence is combined with its p(Yes) by formula instead of going into its prompt. Stage 0 is done (8B cues; frame-grounded cues). Stage 1 signals C (cue presence, both cue sets) and G (graded change gate) are running; then `python scripts/d24_combine.py --samples data/processed/ovo_far_val.jsonl` runs Stages 1 to 3 on the CPU.
Also open: other benchmarks (StreamingBench's proactive output task, ESTP-Bench), to be checked for license, size, released timing labels and whether scoring needs an LLM judge.
Not now: the router (Gate 3), Gemma, quantization, dev/test runs.

## Context that is not written anywhere else
- **People and roles.** The user (branch `abhi`) owns the experiments in this repo. A teammate (jmarsicdev) wrote the first scaffold; quantization and Jetson deployment of the edge model are separate work by other teammates, out of scope here. This is a course project (ITCS 6169/8169).
- **Budget.** About $100 of GPT calls including the benchmark's judge (only needed for ProactiveVideoQA PAUC later). No compute budget was stated; the user asked about UNC Charlotte's research cluster as a possible free source of GPUs (unknown access, GPU type, or whether compute nodes have internet).
- **Open items.** O2 (put the question in the memory prompt or not; Response-G1 does, our notes say it causes early triggers), O5 (a 7B reference run needs a bigger GPU). Everything else in the open list was settled in D18 to D22.
- **References** are in `docs/papers/` (Em-Garde, Response-G1, VideoJev, ProactiveVideoQA, OVO-Bench, VideoLLM-online, Guo et al. calibration, Qwen3-VL). Em-Garde's code is public (github.com/air-embodied-brain/Em-Garde, MIT; a wrapper and its trigger rule are vendored in `third_party/em_garde/`); ProactiveVideoQA's scorer is fetched by `scripts/fetch_proactivevideoqa.sh`, not committed (no license).
- **Things said to the user that were later corrected** (so do not repeat them): the decider's AUC is 0.614 within questions, not 0.71 (pooled); the streaming shortcut is not a small numerical difference but a different prompt ending (fixed by the newline); frame sizes below 448 px do not save compute for the decider and the matcher's frames are upscaled to 256 tokens by the wrapper.
- **What the user cares about:** understanding why each choice was made (every decision has a reason and evidence in the rationale), honest reads over optimistic ones, getting to the question "does the approach work" without drowning in ablations, and keeping the GPU laptop usable.

## To start a fresh session with this context
Paste: "Read docs/HANDOFF.md, then docs/RATIONALE.md section 0, D24 in section 8, and the last sections of 9a. We are on branch abhi. Continue D24 from where 'Where we are' leaves it."
