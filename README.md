# speak-or-escalate

Edge VLM that watches an egocentric video stream and decides, per moment, whether to
**speak** (answer on-device), **stay silent**, or **escalate** to a cloud model when uncertain.

This repo is the **baseline and evaluation setup**, run on a desktop GPU. Quantization and
Jetson deployment of the edge model are separate work by the other half of the team; the
`edge.quant` config key is wired through for them but is not used here.

The Python package is `egogate/`.

## Baseline pipeline

```
query ──► edge VLM parses 3 visual cues (once) ──► embed cues
                                                     │
video ──► 2 s windows @ 2 fps ──► Matcher: cosine(window, cues) ──► surge > θ ? ── no ──► silent
             │                                                            │ yes
             └──► EvidenceMemory: scene-graph triplets every 5 s          ▼
                              (top-k retrieved) ──────────────► edge VLM: p(speak now)
                                                                           │
                                 p ≥ hi: edge answers │ lo < p < hi: cloud answers │ p ≤ lo: silent
```

The cloud model (GPT) is only called in the uncertain band. Cue parsing, evidence
extraction, p(speak), and confident answers are all the edge model.

Built on: VideoLLM-online (speak-or-silent decision per frame), Em-Garde (parse cues once,
cheap embedding matcher, θ = 0.04 surge trigger), Response-G1 (scene-graph evidence memory),
VideoJev (a score rather than generated text as the routing signal).

## Layout

```
egogate/                    the pipeline package (our code)
  types.py      dataclasses shared across stages (Window, MatchResult, Decision, Event)
  config.py     YAML loader (configs/default.yaml + optional overlay + --set overrides)
  video.py      frame sampling + sliding windows
  cues.py       CueParser: query -> visual cues, via the edge model
  embed.py      Embedder (Ops-MM-V1-2B wrapper, RandomEmbedder) and the surge Matcher
  frames.py     causal 2 s frame windows at chosen decision times
  decider_cache.py  run the decider over a video, store raw Yes/No scores, resumable
  replay.py     stored scores -> p(Yes) -> detections, any threshold / temperature / cooldown
  evidence.py   scene-graph triplet memory with top-k retrieval
  edge.py       EdgeVLM: HF wrapper for Qwen3-VL / Gemma 3, MockEdgeVLM; p_speak from Yes/No logits
  cloud.py      OpenAI client with call counting, MockCloud
  router.py     BandPolicy (lo / hi thresholds)
  pipeline.py   run_stream(): glues everything, logs sims/deltas/p_speak per window
  eval/
    metrics.py  Em-Garde online recall/precision/F1 (+/- 2 s), calibration table, run summary
    datasets.py normalised JSONL loader, toy samples
    ovo.py      OVO-Bench forward-task parsing and the decision-time grid
    splits.py   seeded dev/val/test split, grouped by video
    floors.py   no-skill reference systems
configs/default.yaml        every knob with its default
scripts/
  run_baseline.py           run the pipeline -> results/baseline/{summary.json, <id>.jsonl, <id>.calib.json}
  ingest_dataset.py         OVO-Bench annotations -> seeded dev/val/test jsonl
  score_detections.py       score any system's detections with the Em-Garde metric
  eval_floors.py            row 0: always-silent / always-respond / random / oracle
  run_decider_cache.py      row 1 stage 1: edge model alone -> stored Yes/No scores (resumable)
  replay_decider.py         row 1 stage 2: sweep threshold / cooldown on stored scores (CPU)
  ovo_tar_remote.py         fetch only the needed OVO-Bench videos (5 GB) by HTTP range request
  fetch_proactivevideoqa.sh clone the ProactiveVideoQA scorer at a pinned commit (see third_party/)
tests/test_core.py          matcher, router, metric, cue parsing, end-to-end mock run
data/                       raw/ and processed/ stay local (gitignored); only .gitkeep is tracked
results/                    run outputs; small JSON/CSV/JSONL that we read back from git
docs/RATIONALE.md           what we do, why, the experiment ladder, gates and the decision log
docs/papers/                the 8 reference PDFs, named <short-name>_<arxiv-id>.pdf, with a README
                            saying what we take from each. Kept in the repo so they are read page by
                            page when a detail matters, instead of pasted into a chat.
third_party/                code from other projects, kept apart from egogate/ so it is never confused
                            with ours (see third_party/README.md for sources and pinned commits)
  em_garde/                 MIT-licensed Ops-MM embedding wrapper and config from the Em-Garde repo.
                            It is the reference for fixing the placeholder method names in
                            egogate/embed.py and confirms the HF id. Vendored because its license allows it.
  ProactiveVideoQA/         NOT committed: fetched by scripts/fetch_proactivevideoqa.sh. That repo has no
                            license file, so we fetch it rather than copy it. It holds the PAUC scorer,
                            which we run unchanged so our numbers match the benchmark's.
```

Quick glossary, since the config uses it: **BF16 / FP16 / FP32** are number formats for model weights. FP32 uses 4 bytes per
value and is the full-precision reference. FP16 and BF16 both use 2 bytes, so they halve the memory. FP16 keeps more fine
detail but overflows on large values; BF16 keeps FP32's range with less detail and is safer for large models. The baseline
stage uses BF16 and no quantization (INT8 / INT4), which is a separate later experiment.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests, no GPU needed
pip install -e ".[models,cloud]"   # on the GPU box
export OPENAI_API_KEY=...          # only for cloud.backend=openai
pytest -q
```

Dry run with mocks (no models, no video, no key):

```bash
python scripts/run_baseline.py --set data.toy=true
```

Real run, once data and models are in place:

```bash
python scripts/run_baseline.py --set edge.backend=hf --set matcher.embedder=opsmm --set cloud.backend=openai
```

Any config key can be overridden with `--set dotted.key=value`; the resolved config is saved in
`results/<name>/summary.json`.

## Metric

Response timing is scored the way Em-Garde scores OVO-Bench Forward Active Responding: a response is
correct if it lands within **+/- 2 s** of a ground-truth time, one match per event; recall and precision
are averaged per task and combined into F1. No LLM-as-judge. The implementation is
`egogate/eval/metrics.py` and is tested against Em-Garde's own scoring loop. Why these exact rules, and
what a coin-flip system scores on this metric (about 21 F1), is in `docs/RATIONALE.md`.

## Experiments and rationale

`docs/RATIONALE.md` is the document to read first: the problem, the experiment ladder (rows 0-6, one change
per row, compared against the edge-only baseline), the gates, the data and splits, and a decision log that
records each choice with the paper or code behind it.

Evaluation tooling that exists now (no models or video needed):

```bash
python scripts/ingest_dataset.py ovo --ann path/to/ovo_bench_new.json --out-dir data/processed --seed 0
python scripts/eval_floors.py --samples data/processed/ovo_far_val.jsonl     # row 0: no-skill baselines
python scripts/score_detections.py --samples data/processed/ovo_far_val.jsonl --detections <file>.jsonl
```

`data/processed/ovo_far_{all,dev,val,test}.jsonl` hold the OVO-Bench forward-task annotations and the seeded
splits (annotations only, CC BY-NC-SA 4.0; videos are never committed).

## Next steps (in order)

1. **Download videos** on the GPU laptop, only the 134 we use (5 GB, not 44 GB):
   `python scripts/ovo_tar_remote.py fetch --samples data/processed/ovo_far_all.jsonl --out data/raw/ovo/src_videos`.
   Set `data.video_root` to that folder.
2. **Row 1, edge model alone** (built): open `notebooks/row1_decider_alone.ipynb`. It runs the decider every 2 s
   (`scripts/run_decider_cache.py`), stores the Yes/No scores under `results/cache/`, and scores thresholds on CPU
   (`scripts/replay_decider.py`). Push `results/cache/` back so it can be read. Gemma next.
3. **Stage 1a and the on-demand decider** (D13) for rows 3-4.
4. **Rows 2-5**, then the router (row 6).
