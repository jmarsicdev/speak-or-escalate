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
  evidence.py   scene-graph triplet memory with top-k retrieval
  edge.py       EdgeVLM: HF wrapper for Qwen2.5-VL / Gemma 3, MockEdgeVLM; p_speak from Yes/No logits
  cloud.py      OpenAI client with call counting, MockCloud
  router.py     BandPolicy (lo / hi thresholds)
  pipeline.py   run_stream(): glues everything, logs sims/deltas/p_speak per window
  eval/
    metrics.py  online recall/precision with tolerance window, calibration table, run summary
    datasets.py normalised JSONL loader, toy samples
configs/default.yaml        every knob with its default
scripts/
  run_baseline.py           run the pipeline -> results/baseline/{summary.json, <id>.jsonl, <id>.calib.json}
  ingest_dataset.py         raw benchmark -> data/processed/*.jsonl   (OVO parser is a TODO)
  fetch_proactivevideoqa.sh clone the ProactiveVideoQA scorer at a pinned commit (see third_party/)
tests/test_core.py          matcher, router, metric, cue parsing, end-to-end mock run
data/                       raw/ and processed/ stay local (gitignored); only .gitkeep is tracked
results/                    run outputs; small JSON/CSV/JSONL that we read back from git
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

Response timing is scored with online recall/precision: a response counts if it lands within
`[event_t, event_t + tol_s]` (default 4 s) of a ground-truth event, one match per event.
No LLM-as-judge. OVO-Bench's original offline accuracy is gameable (always answering "yes"
scores ~60%), so we don't use it. `summary.json` also reports trigger count, false positives,
and cloud calls per sample; `<id>.calib.json` bins p_speak against whether the trigger was a
real response time.

## Next steps (in order)

1. **Data.** Download OVO-Bench (Forward Active Responding: CRR / SSR / REC) and/or StreamingBench
   Proactive Output. Finish `ingest_ovo()` in `scripts/ingest_dataset.py` against the real field
   names and write `data/processed/dev.jsonl`.
2. **Embedder.** Run `OpsMMEmbedder` once on the GPU box; the `get_text_embeddings` /
   `get_video_embeddings` method names in `egogate/embed.py` are placeholders to fix from the model card.
3. **Edge model.** Validate `HFEdgeVLM` with `Qwen/Qwen2.5-VL-3B-Instruct`, then `google/gemma-3-4b-it`.
   Check the Yes/No token ids and chat template per family.
4. **Baseline numbers.** Run on the dev split; look at F1, false positives, and cloud calls per video.

Experiments from the project doc (random GPT calls vs. confidence, matcher variants, evidence-only
payload, threshold sweeps, decision-by-score vs. generation) are not in the repo yet; they come
after the baseline numbers exist.
