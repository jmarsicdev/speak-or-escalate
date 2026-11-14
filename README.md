# speak-or-escalate

Edge VLM on a Jetson that watches an egocentric video stream and decides, per moment, whether to
**speak** (answer on-device), **stay silent**, or **escalate** to a cloud model (GPT).

The Python package is `egogate/`.

## Pipeline

```
query ──► CueParser (cloud, once) ──► 3 visual cues ──► embed
                                                          │
video ──► 2 s windows @ 2 fps ──► Matcher: cosine(window, cues) ──► surge > θ ?  ── no ──► silent
             │                                                            │ yes
             └──► EvidenceMemory: scene-graph triplets every 5 s          ▼
                              (top-k retrieved) ──────────────► Edge VLM: p(speak now)
                                                                           │
                                            p ≥ hi: edge answers │ lo < p < hi: cloud │ p ≤ lo: silent
```

Literature this is built on: VideoLLM-online (speak-or-silent token), Em-Garde (cue parse once,
cheap embedding matcher, θ = 0.04 surge trigger), Response-G1 (scene-graph evidence memory),
VideoJev (a score, not generated text, as the routing signal).

## Layout

```
egogate/
  types.py      dataclasses shared across stages (Window, MatchResult, Decision, Event)
  config.py     YAML loader; experiment configs overlay configs/default.yaml
  video.py      frame sampling + sliding windows (fps / window_s / stride_s are config knobs)
  cues.py       CueParser: query -> visual cues (backend: cloud | edge | manual)
  embed.py      Embedder (Ops-MM-V1-2B wrapper + RandomEmbedder) and Matcher (surge/absolute, any/mean, cooldown)
  evidence.py   scene-graph triplet memory with top-k retrieval
  edge.py       EdgeVLM: HF wrapper for Qwen2.5-VL / Gemma 3 (+ int4/int8), MockEdgeVLM; p_speak from Yes/No logits
  cloud.py      OpenAI client with call counting, MockCloud
  router.py     policies: band (the method), random, always_cloud, edge_only
  pipeline.py   run_stream(): glues everything, logs sims/deltas/p_speak per window
  eval/
    metrics.py  online recall/precision with tolerance window (Em-Garde), calibration table, run summary
    datasets.py normalised JSONL loader, toy samples
configs/default.yaml           every knob with its default
configs/experiments/*.yaml     one file per experiment in the project doc
scripts/run_experiment.py      run a config -> results/<name>/{summary.json, <id>.jsonl, <id>.calib.json}
scripts/sweep_thresholds.py    (lo, hi) grid -> results/sweep_<name>.csv for the F1/recall heatmap
scripts/ingest_dataset.py      raw benchmark -> data/processed/*.jsonl   (OVO parser is a TODO)
tests/test_core.py             matcher, router, metric, end-to-end mock run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests, no GPU needed
pip install -e ".[models,cloud]"   # on the GPU box / Jetson
export OPENAI_API_KEY=...          # only for cloud.backend=openai
pytest -q
```

Dry run the whole pipeline with mocks (no models, no video, no key):

```bash
python scripts/run_experiment.py configs/experiments/band_confidence.yaml --set data.toy=true
python scripts/run_experiment.py configs/experiments/random_trigger.yaml --set data.toy=true
```

Any config key can be overridden with `--set dotted.key=value`.

## Experiments (from the project doc) → config

| Question | Config / knob |
|---|---|
| Does confidence matter vs. random GPT calls? | `band_confidence.yaml` vs `random_trigger.yaml` (`policy.p_cloud`) |
| Upper / lower bounds | `always_cloud.yaml`, `edge_only.yaml` |
| Is p_speak calibrated? F1/recall heatmap | `scripts/sweep_thresholds.py`; per-sample `*.calib.json` |
| Any-cue vs mean, surge vs absolute, θ | `matcher.aggregate`, `matcher.mode`, `matcher.threshold` (`matcher_mean_absolute.yaml`) |
| Payload: frames vs scene-graph only | `cloud.payload` (`payload_evidence_only.yaml`) |
| Gemma vs Qwen, quantization | `edge.model_id`, `edge.quant` |
| fps / window tuning | `video.fps`, `video.window_s`, `video.stride_s` |

## Metric

Timing is scored with online recall/precision: a response counts if it lands within
`[event_t, event_t + tol_s]` (default 4 s) of a ground-truth event, one match per event.
No LLM-as-judge. OVO-Bench's original offline accuracy is gameable (always answering "yes"
scores ~60%), so we don't use it.

## Status / next steps

Scaffold runs end to end with mocks. Nothing has touched a real model yet. In order:

1. **Data.** Download OVO-Bench (Forward Active Responding: CRR/SSR/REC) and/or StreamingBench
   Proactive Output. Finish `ingest_ovo()` in `scripts/ingest_dataset.py` once the real field
   names are known, write `data/processed/dev.jsonl`.
2. **Embedder.** Run `OpsMMEmbedder` once on the GPU box; the `get_text_embeddings` /
   `get_video_embeddings` method names in `egogate/embed.py` are guesses, fix them from the model card.
3. **Edge model.** Validate `HFEdgeVLM` with `Qwen/Qwen2.5-VL-3B-Instruct`, then
   `google/gemma-3-4b-it`. Check the Yes/No token ids and the chat template per family.
   Don't use Gemma-3n with int8 (AltUp incompatibility, see ISI SmallSat 2026).
4. **Baselines.** Run `edge_only` and `always_cloud` on real data to bracket the problem.
5. **Routing.** Run `band_confidence` vs `random_trigger`, then the threshold sweep.
6. Jetson deployment of the matcher + edge model (int4) comes after the numbers look sane on a desktop GPU.

Conventions: experiment names are the YAML filename; results are gitignored; every run writes
its full resolved config into `summary.json` so it can be reproduced.
