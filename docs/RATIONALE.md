# Rationale: what we are building, why, and how we will know it works

This is a living document. It is written so that someone who has not followed the project can
understand what we did, why we did it, what each experiment tests, and where we are going.
Every design decision has an entry in the **Decision log** (section 8) with the reasoning and
the evidence behind it. When we change our mind, we add a new entry; we do not rewrite history.

Status key used throughout: **verified** (we checked it in a paper's text or code),
**assumed** (reasonable, not yet checked), **open** (undecided on purpose).

---

## 1. The problem in plain words

You wear a camera and ask one question at the start, such as "tell me when the water boils".
The system must **speak up by itself, at the moment the answer appears**, without being asked
again. Existing systems either run a very large model on every frame (accurate, but costly and
slow) or use a small model that is often wrong about *when* to speak.

Our idea: a **small model on the edge device** watches everything and answers when it is
confident. When it is unsure, it **escalates** to a large cloud model (GPT). We want to keep
most of the quality while sending far less video and making far fewer cloud calls.

Two questions follow, and the whole project is built to answer them:

1. **Is the small model's confidence trustworthy?** If it says "I'm 90% sure", is it right about 90% of the time?
2. **Does asking GPT when the model is unsure beat asking GPT at random?** At the same number of
   GPT calls, does choosing *which* moments to escalate by confidence actually help?

## 2. The system, one step at a time

See the architecture diagram: https://claude.ai/artifact/UzmwnghvVgb5XnrienCCUG

1. **Cue parsing (once per question).** The question is turned into about 3 short descriptions of
   what the camera would see at the right moment ("bubbles rising in a pot"). *From Em-Garde.*
2. **Cue matching (cheap, always on).** Every 2 seconds of video is turned into a vector by an
   embedding model; its cosine similarity to each cue's vector is computed. A **sudden rise** in
   similarity (not a high absolute value) marks a candidate moment. *From Em-Garde.*
3. **Memory.** Every ~5 seconds the small model writes down what it sees as scene-graph triplets,
   like `(hand, holds, pot)`. When a decision is needed, the few most relevant triplets are
   retrieved by comparing their text embeddings to the question's. *From Response-G1.*
4. **Decider.** The small model reads the recent frames plus the retrieved triplets and is asked
   "is now the moment to respond? Yes or No". We read the probability of "Yes" directly from the
   model's output scores. We then calibrate that probability (temperature scaling). *Yes/No
   trigger from Response-G1 and VideoLLM-online; reading a score instead of generated text from
   VideoJev; calibration from Guo et al., ICML 2017.*
5. **Router.** High probability: the edge model answers. Low: stay silent. In between: ask GPT.
   *This part is ours. None of the four papers tests escalating to a cloud model.*
6. **What GPT receives** (three options, tested later): raw frames, scene graphs only, or both.

## 3. What we borrow, and from which paper

| Idea | Paper | Where to look |
|---|---|---|
| Parse the question into cues once; match with a cheap embedder; trigger on a similarity surge, threshold 0.04 | Em-Garde (arXiv 2603.19054) | `docs/papers/em-garde_2603.19054.pdf`, section 3.4 and 4.1.1 |
| Online recall / precision metric that needs no LLM judge | Em-Garde | appendix, eq. 10; code in `third_party` notes below |
| Scene-graph memory, query turned into triplet form, top-K retrieval by cosine similarity | Response-G1 (arXiv 2605.07575) | section 3.3 |
| Yes/No token as the "speak now" decision | Response-G1, VideoLLM-online | Response-G1 section 3.3 end |
| A score instead of generated text for the decision | VideoJev (arXiv 2610.18615) | treated as a low-priority extension (Gate 5) |
| Temperature scaling for calibration | Guo et al., ICML 2017 | section 4 |
| The benchmark we start with | OVO-Bench (CVPR 2025) | Forward Active Responding tasks |
| The benchmark we use later (needs a GPT judge) | ProactiveVideoQA (arXiv 2507.09313) | scorer fetched by `scripts/fetch_proactivevideoqa.sh` |

## 4. What we test: the experiment ladder

**Principle:** every row changes exactly one thing relative to the row above or to the edge-only
baseline, so any difference in score can be blamed on that one thing. We do not build the
combined ("hybrid") system until we know what each piece is worth by itself.

| Row | System | The question it answers | Gate |
|---|---|---|---|
| 0 | **Floors**: always silent, always respond, random, oracle | What does "no skill" score? (And does our scoring code work: oracle must be 100, silent must be 0.) | sanity |
| 1 | **Edge model alone** (the baseline): every 2 s, "respond now?" from the last window and the question | How good is the small model by itself? Is its confidence accurate? | **Gate 1** |
| 2 | Row 1 + the cues written into the prompt | Do cues help the decider? | |
| 3 | **Embedding matcher alone**: a similarity surge fires the response, no decider | How far does the cheap path get? | **Gate 2** (how to combine cue scores, what level counts) |
| 4 | Matcher trigger, then decider | Does gating beat either alone? | |
| 5 | Row 4 + scene-graph evidence in the decider prompt. Sub-tests: raw question vs question-as-triplet; K=1 vs K=5; question in the memory prompt or not | Does the memory help? Which details matter? | **Gate 4** (text-only payload) |
| 6 | **Router with GPT**: confidence routing vs random routing at the same number of GPT calls, with edge-only and GPT-everywhere as the two ends | Does confidence earn its place? | **Gate 3** |
| later | Score head instead of Yes/No readout (VideoJev style) | Does a trained score help? | **Gate 5** (low priority) |
| later | Hybrid: the cue trigger also decides when to pull scene graphs for the decider | Do the pieces compound? | |

Rows 0 to 5 call no GPT, so they cost nothing but GPU time. GPT enters at row 6.

## 5. How we measure (no LLM judge)

**Benchmark:** OVO-Bench "Forward Active Responding", three task types:
- **REC** (repetition counting): respond when the repeated action has happened again.
- **SSR** (step-by-step tutorials): respond as each step finishes.
- **CRR** (clue reveal): respond when the clue that answers the question appears.

172 questions on 134 videos (82 REC, 42 SSR, 48 CRR), about 4.4 hours of video to judge.

**Metric:** a response is correct if it lands within **±2 seconds** of a ground-truth time.
Each ground-truth time can be matched once; extra responses near the same event count as false
positives. Per question: recall = correct / events, precision = correct / responses (0 if the
system never responded). Per task: average recall and precision across questions, then take the
F1 of those two averages. Overall: the mean of the three task F1s. This is Em-Garde's scoring
code, copied in logic exactly (decision D3, D4), with a test that checks our implementation
against their loop on 500 random cases.

**Why no LLM judge for now:** the metric scores *when* the system speaks, not what it says. That
is the thing proactive systems get wrong, it is free to compute, it is deterministic, and it is
the metric Em-Garde reports. ProactiveVideoQA, which does judge answer text with GPT-4.1, comes
later and costs money from our budget of about $100.

### What chance scores on this metric (no model, no video)

To know what counts as an improvement, we first measured what simple rules score on this metric, with no
model and no video. At each 2 s decision point, a rule decides "respond" or "stay quiet" without looking at
the video, and the scoring code grades the result against the true response times. Scores below are F1 over
all 172 forward-task questions, at a 2 s decision cadence and a ±2 s tolerance (`scripts/eval_floors.py`).

| Rule that ignores the video | Avg F1 |
|---|---|
| Respond at every decision point | 15.6 |
| Random, 10% chance per decision point | 16.4 ± 1.8 |
| Random, 25% chance | 21.4 ± 1.1 |
| Random, 50% chance | 21.7 ± 0.6 |

The ± is the spread across 10 random seeds. The always-silent and oracle checks only confirm that the
scoring code works (silent must score 0, a system that responds exactly at the true times must score 100),
so they live in the tests (`tests/test_eval.py`) and not here.

**What this tells us**

- A coin flip scores about 21 F1. Em-Garde reports 30.99 for its setup, so the gap between guessing and a
  published result is only about ten points.
- The floor depends on how often a system is allowed to respond. At a 0.5 s cadence, always-respond falls to
  4.3 and the best random setting reaches 19.7. So every model result is compared with floors at the same
  cadence, never with Em-Garde's number directly.

### How we decide an improvement is real

The seed spread above is only the noise from the coin flips. A bigger source of noise is that the splits are
small (43 val, 56 test, 73 dev questions), so a score can move a lot just because of which questions landed
in it. We therefore require:

1. **A confidence interval on every score.** Bootstrap over questions, resampling whole videos, since several
   questions share one video. Models and the random baseline each get an interval.
2. **A paired comparison between systems.** Resample the same questions for both systems and compute the
   interval of the difference. A gain counts only if that interval excludes zero.
3. **Honesty about power.** With 56 test questions, only fairly large gaps can be detected. Results are
   reported per split, and small differences are called inconclusive.

*Status:* the floors are measured. The bootstrap helper and the intervals are not built yet.

## 6. Data and splits

- Source: `data/ovo_bench_new.json` from github.com/JoeLeelyf/OVO-Bench (CC BY-NC-SA 4.0). We keep
  only the annotations of the three forward tasks in `data/processed/ovo_far_*.jsonl` (about
  1 MB, no video). Videos are never committed.
- **Splits** (seed 0): dev 73 questions / 54 videos; val 43 / 39; test 56 / 41. Rebuild with
  `python scripts/ingest_dataset.py ovo --ann <ovo_bench_new.json> --out-dir data/processed --seed 0`.
- **Video download size: 5.0 GB, not 44 GB.** OVO-Bench ships its source videos as one plain tar
  archive cut into five 10 GiB parts (46.3 GB in total), plus a 144 GB chunked copy we never need.
  Because the server supports HTTP range requests, `scripts/ovo_tar_remote.py` reads the tar headers
  remotely, records where every video sits (`data/processed/ovo_src_index.json`, committed), and then
  downloads only the byte ranges of our 134 videos: **5.03 GB** (dev 2.0, val 1.4, test 1.6; by source
  COIN 1.7, thumos 1.6, MovieNet 1.2, perception_test 0.6). One video was fetched and probed as a valid mp4
  to check the method. Decision D11.

## 7. Compute plan (RTX 4060, about 8 GB)

- **What BF16 is.** Model weights are stored as numbers. FP32 uses 4 bytes each; FP16 and BF16
  use 2, halving memory. FP16 keeps finer detail but can overflow on large values. BF16 keeps
  FP32's range with less detail and is the safer default for big models; Em-Garde's code loads
  everything in BF16. INT8/INT4 (quantization) shrink further and are a separate, later experiment.
- **Two-pass design (built for the decider).** A 2B decider (4.3 GB) plus the 2B embedder (about 4.4 GB) is
  8.7 GB of weights against a GPU with 8.6 GB total, so they cannot be loaded together. So each model runs alone over the videos and saves its outputs to disk
  (embeddings and cue similarities; Yes/No log-probabilities per decision point). A second,
  CPU-only pass then applies matcher, calibration and router settings to those saved numbers.
  Trying a new threshold takes seconds and loads no model. GPT replies are the one exception:
  they are fetched the first time a setting needs one and cached.
- **Decision cadence.** The decider runs every 2 s of video: 7,863 decision points over all 172
  questions, against 31,620 at 0.5 s.

## 8. Decision log

Each entry: what we decided, why, the evidence, and its status.

**D1. One variable per experiment, compared against the edge-only baseline.**
*Why:* four papers each claim a component helps, in different setups, on different models and
data. If we combine them first, we cannot tell which part earned the score or which hurt. With
one change per row, the table is the argument. *Status:* decided with the team lead (user).

**D2. Start on OVO-Bench Forward Active Responding, not ProactiveVideoQA.**
*Why:* its score is computed from response times alone, so it needs no LLM judge, is free and
repeatable, and is the benchmark Em-Garde reports numbers on. ProactiveVideoQA's PAUC uses a
GPT-4.1 judge (verified in `judge_llm_scores_openai.py`), which spends budget and adds noise.
*Evidence:* Em-Garde appendix; ProactiveVideoQA scorer. *Status:* verified.

**D3. Use Em-Garde's ground truth and scoring logic exactly, with a tolerance of ±2 s.**
*Why:* a comparison to a published number is only meaningful if the ground truth and matching
rules are identical. Ground truth: CRR uses `clue_time`; REC and SSR use each event's start time
plus 2 s; REC and SSR are judged from 0, CRR from `ask_time`; all stop at the last test point.
*Evidence:* `eval/ovo_bench/ovo_bench_eval_online.py` in the Em-Garde repo (`abs(t - gt) <= 2`);
appendix eq. 10 says "within a 2-second window". *A trap we avoided:* section 3.3.2 of the paper
mentions a **4 s** window after the event, but that is the reward used for *training* the
proposer, not the evaluation metric. The earlier scaffold used 4 s one-sided; we replaced it.
*Status:* verified. A test compares our code to their loop on 500 random cases.

**D4. Aggregate the way Em-Garde does: average recall and precision per task, then F1.**
*Why:* it matches their reported numbers. *Consequence we accept and watch:* a question where
the system never speaks scores precision 0, so silence is punished, and averaging before F1
behaves differently from averaging per-question F1s (a unit test pins this). *Status:* verified.

**D5. Baseline decision cadence is every 2 s.**
*Why:* it cuts decider calls about 4x (7,863 vs 31,620 over the full set), which matters on a
laptop GPU; a response within ±2 s of the event is still credited, so a 2 s grid can still land
inside the window. *Cost:* coarser timing, and floors at this cadence are higher (see section 5),
so every comparison uses same-cadence floors. We will test a finer cadence once the pipeline
runs. *Status:* decided with the team lead.

**D6. Seeded splits grouped by video; dev 40% / val 30% / test 30%.**
*Why grouped:* ten videos carry several questions (one has nine). Splitting by question would let
a threshold tuned on dev see test footage. *Why val is separate:* temperature scaling and the
router thresholds are fit on data the model has not been scored against (Guo et al.). Test is
used once, at the end. *Known weakness:* val has only 7 CRR questions; val windows are plentiful
for temperature scaling, but CRR-specific claims should use dev plus test with care.
*Status:* decided; revisit if val is too thin.

**D7. BF16, no quantization in this stage.** *Why:* it is the precision Em-Garde runs
(`torch_dtype=torch.bfloat16` in their model code), and keeping precision fixed means any score
difference is not a precision effect. Quantization is a later experiment on both edge models.
The scaffold's default of FP16 was changed to BF16. *Status:* verified for Em-Garde. Whether a decider fits in BF16 on the 8 GB GPU turned out to depend on size: see D16.

**D8. Score floors first (row 0).** *Why:* the metric turned out to be generous (a coin flip
scores about 21), so a model result is uninterpretable without the floor at the same cadence;
the oracle row also proves the scoring code works. *Status:* done.

**D9. Two-pass cache-and-replay architecture.** *Why:* 8 GB cannot hold both models, and we
want threshold sweeps and the random-routing control (Gate 3) to cost nothing. *Status:* decided;
not yet built (next step).

**D10. Vendor MIT-licensed code, fetch the rest.** Em-Garde's Ops-MM wrapper is copied into
`third_party/em_garde/` with its license. ProactiveVideoQA has no license file, so its scorer is
fetched at a pinned commit and not committed. *Status:* done.

**D11. Fetch only the needed videos by HTTP range request instead of downloading the archive.**
*Why:* the full archive is 46.3 GB and the laptop would need it all just to extract 134 videos.
The archive is an uncompressed tar, so a video's location can be found by reading 512-byte headers,
and Hugging Face honours `Range:` requests (verified: HTTP 206, valid tar header at byte 0, 645 files
indexed). The dataset revision is pinned (`fec29e3`) so the index cannot silently go stale.
*Cost:* about 650 small requests to build the index once; the index is committed so nobody repeats it.
*Status:* verified end to end on one video; the full 5 GB fetch has not been run yet.

**D12. The cache-and-replay design is an evaluation harness only, with two validity rules and one check.**
*What it is:* we run each model once on a recorded benchmark video, store its raw outputs, and then
combine and re-score them under many settings. *What it is not:* the deployed system. The deployed
system is streaming and causal (frames up to time `t` only), with both models resident and running
concurrently. *Rule 1, causality:* every stored number at time `t` may use only video up to `t`.
*Rule 2, no feedback:* a stored stage's inputs must not depend on the system's own earlier decisions
(cooldowns and the router are applied in stage 2 for that reason). *Check:* on a few videos, run the
real frame-by-frame loop (`run_stream`) and the replay with identical settings and require identical
detections; this runs once after both exist, to confirm the shortcut matches the real thing.
*Also:* latency is logged per component (matcher, decider, memory) in the live loop, since replay
deliberately ignores latency, concurrency and memory management; those belong to the hybrid build.
*Status:* decided; the check and timers are not yet built.

**D13. Decider calls for the combined rows run on demand at the exact trigger time, and are remembered.**

*The catch.* The cue matcher steps every 0.5 s (Em-Garde's setting) but the baseline decider is stored
only every 2 s (D5). When the matcher fires at, say, 13.5 s, no stored decider answer exists for that moment.

*Options considered.*

| Option | What it does | Trade-off |
|---|---|---|
| a | Store the decider every 0.5 s | Most faithful, but up to 31,620 decider calls instead of 7,863 (about 4x GPU time) |
| b | Wait for the next 2 s tick | Cheap, but adds up to 2 s delay, which can push a correct trigger outside the +/- 2 s tolerance |
| c | **Run the decider at the trigger time, remember the result** | Faithful to a gated system (the decider runs only when the matcher fires); costs GPU time the first time a new trigger time appears |
| d | Reuse the verdict from the last 2 s tick | Free, but up to 2 s stale, and it assumes the decider runs continuously, which hides the compute saving the gate is meant to give |

*Why c.* It is the only option that tests the real gated design: the decider sees the window that ends
at the moment of the trigger, exactly as the live loop would show it. It also avoids paying for 0.5 s
decider calls at moments no setting ever triggers on.

*What this changes, and what we can and cannot assume.*
- **Stage 2 is no longer purely CPU.** The first time a (video, time, prompt variant) is requested, the
  decider runs on the GPU and the result is stored. Later requests, including every re-score with the same
  triggers, are free. Lowering the matcher threshold creates new trigger times and so new calls. The
  stored results are keyed by model id, precision, prompt text and frame sampling, so a changed prompt can
  never reuse an old answer. A per-run call budget is reported.
- **Row 1 and row 4 call the decider at different times** (the 2 s grid vs the trigger times). A difference
  between them therefore mixes two effects: the gate, and a small timing-alignment effect. To separate them
  we also report option d as a free control (row 4 using the stale grid verdict). If c and d agree, the
  alignment effect is negligible; if not, we say so.
- **The "gating cannot add recall" argument is approximate.** With the decider asked at identical times in
  both rows, a combined detection (trigger AND decider says Yes) could never beat the decider alone on
  recall. Under c the times differ, so the bound holds only approximately.
- **Live and replay agree by construction for decider calls.** Under c the decider is called with the
  window ending at the trigger time, as in the live loop, which makes the live-versus-replay check (D12) a
  cleaner test.
- **Cost is unknown until the matcher has run.** The ceiling is the 31,620 calls of option a. We will
  time a few videos first and report the actual count.

*Status:* decided. Not yet built.

**D14. How each component is graded.** The benchmark gives only the correct response times (one or a few
per question), so every system is graded the same way: by the times at which it responds, scored with the
+/- 2 s metric (D3). Nothing grades a component's internals. In practice:
- *Decider alone (row 1):* its Yes decisions are the detections.
- *Cue matcher alone (row 3):* its trigger times are the detections, with no answer text, since the metric
  scores timing only. The cue text itself is never graded; cue quality shows up only through whether the
  triggers land near events.
- *Combined (row 4):* detections are the moments where the trigger fired and the decider said Yes.
- *Diagnostics we also derive from the same ground truth* (not extra labels): a moment counts as
  "positive" if it is within +/- 2 s of an event. That gives per-moment labels for the Gate 1 calibration
  heatmap (is p(Yes) high where it should be?) and for Gate 2 (do cue similarities rise near events, and by
  how much compared with elsewhere?).
*Caveat:* one-to-one matching means a second detection near the same event counts as a false positive in
the F1 score, while the diagnostic labels treat every moment near an event as positive. The two views will
not agree exactly, and the Gate 1 heatmap states which one it uses.

**D15. What the decider is shown in row 1, and what we verified.**
*Input:* the user's question text and the last 4 frames (2 s at 2 fps), each resized so the long side is at
most 448 px. No cues (row 2), no scene-graph evidence (row 5), and no earlier video, so it cannot count
repetitions or track tutorial steps from history; that gap is what the memory is meant to fill.
*Question asked:* "Respond only at the moment the answer becomes available ... Is RIGHT NOW such a moment? Answer
with a single word: Yes or No." (exact text is `build_speak_prompt` in `egogate/edge.py`, version `speak-v1`).
*Score:* the log-probabilities of the tokens "Yes" and "No" over the whole vocabulary, stored raw; p(Yes) is
computed later as sigmoid((lp_yes - lp_no) / T). We also keep the total probability on those two tokens; if it
is far below 1 the model is not answering in the format asked and the prompt needs work.
*Why these choices:* four frames keep a call cheap enough for a decision every 2 s (D5); the wording is our own
since none of the four papers fixes one for this setup, so it is versioned and hashed into the cache folder name
(changing it creates a new cache, never silently reusing old scores). The wording is a free design choice that
can move results, and we expect to ablate it.
*What we verified:* a CPU smoke test of Qwen2.5-VL-3B on one 20 s val video (10 decisions, frames shrunk to 224 px)
ran end to end: the Yes/No tokens are single tokens, 99.6-99.7% of the probability sat on them, and each call took
about 1.5-2 s on CPU. The p(Yes) values (0.44-0.62) show nothing about accuracy; this only checks the plumbing.
*Status:* verified for plumbing; accuracy on the GPU not yet measured.

**D16. The row 1 decider is a 2B model, not the 3B we started with.**
*What happened:* the first GPU run of Qwen2.5-VL-3B took a median of 14.4 s per decision (projected 30 hours for
all 7,863). A diagnostic on the RTX 4060 laptop GPU (8.6 GB total, 7.5 GB free) showed the weights alone take
7.63 GB, and time per decision exploded with input size: 2.4 s at 241 tokens (224 px frames), 7.0 s at 449
tokens (336 px), 37.5 s at 689 tokens (448 px), while preparing the inputs took 11-14 ms. That pattern, with
peak memory at 7.7-7.9 GB, means the extra memory each decision needs spills out of GPU memory, which is slow.
*Decision:* use one roughly 2B-parameter decider, **Qwen3-VL-2B-Instruct** (4.26 GB of weights, which leaves GPU
memory for the frames). It fits, it is the newest of the options, and Qwen3-VL is one of our references, so it is
the model most likely to be kept. The goal of this stage is to learn whether the *approach* works (the gates), not
to compare models, so the whole experiment ladder runs on this one model first.
*Deferred, only if needed:* Qwen2-VL-2B-Instruct (4.42 GB), the model Em-Garde used for its direct Yes/No ablation,
is run only if row 1 looks suspicious and we need to separate "our setup" from "this model". A Gemma comparison
waits for a larger GPU or a reason: Qwen2.5-VL-3B (7.51 GB) and Gemma 3 4B (about 8.6 GB) do not fit in BF16 on
this GPU.
*What we verified:* a CPU smoke test of Qwen3-VL-2B on one validation video ran end to end (10 decisions, 99.9-100%
of the probability on Yes/No), so the template, the Yes/No tokens and the runner work for this model. Speed and
accuracy on the GPU are not measured yet.
*Em-Garde's own numbers for this baseline (their Table 5, OVO-Bench forward tasks, F1):* full system 31.0; a direct
Yes/No decision module with Qwen2-VL-2B in place of the embedding matcher 14.6; a "sliding window" baseline (current
query plus the last 2 s, decide whether to answer) 21.0. Their baselines sit at or below what a coin flip scores in
our floor table, so a weak row 1 is plausible, and it is why section 5 requires intervals before we call anything
an improvement. Their cadence differs from ours, so we do not compare numbers directly.

### Open on purpose

**O1. Who parses the cues: GPT once, or the edge model?** The paper and project notes use GPT;
the first scaffold used the edge model. GPT costs a few cents per question but needs the network;
the edge model keeps everything on-device. We will run both as a factor in rows 2-5.

**O2. Put the question into the memory (scene-graph) prompt or not?** Response-G1 does
(query-guided generation, section 3.2). Our notes say it makes the model hallucinate early
triggers. Direct disagreement, tested as a sub-row of row 5.

**O3. Which trigger rule.** Em-Garde's paper: any cue rose by more than 0.04 since the previous
step. Their released code adds two stricter conditions (compared with the value `fps` steps back,
and above the recent average). Both are tested; the saved similarities make that free.

**O4. Cue score: any cue vs the average across cues** (Gate 2).

**O5. The 7B reference run.** Em-Garde's proposer is a 7B model that does not fit on the 4060 in
BF16 (about 15 GB). Needs a larger GPU or a quantized load, which conflicts with D7.

## 9. Not yet verified

- Em-Garde's 52.3 PAUC reproduction on ProactiveVideoQA EGO (their code and weights are public;
  we have not run them).
- That the embedder wrapper's method names work with our wiring (we vendored their file but have
  not run it).

## 10. Glossary

- **Edge model / edge VLM**: the small vision-language model that runs on the device.
- **Cue**: a short description of what the camera would see at the moment of the answer.
- **Embedding**: a list of numbers representing a piece of text or video so similarity can be measured.
- **Cosine similarity**: how aligned two embeddings are, from -1 to 1.
- **Scene graph / triplet**: a note like `(hand, holds, pot)`: subject, relation, object.
- **p(Yes)**: the model's probability of answering "Yes" to "respond now?", read from its output scores.
- **Calibration / temperature scaling**: adjusting p(Yes) so that "90%" means right about 90% of the time.
- **Router**: the rule that sends a moment to the edge, to silence, or to GPT based on p(Yes).
- **Floor / oracle**: the score of a system with no skill / one that knows the answer times.
- **Gate**: a checkpoint experiment that must pass before we trust the next step.

## 11. Changelog

- 2026-11-15: Section 5 rewritten: chance-level table trimmed to the rules that ignore the video, and the rules for calling an improvement real (bootstrap intervals, paired comparison, power) added.
- 2026-11-15: D16 (switch the decider to a 2B model after the 3B spilled GPU memory) added.
- 2026-11-15: D15 (row 1 decider input and plumbing check) added; row 1 runner, replay and notebook added.
- 2026-11-15: D12 (replay is an evaluation harness only), D13 (on-demand decider calls at trigger times) and D14 (how each component is graded) added.
- 2026-11-15: Added `scripts/ovo_tar_remote.py` and the committed video index; the video download
  drops from about 44 GB to 5 GB (D11).
- 2026-11-15: First version. Ingestion, seeded splits, Em-Garde-faithful metric, floors, and a
  scoring CLI added. Tolerance moved from 4 s one-sided to ±2 s. Default dtype set to BF16.
