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

### Floors that match the model's settings (added after the first val run)

The first row 1 run showed that the cooldown (dropping a response that comes within N seconds of the previous one)
raises F1 by roughly ten points for almost any model setting, because it removes duplicate responses around one
event. So a model row must be compared with a floor that has the same cooldown, or the same number of responses.
The simplest such floor is a **timer**: respond every k seconds, with no model and no video. On the val split
(43 questions, 30 random phases):

| Rule that ignores the video | Avg F1 | Responses |
|---|---|---|
| Respond every 4 s | 27.0 ± 0.3 | about 990 |
| Respond every 8 s | 27.6 ± 2.2 | about 495 |
| Respond every 12 s | 23.1 ± 3.5 | about 330 |
| Respond every 16 s | 20.2 ± 3.6 | about 250 |
| Random 50% + 4 s cooldown | 26.7 ± 1.8 | about 670 |
| Random 25% + 8 s cooldown | 21.3 ± 2.7 | about 290 |

Reading a model row therefore means comparing it with the timer that makes a similar number of responses.
(`scripts/eval_floors.py` now includes the timer rows; `egogate/eval/floors.py` accepts a cooldown.)

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

**D16. The row 1 decider is Qwen3-VL-2B-Instruct (it started as a 3B model that did not fit).**

*What happened.* Our first choice was Qwen2.5-VL-3B-Instruct. On the RTX 4060 laptop GPU (8.6 GB total, 7.5 GB free)
it took a median of 14.4 s per decision, which projects to about 30 hours for all 7,863 decisions. A diagnostic
(`scripts/diagnose_speed.py`) found why. The weights alone take 7.63 GB, so almost nothing is left for the frames,
and the time per decision grew far faster than the input did while input preparation stayed at 11-14 ms:

| Frame long side | Tokens | Forward time | Peak GPU memory |
|---|---|---|---|
| 224 px | 241 | 2.4 s | 7.72 GB |
| 336 px | 449 | 7.0 s | 7.80 GB |
| 448 px | 689 | 37.5 s | 7.89 GB |

That pattern, with peak memory at 7.7-7.9 GB of 8.6, means the memory each decision needs spills out of GPU memory
into slower shared system memory. Shrinking the frames would not have fixed it, because the weights themselves
were the problem.

*Options considered.*

| Model | Weights (BF16) | Fits with room for frames on this GPU? | Decision |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | 7.51 GB | No: measured above | dropped |
| **Qwen3-VL-2B-Instruct** | 4.26 GB | Yes: measured below | **chosen** |
| Qwen2-VL-2B-Instruct | 4.42 GB | Yes (not measured) | deferred; run only if row 1 looks suspicious (it is the model Em-Garde used for its direct Yes/No ablation, so it separates "our setup" from "this model") |
| Gemma 3 4B | about 8.6 GB | No | deferred until a larger GPU or a reason |

*Why Qwen3-VL-2B.* It fits with room to spare, it is the newest of the options, and Qwen3-VL is one of our
references, so it is the model most likely to be kept. The goal of this stage is to learn whether the approach
works (the gates), not to compare models, so the whole experiment ladder runs on this one model first. Comparing
models is a later, optional step, not part of getting the core result.

*Measured on the chosen model (same diagnostic, same GPU, BF16, random-noise frames):*

| Frame long side | Tokens | Forward time | Peak GPU memory |
|---|---|---|---|
| 224 px | 366 | 161 ms | 4.43 GB |
| 336 px | 366 | 159 ms | 4.43 GB |
| 448 px (our default) | 550 | 257 ms | 4.51 GB |

The weights take 4.26 GB, the peak is 4.5 GB, and about 4 GB of the GPU is left over, so there is no spill. At
roughly 0.27 s per decision, all 7,863 decisions take about 35 minutes of model time (val alone, 1,983 decisions,
about 9 minutes), plus video decoding. The 224 px and 336 px rows have the same token count because the model
enforces a minimum image size, so frames smaller than about 448 px would not have saved anything; we stay at 448.

*What this does to other decisions.* The 2B decider (4.26 GB) plus the 2B embedder (about 4.4 GB) is about 8.7 GB
of weights against 8.6 GB of GPU, so the two cannot be loaded together. This confirms the sequential two-pass
design (D9, D12) as a requirement on this hardware, not only a convenience. On the larger GPU we may get later, a
bigger decider or a Gemma comparison becomes possible.

*Em-Garde's own numbers for this baseline (their Table 5, OVO-Bench forward tasks, F1):* full system 31.0; a direct
Yes/No decision module with Qwen2-VL-2B in place of the embedding matcher 14.6; a "sliding window" baseline (current
query plus the last 2 s, decide whether to answer) 21.0. Their baselines sit at or below what a coin flip scores in
our floor table, so a weak row 1 is plausible, and it is why section 5 requires intervals before we call anything an
improvement. Their cadence differs from ours, so we do not compare numbers directly.

*Status.* Plumbing verified by a CPU smoke test (10 decisions, 99.9-100% of the probability on Yes/No) and speed and
memory verified on the GPU. Accuracy is not measured yet.

**D17. How the cue matcher embeds video (stage 1a), and where it differs from Em-Garde's code.**
*What we do:* every 0.5 s, the last 2 s of video (4 frames at 2 fps) are embedded with Ops-MM-embedding-v1-2B
(`OpenSearch-AI/Ops-MM-embedding-v1-2B`, 1536 dimensions, length-normalised) and compared by cosine similarity with each
cue's text embedding. *Why:* paper section 4.1.1 ("embeds 2-second, 2 fps video segments at a process rate of 2 fps") and
their code (`streaming_fps=2`, `streaming_length=2`, so a 4-frame window) agree. The embedder is the vendored wrapper
(`third_party/em_garde`), checked to load and give deterministic, unit-length vectors.
*Differences from Em-Garde's code:*
- **No streaming shortcut.** Their code caches the visual features of earlier frames and encodes only one new frame per
  step (the paper says this makes the matcher 2-3 times faster). That patch reaches into model internals, pins
  `transformers==4.52.3`, and fails on current versions (we hit `'BaseModelOutputWithPooling' object has no attribute
  'shape'` on transformers 5.23). We therefore embed each window from scratch with the model's standard path. *Cost:* about
  2-3 times more compute per step, and small numerical differences are possible (for instance their streaming prompt ends
  with one extra newline), so our similarities are not guaranteed to match theirs bit for bit. We check the result through
  the Gate 2 diagnostic rather than assuming it.
- **Attention.** Their default is `flash_attention_2`, which is not available on this laptop setup; we use `sdpa`.
- **Frame size.** Their loader reads frames at about 280 x 280 pixels (78k); ours are resized to a long side of 448 (448 x
  252, 113k). Similar scale, not identical.
- **The first 1.5 s of a video** have no full 4-frame window and are skipped, not padded; no true event is that early.
*Status:* verified on CPU (one real window embedding, deterministic); GPU speed not measured yet.

**D18. Cues come from the edge model reading only the question (this settles O1).** Qwen3-VL-2B writes 3 short visual cues
per question (`egogate/cues.py`, versioned by a hash of its prompt), stored once in `results/cache/cues/` and reused by every
later row. *Why the edge model:* it keeps the whole system on the device, and the stage costs nothing in API money.
*What is weaker than Em-Garde's version:* their cue proposer is a 7B model trained with supervised learning and
reinforcement learning, it sees 5 s of video history at 1 fps as well as the question, and its proposals can include
"negative" cues (things that should not be visible) combined by a logic rule. Ours is an untrained 2B model, text only,
positive cues only, any one of them counts. The embedding model itself is the same as theirs and is not trained for this: the paper says they "employ an
off-the-shelf embedding model, Ops-MM-V1, for the LPMM without any training" (section 4). What they train is the cue
writer, so that its cues are "perception-friendly" for that embedder. *Evidence on how much that matters (their Table 5,
OVO-Bench forward tasks, F1):* full system 31.0; replacing the trained proposer with plain prompt rewriting by an untrained
Qwen2.5-VL-7B 24.0. Our writer is smaller and untrained, so we should expect to land at or below that 24.0 level, and their
limitations section notes that the embedder struggles to tell subtly different cues apart.
*Expected consequence:* weaker cues, so a weaker matcher than the published one.
In the one question we looked at, the cues were plausible but invented specifics ("hand strikes a metal surface with a
hammer" for "hitting something against/with something"). *Escape hatch, if row 3 is poor and the cues look like the cause:*
give the proposer a few frames, or have GPT write the cues (a few cents per question; the cue file format is the same).
Not pursued unless the matcher results point at it.
*How we decide:* read the printed cues and the Gate 2 diagnostic (AUC of the similarity near events) before the full run is
trusted. Because the window embeddings are stored locally, trying a different set of cues later costs only embedding the
new cue text, not re-running the video. The edge model as writer saves no memory or compute worth mentioning (it runs once per
question and is the same model as the decider), so the only real argument for it is keeping the system on-device; if cue
quality is the bottleneck, GPT-written cues (one call per question, as the project notes originally assumed) are the first
alternative.

**D19. Trigger rules and what we sweep for row 3 (this settles O3 and O4).** The matcher is scored by treating every trigger
as a response. Everything below is computed from the stored similarities, on CPU, so we try all of it in one table:
- *Rule "paper":* trigger when any cue's similarity rose by more than theta since the previous step (paper section 3.4, theta
  0.04). *Rule "code":* the rule in their released code: the similarity must be more than theta above its value one second
  earlier and above the average of the older history (history trimmed to the latest 500-600 steps). A unit test confirms our
  version gives the same triggers as their `trigger_detection.py` on random data, including past 600 steps.
- *Cue combination (Gate 2):* apply the rule to each cue and fire if any cue fires ("any"), or to the average similarity of
  the cues ("mean").
- *Duplicate handling:* a time-based cooldown (0, 4 or 8 s), or Em-Garde's rule (skip a trigger within 5 s of the last one if
  the window looks at least 0.8 similar to the window at that trigger; needs the stored window vectors).
- *Thresholds:* 0.01 to 0.08 around their 0.04 (their sweep covered 0.03 to 0.04 after reinforcement training of the
  proposer, so we treat it as a starting point, not a target).
To avoid an ablation maze, the sweep is one table; we carry a single combination, chosen on val, into row 4. A diagnostic
accompanies it: how well the similarity, and its rise, separate steps near a true event from the rest (AUC; 0.5 means no
separation), which tells us whether the cues carry any signal before any threshold is applied.

### Open on purpose

**O1.** Settled: see D18 (edge model, text only).

**O2. Put the question into the memory (scene-graph) prompt or not?** Response-G1 does
(query-guided generation, section 3.2). Our notes say it makes the model hallucinate early
triggers. Direct disagreement, tested as a sub-row of row 5.

**O3, O4.** Settled: see D19 (both trigger rules and both cue combinations are swept in one table).

**O5. The 7B reference run.** Em-Garde's proposer is a 7B model that does not fit on the 4060 in
BF16 (about 15 GB). Needs a larger GPU or a quantized load, which conflicts with D7.

## 9. Not yet verified

- Em-Garde's 52.3 PAUC reproduction on ProactiveVideoQA EGO (their code and weights are public;
  we have not run them).
- That the embedder wrapper's method names work with our wiring (we vendored their file but have
  not run it).

## 9a. Results so far

### Row 1, edge model alone, validation split (first run)

**Setup.** Qwen3-VL-2B-Instruct in BF16 (D16). Every 2 s the model is shown the question and the last 4 frames
(448 px) and asked "respond now?" (D15). The 43 validation questions (1,983 decisions) are scored with the Em-Garde
metric, tolerance ±2 s. Median 300 ms per decision on the RTX 4060 laptop GPU. The model puts essentially all its
probability on "Yes" or "No" (at least 99.95% on every decision), so the prompt works as intended.
Source: `results/cache/decider/Qwen3-VL-2B-Instruct__query_only__1a7df950/` (`replay_ovo_far_val.json` has every setting).

**What the table shows.** Each cell is average F1, with the number of responses in brackets. "Threshold" is how sure
the model must be before we count a Yes (p(Yes) at or above it). "Cooldown" drops a response that comes within that many
seconds of the previous one, because one event usually looks like "Yes" on two consecutive ticks and the second would
count as a false alarm.

| Threshold | No cooldown | 4 s cooldown | 8 s cooldown |
|---|---|---|---|
| 0.05 | 16.8 (1980) | 26.7 (1002) | 26.5 (512) |
| 0.20 | 17.8 (1857) | 27.8 (953) | 28.2 (492) |
| 0.50 | 17.6 (1487) | 26.3 (788) | 25.7 (422) |
| 0.70 | 20.8 (1027) | 29.0 (580) | 27.8 (330) |
| 0.80 | 24.5 (540) | 29.7 (345) | **32.0 (212)** |
| 0.90 | 13.0 (76) | 13.8 (58) | 13.9 (46) |

**The headline, and why not to trust it yet.** The best cell is 32.0 F1 (threshold 0.80, 8 s cooldown). It is the best of
33 settings tried on only 43 questions, picked using those same questions, so it is flattered by the selection. A cleaner
reading is below.

**Reading 1: the model leans toward "Yes".** Its median p(Yes) is 0.73, and 75% of decisions are at or above 0.5. At a very
low threshold it responds almost every time (1,980 of 1,983 decisions) and scores 16.8, which is exactly what the
"respond at every decision point" floor scores (16.8). So a low threshold adds nothing.

**Reading 2: the cooldown alone is worth about ten points.** Going from "no cooldown" to "4 s cooldown" lifts nearly
every row from about 17 to about 27, with no change to the model. That is not skill; it is removing duplicate responses.
This is why a model row has to be compared with a floor that has the same cooldown (section 5).

**Reading 3: against timers with a similar number of responses, there is a modest edge at high thresholds.**

| Model setting | Responses | Model F1 | Timer with a similar number of responses | Timer F1 |
|---|---|---|---|---|
| threshold 0.05, 4 s cooldown | 1002 | 26.7 | respond every 4 s (about 990) | 27.0 ± 0.3 |
| threshold 0.50, 8 s cooldown | 422 | 25.7 | respond every 8 s (about 495) | 27.6 ± 2.2 |
| threshold 0.80, 4 s cooldown | 345 | 29.7 | respond every 12 s (about 330) | 23.1 ± 3.5 |
| threshold 0.80, 8 s cooldown | 212 | 32.0 | respond every 16 s (about 250) | 20.2 ± 3.6 |

At low and middle thresholds the model is no better than speaking on a timer. At threshold 0.80, where it responds only
when quite sure, it scores 7 to 12 points above a timer making a similar number of responses. (Timer rows are the 30-phase
figures from section 5; the ± is the spread across phases, not a confidence interval.)

**Reading 4: the ranking signal is real but weak.** Taking every decision, and labelling it "positive" when it falls within
2 s of a true event (313 of 1,983, or 15.8%), we asked how well p(Yes) ranks positives above the rest. AUC is the chance
that a random positive outranks a random negative: 0.5 means no skill, 1.0 is perfect.

| Task | AUC of p(Yes) | Mean p(Yes) near an event | Mean p(Yes) elsewhere |
|---|---|---|---|
| All three tasks | 0.71 | | |
| REC (repeated actions, 24 questions) | 0.71 | 0.81 | 0.68 |
| SSR (tutorial steps, 12) | 0.61 | 0.67 | 0.60 |
| CRR (clue reveal, 7) | 0.61 | 0.56 | 0.45 |

**Per task for the best cell** (threshold 0.80, 8 s cooldown): REC 50.4, SSR 29.9, CRR 15.6. The timer that responds every
16 s scores REC 28.5, SSR 15.7, CRR 13.9 (10 phases, `scripts/eval_floors.py`). REC accounts for most of the gap; CRR (7 questions, long videos) is barely above the timer.

**What this means.**
- The decider alone is weakly informative: when it is very sure, it is right more often than a timer would be. It is not
  good at deciding *when*, and it says Yes far too often.
- That fits the plan. A model that over-answers Yes is exactly what a cheap gate (the cue matcher, row 3 to 4) should
  help with, and calibration (Gate 1) should help with the threshold. We have not tested either yet.
- Nothing here shows the approach works. It shows the baseline is weak, which is the starting point we expected
  (Em-Garde's own sliding-window baseline scores 21.0 on its metric, and its direct Yes/No module 14.6).

**Caveats.**
- The setting was chosen on val and scored on val, so the numbers are optimistic. A fair number needs a split the
  setting was not chosen on (dev, 73 questions, or test, 56).
- 43 questions is small; CRR has 7. No confidence intervals yet (section 5 describes the plan).
- One prompt wording, one model, one frame size. The prompt is a free choice that can move results.
- The test split has not been touched.

**Next.** Stage 1a: the cue matcher (row 3), then the matcher-gated decider (row 4). Confirming on dev is optional
and can wait.

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
- 2026-11-15: D17-D19 added (cue matcher design: embedding path, cue source, trigger rules); stages 0 and 1a built.
- 2026-11-15: Section 9a added: first row 1 results on val, with the cooldown and timer comparison.
- 2026-11-15: Added timer and cooldown-matched floors to section 5 after the first val run showed a cooldown alone adds about ten F1.
- 2026-11-15: D16 added and then rewritten with the options table and GPU measurements: the decider is Qwen3-VL-2B because the 3B spilled GPU memory; other models deferred.
- 2026-11-15: D15 (row 1 decider input and plumbing check) added; row 1 runner, replay and notebook added.
- 2026-11-15: D12 (replay is an evaluation harness only), D13 (on-demand decider calls at trigger times) and D14 (how each component is graded) added.
- 2026-11-15: Added `scripts/ovo_tar_remote.py` and the committed video index; the video download
  drops from about 44 GB to 5 GB (D11).
- 2026-11-15: First version. Ingestion, seeded splits, Em-Garde-faithful metric, floors, and a
  scoring CLI added. Tolerance moved from 4 s one-sided to ±2 s. Default dtype set to BF16.
