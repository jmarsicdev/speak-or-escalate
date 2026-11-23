# Rationale: what we are building, why, and how we will know it works

This is a living document. It is written so that someone who has not followed the project can
understand what we did, why we did it, what each experiment tests, and where we are going.
Every design decision has an entry in the **Decision log** (section 8) with the reasoning and
the evidence behind it. When we change our mind, we add a new entry; we do not rewrite history.

Status key used throughout: **verified** (we checked it in a paper's text or code),
**assumed** (reasonable, not yet checked), **open** (undecided on purpose).

---

## 0. Where we stand (written 2026-11-22; everything below is on the validation split, 43 questions, with the test split untouched)

**Bottom line.** Nothing yet shows that the approach works. Each edge-only part finds events modestly better than a timer that speaks every few seconds. Combining the parts did not compound, and the part that is the point of the project, routing
uncertain moments to GPT, has not been tested. The one question that decides what to do next is whether there is headroom at all: does a much stronger judge do far better than our 2B model on the same moments?

### What we ran

| Row | System | Result (average F1; "timer" = respond every k seconds with no model, same number of responses) | Verdict |
|---|---|---|---|
| 0 | No-skill floors | A coin flip scores about 21; a timer every 4 to 8 s about 27 on val | The metric is generous |
| 1 | 2B decider alone, every 2 s, question only | Best 32.0 (tuned on val); about 30 at 200 to 600 responses against a timer's 20 to 27; within-question AUC 0.61; says Yes to 75 to 80% of moments | Weak |
| 3 | Cue matcher alone (Em-Garde's embedder, cues written by our 2B model) | Best 31.8; about 5 to 7 points above a timer at 270 to 430 responses, no gain at 540 or more; AUC 0.61 | Modest, uneven |
| 4 | Matcher fires AND decider says Yes | Best cell is the matcher alone; the decider only moves the score a point or two either way | No compounding |
| 2 | Decider with cues in its prompt | AUC change +0.003, interval [-0.038, +0.041]; hurts tutorials (-0.08); row 4 redone with it is worse | No help |

By kind of video (matcher alone, against the closest timer): sports clips 57.0 vs 32.7; tutorials 26.2 vs 18.2 (many false alarms); everyday-object counting 40.2 vs 38.4; clue reveal 21.5 vs 20.3. Groups of 7 to 14 questions, so indications only.

### What we have learned

*Fairly solid:*
1. The score is generous, so every number has to be read against a timer with the same number of responses. Duplicate removal alone (a cooldown) is worth about ten points.
2. The 2B decider, shown the last 2 s with no history, ranks moments only weakly (AUC about 0.6), and prompting it differently (rows 1, 2, 4) has not changed that.
3. The matcher's signal is modest and uneven: clearly useful on visually distinctive clips, near nothing on clue-reveal questions.
4. Gating the matcher with this decider does not help; the decider is informative at the trigger moments (moments it is very sure about are near an event 40% of the time against 7% when unsure) but too weakly to improve the F1.

*Tentative:* that cue quality explains the matcher's differences between groups (it is confounded with the kind of video); that cues hurt the decider on tutorials (an unadjusted group interval).

*Engineering findings that shaped the setup:* the 3B model did not fit the 8.6 GB GPU (D16); Em-Garde's embedding wrapper upscales every frame to 256 tokens and uses a prompt detail that changes the vectors (D17); averaging AUC over pooled questions misleads (corrected in 9a).

### What limits each part

| Part | Limit | Evidence | How we would test it |
|---|---|---|---|
| Cue writer | Text only (never sees the video), small, untrained; cannot know the answer in clue-reveal questions; vague activity names | Invented details, "red" in 22 of 43 cue sets, matcher gains vanish where cues were weakest | Raw question as the only cue; GPT-written cues; a few hand-written cues, all from the stored window vectors |
| Matcher | Weak step-by-step signal (AUC 0.55 to 0.62); expensive (about 720 to 900 ms per step, 114 minutes for val) | Gate 2 diagnostic | Better cues first; Em-Garde's trained cue writer or a 7B proposer would need more compute |
| Decider | 2B model, 2 s window, no memory, says Yes too often | AUC 0.61 under three different setups | **GPT on the same frames and prompt** (model size or task?); scene-graph memory (row 5) for history |
| Scene-graph memory (row 5) | Not tested; the only part that gives the decider history (matters for counting repetitions and tracking steps) | None yet | About 40 minutes of GPU for generation on val (unmeasured estimate) plus decider reruns |
| Router and GPT (rows 6, Gates 3 and 4) | Not tested; can be no better than the confidence it routes on, which is weak | Row 4 calibration table | Confidence routing against random routing at equal GPT calls, once a decider worth routing exists |
| Evaluation | 43 questions, clue reveal only 7; settings chosen and scored on the same split; F1 intervals not yet built (AUC intervals are) | Throughout | Intervals on F1; confirm chosen settings on dev, then test once |
| Compute | 8.6 GB laptop GPU: no model above about 2B, a 6-hour matcher run for all splits | D16, D17 | A cluster or rented GPU for dev, test, a 7B decider and Gemma |

**Update 2026-11-23:** by decision (D22), the GPT ceiling check and the router are deferred until the edge parts are informative. The order is now: the cue-quality ceiling, a scene-graph pilot, then cues aligned to the memory.

### Still to test, in the order I would do it

1. **GPT ceiling check.** The same frames and prompt on the same moments (about 2,000 calls on val, a few dollars). It also gives the "GPT everywhere" end of the Gate 3 plot. High GPT AUC means there is headroom for routing to capture; AUC near 0.6 means the task or labels are the limit.
2. **Cue-source checks** (raw question, GPT cues, hand cues): cheap, and they say whether the matcher's weakness is the writer or the embedder.
3. **Row 5, scene-graph memory** (with the sub-tests K = 1 or 5, and the question as triplets or raw).
4. **Row 6 and Gate 3, the router.** Only after step 1 shows there is something to route.
5. **Gate 4** (text-only GPT payloads) and **Gate 5** (a trained score head), both low priority.
6. **Confirmation:** F1 intervals, the chosen settings on dev, and test once.
7. **Systems work for the hybrid:** live-versus-replay equivalence check, latency per component, running the matcher and decider together, memory management, the streaming shortcut's real speed, quantization (teammates).

### What I would not do

- More prompt variants for the 2B decider: three results say the prompt is not the lever.
- Finer tuning of thresholds and cooldowns on val: 112 matcher settings and dozens of decider settings have been tried on 43 questions, and selection noise already dominates.
- Quoting a best cell as a result, or comparing with Em-Garde's 31.0 directly (their cadence, their trained cue writer, the full benchmark).
- Treating group-level numbers as findings.
- Running dev and test embeddings (about 6 hours) before we know which design is worth it.
- Quantization, or a Gemma comparison, before the strategy shows any promise.

### How the next result changes the plan

If the GPT ceiling is high, the project is promising: the work becomes making the edge confidence informative (scene-graph memory, a larger decider with more compute) and then testing routing. If it is low, the decider's job as posed (judge timing from a short window) is the problem, and we should
rethink what the decider is asked, lean on the matcher, and use GPT where it can see more than the edge model, for example on a longer clip.

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

**Metric:** a response is correct if it lands within **Â±2 seconds** of a ground-truth time.
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
all 172 forward-task questions, at a 2 s decision cadence and a Â±2 s tolerance (`scripts/eval_floors.py`).

| Rule that ignores the video | Avg F1 |
|---|---|
| Respond at every decision point | 15.6 |
| Random, 10% chance per decision point | 16.4 Â± 1.8 |
| Random, 25% chance | 21.4 Â± 1.1 |
| Random, 50% chance | 21.7 Â± 0.6 |

The Â± is the spread across 10 random seeds. The always-silent and oracle checks only confirm that the
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
| Respond every 4 s | 27.0 Â± 0.3 | about 990 |
| Respond every 8 s | 27.6 Â± 2.2 | about 495 |
| Respond every 12 s | 23.1 Â± 3.5 | about 330 |
| Respond every 16 s | 20.2 Â± 3.6 | about 250 |
| Random 50% + 4 s cooldown | 26.7 Â± 1.8 | about 670 |
| Random 25% + 8 s cooldown | 21.3 Â± 2.7 | about 290 |

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

**D3. Use Em-Garde's ground truth and scoring logic exactly, with a tolerance of Â±2 s.**
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
laptop GPU; a response within Â±2 s of the event is still credited, so a 2 s grid can still land
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
*Differences from Em-Garde's code, and what we measured about the main one:*
- **No streaming shortcut.** Their code caches the visual features of earlier frames and encodes only one new frame per
  step (the paper says this makes the matcher 2-3 times faster). The patch reaches into model internals, pins
  `transformers==4.52.3`, and fails on current versions (`'BaseModelOutputWithPooling' object has no attribute 'shape'` on
  transformers 5.23). We therefore embed each window from scratch with the model's standard path.
- **That is not the same vector unless one detail is matched.** We ran their shortcut under transformers 4.52.3 (it works
  there) and compared it, window by window, with embedding the same 4 frames from scratch. The two disagreed: cosine
  similarity 0.91-0.93 on 12 windows. The cause is the prompt, not the caching. Their streaming prompt ends with one
  extra newline after the end-of-text token, the standard prompt does not, and the embedding is read from the final token.
  With the same newline added to the from-scratch prompt, the cosine becomes 0.998-1.00 on the 3 windows we tested (values just
  above 1 are rounding in the model's 16-bit arithmetic). So embedding from scratch **with that newline reproduces their
  vectors** without the old library. Our first version of this stage (no newline) was a different matcher from theirs;
  the earlier statement that the difference would be "small numerical differences" was wrong. The window embedder now adds the
  newline (cue texts keep the standard prompt, as theirs do), and the cache folder name changes, so nothing computed before
  can be mixed in.
- **Cost of not using the shortcut.** Each step re-encodes 4 frames instead of 1, so more compute per step. How much more is not
  settled: on a 4-core CPU we measured 4.5 s per step for the shortcut (under 4.52.3) against 44-52 s from scratch under that
  same version, but only 6-8 s from scratch under 5.23, so the CPU timings contradict each other and tell us little about a GPU.
  `scripts/check_streaming.py` repeats the comparison and the timing on any machine; run it on the 4060 (in an environment
  with transformers 4.52.3, since that is the only place the shortcut runs) to get the real ratio.
- **Attention.** Their default is `flash_attention_2`, which is not available on this laptop setup; we use `sdpa`.
- **Frame size, and a hidden upscaling step.** Their loader decodes frames with Qwen's smart_resize to an area of at most 280 x
  280 = 78,400 pixels (364 x 196 for a 16:9 frame, 91 visual tokens). We use the same rule (`matcher.max_pixels: 78400`); the
  first version reused the decider's 448 x 252 frames, an inherited default, not a choice. **However**, the embedding wrapper then
  resizes every frame *up* to a minimum of 256 x 28 x 28 = 200,704 pixels (256 visual tokens) before the model sees it, in both
  their streaming path and the standard path. A 4-frame window is therefore about 1,085 tokens, not the roughly 400 that the
  decoded size suggests. I had not noticed this; the first GPU timing (720 ms per step, about 95 minutes for the val split) is
  what exposed it.
- **What the upscaling costs and changes (`scripts/diagnose_embed_speed.py`, measured on a 4-core CPU, so read the ratios, not
  the milliseconds):** without it a window is 393 tokens and a step takes 2.1 s against 7.1 s with it (3.3 times faster), but the
  vector changes: cosine 0.94 against the upscaled one. That is a larger change than the cosine of 0.998-1.00 we accepted for
  the newline fix, so we do not assume it is harmless. *Decision:* keep Em-Garde's upscaling as the default, because the goal
  of row 3 is to test their matcher as published, and the run is a one-time, resumable cost. A cheaper variant exists
  (`matcher.min_pixels`, cache keyed separately) and can be compared on the Gate 2 diagnostic later if time becomes the constraint.
- **The first 1.5 s of a video** have no full 4-frame window and are skipped, not padded; no true event is that early.
*Status:* the shortcut-vs-from-scratch equivalence is verified on CPU. Measured on the RTX 4060: median 720 ms per step from scratch, about 95 minutes for the 7,900 val steps and roughly 6 hours for all 31,600 steps of the three splits.

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

**D20. What row 4 takes from row 3, and how it is run.** Row 4 asks the decider "respond now?" only at the moments the matcher fires, and reports a response where
both agree. *Candidates:* the trigger times of the released-code rule with "any cue", a threshold of 0.02 and no duplicate removal. *Why this setting:* for this rule, a higher threshold fires at a subset of the steps a lower one fires at, so one set of
decider answers at the 0.02 triggers serves every threshold from 0.02 up, and duplicates and cooldowns can be applied afterwards, as in row 1. Lower thresholds than 0.02 gave no gain over a timer in row 3, so they are not worth the cost. *Cost:*
1,841 decider calls on val, about 10 minutes at 300 ms. *Chosen, not tuned:* the same Yes/No threshold sweep and cooldown grid as row 1; thresholds picked on val only, and the test split stays untouched. *What it measures:* whether gating a
decider that says Yes too often with a weak cue trigger raises precision more than it loses in recall (a combined response needs both to agree, so recall cannot go up).

**D21. Row 2: the decider with the cues in its prompt, and how we will call the difference real.**
*What changes:* the decider (Qwen3-VL-2B, same frames, same question, same wording) gets one extra block in its prompt, "Visual cues that mean it is time to respond:" followed by
the three cues from row 3's cue file. Nothing else differs, so any change is due to the cues. The cues' identity (a hash of the cue file) is part of the decider's cache key, so decider answers computed with one set of cues can never be
reused for another. *Why this row before the scene-graph row:* it needs no new model run beyond the decider (about 10 minutes on val for the regular ticks, about 8 more at the matcher's trigger moments), and it asks the cheapest version of the question
"does giving the decider more context help?". *How we compare:* for each question, the AUC of p(Yes) for separating moments near an event from the rest (no threshold or cooldown is chosen, so nothing on the val set is selected to flatter the result), averaged over questions;
the change from row 1 to row 2 comes with a 95% interval from resampling whole videos (several questions share one video), for all questions and for each group. *Rule:* a gain counts only if the interval excludes zero (section 5). This is the first use of the interval
tooling (`egogate/eval/bootstrap.py`, `scripts/compare_deciders.py`); it is checked with a synthetic test that it finds a real gain and does not invent a null one. *What each outcome would mean:* if the cues help the decider, row 4 is redone with the cue-aware decider (included in the same
notebook); if they do not, the next thing to try is scene-graph evidence (row 5), not more prompt variants.
*Status:* built; the runner was checked on CPU with the real model on one video (the cues change p(Yes) from the question-only values, and 100% of the probability stays on Yes/No); not yet run on the GPU.

**D22. Fix the base before routing: cue quality first, then scene-graph memory, then cues aligned to the memory (decided 2026-11-23).**
*Decision:* the GPT ceiling check and the router (rows 6, Gate 3) are deferred. *Why:* the router can only be as good as the edge confidence it routes on, and the edge decider is overconfident and the two things meant to ground it (the cue writer and the matcher) are
both weak. Testing routing now would measure the weakness of the base, not the idea. *What makes the matcher weak (evidence from the stored similarities, 43 val questions):* the cue similarity takes values from 0.17 to 0.51 (median 0.37); within one question its highest value spreads over about 0.18; moments near an
event average 0.399 against 0.365 elsewhere; and the largest one-step rise near an event has a median of 0.0071 against 0.0053 elsewhere (90th percentile 0.030 against 0.028), while the trigger thresholds are 0.02 to 0.04. In other words, what the matcher sees at an event is about the same size as its ordinary
wiggle. *Likely reasons, not yet separated:* the embedder measures how much a clip resembles a sentence, not that something has just happened; our cues were not trained for this embedder (Em-Garde's ablation: an untrained 7B writer cost 7 points); the cues are text only; and the benchmark is hard in itself (Em-Garde's own full
system scores 31.0). *Plan, in order, each stage deciding whether the next is worth building:*
- **A. Cue-quality ceiling (built, not run).** Re-score the matcher with other cues on the stored window vectors, no video: the question text alone, and "oracle" cues taken from the benchmark's own annotations (tutorial steps, the activity phrase, the clue answer), which no deployed system has (`scripts/recompute_matcher_sims.py`,
  `egogate/cue_sources.py`). *Rule:* if oracle cues clearly lift the matcher (judged with intervals), cue wording is a main limit and a better-grounded cue writer is worth building; if they do not, the embedder is the limit and rewriting cues will not fix it.
- **B. Scene-graph pilot.** Have the 2B model write (subject, relation, object) triplets every 5 s for a few val questions; read them, time the generation, and measure a quality proxy: how well the text of the triplets near an event resembles the oracle cue compared with elsewhere. Without the question in the prompt, as our earlier notes
  recommend (O2).
- **C. If the graphs are usable:** (i) retrieve with the question rewritten as a triplet, as Response-G1 does; (ii) the decider with the retrieved triplets as evidence (row 5); (iii) a cue writer that sees the memory and writes short cues in the same subject-relation-object form, describing what should appear *next* given what is already
  there, so they are grounded in the scene and not over-detailed. Test (iii) against the baseline cues with intervals, and also match the cues against the graph text directly (Gate 4).
*Risks:* one small model writing both the graphs and the cues can compound its own errors; we have no ground-truth scene graphs; clue-reveal answers stay unknowable from text and memory, so cues there remain guesses about what is not yet in the scene.
*Status:* A built, not run; B and C not built.

### Open on purpose

**O1.** Settled: see D18 (edge model, text only).

**O2. Put the question into the memory (scene-graph) prompt or not?** Response-G1 does
(query-guided generation, section 3.2). Our notes say it makes the model hallucinate early
triggers. Direct disagreement, tested as a sub-row of row 5.

**O3, O4.** Settled: see D19 (both trigger rules and both cue combinations are swept in one table).

**O5. The 7B reference run.** Em-Garde's proposer is a 7B model that does not fit on the 4060 in
BF16 (about 15 GB). Needs a larger GPU or a quantized load, which conflicts with D7.

### D23: larger-model reference through an API, and the drag-and-drop runner (2026-11-23)

**Decision.** Keep the 2B model as the local edge baseline and make a larger model a configuration change, not a code change. We never use model weights in these experiments (no fine-tuning, no activations), only the next-token probability of Yes against No, so any endpoint that returns log-probabilities can stand in. Added: an `api` edge backend (`egogate/edge.py`, `APIEdgeVLM`, any OpenAI-compatible endpoint, OpenRouter by default; the key comes from an environment variable and is never stored), `scripts/run_reference_model.py` (one command: check log-probabilities, run the decider on a small question subset, compare with the 2B model by within-question AUC and bootstrap), `docs/RUN_WITH_LARGER_MODEL.md` (instructions for someone outside the project) and five offline tests (43 pass).

**Why.** The laptop cannot hold a model above about 2B in BF16 (D16), and quantizing is a later step. An API or a bigger GPU gives a reference ceiling: if a much larger model ranks moments no better than the 2B one (about 0.6 AUC), the limit is the task and the 2 s view, not model size. **Not yet run**: no API key has been used; the code is tested only against a fake client. What would break it: models or providers that do not return log-probabilities, and "thinking" models.

**What stays local.** The embedder, the cue matcher, and anything that needs the model's internals (the scene-graph extraction script uses the Hugging Face model directly and has no API route yet).
## 9. Not yet verified

- Em-Garde's 52.3 PAUC reproduction on ProactiveVideoQA EGO (their code and weights are public;
  we have not run them).
- That the embedder wrapper's method names work with our wiring (we vendored their file but have
  not run it).

## 9a. Results so far

### Row 1, edge model alone, validation split (first run)

**Setup.** Qwen3-VL-2B-Instruct in BF16 (D16). Every 2 s the model is shown the question and the last 4 frames
(448 px) and asked "respond now?" (D15). The 43 validation questions (1,983 decisions) are scored with the Em-Garde
metric, tolerance Â±2 s. Median 300 ms per decision on the RTX 4060 laptop GPU. The model puts essentially all its
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
| threshold 0.05, 4 s cooldown | 1002 | 26.7 | respond every 4 s (about 990) | 27.0 Â± 0.3 |
| threshold 0.50, 8 s cooldown | 422 | 25.7 | respond every 8 s (about 495) | 27.6 Â± 2.2 |
| threshold 0.80, 4 s cooldown | 345 | 29.7 | respond every 12 s (about 330) | 23.1 Â± 3.5 |
| threshold 0.80, 8 s cooldown | 212 | 32.0 | respond every 16 s (about 250) | 20.2 Â± 3.6 |

At low and middle thresholds the model is no better than speaking on a timer. At threshold 0.80, where it responds only
when quite sure, it scores 7 to 12 points above a timer making a similar number of responses. (Timer rows are the 30-phase
figures from section 5; the Â± is the spread across phases, not a confidence interval.)

**Reading 4: the ranking signal is real but weak.** (*Correction added after row 3:* the figures below pool every decision from every question, which mixes in differences between questions, such as REC questions having higher p(Yes) everywhere. Averaging the AUC within each question instead gives 0.614, not 0.71. That is the number to compare with the matcher; see the row 3 section.) Taking every decision, and labelling it "positive" when it falls within
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

**Next.** Row 4 is below.

### Row 3, the cue matcher alone, validation split (first run)

**Setup.** Cues written by Qwen3-VL-2B from the question text only (D18); windows embedded by Ops-MM-embedding-v1-2B exactly as Em-Garde
runs it, including the upscaling and the trailing newline (D17); every 0.5 s, 7,903 steps over the 43 val questions. Each trigger is
counted as a response and scored with the same metric as row 1. The run took 6,845 s (114 minutes), against 95 minutes projected: the median time per step
drifted from 727 ms on the first question to 917 ms on the last, which looks like heat or power throttling (not verified). Source: the saved
output of `notebooks/row3_cue_matcher.ipynb`; the stored similarities are not in the repository yet.

**Gate 2 diagnostic: how well does the similarity itself separate steps near an event from the rest?** (AUC; 0.5 means no separation, 1.0 is perfect.)

| Signal | AUC |
|---|---|
| highest similarity among the cues | 0.598 |
| average similarity across the cues | 0.617 |
| largest one-step rise (the quantity the paper's rule thresholds) | 0.540 |
| *for comparison, row 1: the decider's p(Yes)* | *0.706* |

Those figures pool all steps from all questions, which mixes in differences *between* questions (each question has its own cues and so its own typical
similarity level). Averaging the AUC within each question instead gives a cleaner picture, and it changes the comparison with row 1:

| Signal (AUC computed within each question, then averaged; 43 questions) | AUC |
|---|---|
| matcher: highest similarity among the cues | 0.612 |
| matcher: average similarity across the cues | 0.624 |
| matcher: largest one-step rise | 0.554 |
| decider (row 1): p(Yes) | 0.614 |

So the matcher and the decider are about equally informative, and both only weakly (0.5 is no skill). The 0.71 quoted for row 1 above was inflated by the pooling. The rise, which is
what a trigger rule actually looks at, is the weakest of the four. This measure averages over all steps; the F1 below depends only on the strongest triggers, so the two need not agree.

**What the sweep shows.** Average F1 (number of responses in brackets) for the rule in Em-Garde's released code, any cue may fire, with a 4 s
cooldown, across thresholds (higher threshold means fewer, more selective triggers):

| Threshold | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 |
|---|---|---|---|---|---|---|---|
| F1 (responses) | 28.6 (698) | 29.4 (539) | **31.8 (426)** | 29.9 (329) | 26.2 (270) | 24.2 (210) | 16.1 (130) |

**Reading 1: it beats a timer, modestly, at moderate response counts.** Compared with the "respond every k seconds" floors of section 5 (30 phases) at
a similar number of responses (interpolating between them where needed, so approximate):

| Matcher cell | Responses | Matcher F1 | Timer with a similar count | Gain |
|---|---|---|---|---|
| threshold 0.01 | 698 | 28.6 | about 27 | about 1 (nothing) |
| threshold 0.02 | 539 | 29.4 | about 27.6 | about 2 (nothing) |
| threshold 0.03 | 426 | 31.8 | about 25.7 | about 6 |
| threshold 0.04 | 329 | 29.9 | 23.1 Â± 3.5 | about 7 |
| threshold 0.05 | 270 | 26.2 | about 21 | about 5 |

With many responses it is no better than speaking on a timer; with 270 to 430 responses it is roughly 5 to 7 points above one. The 31.8 best cell
is the best of 112 settings on 43 questions, so it is flattered; the neighbouring cells (29 to 30) are the more honest level. The similar scores of row 1
(32.0 best, about 30 at comparable response counts) and row 3 mean the two parts are about equally good on their own.

**Reading 2 (Gate 2, how to combine the cues).** At the paper's threshold of 0.04, "any cue may fire" is far better than "average of the cues" (29.9 against 11.3 with a 4 s cooldown), because
averaging smooths out the rise. But the average's best setting is a lower threshold (31.6 at 0.02), which ties "any cue" at its best (31.8). We cannot separate them with 43
questions, so we keep "any cue" (the paper's choice, and it is robust to the threshold).

**Reading 3 (which rule).** The released code's rule is as good as the paper's at their best settings (31.8 against 31.6) and much steadier across thresholds: 26 to 32 from 0.01 to 0.05, against a fall from 31.6 to 20.0 for the paper's rule.

**Reading 4 (duplicates).** Removing near-duplicate triggers matters a lot: with no cooldown the best is 22.3, against 31.8 with a 4 s cooldown. Em-Garde's own embedding-based rule is about the same as the 4 s cooldown (31.0).

**Reading 5: where it works (the cue-quality question from D18).** Splitting the best setting (released rule, any cue, threshold 0.03, 4 s cooldown) by question type. The
type and the video source coincide: clue-reveal questions are all MovieNet clips, tutorials all COIN, and the counting questions split into sports clips (thumos) and everyday-object clips (perception_test).
The timer is the "respond every k seconds" system whose number of responses is closest to the matcher's in that group (30 random phases).

| Group | Questions | Matcher F1 | Recall / precision | Responses | Closest timer | AUC within question (matcher highest similarity / decider) |
|---|---|---|---|---|---|---|
| Counting, sports clips | 10 | **57.0** | 68% / 49% | 99 | 32.7 Â± 5.2 (103 responses) | 0.67 / 0.70 |
| Counting, everyday objects | 14 | 40.2 | 38% / 43% | 37 | 38.4 Â± 6.3 (40) | 0.60 / 0.62 |
| Tutorial steps | 12 | 26.2 | 68% / 16% | 249 | 18.2 Â± 2.4 (271) | 0.67 / 0.60 |
| Clue reveal | 7 | 21.5 | 71% / 13% | 41 | 20.3 Â± 8.5 (38) | 0.46 / 0.50 |

This matches what the cues looked like (D18): sports clips, where the cues were concrete and on topic, gain about 24 points over a timer; tutorials gain about 8 but with many
false alarms (precision 16%); everyday-object counting and clue reveal gain nothing, and for those the cues were the weakest (invented scenes, or guesses at an answer that a text-only writer cannot know).
For clue-reveal questions neither the matcher nor the decider separates event from non-event at all (AUC 0.46 and 0.50). *Caution:* this cannot separate "better cues" from "different kind of video"; sports
broadcasts are visually distinctive and may suit an embedding model regardless of cue wording. The groups are 7 to 14 questions each, so these are indications, not measurements.

**What this means.** The matcher with our small, untrained cue writer is not useless: it finds events better than a timer in some kinds of video (clearly in sports clips, somewhat in tutorials) and
not at all in others. Within a question, it is about as informative as the decider alone (AUC 0.61 each), and the two differ by group (the matcher is better on tutorials, the decider on sports and everyday objects),
which is some reason to think they could complement each other; row 4 tests that directly. Nothing has been combined yet.

**Caveats.** One split of 43 questions; settings chosen and scored on the same questions; no confidence intervals yet; timer figures interpolated; the per-group numbers come from `scripts/analyze_matcher.py` on the stored similarities (now in the repository).


### Row 4, matcher fires AND decider says Yes, validation split (first run)

**Setup.** Candidates are the matcher's trigger moments (released-code rule, any cue, threshold 0.02, no duplicate removal): 1,841 moments over the 43 val questions (D20). The decider (Qwen3-VL-2B, question only, as in row 1)
was run at exactly those moments: 1,395 new on-demand calls at a median of 258 ms, plus the stored answers from row 1 where a trigger fell on a 2 s tick. A response needs the matcher to fire and p(Yes) to reach a threshold tau;
a cooldown is applied to the responses afterwards, as in row 1. tau = 0 is the matcher alone.

**Result: adding the decider did not help.** Average F1 (responses in brackets) for matcher threshold 0.03 with a 4 s cooldown:

| tau (how sure the decider must be) | 0 (matcher alone) | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| F1 (responses) | **31.8 (426)** | 30.1 (403) | 28.2 (377) | 28.2 (342) | 28.7 (299) | 26.2 (193) | 12.6 (43) |

The best cell in the whole table is the matcher alone. Across the other thresholds and cooldowns the decider moves the score by a point or two in either direction (for example 25.0 to 26.8 at threshold 0.02 with an 8 s cooldown and tau 0.8, but 29.4 down to 27.8 with a 4 s cooldown),
which is within the noise of a 43-question split. Comparing with the matcher alone at a similar number of responses (matcher at threshold 0.04: 29.9 with 329 responses; matcher plus decider at 0.03 and tau 0.7: 28.7 with 299) shows no gain either.
So on this data the two parts do not compound.

**Why: the decider carries some information at these moments, but not enough.** Among the 1,841 candidate moments, 19.8% are within 2 s of an event. Grouping them by the decider's p(Yes):

| p(Yes) | Moments | Share really near an event |
|---|---|---|
| below 0.3 | 138 | 9.4% |
| 0.3 to 0.5 | 162 | 7.4% |
| 0.5 to 0.7 | 395 | 7.6% |
| 0.7 to 0.8 | 489 | 20.0% |
| 0.8 to 0.9 | 554 | 30.9% |
| above 0.9 | 103 | 39.8% |

So the decider does rank: moments it is very sure about are near an event four times as often as moments it is unsure about, and the ranking is steeper than "on average 0.78 vs 0.67" suggests. Within a question the AUC is 0.62 (pooled over questions
0.70), the same as on all moments in row 1 (0.61). But it is a weak ranking, with a big jump at 0.7 and not much below it, and it is weakest where the matcher is also weakest.

**By kind of question** (threshold 0.03, 4 s cooldown; F1 with responses in brackets):

| Group | matcher alone | tau 0.5 | tau 0.7 | tau 0.8 | AUC of p(Yes) among candidates |
|---|---|---|---|---|---|
| Counting, sports clips | 57.0 (99) | 55.9 (91) | 56.6 (76) | 47.6 (54) | 0.69 |
| Counting, everyday objects | 40.2 (37) | 40.2 (37) | 42.1 (33) | 36.3 (24) | 0.65 |
| Tutorial steps | 26.2 (249) | 24.0 (226) | 24.4 (171) | 30.3 (105) | 0.64 |
| Clue reveal | 21.5 (41) | 13.5 (23) | 13.5 (19) | 7.1 (10) | 0.40 |

The decider is neutral on sports clips, perhaps slightly helpful on everyday objects at tau 0.7 (+2 points, within noise), helpful on tutorials only at tau 0.8, and harmful on clue reveal, where it points the wrong way (AUC 0.40 on 5 questions).

**One more thing tried (exploratory, after seeing the results, on val only, so treat it as a hint).** Instead of "the first trigger in a time window wins", keep the best-scoring one. Using the matcher's own rise as the score:
32.3 (4 s window) against 29.4 for first-wins; using the decider's p(Yes): 29.2. The decider's score does not choose better than the matcher's own.

**What this means.** With a generic question-only prompt and a 2B model, the decider is too weak to improve on the matcher. That is a result about this decider, not about gating as an idea. Its prompt contains neither the cues nor any scene evidence,
and those are exactly the things rows 2 and 5 add. Gate 1 is also touched: the decider's confidence is informative but weak (AUC about 0.62), and a router that sends uncertain moments to GPT can be no better than the signal it is routing on, so Gate 3 may show little advantage for
confidence-based routing over random routing unless the decider improves. We will find that out, not assume it.

**Caveats.** One split of 43 questions; thresholds read off val; no confidence intervals; the clue-reveal group is 7 questions (5 with both kinds of moment); the tau, cooldown and window comparisons are post hoc.

**Next.** Row 2: give the decider the cues in its prompt (a decider rerun, about 10 minutes on val, no new embedding), and compare with row 1 on the same moments.

### Row 2, the decider with the cues in its prompt, validation split (first run)

**Setup.** Exactly row 1 (Qwen3-VL-2B, same frames and question) with the three cues written into the prompt (D21). Speed is unchanged: median 242 ms per decision on the regular ticks and 246 ms at the matcher's trigger moments
(row 1: 300 and 258 ms; the differences are within what the GPU's state could explain). Compared with row 1 by the within-question AUC of p(Yes), with 95% intervals that resample whole videos.

**The pre-planned test, all 43 questions: no change.** AUC 0.614 without cues, 0.618 with: a difference of +0.003, interval [-0.038, +0.041]. The cues did not make the decider better at telling moments near an event from the rest. The share of moments it says Yes to
changed little (80% to 76%).

**By group (not pre-planned, ten unadjusted intervals in the two tables, so one or two crossing zero by luck is expected):**

| Group | Questions | AUC without cues | AUC with cues | Change | 95% interval |
|---|---|---|---|---|---|
| Counting, sports clips | 10 | 0.699 | 0.735 | +0.036 | [+0.003, +0.068] |
| Counting, everyday objects | 14 | 0.618 | 0.627 | +0.009 | [-0.055, +0.084] |
| Tutorial steps | 12 | 0.604 | 0.525 | **-0.080** | [-0.162, -0.016] |
| Clue reveal | 7 | 0.504 | 0.592 | +0.088 | [-0.004, +0.131] |

Cues look slightly useful for sports clips, where the cues were good (the interval only just clears zero), and **harmful for tutorials** (the decider drops to 0.525, almost no skill), where the cues were generic or wrong. For clue reveal the change is large but uncertain with 7 questions, and
the cues there were guesses, so I would not read it as a benefit.

**At the matcher's trigger moments** (38 questions with both kinds of moment): all questions 0.619 without cues, 0.569 with (change -0.050, interval [-0.123, +0.015]); tutorials 0.636 to 0.506 (change -0.130, interval [-0.215, -0.059]). Again no overall gain, and again a clear loss on tutorials.

**Row 2 on its own, scored like row 1.** Best on val: 27.4 F1 (cooldown 8 s, p(Yes) at least 0.1), against row 1's 32.0 (tuned on the same val, cooldown 8 s, p(Yes) at least 0.8). Row 1's score *rose* as the decider was made more demanding (0.7 to 0.8); with cues it *falls* (23.2 and 19.9
at 0.7 and 0.8 with an 8 s cooldown). So with cues, the moments the decider is most sure about are less often the right ones. At a 4 s cooldown the cue decider scores 26 to 27 at low thresholds, the same as a timer making a similar number of responses (about 27).

**Row 4 redone with the cue-aware decider: worse.** Matcher threshold 0.03, 4 s cooldown, F1 by how sure the decider must be: matcher alone 31.8; with cues in the decider 31.5 (0.3), 26.1 (0.5), 21.8 (0.7), 15.0 (0.8). Row 4 with the question-only decider gave 30.1, 28.2, 28.7, 26.2 at the same
thresholds. So the cue-aware decider removes true events faster than false alarms, and the best cell in the whole table is still the matcher alone.

**What this means.**
- Giving a 2B decider the cues does not make it a better judge of timing. On the questions where the cues are weak it makes it worse, as if it were agreeing with plausible-sounding but wrong descriptions.
- Rows 1, 2 and 4 together say the same thing: the 2B decider, shown the last 2 s of video, ranks moments only weakly (AUC about 0.6) however we prompt it, and it does not add to what the matcher already finds.
- This is the point where it is fair to ask what limits the decider, because there are two quite different answers: the model is too small, or the task as posed (judging timing from a 2 s window with no history) is the limit. A scene-graph memory (row 5) addresses the second; a stronger decider addresses the first.
  A cheap way to tell them apart is to run a strong model, GPT with the same frames and prompt, on the same moments. That also gives the "GPT everywhere" end of the Gate 3 plot. If GPT also lands near AUC 0.6, the limit is the task or the labels, and routing to GPT would not help; if GPT is much better, there is headroom that routing could capture.
  Not yet run; it needs an OpenAI key and costs very little (about 2,000 calls on val).

**Caveats.** One split of 43 questions; the group intervals are unadjusted and not pre-planned; the F1 comparisons use settings chosen on val; no test split has been used.

### Step A (D22): cue-quality ceiling, matcher re-scored with other cues on the stored window vectors (val, 43 questions, 2026-11-22)

**What was done.** The 2 s window vectors already stored for val were re-scored against different cue texts (no video decoded, no GPU decision): the edge model's own cues (baseline), the benchmark's annotations as cues ("oracle": tutorial step list, activity phrase, clue answer; no deployed system has these, so this is an upper bound on cue wording), and the raw question text as the single cue. All other settings are unchanged (Em-Garde's released rule, any cue, 4 s cooldown, threshold 0.03 for the by-group table).

**Gate 2 diagnostic (7,903 steps, 14.7% within 2 s of an event; 0.5 = no separation).**

| Cues | max-similarity AUC | mean-similarity AUC | max-rise AUC |
|---|---|---|---|
| Edge model's cues (baseline) | 0.598 | 0.617 | 0.540 |
| Oracle (benchmark annotations) | 0.612 | 0.638 | 0.525 |
| Raw question text | 0.547 | 0.547 | 0.543 |

(These AUCs are pooled over all steps, so they mix questions; read them only as a rough ordering. No bootstrap interval was computed.)

**By group at threshold 0.03, F1 against the closest timer (timer ± is its spread over start offsets).**

| Group | Baseline cues | Oracle cues | Question text |
|---|---|---|---|
| Sports counting (10) | 57.0 vs 32.7 (99 resp) | 35.4 vs 23.4 (57 resp) | 27.7 vs 23.4 (45 resp) |
| Everyday-object counting (14) | 40.2 vs 38.4 (37 resp) | 30.1 vs 25.8 (21 resp) | 32.9 vs 17.0 (15 resp) |
| Tutorials (12) | 26.2 vs 18.2 (249 resp) | 27.0 vs 18.5 (233 resp) | 27.2 vs 17.5 (185 resp) |
| Clue reveal (7) | 21.5 vs 20.3 (41 resp) | 28.0 vs 15.1 (27 resp) | 23.0 vs 17.7 (33 resp) |

**Reading.** Oracle cues do not clearly lift the matcher. The mean-similarity AUC moves from 0.617 to 0.638, within what 43 questions can move by chance. Tutorials, where the oracle gives the exact step list, gain 0.8 F1 over the baseline (26.2 to 27.0), and the gain over the timer is the same (+8.0 to +8.5). Sports counting, the group where the baseline cues were good, gets *worse* with oracle cues (57.0 to 35.4), mostly because fewer responses are made at the same threshold (99 to 57; the similarity scale changes with the cue text), so a fixed threshold is not a fair comparison there. Clue reveal looks better (+12.9 over the timer against +1.2) but it is 7 questions with a timer spread of ±9, so it is not evidence. The best cell of the oracle table (40.0 F1, mean rule, threshold 0.02) is the best of many cells and was not compared at equal response count, so it is not quoted as a result. The question-text-only cue is the weakest on the diagnostic, as expected.

**Decision rule outcome.** Per D22, if oracle cues do not clearly lift the matcher, the embedder (not the cue wording) is the main limit, and building a memory-aligned cue writer is not justified on this evidence. Cue wording is a minor factor on this embedder with these 2 s windows. Caveats: one split, 43 questions, no bootstrap interval on the oracle comparison, fixed-threshold comparison with differing response counts, for tutorials the oracle cues are the benchmark's step descriptions, one cue per step (the matcher fires on any of them), which are short phrases and may still be far from what the embedder separates visually.
### Step B (D22): scene-graph pilot, 4 val questions (2026-11-22)

**What was done.** `scripts/pilot_scene_graph.py`: the 2B edge model (Qwen3-VL-2B, BF16) was shown 5 frames (1 per second, the last 5 s) every 5 s and asked, with the existing `TRIPLET_PROMPT` and without the question, for up to 8 (subject, relation, object) triplets. One question from each source (tennis counting, bowl-hitting counting, slime tutorial, a movie clip), the first 90 s of each: 31 ticks. Triplets are stored in `results/cache/scene_graph_pilot/`.

**Cost.** 4 to 9 s per tick (median 8.8 s), because the model fills the whole 160-token budget. At that rate the 43 val questions (about 790 ticks) would take roughly 1.5 to 2 hours, against about 10 minutes for the decider.

**Quality (read by hand; the proxies are not meaningful, see below).** The triplets are mostly unusable as written.
- **Repetition loops.** The same triplet repeats until the token limit, for example `(liquid, pour, cup)` 20 times, `(player, is holding, racket)` 8 times. After removing duplicates a tick often holds 1 to 4 distinct facts.
- **Template leakage and wrong language.** `(wooden table, relation, has)`, `(3, relation, 3)`, and a whole tick in Chinese (`(杯子, 位于, 桌子)`).
- **Static content.** What is described is mostly objects and clothing ("player wearing white shirt"), not what changed. Only the tennis clip names the action ("serving", "preparing to serve"). The slime clip's triplets (ink bottle, beaker, glass of liquid) do not match the steps (add ingredients, mix, rub and drag).
- **Parse rate** 0.97 and 4.0 triplets per tick, but these count the repeats, so they overstate the content.

**Proxies.** Novelty (share of triplets new compared with the previous tick) was 0.97 near events and 0.91 elsewhere. Cue-word hit (any content word of the oracle cue in the triplets) was 0.12 near events and 0.21 elsewhere, i.e. *lower* near events. With 31 ticks this is noise, and novelty is inflated because wording changes from tick to tick without the scene changing. The point is only that there is no sign of an event signal.

**Reading.** With this prompt and this 2B model the scene graph is not a usable memory: it is slow, loops, and describes the static scene rather than the change a question is about. This does not show that scene graphs cannot help; it shows this extraction does not. Cheap fixes that are not tried yet: a repetition penalty or a stop after 8 lines, a shorter token budget (about 64), an English-only instruction, and asking for what *changed* since the last clip. Whether to try these before step C is a decision for the user. Step C (memory-aligned cues) is not supported by this pilot.

**Caveats.** 4 questions, 31 ticks, one prompt, one sampling setting, hand-read.
### Step B, second try: generation fixes and a relevance check (same 4 val questions, 31 ticks, 2026-11-22)

**Changes to the extraction** (`scripts/pilot_scene_graph.py --variant fixed --rep-penalty 1.15 --max-new 80`): a rewritten prompt (English only, at most 8 lines, describe what people DO and what changed, every line different, no "relation" placeholder), the previous tick's triplets given as text with "say what is new or different", a repetition penalty of 1.15 and an 80-token budget. The question is still not in the prompt.

**Result.** The loops, the Chinese output and the placeholder words are gone, and generation takes 4.3 s per tick (median 4.5 s) against 7.6 s. The triplets now name actions: tennis "serves ball", "swings racket", "hits ball"; slime "pouring liquid from bottle into cup", then "stirring blue liquid in cup", "removing wooden stirrer"; bowl clip "hand lifts bowl", "bowl is put down on table". Two of 31 ticks parsed to nothing because the model dropped the "Subject:" label on some lines (a parser issue, not empty output). Still weak: some ticks repeat the previous tick almost line for line (bowl clip at 15 s and 20 s), the model sometimes invents a second player, and it cannot say how many times something happened. At 4.3 s per tick, full val would still take about 1 hour.

**Relevance check** (`scripts/pilot_triplet_relevance.py`, text against text with Ops-MM, separate process). For each tick, the highest cosine between any of its triplets and the cue (oracle phrase, or the question). Within-question AUC of near-event ticks against other ticks: 0.53 with the oracle cue, 0.65 with the question, averaged over 4 questions, one of which has a single near-event tick. That is noise at this size and does not support using a similarity threshold to drop triplets. Raw similarities were low (0.25 to 0.52), so any threshold would also have thrown away most ticks.

**Reading.** The extraction problem was mostly prompt and decoding, and is largely fixed. Whether the triplets carry the event signal is not shown: the one proxy we have is the same embedder that is weak on video, and 31 ticks cannot settle it. A better test of the memory is downstream (does retrieval of triplets help the decider, row 5), not a similarity filter.
### Row 5 short validation: decider with scene-graph evidence, and "what is new" scoring (14 and 12 val questions, 2026-11-23)

**Why short.** Before an hour of extraction for all of val, both ideas were tried on a small subset: the 3 shortest questions of each source plus the 2 other clue-reveal questions already extracted (14 questions, 112 five-second ticks, 8 minutes of extraction). Settings are those of the step B second try (`scripts/run_scene_graph_cache.py`). The full-val extraction was started and stopped after 4 questions; those are kept.

**Test 1, the decider with evidence** (`scripts/run_decider_evidence.py`; question plus the triplets of the last two extraction ticks, up to 10 lines, no cues, only past ticks; 111 s). Compared with row 1 on the same 14 questions by within-question AUC with a video-level bootstrap interval (`scripts/compare_deciders.py`):

| Group | Questions | Row 1 AUC | With evidence | Change | 95% interval |
|---|---|---|---|---|---|
| All | 14 | 0.574 | 0.454 | **-0.120** | [-0.240, -0.017] |
| Sports counting | 3 | 0.802 | 0.722 | -0.079 | [-0.236, +0.079] |
| Everyday-object counting | 3 | 0.453 | 0.322 | -0.131 | [-0.393, +0.083] |
| Tutorials | 3 | 0.596 | 0.300 | **-0.296** | [-0.491, -0.063] |
| Clue reveal | 5 | 0.496 | 0.466 | -0.031 | [-0.312, +0.145] |

The share of moments where the decider says Yes rose from 0.78 to 0.89. So the evidence made the decider say Yes more often and rank moments worse, the same pattern as cues in row 2 (more text in the prompt, more agreement, less discrimination). 14 questions and unadjusted intervals: the overall interval only just excludes zero, so this is a warning, not a firm loss; but there is no sign of a gain.

**Test 2, scoring what is new** (`scripts/pilot_triplet_novelty.py`; text against text with Ops-MM; a tick counts as near an event if an event lies in [t-5, t+2]; within-question AUC averaged over 12 questions with both kinds of tick):

| Score | Oracle cues | Question text as cue |
|---|---|---|
| relevance: best cosine between a triplet and a cue | 0.664 | 0.577 |
| new: how unlike the previous tick the triplets are (no cue) | 0.557 | 0.557 |
| new and relevant: cue cosine times novelty | 0.600 | 0.540 |
| rise in relevance since the previous tick | 0.402 | 0.436 |

Scoring what is new does not beat plain relevance (0.60 and 0.54 against 0.66 and 0.58), novelty without any cue is close to chance, and the rise is below 0.5. Small sample, no bootstrap interval; relevance with oracle cues at 0.66 is the only score that looks above chance, and it needs cues no deployed system has.

**Reading.** Neither idea is validated at this size. Putting triplets in the decider's prompt hurt, and the embedder on triplets gives no better event signal than it gave on video. Taken with step A and B, the weak part is not specifically the video embedder, the cue wording or the extraction format: neither the matcher, the decider, nor a text comparison picks event moments out reliably from a 2 s to 5 s view of the stream. We have not yet tried: longer or smarter views (several windows, frame differences), a different embedder, or a larger decider.
### Several-window test on the stored window vectors (val, 38 questions, CPU, 2026-11-23)

**Question.** Is the 2 s window too little context? Using the stored val window vectors (one per 0.5 s step) and cue similarities, no video and no model (`scripts/pilot_context_windows.py`): score each step from more of the past stream and compare, by within-question AUC (steps within 2 s of an event against the others, 10 s of history required for every score so the same steps are compared), with a paired difference from the matcher's current score and a video-level 95% interval. Scores: base (best cue similarity now); roll_k (that similarity averaged over the last k steps, k=4, 8, 16, i.e. 2, 4, 8 s); contrast_k (similarity minus its average over the k steps before, k=8, 20); change_k (1 - cosine between the window vector now and k steps ago, k=4, 8: visual change, no cue at all).

| Score | AUC, edge cues | Change from base | AUC, oracle cues | Change from base |
|---|---|---|---|---|
| base | 0.627 | | 0.613 | |
| roll_4 (2 s) | 0.594 | -0.033 [-0.057, -0.008] | 0.577 | -0.036 [-0.062, -0.006] |
| roll_8 (4 s) | 0.547 | -0.079 [-0.123, -0.037] | 0.530 | -0.083 [-0.129, -0.033] |
| roll_16 (8 s) | 0.518 | -0.109 [-0.173, -0.050] | 0.498 | -0.115 [-0.183, -0.050] |
| contrast_8 | 0.648 | +0.021 [-0.041, +0.081] | 0.631 | +0.018 [-0.051, +0.084] |
| contrast_20 | 0.662 | +0.035 [-0.013, +0.080] | 0.621 | +0.008 [-0.035, +0.050] |
| change_4, no cue | 0.590 | -0.036 [-0.124, +0.047] | 0.590 | -0.023 [-0.113, +0.062] |
| change_8, no cue | 0.578 | -0.049 [-0.144, +0.045] | 0.578 | -0.035 [-0.125, +0.060] |

**Reading.**
- **Averaging over a longer span hurts, clearly** (the interval excludes zero for all three): the event is brief and a longer average smears it. So "give it more context" in this simple form is the wrong direction.
- **Comparing against what was just before helps slightly, but not reliably** (+0.02 to +0.035, intervals include zero). It is the only direction that points up, and it is the "rise" idea with a local baseline instead of one step back.
- **Visual change alone, with no cue, scores 0.58 to 0.59**, close to the cue-based 0.63. This says a good part of what the matcher picks up is that the picture is changing, not that it matches the cue. That fits step A: better cue wording barely moved the score.
- Same caveats as before: unadjusted, 38 questions, 8 scores shown, within-question AUC is about ranking steps and not a final score.
### Contrast rule as a trigger, scored against a matched timer (val, 43 questions, CPU, 2026-11-23)

**Question.** The several-window test left one direction slightly up: compare the cue similarity with its own recent past. As a trigger rule, does it beat Em-Garde's released rule ("code") at the same number of responses? `scripts/pilot_contrast_rule.py` sweeps a threshold grid for three rules, all with a 4 s cooldown: "code" (the matcher now), "contrast" (a cue's similarity more than theta above its average over the previous 10 s) and "change" (the same on the cue-free visual change signal). The whole grid is shown; the "gap" is F1 minus the F1 of the timer with the closest number of responses (timer spread over start offsets is 1.7 to 3.6).

Note on the scale: this script pools all questions into one F1, which is not the average over task groups used in the earlier tables, so these F1 values (for example code at 0.03: 41.3 here, 31.8 there, same 426 responses) are not comparable with section 9a above. Only the gaps within this table are.

| Rule | theta | Edge cues: F1 (responses), gap | Oracle cues: F1 (responses), gap |
|---|---|---|---|
| code | 0.01 | 40.1 (698), +3.7 | 45.8 (612), +9.4 |
| code | 0.02 | 40.7 (539), +6.4 | 48.9 (474), +14.6 |
| code | 0.03 | 41.3 (426), +9.1 | 35.0 (338), +4.9 |
| code | 0.04 | 35.0 (329), +4.8 | 28.7 (246), +3.5 |
| code | 0.05 | 25.8 (270), +0.7 | 27.3 (195), +6.2 |
| contrast | 0.01 | 33.8 (667), -2.6 | 38.6 (597), +2.2 |
| contrast | 0.02 | 36.9 (561), +2.6 | 32.5 (487), -1.8 |
| contrast | 0.03 | 40.7 (476), +6.4 | 30.6 (380), -1.7 |
| contrast | 0.04 | 35.5 (383), +3.3 | 21.9 (297), -8.3 |
| contrast | 0.05 | 33.5 (311), +3.4 | 20.1 (231), -5.0 |
| contrast | 0.06 | 26.2 (246), +1.1 | 18.4 (176), -2.8 |
| change (no cue) | 0.02 to 0.15 | 11.6 to 32.4, gap -1.9 to -10.0 | not applicable |

**Reading.**
- **The contrast rule does not beat the released rule.** With edge cues its gaps (+1 to +6) are at or below the released rule's (+1 to +9); with oracle cues it is at or below the timer. The small AUC hint did not turn into a better trigger.
- **The cue-free visual change rule loses to a timer at every threshold** (gap -2 to -10). So although visual change alone had an AUC of 0.59, as a trigger it is worse than doing nothing clever; the cue is what makes the trigger better than a timer.
- **Oracle cues look better here than in step A.** With the released rule, oracle gaps are +9.4, +14.6, +4.9, +3.5, +6.2 against +3.7, +6.4, +9.1, +4.8, +0.7 for edge cues: larger at four of five thresholds, mean gap 7.7 against 4.9, best 14.6 against 9.1. Step A, at one threshold and by group, saw no clear lift. This pooled view hints that perfect cue wording may be worth a few F1 points; with a timer spread of 2 to 3 points, 43 questions, a different aggregation and no bootstrap interval, I do not treat it as established. It softens the step A conclusion: cue wording is probably a smaller limit than the embedder, not necessarily none.
### Paired bootstrap of the cue-wording gap (val, 43 questions, CPU, 2026-11-23)

**Question.** The contrast-rule table hinted that oracle cues beat the edge model's cues; step A had seen no clear lift. `scripts/bootstrap_cue_gap.py` settles it the way the project rules ask: for each threshold of the released rule (4 s cooldown), the matcher's averaged F1 (mean over the three tasks, as in the earlier tables) minus the F1 of the timer with the closest number of responses (10 random phases), for each cue source; then oracle's gap minus the edge cues' gap with a 95% interval from resampling whole videos.

| Threshold | Edge: responses, gap | Oracle: responses, gap | Oracle minus edge | 95% interval |
|---|---|---|---|---|
| 0.01 | 698, -0.1 | 612, +9.4 | **+9.5** | [+1.8, +14.3] |
| 0.02 | 539, +2.0 | 474, +11.3 | **+9.3** | [+2.1, +15.4] |
| 0.03 | 426, +7.0 | 338, +6.3 | -0.7 | [-8.4, +7.0] |
| 0.04 | 329, +6.6 | 246, +4.4 | -2.2 | [-12.5, +8.8] |
| 0.05 | 270, +3.5 | 195, +5.3 | +1.9 | [-7.2, +11.2] |
| mean over thresholds | | | +3.6 | [-2.6, +10.1] |

**Reading.** Oracle cues clearly help when the matcher is set loose (thresholds 0.01 and 0.02, where it fires 470 to 700 times): the edge cues are then no better than a timer (gap about 0 to +2) and the oracle cues are about 9 points better, interval above zero. At the better thresholds (0.03 and up) the two are the same: the edge cues reach a gap of +7.0 at 0.03, and no oracle setting beats that by a margin the data can see. The average over thresholds, which avoids picking one, is +3.6 with an interval that includes zero. So perfect wording makes the matcher more robust to the threshold, but does not raise its ceiling. With five thresholds looked at, two intervals excluding zero is not strong evidence; the safe statement is that cue wording matters at loose thresholds and not at the best one. Step A's conclusion stands in the form: cue wording is not what limits the best matcher result; the embedder is.
### A different embedder: SigLIP2 in place of Ops-MM (val, 38 questions, 2026-11-23)

**What was done.** `scripts/run_embed_alt.py` re-ran the matcher's stored steps with `google/siglip2-so400m-patch14-384`, a standard image-text model: each frame (one per 0.5 s) embedded once, a step's window vector is the normalised mean of its 4 frame vectors (the same 2 s, 4 frames as the Ops-MM windows), cosine to each cue, same edge cues and (second folder) oracle cues. Cost: 5.7 minutes for all of val on the idle GPU (Ops-MM took 114 minutes: the wrapper's 256-token frames dominate, SigLIP2 works at its own 384 px). `scripts/compare_embedders.py` compares them question by question: within-question AUC of the best-cue similarity for steps within 2 s of an event against the other steps (10 s of history required), paired difference with a video-level 95% interval.

| Group | Questions | Ops-MM AUC | SigLIP2 AUC | SigLIP2 - Ops-MM | 95% interval |
|---|---|---|---|---|---|
| All, edge cues | 38 | 0.627 | 0.618 | -0.009 | [-0.071, +0.049] |
| Clue reveal | 4 | 0.498 | 0.649 | +0.152 | [+0.025, +0.278] |
| Everyday-object counting | 12 | 0.590 | 0.572 | -0.018 | [-0.128, +0.094] |
| Sports counting | 10 | 0.693 | 0.656 | -0.036 | [-0.145, +0.069] |
| Tutorials | 12 | 0.652 | 0.621 | -0.031 | [-0.102, +0.028] |
| All, oracle cues | 38 | 0.613 | 0.628 | +0.015 | [-0.056, +0.084] |

**Reading.** The two embedders rank steps equally well (difference -0.009 with edge cues, +0.015 with oracle cues, both intervals centred on zero). A model a fraction of the size, with none of the wrapper's cost, does no worse. So the weak event signal is not a property of Ops-MM specifically: two quite different embedders both give about 0.62 to 0.63, which points at the approach (similarity of a short window to a text cue) more than at the model. The clue-reveal gain is 4 questions and one of 20 intervals, so I do not read it as a benefit. Not done: the F1-against-timer comparison for SigLIP2 (its similarity scale differs, so the thresholds would need rescaling); the within-question AUC, which does not depend on scale, is the comparison used. Practical consequence: SigLIP2 is about 20 times cheaper to run, so it is a better vehicle for further matcher experiments than Ops-MM.
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
- 2026-11-23: D22 (fix the base before routing; staged plan) added; cue-source tooling built.
- 2026-11-22: Section 0 (overall status, bottlenecks, what to test and what not to do) added.
- 2026-11-22: Row 2 val results added to 9a: cues do not help the decider overall and hurt it on tutorials; row 4 redone with cues is worse.
- 2026-11-21: D21 (row 2 design) added; row 2 notebook, cue-aware decider cache, interval tooling built.
- 2026-11-21: Row 4 val results added to 9a: the decider did not improve on the matcher.
- 2026-11-16: Row 3 val results added to 9a; D20 (row 4 design).
- 2026-11-15: D17-D19 added (cue matcher design: embedding path, cue source, trigger rules); stages 0 and 1a built.
- 2026-11-15: Section 9a added: first row 1 results on val, with the cooldown and timer comparison.
- 2026-11-15: Added timer and cooldown-matched floors to section 5 after the first val run showed a cooldown alone adds about ten F1.
- 2026-11-15: D16 added and then rewritten with the options table and GPU measurements: the decider is Qwen3-VL-2B because the 3B spilled GPU memory; other models deferred.
- 2026-11-15: D15 (row 1 decider input and plumbing check) added; row 1 runner, replay and notebook added.
- 2026-11-15: D12 (replay is an evaluation harness only), D13 (on-demand decider calls at trigger times) and D14 (how each component is graded) added.
- 2026-11-15: Added `scripts/ovo_tar_remote.py` and the committed video index; the video download
  drops from about 44 GB to 5 GB (D11).
- 2026-11-15: First version. Ingestion, seeded splits, Em-Garde-faithful metric, floors, and a
  scoring CLI added. Tolerance moved from 4 s one-sided to Â±2 s. Default dtype set to BF16.
