# Architecture report: what the system is now, what works, what does not (2026-11-26)

A consolidated report of the experiments in `docs/RATIONALE.md` (decisions D1 to D25, results in section 9a). Every number below comes from a stored run; the script and output file are named in the rationale section cited in brackets. Data: OVO-Bench forward tasks, validation split (43 questions: 24 counting, 12 tutorials, 7 clue reveal; the test split is untouched), and ProactiveVideoQA EGO (a seeded random 60 of 326 egocentric questions). Intervals are 95%, from resampling whole videos. "Timer" is a rule that ignores the video and responds at regular intervals, matched to the model's number of responses (per question unless stated).

## 1. Architecture

### 1.1 As planned (D1 to D22)

| Component | Role |
|---|---|
| Cue writer | turns the question into short descriptions of what the camera would see (Em-Garde) |
| Cue matcher | embeds each 2 s window and fires on a rise in similarity to a cue (Em-Garde) |
| Memory | scene-graph triplets every 5 s, retrieved for the decider (Response-G1) |
| Decider | small VLM, "is now the moment? Yes/No", p(Yes) from log-probabilities |
| Router | uncertain moments to a cloud model; high-confidence moments answered on the edge |

### 1.2 As supported by the evidence now

```
video, last 2 s (4 frames at 2 fps, 448 px), every 2 s
        |
        v
Edge decider: Qwen3-VL-8B (through an API as a stand-in for an on-device model)
  question: "is now the moment?" or a change question
  ("did something change that matters for the question?" with two earlier frames;
   "did something happen between the clip from about 5 s ago and the last 2 s that answers the question?")
        |  p(Yes)
        v
Response policy: a threshold set per model (p(Yes) >= tau on a fixed share of its own moments)
  one-moment questions: respond the first time p(Yes) >= tau, then stop
  ongoing-activity questions: respond on the upward crossing, and again every 6 s while p(Yes) stays >= tau (exploratory, section 3.3)
        |
        v
Router (validated in simulation, deferred): the edge model's uncertain moments to a stronger cloud model
```

Removed or demoted, with the evidence in section 4: the embedding matcher and cue writer, scene-graph memory, any evidence written into the decider's prompt, separate cue-presence checks, goal tracking (fixed and rolling), change-point accumulation, weighted combinations of signals.

## 2. Evaluation conventions that turned out to be necessary

| Convention | Reason (evidence) |
|---|---|
| Compare every model with a timer that makes the same number of responses | the metric is generous: a coin flip scores about 21 F1 and a timer about 27 on OVO val; a cooldown alone adds about 10 F1 (section 5) |
| Per-question timer, not one matched on the total | a model could gain 3 to 10 points just by putting more responses into questions with more events (9a, rescoring) |
| Within-question AUC, never pooled | the 2B decider's AUC is 0.71 pooled but 0.61 within questions (9a, row 3 correction) |
| A threshold set per model (share of its own moments) | models say Yes on 6% (8B) to 80% (2B) of moments, so one fixed threshold is unfair (9a, rescoring) |
| Recall compared at a fixed false-alarm rate | recall alone rewards responding more (9a, recall at equal false-alarm rates) |
| Settings written down before results | D24 and D25 were committed before their runs; post hoc choices are labelled as such |
| Clue reveal reported apart | it needs plot understanding; every model and signal scores 0.40 to 0.63 AUC there |

## 3. What works

### 3.1 The edge decider has real signal: modest on exact moments, strong on egocentric activity

| Setting | Result | Rationale section |
|---|---|---|
| OVO counting + tutorials (29 questions), per-model threshold, responses on upward crossings, repeats not punished; F1 gap over the per-question timer | 2B **+10.1 to +17.0 at every threshold share** (all intervals exclude zero); 8B +19.6 and +15.1 at the 20% and 30% shares (not significant below); 235B up to **+26.5 [+16.7, +36.6]** | 9a, rescoring intervals |
| OVO counting + tutorials (36), recall at <= 1 / <= 2 false alarms per minute | 8B **34.7 / 53.5%**, 2B 40.3 / 45.1%, timer 6.2 / 20.1% | 9a, recall at equal false-alarm rates |
| ProactiveVideoQA EGO (60), 8B decider | within-question AUC **0.743**; interval F1 gap over the per-question timer **+17.8 [+10.8, +24.9] / +19.4 [+13.4, +25.6] / +12.4 [+6.5, +18.5]** at the 5 / 15 / 30% shares | 9a, pipelines |
| ProactiveVideoQA WEB (150 short clips) | AUC 0.816; gap +1.5 / +2.5 / +6.6 (two of three intervals exclude zero) | 9a, pipelines |

### 3.2 Change questions beat "is now the moment?" on state changes

| Signal | Result | Rationale section |
|---|---|---|
| Before/after (clip from about 5 s ago against the last 2 s) on OVO | F1 gap over the timer at 15% **+19.2** against the decider's +8.9: **+10.3 [+2.1, +18.0]**; tutorials AUC **0.746 vs 0.655 (+0.091)**; tutorial recall at <= 2 false alarms per minute **51.2% vs 39.0%** | 9a, D25 C |
| Graded change gate on OVO | tutorials AUC **0.744 vs 0.659 (+0.085 [+0.021, +0.141])**; overall 0.630 vs 0.618 (+0.011, not significant) | 9a, D24 Stages 1 to 3 |
| Question-aware change sentence | moments it flags as a relevant change are near an event **24.0% vs 8.3%**; used as a separate gate it beats the same sentence placed in the prompt by **+0.051 AUC [+0.011, +0.098]** | 9a, decider history; change gate |
| On EGO | 8B decider 0.743, change gate **0.752**, before/after 0.749; paired differences from the decider +0.009 [-0.012, +0.030] and +0.005 [-0.013, +0.023]; F1 gaps at 15%: decider +19.4, change gate **+23.0**, before/after +21.1 | 9a, pipelines |

On egocentric activity questions the three questions rank moments about equally; on OVO tutorials the change questions are clearly better.

### 3.3 On EGO, recall was limited by the response rule, not by the model

**What is provided to the model.** Every 2 s of video (a "tick"), the model gets the question and frames ending at that tick: the last 4 frames (2 s) for the decider; for the change gate, two earlier frames (about 10 s and 5 s back) plus the last 4; for before/after, 4 frames from about 6.5 to 5 s back plus the last 4. It returns one number, p(Yes), from its log-probabilities for "Yes" and "No".

**What is set up.** A threshold tau per model, chosen so that p(Yes) >= tau on a fixed share of that model's own moments. Two response rules turn the p(Yes) series into response times:
- *crossing* (used everywhere before): respond at a tick where p(Yes) >= tau and the previous tick was below tau; nothing more until p(Yes) drops below tau and rises again;
- *crossing, then every 6 s while above*: respond at the upward crossing, and then again at any later tick that is still >= tau and at least 6 s after the previous response (every third tick at the 2 s cadence); when p(Yes) falls below tau the rule resets. A 10 s variant was tested alongside, and the pre-registered re-arm rule (respond again after the log-odds fall 1.0 below the stretch's peak and recover 1.0).

**What is measured.** EGO questions have about 5 reply intervals each (median 34% of the clip covered). An interval is caught if any response falls inside it; a false alarm is a response outside every interval. For each model and rule, tau is swept over 70 values and the highest share of intervals caught with at most 1, 2 or 3 false alarms per minute is reported, against a timer swept the same way.

**What happens.** With the crossing rule the decider stays above tau through long stretches of activity, so one response covers a stretch that contains several reply intervals, and recall stops rising (about 60%) however many false alarms are allowed:

| Recall of reply intervals at <= 1 / <= 2 / <= 3 false alarms per minute | Crossing | Re-arm | Every 6 s while above | Every 10 s while above |
|---|---|---|---|---|
| Timer | 49.3 / 71.8 / 80.9% | | | |
| 8B decider | 58.1 / 59.7 / 59.7% | 63.8 / 78.5 / 84.2% | 57.4 / **83.6** / 86.6% | 60.4 / 79.9 / 86.2% |
| Before/after | 58.7 / 58.7 / 58.7% | 71.5 / 81.2 / 82.6% | 70.5 / **85.2** / 92.3% | 70.8 / 83.9 / 87.6% |
| Change gate | 63.1 / 63.1 / 63.1% | 72.1 / 79.2 / 79.2% | 73.5 / **88.3** / 92.3% | 74.2 / 85.9 / 88.3% |

Because the comparison is at a fixed false-alarm rate, the gain is not just from responding more: at 2 false alarms per minute the 8B with the 6 s rule catches 84% of intervals against the timer's 72%. The rule is meant for questions about an ongoing activity; on OVO's single-moment events re-firing did not help (re-arm: 0 to -3.5 F1).

**Status.** The re-arm rule was fixed in advance (D24 (v)). The 6 s and 10 s rules were chosen after the 60% ceiling was seen, so they are exploratory until confirmed on questions they were not chosen on. The figures have no intervals yet, and the precision and F1 of the 6 s rule have not been reported.

### 3.4 The edge model's uncertainty is a useful basis for escalation (simulation)

The 8B's and 235B's stored p(Yes) on the same moments (29 counting and tutorial questions), each as its percentile among its own moments; a budget of moments takes the 235B's value; responses at the 15% share; F1 gap over the per-question timer (9a, D25 A):

| Moments routed | Random | Routed where the 8B is nearest its threshold | Difference |
|---|---|---|---|
| 0% | +8.9 | | |
| 10% | +8.3 | +15.9 | +7.6 [-0.6, +16.1] |
| 20% | +8.8 | +14.8 | +6.0 [-0.3, +14.1] |
| 30% | +9.8 | +17.1 | **+7.3 [+0.5, +15.0]** |
| 50% | +11.8 | **+21.1** | **+9.3 [+3.0, +15.8]** |
| 100% | +20.4 | | |

Routing half the moments matches routing all of them. Routing the moments the change gate flags while the decider does not is no better than random (-3.2 to 0.0). The 235B was used because it was stored (it was run as the model-size ceiling); a deployed cloud model would be stronger. Further routing work is deferred until the edge model finds more events (user's decision).

## 4. What does not work

### 4.1 The embedding matcher

| Test | Result | Section |
|---|---|---|
| Ops-MM matcher alone (Em-Garde's setup) | within-question AUC 0.61 to 0.62; the largest one-step rise near an event has median 0.0071 against 0.0053 elsewhere, below the 0.02 to 0.04 trigger thresholds | row 3; D22 |
| Benchmark annotations as cues (oracle) | no lift at the best threshold; mean over thresholds +3.6 F1 [-2.6, +10.1]; helps only at loose thresholds | step A; paired bootstrap |
| SigLIP2 instead of Ops-MM | -0.009 [-0.071, +0.049], 20 times cheaper | SigLIP2 |
| X-CLIP (video-native) | **0.526 vs 0.624 (-0.099 [-0.186, -0.015])**; the same with plain text embeddings (0.528) | X-CLIP |
| Window length | 1, 2 or 4 frames equal (0.588 / 0.594 / 0.593); averaging over 2 to 8 s worse (-0.033 to -0.109) | shorter windows; several-window test |
| Scene-level cues | mean similarity 0.109 near events, 0.098 elsewhere, against 0.107 / 0.096 for oracle cues: higher everywhere, no better separation | shorter windows and scene-level cues |
| Wider "around now" target | AUC flat or falling from +/- 2 s to +/- 15 s | label radius |
| Matcher gated by the decider | the best cell is the matcher alone; the decider moves F1 by 1 to 2 points | row 4 |
| 8B-written cues / frame-grounded cues | -0.032 (n.s.) / **-0.097 [-0.165, -0.029]** against the 2B's cues | D24 Stage 0 |

### 4.2 Evidence written into the decider's prompt

| Added text | Result | Section |
|---|---|---|
| Cues (2B) | AUC +0.003 [-0.038, +0.041]; tutorials **-0.080 [-0.162, -0.016]** | row 2 |
| Scene-graph triplets (2B, 14 questions) | **-0.120 [-0.240, -0.017]**; Yes-rate 0.78 to 0.89 | row 5 |
| Neutral "what changed" sentence (8B) | **-0.067 [-0.115, -0.018]**; Yes-rate 6% to 38%; the sentence reports a change 97.5% of the time | decider history |
| Question-aware "what changed" sentence (8B) | the decider follows it (mean p(Yes) 0.00 after "nothing relevant changed", 0.72 otherwise); AUC 0.581 vs 0.618 | decider history |
| Recall with these additions | 18.8 to 36.8% at <= 2 false alarms per minute, against 53.5% for the plain 8B | recall summary |

### 4.3 Model size alone

| Comparison | Result | Section |
|---|---|---|
| 8B vs 2B, same setup | +0.004 [-0.037, +0.045] | larger deciders |
| 235B vs 2B (36 questions) | +0.069 [+0.021, +0.119] | larger deciders |
| Recall at <= 1 / <= 2 false alarms per minute | 235B 37.7 / 49.1% (29 questions) against 8B 34.7 / 53.5% | recall summary |

### 4.4 Other signals and combinations

| Approach | Result | Section |
|---|---|---|
| Cue presence (8B checks each cue separately) | AUC 0.624 vs 0.631; learned weight -0.03 | D24 |
| Log-odds vote, gate nudge | AUC 0.638 to 0.670; scored as responses no reliable gain (best +7.0 [-0.9, +13.6] at 15%; several significantly worse at 30%) | D24 |
| "Evidence lowers the bar", as pre-registered | invalid: alpha was set in probability units without knowing the 8B's threshold (0.037), so the bar fell below zero; corrected post hoc version equals the decider at the same number of responses (+19.7 vs +19.7) | D24 |
| Two thresholds | 30.6% of events at 0.98 false alarms per minute, against 34.7% at 1.39 for the decider; F1 -0.1 | D24 |
| Learned weights, leave-one-video-out | held-out AUC +0.002 [-0.034, +0.034]; F1 under the same cross-validation +4.2 [-0.4, +8.8]; weights decider +0.84, change gate +0.17, similarity +0.16, cue presence -0.03 | D24 |
| Fixed goal state (C1) | AUC 0.643; its rise worse on tutorials (-0.092) | D25 C |
| Anticipation (C3) | tutorials +0.027; no rescue of missed events | D25 C |
| Rolling goal (C4) | **0.514 vs 0.631 (-0.117 [-0.196, -0.038])**; rewritten 16 times in 100 s in one tutorial; the check says yes on 6% of ticks | D25 C |
| Union rule (decider or signal) | never catches more events than the decider at the same number of responses | D25 C |
| CUSUM change-point accumulation | on the decider alone -9.4 and -12.2 at the 15% and 30% shares | D25 B |
| Re-arm on OVO | 0 to -3.5 F1 | rescoring additions |
| History frames | tutorials **+0.062 [+0.013, +0.112]**, sports counting **-0.106 [-0.164, -0.054]** | decider history |
| One frame instead of 2 s | **-0.056 [-0.105, -0.007]** | shorter windows |

## 5. Benchmarks

| Benchmark | Status |
|---|---|
| OVO-Bench forward tasks | main development set; strict +/- 2 s single moments; clue reveal reported apart |
| StreamingBench Proactive Output | weak test of timing: answers come 4 to 24 s after the question, and "answer 6 s after the question" scores **54.9%** with no video, against the 8B's 45.1% and the paper's GPT-4o 56.9%. The Hugging Face question numbers are not the video folders; the benchmark's own `questions_proactive_50.json` maps them |
| ProactiveVideoQA EGO / WEB | egocentric and web clips with several reply intervals per question; scored on timing only (its LLM judge grades only answer text) |
| ESTP-Bench | suitable labels (about 4 valid intervals per question, egocentric) but not released |

## 6. Not yet established

| Item | What is missing | Why it matters | How to settle it | Cost |
|---|---|---|---|---|
| 1. Confirmation on fresh questions | EGO results come from one seeded sample of 60 questions; the 6 s rule was chosen after seeing that sample; all OVO work is on val | the main recall result and the change-question results could partly fit this sample | a new seeded sample of ProactiveVideoQA EGO questions not in the 60, with the rules and their settings fixed in writing first (8B decider, change gate, before/after; crossing, re-arm, every 6 s); later OVO test once | about 10,000 API calls per signal |
| 2. Intervals on recall | the recall-at-fixed-false-alarm figures (OVO and EGO) are point estimates | differences of a few points (e.g. 84% vs 88%) may be noise | bootstrap over videos, with the threshold chosen inside each resample | CPU, minutes |
| 3. Precision and F1 of the 6 s rule | only recall at fixed false-alarm rates is reported for it | a rule can raise recall while the answers it adds are redundant | the interval F1 over the per-question timer, as for the other rules | CPU |
| 4. The full chain | the router was simulated with stored scores and a 235B stand-in; no answer text was generated or graded | the project's claim is about quality and cost of the whole system | edge decider finds moments, uncertain ones go to a stronger cloud model, answers are graded (ProactiveVideoQA's judge for answer text) | API calls and judge cost; deferred |
| 5. On-device reality | the 8B ran through an API; a quantized 8B on the laptop GPU, latency per decision, and the live streaming loop have not been tested | the edge model must fit and run in time | quantized 8B load and timing; the live-versus-replay check (D12) | GPU time |
| 6. Why EGO is easier than OVO | EGO has forgiving targets (about 5 intervals per question, a third of the clip, repeats not punished) and activity content; OVO has single +/- 2 s moments, counting and clue reveal | to know whether the edge model detects moments or only activity | score OVO with interval-style targets, or EGO with point targets around interval starts | CPU |
| 7. Calibration | p(Yes) is used as a ranking; temperature scaling was never applied | routing by confidence assumes the scale means something | fit temperature on dev, check reliability on val | CPU |
| 8. Selection on val | thresholds, shares and rules were looked at on the same 43 questions; many comparisons were made | some significant cells will be chance | the fresh-sample confirmation (item 1) and one test run | as item 1 |
