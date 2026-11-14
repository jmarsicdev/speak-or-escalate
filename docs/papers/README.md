# Reference papers

All downloaded from arxiv.org. Titles were checked against each PDF's first page.

| File | Paper | What we take from it |
|---|---|---|
| `em-garde_2603.19054.pdf` | Em-Garde: A Propose-Match Framework for Proactive Streaming Video Understanding (preprint) | Cue parsing once per query, embedding matcher on 2 s windows, surge-style trigger, online precision/recall metric on OVO-Bench, 52.3 PAUC target on ProactiveVideoQA EGO |
| `response-g1_2605.07575.pdf` | Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding (preprint) | Scene-graph triplet memory, triplet-form query retrieval, top-K, Yes/No response trigger |
| `videojev_2610.18615.pdf` | VideoJev: System One Model for Video Understanding and Interaction (preprint) | Score-readout idea for the speak decision (Gate 5) |
| `proactivevideoqa_2507.09313.pdf` | ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video LLMs (preprint) | PAUC metric, GPT-4.1 judge, EGO split |
| `ovo-bench_2501.05510.pdf` | OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? (CVPR 2025) | Forward Active Responding tasks for the first evaluation |
| `videollm-online_2406.11816.pdf` | VideoLLM-online: Online Video LLM for Streaming Video (CVPR 2024) | Speak-or-silent decision per frame |
| `guo-calibration_1706.04599.pdf` | On Calibration of Modern Neural Networks, Guo et al. (ICML 2017) | Temperature scaling for p(Yes) |
| `qwen3-vl_2511.21631.pdf` | Qwen3-VL Technical Report (preprint) | Candidate edge decider |

Still unverified: whether Em-Garde / ProactiveVideoQA code and weights are released, and the exact HF id of the embedding model.
