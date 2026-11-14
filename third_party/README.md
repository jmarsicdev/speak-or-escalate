# Third-party code

## Vendored: `em_garde/` (MIT)
From https://github.com/air-embodied-brain/Em-Garde at commit `8130867` (2026-10-17).
- `ops_mm_embedding_v1.py`: Ops-MM-embedding wrapper (their file says it is adapted from the
  Ops-MM-Embedding HF repo; we have not checked that upstream license).
- `ops_mm_v1_2B.yaml`: confirms the HF id `OpenSearch-AI/Ops-MM-embedding-v1-2B`.
- `LICENSE`: Em-Garde's MIT license.

## Not vendored: ProactiveVideoQA PAUC scorer
https://github.com/yellow-binary-tree/ProactiveVideoQA has no LICENSE file, so copying its code
is unclear. Fetch it instead, pinned to commit `6fe9a33` (2026-01-08):

    bash scripts/fetch_proactivevideoqa.sh

This clones it into `third_party/ProactiveVideoQA/` (gitignored). The scorer is
`pauc/judge_llm_scores_openai.py` (judge `gpt-4.1` via the OpenAI Batch API, omega = 0.5).
