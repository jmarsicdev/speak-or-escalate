# Running the decider with a larger model (for someone who is not on the project)

**Goal.** Row 1 of the experiment ladder with a bigger model than our 2B one: at every 2 s the model sees the last 4 frames and the user's question and
says Yes or No to "respond now?"; we read its probability of "Yes". We then ask: does a bigger model rank the right moments higher than the 2B model does?
Nothing here changes the model; it is only run. (Background: `docs/RATIONALE.md` sections 0 and 8, decisions D9, D13, D16.)

## What you need
- The repository on branch `abhi`, a Python 3.10+ environment, and the val data and videos (ask Abhi for `data/processed/ovo_far_val.jsonl` and `data/raw/ovo/src_videos/`; a few GB).
- `pip install -e ".[api]"` for the API route, or `pip install -e ".[models]"` for the local-GPU route.
- Do **not** touch the test split (`ovo_far_test.jsonl`). Everything below uses val only.

## Route A: through an API (no GPU, no download). OpenRouter by default.
1. Get an API key and put it in an environment variable. Never paste it into a file or commit it.
   - Windows PowerShell: `$env:OPENROUTER_API_KEY = "..."`   Linux/macOS: `export OPENROUTER_API_KEY=...`
2. Count the calls first (free): `python scripts/run_reference_model.py --estimate --limit-questions 6`
   (6 questions = 111 decisions = 444 images; all of val = 1,983 decisions = 7,932 images). Multiply the images by the provider's price per image.
3. Small run: `python scripts/run_reference_model.py --model qwen/qwen3-vl-8b-instruct --limit-questions 6`
   It first checks the model returns Yes/No probabilities, then runs, then prints the comparison with the 2B model on the same questions.
4. Only if that looked sensible: `--limit-questions 0` for all 43 val questions (resumable, run the same command again after a break).

**The one requirement: log-probabilities.** The decider reads P("Yes") against P("No") from the model's next-token probabilities, so the
endpoint must return `logprobs`. Many models on OpenRouter do not, or only for some providers; the check in step 3 tells you immediately
("returned no log-probabilities") and no money is spent beyond one call. If a model fails, try the same model through another provider,
or another model. Models that "think" before answering are not suited to this.

Other endpoints work the same way: `--base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model gpt-4o-mini` (OpenAI returns top log-probs),
or a vLLM server you run yourself. **Privacy:** the video frames are sent to the provider. They are from public benchmark datasets, but check that
your provider's terms are acceptable.

## Route B: a bigger GPU with the weights
`python scripts/run_reference_model.py --backend hf --model Qwen/Qwen3-VL-8B-Instruct --limit-questions 6`. About 17 GB of GPU memory in bf16 for the 8B model; our
laptop (8.6 GB) cannot do this. No quantization for now: it is a later, separate step.

## What to send back
Commit nothing yourself. Send Abhi the printed comparison (the table from `compare_deciders.py`) and the folder `results/cache/decider/<model name>__query_only__<key>/`
(the stored scores). How to read it: the AUC is within each question (0.5 = no skill; the 2B model is about 0.6); the "B - A" interval resamples whole videos;
**an improvement only counts if the interval excludes zero**, and with 6 questions it will rarely be narrow. A small run tells you whether a larger run is worth it, not more.

## Cost and time
A call is 4 JPEG frames (at most 448 px on the long side) and one output token. Time is mostly network latency (about 1 to 3 s per call), so the 111-decision trial takes a few minutes.
