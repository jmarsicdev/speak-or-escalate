"""Edge VLM wrapper (Gemma / Qwen multimodal via transformers) plus a mock.

Three calls the pipeline makes on the edge model:

  complete(prompt, images)             -> free text   (cue parsing, scene-graph triplets)
  speak_now(query, images, evidence, cues) -> EdgeOutput with p_speak
  answer(query, images, evidence, cues)    -> the on-device response text

`p_speak` is read from the next-token logits over {"Yes", "No"} after a
"Should I respond now?" prompt, not parsed from generated text, so it is a number the
router can threshold and we can check for calibration.

This repo is the desktop-GPU baseline. `quant` is passed through for the Jetson /
quantization work happening separately. (ISI SmallSat 2026: Gemma-3n cannot be INT8-quantized
because of AltUp; Gemma 3 and Qwen2.5-VL are fine.)
"""
from __future__ import annotations

import math

import numpy as np

from .types import EdgeOutput

SPEAK_PROMPT_VERSION = "speak-v1"


def build_speak_prompt(query: str, cues: list[str] | None = None, evidence: list[str] | None = None,
                       n_frames: int | None = None, window_s: float | None = None) -> str:
    """The "respond now?" question. Cues and evidence are included only when given, so the same builder
    serves row 1 (query only), row 2 (+ cues) and row 5 (+ scene-graph evidence). Changing any wording
    means a new SPEAK_PROMPT_VERSION, which makes the stored results use a new cache (see decider_cache.py)."""
    parts = ["You are an assistant watching a first-person video stream."]
    if n_frames:
        span = f", covering the last {window_s:g} seconds" if window_s else ""
        parts.append(f"You are shown the most recent {n_frames} frames{span}.")
    parts.append(f'The user asked earlier: "{query}"')
    if cues:
        parts.append("Visual cues that mean it is time to respond:\n" + "\n".join(f"- {c}" for c in cues))
    if evidence:
        parts.append("Recent scene evidence:\n" + "\n".join(f"- {e}" for e in evidence))
    parts.append("Respond only at the moment the answer becomes available: when what the user asked about "
                 "has just happened or has just become visible. Is RIGHT NOW such a moment?\n"
                 "Answer with a single word: Yes or No.")
    return "\n".join(parts)


def prompt_fingerprint() -> str:
    """Hash of the prompt wording, stored with every cache so a changed prompt can never reuse old answers."""
    import hashlib  # noqa: PLC0415

    text = SPEAK_PROMPT_VERSION + build_speak_prompt("{q}", ["{c}"], ["{e}"], 4, 2.0) + build_speak_prompt("{q}")
    return hashlib.sha1(text.encode()).hexdigest()[:10]


ANSWER_PROMPT = """You are an assistant watching a first-person video stream.
The user asked earlier: "{query}"
{evidence_block}
It is now time to respond. Give a concise, specific answer based on the recent frames."""


def _list_block(items: list[str] | None, header: str = "") -> str:
    if not items:
        return ""
    body = "\n".join(f"- {x}" for x in items)
    return f"{header}\n{body}\n" if header else body


class EdgeVLM:
    """Common interface. Subclasses implement `_generate` and `_yes_no_logprobs`."""

    name = "base"

    def complete(self, prompt: str, images=None, max_new_tokens: int = 128) -> str:
        return self._generate(prompt, images, max_new_tokens)

    def speak_now(self, query: str, images, evidence: list[str] | None = None, cues: list[str] | None = None,
                  window_s: float | None = None) -> EdgeOutput:
        prompt = build_speak_prompt(query, cues, evidence, n_frames=len(images or []) or None, window_s=window_s)
        lp_yes, lp_no = self._yes_no_logprobs(prompt, images)
        p = 1.0 / (1.0 + math.exp(lp_no - lp_yes))  # softmax over the two options
        return EdgeOutput(text="Yes" if p >= 0.5 else "No", p_speak=p, raw={"lp_yes": lp_yes, "lp_no": lp_no})

    def answer(self, query: str, images, evidence: list[str] | None = None) -> str:
        prompt = ANSWER_PROMPT.format(query=query, evidence_block=_list_block(evidence, "Recent scene evidence:"))
        return self._generate(prompt, images, 96)

    # ---- to implement
    def _generate(self, prompt, images, max_new_tokens) -> str:
        raise NotImplementedError

    def _yes_no_logprobs(self, prompt, images) -> tuple[float, float]:
        raise NotImplementedError


class MockEdgeVLM(EdgeVLM):
    """No-GPU stand-in. p_speak comes from an optional oracle so routing can be tested."""

    name = "mock"

    def __init__(self, p_speak_fn=None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.p_speak_fn = p_speak_fn  # callable(images) -> float, or None for random

    def _generate(self, prompt, images, max_new_tokens):
        if "triplet" in prompt.lower():
            return "(hand, holds, pot)\n(water, in, pot)\n(stove, under, pot)"
        if "cues" in prompt.lower():
            return '{"cues": ["bubbles rising in pot", "steam above pot", "lid rattling"]}'
        return "mock answer"

    def _yes_no_logprobs(self, prompt, images):
        p = self.p_speak_fn(images) if self.p_speak_fn else float(self.rng.uniform())
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p), math.log(1 - p)


class HFEdgeVLM(EdgeVLM):
    """transformers-based wrapper for Qwen2.5-VL and Gemma 3 style chat VLMs.

    TODO(models): chat-template / image-placeholder details differ per family.
    Validate on the dev GPU with one real model before trusting numbers.
    """

    name = "hf"

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16", quant: str | None = None,
                 max_pixels: int | None = None):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        import transformers  # noqa: PLC0415
        from packaging.version import Version  # noqa: PLC0415

        # newer transformers renamed torch_dtype -> dtype
        dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        kwargs = {dtype_key: getattr(torch, dtype)}
        if quant in ("int4", "int8"):
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=(quant == "int4"), load_in_8bit=(quant == "int8"))
            kwargs["device_map"] = "auto"
        proc_kwargs = {"max_pixels": max_pixels} if max_pixels else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
        if "device_map" not in kwargs:
            self.model.to(device)
        tok = self.processor.tokenizer
        yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError(f"'Yes'/'No' must each be one token for this tokenizer, got {yes} / {no}")
        self.yes_id, self.no_id = yes[0], no[0]

    def _inputs(self, prompt, images):
        from PIL import Image  # noqa: PLC0415

        pil = [Image.fromarray(f) for f in (images or [])]
        content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        return self.processor(text=[text], images=pil or None, return_tensors="pt").to(self.model.device)

    def _generate(self, prompt, images, max_new_tokens):
        inputs = self._inputs(prompt, images)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _yes_no_logprobs(self, prompt, images):
        inputs = self._inputs(prompt, images)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        lp = self.torch.log_softmax(logits.float(), dim=-1)  # over the whole vocabulary
        out = float(lp[self.yes_id]), float(lp[self.no_id])
        if str(self.model.device).startswith("cuda"):
            self.torch.cuda.synchronize()   # so wall-clock timing around this call is honest
        return out


def load_env_file(path=None):
    """Read KEY=VALUE lines from the repository's .env (gitignored) into the environment, without overriding variables already set.
    No dependency on python-dotenv; values are never printed."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


class APIEdgeVLM(EdgeVLM):
    """The edge model behind any OpenAI-compatible chat endpoint (OpenRouter, OpenAI, a vLLM server): no weights are downloaded.

    The decider needs the model's probability of "Yes" against "No" as the next token, so the endpoint must return log-probabilities
    (`logprobs` with `top_logprobs`). Not every model or provider on OpenRouter does; `check_logprobs()` tells you before a long run.
    Frames are sent as JPEG data URLs, so the video frames leave the machine; the key is read from an environment variable and is never stored.
    """

    name = "api"

    def __init__(self, model_id: str, base_url: str = "https://openrouter.ai/api/v1", api_key_env: str = "OPENROUTER_API_KEY",
                 jpeg_quality: int = 85, top_logprobs: int = 20, client=None):
        import os  # noqa: PLC0415

        if client is None:
            from openai import OpenAI  # noqa: PLC0415

            load_env_file()
            if api_key_env not in os.environ:
                raise SystemExit(f"set the environment variable {api_key_env} (an API key) before running with edge.backend=api")
            # max_retries: the SDK retries 429s and connection errors with exponential backoff; the default of 2 was not enough
            # for a 2,000-call run against a busy provider (rate-limit stop after 1,578 calls on the 235B model).
            client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env], max_retries=10)
        self.client, self.model, self.q, self.top = client, model_id, jpeg_quality, top_logprobs
        # OpenRouter may route a request to any provider of the model, including ones that silently drop `logprobs`;
        # require_parameters restricts it to providers that support every parameter we send. Other endpoints reject the field.
        self.extra = {"extra_body": {"provider": {"require_parameters": True}}} if "openrouter.ai" in base_url else {}
        self.n_calls = 0
        self.n_images = 0

    def _content(self, prompt, images):
        import base64  # noqa: PLC0415
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        parts = []
        for f in images or []:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=self.q)
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        return parts + [{"type": "text", "text": prompt}]

    def _call(self, prompt, images, **kw):
        self.n_calls += 1
        self.n_images += len(images or [])
        return self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": self._content(prompt, images)}],
                                                   temperature=0, **self.extra, **kw)

    def _call_ok(self, prompt, images, tries=5, **kw):
        """Retry when a provider returns a response without choices (seen once in a 2,000-call run), with a short growing wait."""
        import time  # noqa: PLC0415

        for k in range(tries):
            r = self._call(prompt, images, **kw)
            if getattr(r, "choices", None):
                return r
            time.sleep(2 * (k + 1))
        raise RuntimeError(f"{self.model}: {tries} empty responses in a row")

    def _generate(self, prompt, images, max_new_tokens):
        return (self._call_ok(prompt, images, max_tokens=max_new_tokens).choices[0].message.content or "").strip()

    def _yes_no_logprobs(self, prompt, images):
        r = self._call_ok(prompt, images, max_tokens=1, logprobs=True, top_logprobs=self.top)
        lp = getattr(r.choices[0], "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if not content:
            raise RuntimeError(f"{self.model} returned no log-probabilities: choose a model/provider that supports them (see docs/RUN_WITH_LARGER_MODEL.md)")
        return yes_no_from_top(content[0].top_logprobs)

    def check_logprobs(self):
        """Raise a clear error now, not 2,000 calls into a run, if this model does not return usable Yes/No probabilities."""
        import numpy as np  # noqa: PLC0415

        lp_yes, lp_no = self._yes_no_logprobs('Answer with a single word: Yes or No. Is the sky blue?', None)
        if max(lp_yes, lp_no) < -10:
            raise RuntimeError(f"{self.model}: neither Yes nor No is among its top tokens (best log-prob {max(lp_yes, lp_no):.1f}); the decider cannot use it")
        return lp_yes, lp_no


def yes_no_from_top(top) -> tuple[float, float]:
    """Log-probability mass on "Yes" and on "No" among the top candidates (case and spaces ignored). A side with no candidate gets -30."""
    import numpy as np  # noqa: PLC0415

    def mass(word):
        v = [math.exp(t.logprob) for t in top if t.token.strip().lower() == word]
        return float(np.log(sum(v))) if v else -30.0

    return mass("yes"), mass("no")


def build_edge_model(cfg: dict) -> EdgeVLM:
    backend = cfg.get("backend", "mock")
    if backend == "mock":
        return MockEdgeVLM()
    if backend == "api":
        return APIEdgeVLM(cfg["model_id"], cfg.get("base_url", "https://openrouter.ai/api/v1"), cfg.get("api_key_env", "OPENROUTER_API_KEY"))
    if backend == "hf":
        return HFEdgeVLM(cfg["model_id"], cfg.get("device", "cuda"), cfg.get("dtype", "bfloat16"), cfg.get("quant"),
                         cfg.get("max_pixels"))
    raise ValueError(f"unknown edge backend {backend}")
