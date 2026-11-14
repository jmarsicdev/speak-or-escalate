"""Edge VLM wrappers (Gemma / Qwen multimodal via transformers) plus a mock.

Two calls the pipeline makes:

  complete(prompt, images)        -> free text  (cue parsing, triplets, answers)
  speak_now(query, images, evidence) -> EdgeOutput with p_speak

`p_speak` is read from the next-token logits over {"Yes", "No"} after a
"Should I respond now?" prompt, NOT from parsing generated text. That gives a
calibratable number we can plot (project doc: "is the model's confidence accurate?",
"classification f1 recall heatmap").

Quantization: pass `quant="int4"` / `"int8"` to load with bitsandbytes. Note from the
ISI paper: Gemma-3n cannot be INT8-quantized (AltUp); use Gemma 3 or Gemma 4 E2B / Qwen2.5-VL.
"""
from __future__ import annotations

import math

import numpy as np

from .types import EdgeOutput

SPEAK_PROMPT = """You are an assistant watching a first-person video stream.
The user asked earlier: "{query}"
{evidence_block}
Based on the most recent frames, is RIGHT NOW the moment to respond to the user?
Answer with a single word: Yes or No."""

ANSWER_PROMPT = """You are an assistant watching a first-person video stream.
The user asked earlier: "{query}"
{evidence_block}
It is now time to respond. Give a concise, specific answer based on the recent frames."""


def _evidence_block(evidence: list[str] | None) -> str:
    if not evidence:
        return ""
    return "Recent scene evidence:\n" + "\n".join(f"- {e}" for e in evidence) + "\n"


class EdgeVLM:
    """Common interface. Subclasses implement `_generate` and `_yes_no_logprobs`."""

    name = "base"

    def complete(self, prompt: str, images=None, max_new_tokens: int = 128) -> str:
        return self._generate(prompt, images, max_new_tokens)

    def speak_now(self, query: str, images, evidence: list[str] | None = None) -> EdgeOutput:
        prompt = SPEAK_PROMPT.format(query=query, evidence_block=_evidence_block(evidence))
        lp_yes, lp_no = self._yes_no_logprobs(prompt, images)
        p = 1.0 / (1.0 + math.exp(lp_no - lp_yes))  # softmax over the two options
        return EdgeOutput(text="Yes" if p >= 0.5 else "No", p_speak=p, raw={"lp_yes": lp_yes, "lp_no": lp_no})

    def answer(self, query: str, images, evidence: list[str] | None = None) -> str:
        prompt = ANSWER_PROMPT.format(query=query, evidence_block=_evidence_block(evidence))
        return self._generate(prompt, images, 96)

    # ---- to implement
    def _generate(self, prompt, images, max_new_tokens) -> str:
        raise NotImplementedError

    def _yes_no_logprobs(self, prompt, images) -> tuple[float, float]:
        raise NotImplementedError


class MockEdgeVLM(EdgeVLM):
    """No-GPU stand-in. p_speak is driven by an optional oracle so routing logic can be tested."""

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
    """transformers-based wrapper. Works for Qwen2.5-VL and Gemma 3 style chat VLMs.

    TODO(models): the chat-template / image-placeholder details differ per family.
    Validate on the dev GPU with one real model before trusting numbers.
    """

    name = "hf"

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16", quant: str | None = None):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        self.device = device
        kwargs = {"torch_dtype": getattr(torch, dtype)}
        if quant in ("int4", "int8"):
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quant == "int4"), load_in_8bit=(quant == "int8")
            )
            kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
        if "device_map" not in kwargs:
            self.model.to(device)
        tok = self.processor.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]

    def _inputs(self, prompt, images):
        from PIL import Image  # noqa: PLC0415

        pil = [Image.fromarray(f) for f in (images or [])]
        content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.processor(text=[text], images=pil or None, return_tensors="pt").to(self.model.device)

    def _generate(self, prompt, images, max_new_tokens):
        inputs = self._inputs(prompt, images)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()

    def _yes_no_logprobs(self, prompt, images):
        inputs = self._inputs(prompt, images)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        lp = self.torch.log_softmax(logits.float(), dim=-1)
        return float(lp[self.yes_id]), float(lp[self.no_id])


def build_edge_model(cfg: dict) -> EdgeVLM:
    backend = cfg.get("backend", "mock")
    if backend == "mock":
        return MockEdgeVLM()
    if backend == "hf":
        return HFEdgeVLM(cfg["model_id"], cfg.get("device", "cuda"), cfg.get("dtype", "float16"), cfg.get("quant"))
    raise ValueError(f"unknown edge backend {backend}")
