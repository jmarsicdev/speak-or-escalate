"""Cloud model client (GPT) with call counting so every experiment reports its cloud budget.

Set OPENAI_API_KEY in the environment. `MockCloud` returns canned answers and still
counts calls, so cost curves can be produced in dry runs.
"""
from __future__ import annotations

import base64
import io
import os


class CloudClient:
    def __init__(self):
        self.n_calls = 0
        self.n_images = 0

    def complete(self, prompt: str, images=None, max_tokens: int = 200) -> str:
        self.n_calls += 1
        self.n_images += len(images or [])
        return self._complete(prompt, images, max_tokens)

    def _complete(self, prompt, images, max_tokens) -> str:
        raise NotImplementedError


class MockCloud(CloudClient):
    def _complete(self, prompt, images, max_tokens):
        if "cues" in prompt.lower():
            return '{"cues": ["bubbles rising in pot", "steam above pot", "lid rattling"]}'
        return "cloud answer"


class OpenAICloud(CloudClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__()
        from openai import OpenAI  # noqa: PLC0415

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    @staticmethod
    def _b64(frame) -> str:
        from PIL import Image  # noqa: PLC0415

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _complete(self, prompt, images, max_tokens):
        content = [{"type": "text", "text": prompt}]
        for f in images or []:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._b64(f)}"}})
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()


def build_cloud(cfg: dict) -> CloudClient:
    backend = cfg.get("backend", "mock")
    if backend == "mock":
        return MockCloud()
    if backend == "openai":
        return OpenAICloud(cfg.get("model", "gpt-4o-mini"))
    raise ValueError(f"unknown cloud backend {backend}")
