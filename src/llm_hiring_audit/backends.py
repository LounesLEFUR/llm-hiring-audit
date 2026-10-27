"""Language-model backends.

Every backend exposes the same tiny interface, so the audit logic never
depends on a specific provider. `MockBackend` returns scores with a bias
we plant ourselves: it lets us check that the analysis recovers a known
effect before we trust it on a real model.
"""

import hashlib
import os
from collections.abc import Mapping
from typing import Any, Protocol

import httpx
import numpy as np

from llm_hiring_audit.prompts import SYSTEM_PROMPT


class LLMBackend(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, prompt: str, *, temperature: float, seed: int) -> str: ...


def _messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


class OllamaBackend:
    """A model served locally by Ollama (https://ollama.com)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=300)

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def complete(self, prompt: str, *, temperature: float, seed: int) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": _messages(prompt),
                "stream": False,
                "options": {"temperature": temperature, "seed": seed},
            },
        )
        response.raise_for_status()
        content: str = response.json()["message"]["content"]
        return content


class MistralBackend:
    """A model from the Mistral API. Reads the key from MISTRAL_API_KEY."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError("Set the MISTRAL_API_KEY environment variable.")
        self.model = model
        self._client = client or httpx.Client(
            base_url="https://api.mistral.ai",
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )

    @property
    def name(self) -> str:
        return f"mistral/{self.model}"

    def complete(self, prompt: str, *, temperature: float, seed: int) -> str:
        response = self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": _messages(prompt),
                "temperature": temperature,
                "random_seed": seed,
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        content: str = data["choices"][0]["message"]["content"]
        return content


class MockBackend:
    """Fake model whose score depends on planted penalties.

    The score is `base + sum of penalties for words found in the prompt +
    noise`. With `penalties={"Sevran": -5}`, every CV with a Sevran address
    loses 5 points on average.
    """

    def __init__(
        self,
        penalties: Mapping[str, float] | None = None,
        base: float = 70.0,
        noise_sd: float = 5.0,
    ) -> None:
        self.penalties = dict(penalties or {})
        self.base = base
        self.noise_sd = noise_sd

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, prompt: str, *, temperature: float, seed: int) -> str:
        # Seed the noise from the prompt itself so results are reproducible.
        digest = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        score = self.base + rng.normal(0, self.noise_sd)
        score += sum(v for word, v in self.penalties.items() if word in prompt)
        return str(int(np.clip(round(score), 1, 100)))
