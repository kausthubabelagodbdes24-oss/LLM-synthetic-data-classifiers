"""Talk to the open LLM.

Backends
- ollama         local Ollama server (default): `ollama pull qwen3:8b`, then `ollama serve`
- openai_compat  any OpenAI-style /chat/completions server (vLLM on Kaggle/Colab, hosted open models)
- fake           deterministic stand-in for tests and dry runs; produces obviously fake text
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, cfg: dict[str, Any], model: str | None = None) -> None:
        self.model = model or cfg["model"]
        self.url = cfg.get("ollama_url", "http://localhost:11434").rstrip("/")
        self.options = {"temperature": cfg["temperature"], "top_p": cfg["top_p"]}
        self.think = cfg.get("think")
        self.timeout = cfg.get("timeout_s", 180)

    def complete(self, messages: list[dict[str, str]], schema: dict | None = None, meta: dict | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
        }
        if self.think is not None:
            payload["think"] = self.think
        if schema is not None:
            payload["format"] = schema  # Ollama structured output: forces this JSON shape
        try:
            resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}. Is `ollama serve` running?") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()["message"]["content"]

    def describe(self) -> str:
        """Model name plus digest, for the report."""
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=10).json().get("models", [])
            for m in tags:
                if self.model in (m.get("name"), m.get("model")) and m.get("digest"):
                    return f"{self.model} ({m['digest'][:12]})"
        except Exception:
            pass
        return self.model


class OpenAICompatClient:
    def __init__(self, cfg: dict[str, Any], model: str | None = None) -> None:
        base = cfg.get("openai_base_url")
        if not base:
            raise LLMError("Set openai_base_url in configs/generation.yaml for the openai_compat backend.")
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = model or cfg["model"]
        self.key = os.environ.get("OPENAI_API_KEY", "not-needed")
        self.temperature = cfg["temperature"]
        self.top_p = cfg["top_p"]
        self.timeout = cfg.get("timeout_s", 180)

    def complete(self, messages: list[dict[str, str]], schema: dict | None = None, meta: dict | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            self.url, json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.key}"},
        )
        if resp.status_code != 200:
            raise LLMError(f"Server returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]

    def describe(self) -> str:
        return self.model


class FakeClient:
    """Returns obviously fake but contract-valid examples. Never use for results."""

    model = "fake-llm"

    def __init__(self, seed: int = 0, fail_every: int = 0) -> None:
        self.rng = random.Random(seed)
        self.calls = 0
        self.fail_every = fail_every

    def complete(self, messages: list[dict[str, str]], schema: dict | None = None, meta: dict | None = None) -> str:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            return "Sorry, I can't help with that."  # exercises the retry path
        meta = meta or {}
        n = int(meta.get("n", 3))
        label = meta.get("label", "label")
        if meta.get("source_text"):
            base = meta["source_text"]
            outs = [f"put another way {i + 1}, {base}" for i in range(n)]
        else:
            words = ["river", "budget", "violin", "harbour", "planet", "recipe", "garden", "engine",
                     "teacher", "festival", "winter", "museum", "camera", "station", "letter"]
            outs = [
                f"fake {label} example about the {self.rng.choice(words)} and the "
                f"{self.rng.choice(words)} number {self.rng.randint(1, 10_000)}"
                for _ in range(n)
            ]
        time.sleep(0)  # keep the call shape; real clients block on the network
        return json.dumps({"examples": outs})

    def describe(self) -> str:
        return self.model


def make_client(cfg: dict[str, Any], dry_run: bool = False, model: str | None = None):
    if dry_run or cfg.get("backend") == "fake":
        return FakeClient()
    if cfg.get("backend", "ollama") == "ollama":
        return OllamaClient(cfg, model)
    if cfg["backend"] == "openai_compat":
        return OpenAICompatClient(cfg, model)
    raise LLMError(f"Unknown backend '{cfg['backend']}'")
