from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib import request


@dataclass(frozen=True)
class ModelResponse:
    text: str
    latency_seconds: float
    raw: dict


class TextBackend(Protocol):
    name: str
    model_id: str

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> ModelResponse: ...


class MockTextBackend:
    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        responder: Callable[[str, str | None, float, int], str],
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._responder = responder

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> ModelResponse:
        started = time.perf_counter()
        text = self._responder(prompt, system_prompt, temperature, max_new_tokens)
        return ModelResponse(
            text=text,
            latency_seconds=round(time.perf_counter() - started, 6),
            raw={"mock": True},
        )


class RemoteHTTPTextBackend:
    def __init__(self, *, name: str, model_id: str, url: str, timeout_seconds: float = 90.0):
        self.name = name
        self.model_id = model_id
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> ModelResponse:
        payload = json.dumps(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        ).encode("utf-8")
        started = time.perf_counter()
        req = request.Request(
            self.url + "/generate",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return ModelResponse(
            text=body["text"],
            latency_seconds=round(time.perf_counter() - started, 6),
            raw=body,
        )

    def health(self) -> dict:
        with request.urlopen(self.url + "/health", timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
