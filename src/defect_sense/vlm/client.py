"""Minimal Ollama chat client for vision-language models.

Talks to the local Ollama server over its REST API. The HTTP transport is
injectable so the adjudicator and pipeline can be tested without a running
server or a GPU.
"""
import base64
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

Transport = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:8b"


@dataclass(frozen=True)
class VLMResponse:
    text: str
    model: str
    latency_s: float


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 300.0,
        retries: int = 2,
        transport: Transport | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._transport = transport or self._http_transport

    def _http_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        prompt: str,
        images: list[bytes] = (),
        format_schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """Send one user message (optionally with images) and return the reply.

        `format_schema` is a JSON schema passed to Ollama's structured-output
        `format` field, constraining the model to emit valid JSON.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = [base64.b64encode(img).decode("ascii") for img in images]
        messages.append(user_msg)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_schema is not None:
            payload["format"] = format_schema

        last_err: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                data = self._transport(payload)
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(2**attempt)
        else:
            raise ConnectionError(
                f"Ollama unreachable at {self.host} after {self.retries + 1} attempts. "
                f"Is `ollama serve` running and `{self.model}` pulled?"
            ) from last_err

        text = data.get("message", {}).get("content", "")
        return VLMResponse(text=text, model=self.model, latency_s=time.perf_counter() - t0)
