"""Ollama-backed :class:`LLMCaller` for production / dev.

Talks directly to the Ollama HTTP API at ``POST /api/generate`` with
``format="json"`` to force structured JSON output. No LiteLLM wrapper
in v1 — we have one production provider (Ollama) and the LiteLLM
overhead (a heavy dep, complex provider-routing layer) isn't earning
its keep until we wire the opt-in cloud providers.

API contract (Ollama):
  Request : POST {base_url}/api/generate
            body = {model, prompt, format: "json", stream: false}
  Response: {response: "<json-string>", prompt_eval_count, eval_count,
             total_duration (ns), model, ...}

We normalise that into :class:`auxima_ai.intake.llm.LLMResponse`:
  - ``payload`` = ``json.loads(response)`` — the structured fields
  - ``prompt_tokens`` = ``prompt_eval_count`` (Ollama's name)
  - ``completion_tokens`` = ``eval_count`` (Ollama's name)
  - ``latency_ms`` = ``total_duration // 1_000_000`` (ns -> ms)
  - ``model_version`` = the model string returned (may include the
    Ollama digest tag; useful for the AI Run Log)

Errors are normalised into :class:`OllamaError` subclasses so the
intake service / circuit-breaker layer above sees a single typed
exception space rather than httpx internals.

The HTTP transport is injectable (:class:`httpx.MockTransport` in
tests) so the unit-test suite never touches the network.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from auxima_ai.intake.llm import LLMResponse

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_PATH = "/api/generate"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OllamaError(RuntimeError):
    """Base — every failure raises a subclass of this."""


class OllamaConnectionError(OllamaError):
    """Network / DNS / TLS failure; treat as transient."""


class OllamaTimeoutError(OllamaError):
    """Upstream took longer than the configured timeout."""


class OllamaBadStatusError(OllamaError):
    """Ollama returned a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Ollama returned HTTP {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class OllamaMalformedResponseError(OllamaError):
    """Response missing required fields, or ``response`` not parseable as JSON."""


# ---------------------------------------------------------------------------
# Caller
# ---------------------------------------------------------------------------


@dataclass
class OllamaLLMCaller:
    """Ollama HTTP client satisfying :class:`LLMCaller`.

    Construct once at app startup; reuse the underlying ``httpx.Client``
    so the connection pool warms up.
    """

    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise OllamaError("base_url must be a non-empty string")
        if self.timeout_seconds <= 0:
            raise OllamaError(f"timeout_seconds must be > 0; got {self.timeout_seconds}")
        # Strip trailing slash so we can append DEFAULT_PATH without doubling up.
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> "OllamaLLMCaller":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- LLMCaller protocol -----------------------------------------------

    def call(self, *, model_id: str, prompt: str) -> LLMResponse:
        """Issue one /api/generate call and return the normalised response."""
        if not isinstance(model_id, str) or not model_id:
            raise OllamaError("model_id must be a non-empty string")
        if not isinstance(prompt, str):
            raise OllamaError(f"prompt must be str; got {type(prompt).__name__}")

        # The model alias in the policy / pricing layer is namespaced
        # ("ollama/qwen2.5:32b"); strip the provider prefix before sending
        # to Ollama, which only knows the bare model name.
        bare_model = model_id.split("/", 1)[1] if "/" in model_id else model_id

        body = {
            "model": bare_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        try:
            resp = self._client.post(DEFAULT_PATH, json=body)
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Ollama call timed out after {self.timeout_seconds}s"
            ) from e
        except httpx.HTTPError as e:
            raise OllamaConnectionError(f"Ollama connection failed: {e}") from e

        if resp.status_code >= 300 or resp.status_code < 200:
            raise OllamaBadStatusError(resp.status_code, resp.text)

        try:
            raw = resp.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise OllamaMalformedResponseError(
                f"Ollama response was not valid JSON: {e}"
            ) from e

        return _normalise_response(raw, model_id=model_id)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalise_response(raw: dict[str, Any], *, model_id: str) -> LLMResponse:
    """Map Ollama's response shape into :class:`LLMResponse`.

    Raises :class:`OllamaMalformedResponseError` if required keys are
    missing or have the wrong type.
    """
    if not isinstance(raw, dict):
        raise OllamaMalformedResponseError(
            f"Ollama response root must be a JSON object; got {type(raw).__name__}"
        )
    inner_text = raw.get("response")
    if not isinstance(inner_text, str):
        raise OllamaMalformedResponseError(
            f"Ollama response.response missing or not a string; got {type(inner_text).__name__}"
        )

    # With format="json" Ollama returns a JSON-encoded string in
    # ``response`` — decode it into the structured payload.
    try:
        payload = json.loads(inner_text)
    except json.JSONDecodeError as e:
        raise OllamaMalformedResponseError(
            f"Ollama returned format=json but inner response failed to parse: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise OllamaMalformedResponseError(
            f"Inner Ollama payload must be a JSON object; got {type(payload).__name__}"
        )

    prompt_tokens = _coerce_token_count(raw.get("prompt_eval_count"), "prompt_eval_count")
    completion_tokens = _coerce_token_count(raw.get("eval_count"), "eval_count")
    latency_ms = _coerce_latency_ns_to_ms(raw.get("total_duration"))

    # Ollama echoes the model string in the response — fall back to the
    # request's model_id if the field is absent so the AI Run Log always
    # has something to record.
    model_version = raw.get("model") if isinstance(raw.get("model"), str) else model_id

    return LLMResponse(
        payload=payload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        model_version=model_version,
    )


def _coerce_token_count(value: Any, name: str) -> int:
    """Accept missing -> 0 (defensive), else require non-negative int."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise OllamaMalformedResponseError(
            f"{name} must be int; got {type(value).__name__}"
        )
    if value < 0:
        raise OllamaMalformedResponseError(f"{name} must be >= 0; got {value}")
    return value


def _coerce_latency_ns_to_ms(value: Any) -> int:
    """Ollama reports ``total_duration`` in nanoseconds — convert to ms."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise OllamaMalformedResponseError(
            f"total_duration must be int (ns); got {type(value).__name__}"
        )
    if value < 0:
        raise OllamaMalformedResponseError(
            f"total_duration must be >= 0; got {value}"
        )
    return value // 1_000_000


__all__ = (
    "DEFAULT_BASE_URL",
    "DEFAULT_PATH",
    "DEFAULT_TIMEOUT_SECONDS",
    "OllamaBadStatusError",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaLLMCaller",
    "OllamaMalformedResponseError",
    "OllamaTimeoutError",
)
