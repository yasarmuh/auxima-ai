"""OpenRouter-backed :class:`LLMCaller` for dev / opt-in cloud.

OpenRouter exposes an OpenAI-compatible Chat Completions API, so — like
:mod:`auxima_ai.intake.ollama` — we talk to it directly over httpx rather
than pulling in LiteLLM. One provider, one HTTP shape, no extra dependency.

API contract (OpenRouter):
  Request : POST {base_url}/chat/completions
            headers = Authorization: Bearer <key>
            body    = {model, messages:[{role,content}], max_tokens, temperature}
  Response: {choices:[{message:{content:"..."}}], usage:{prompt_tokens, completion_tokens}}

The model is expected to return a JSON object as its message content (we ask
for it in the prompt). We parse that content into ``LLMResponse.payload``,
tolerating a ```json ...``` markdown fence since some models wrap output.

The HTTP transport is injectable (:class:`httpx.MockTransport`) so the unit
tests never hit the network, and the API key is injected (never read from a
module global) so a test can run without a real key.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from auxima_ai.intake.llm import LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_PATH = "/chat/completions"

#: env var holding the OpenRouter key. The repo .env names it
#: HUNZI_OPENROUTER_API_KEY; deployment maps it to this canonical name.
API_KEY_ENV = "OPENROUTER_API_KEY"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class OpenRouterError(RuntimeError):
	"""Base — every failure raises a subclass of this."""


class OpenRouterAuthError(OpenRouterError):
	"""Missing/invalid API key (401/403). NOT transient — don't retry."""


class OpenRouterConnectionError(OpenRouterError):
	"""Network / DNS / TLS failure; transient — a fallback caller may succeed."""


class OpenRouterTimeoutError(OpenRouterError):
	"""Upstream took longer than the configured timeout; transient."""


class OpenRouterRateLimitedError(OpenRouterError):
	"""429 / spend-limit / provider-busy; transient — try a fallback model."""


class OpenRouterBadStatusError(OpenRouterError):
	"""Non-2xx that isn't auth or rate-limit (e.g. 404 model not found, 5xx)."""

	def __init__(self, status_code: int, body: str) -> None:
		super().__init__(f"OpenRouter returned HTTP {status_code}: {body[:200]}")
		self.status_code = status_code
		self.body = body


class OpenRouterMalformedResponseError(OpenRouterError):
	"""Response missing choices, or message content not parseable as JSON."""


@dataclass
class OpenRouterLLMCaller:
	"""OpenRouter Chat Completions client satisfying :class:`LLMCaller`.

	``api_key`` is required (falls back to the ``OPENROUTER_API_KEY`` env var
	only if not passed). Construct once at startup; reuse the httpx pool.
	"""

	api_key: str | None = None
	base_url: str = DEFAULT_BASE_URL
	timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
	temperature: float = 0.4
	max_tokens: int = 700
	transport: httpx.BaseTransport | None = None
	_client: httpx.Client = field(init=False)

	def __post_init__(self) -> None:
		self.api_key = self.api_key or os.environ.get(API_KEY_ENV)
		if not self.api_key:
			raise OpenRouterAuthError(
				f"OpenRouter API key required (pass api_key= or set {API_KEY_ENV})"
			)
		if self.timeout_seconds <= 0:
			raise OpenRouterError(f"timeout_seconds must be > 0; got {self.timeout_seconds}")
		self.base_url = self.base_url.rstrip("/")
		self._client = httpx.Client(
			base_url=self.base_url,
			timeout=self.timeout_seconds,
			transport=self.transport,
			headers={
				"Authorization": f"Bearer {self.api_key}",
				# OpenRouter etiquette headers — identify the app, don't leak data.
				"HTTP-Referer": "https://insure.auxima.ai",
				"X-Title": "Auxima Insure",
			},
		)

	def close(self) -> None:
		self._client.close()

	def __enter__(self) -> "OpenRouterLLMCaller":
		return self

	def __exit__(self, *exc_info) -> None:
		self.close()

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		"""Issue one chat-completion and return the normalised JSON payload."""
		if not isinstance(model_id, str) or not model_id:
			raise OpenRouterError("model_id must be a non-empty string")
		if not isinstance(prompt, str):
			raise OpenRouterError(f"prompt must be str; got {type(prompt).__name__}")

		body = {
			"model": model_id,
			"messages": [{"role": "user", "content": prompt}],
			"max_tokens": self.max_tokens,
			"temperature": self.temperature,
		}
		try:
			resp = self._client.post(DEFAULT_PATH, json=body)
		except httpx.TimeoutException as e:
			raise OpenRouterTimeoutError(
				f"OpenRouter call timed out after {self.timeout_seconds}s"
			) from e
		except httpx.HTTPError as e:
			raise OpenRouterConnectionError(f"OpenRouter connection failed: {e}") from e

		if resp.status_code in (401, 403):
			raise OpenRouterAuthError(f"OpenRouter auth failed: HTTP {resp.status_code}")
		if resp.status_code == 429:
			raise OpenRouterRateLimitedError("OpenRouter rate-limited (HTTP 429)")
		if resp.status_code == 402:
			# spend-limit / paid-provider route on a "free" model — treat as transient
			# so the fallback chain moves to the next (free or local) model.
			raise OpenRouterRateLimitedError("OpenRouter spend limit hit (HTTP 402)")
		if resp.status_code < 200 or resp.status_code >= 300:
			raise OpenRouterBadStatusError(resp.status_code, resp.text)

		try:
			raw = resp.json()
		except (ValueError, json.JSONDecodeError) as e:
			raise OpenRouterMalformedResponseError(
				f"OpenRouter response was not valid JSON: {e}"
			) from e
		return _normalise_response(raw, model_id=model_id)


def _strip_fence(text: str) -> str:
	"""Remove a leading ```json / trailing ``` fence some models wrap output in."""
	return _FENCE_RE.sub("", text).strip()


def _normalise_response(raw: dict[str, Any], *, model_id: str) -> LLMResponse:
	if not isinstance(raw, dict):
		raise OpenRouterMalformedResponseError(
			f"OpenRouter response root must be a JSON object; got {type(raw).__name__}"
		)
	# An error object can come back with HTTP 200 in some provider routes.
	if "error" in raw and "choices" not in raw:
		msg = raw.get("error", {})
		raise OpenRouterRateLimitedError(f"OpenRouter provider error: {msg}")
	choices = raw.get("choices")
	if not isinstance(choices, list) or not choices:
		raise OpenRouterMalformedResponseError("OpenRouter response missing choices[]")
	content = (choices[0] or {}).get("message", {}).get("content")
	if not isinstance(content, str) or not content.strip():
		raise OpenRouterMalformedResponseError("OpenRouter choices[0].message.content empty")

	try:
		# strict=False tolerates literal control chars (unescaped newlines/tabs) inside string
		# values — smaller instruct models routinely emit these in multi-line bodies, and a strict
		# parse would 500 on otherwise-usable output.
		payload = json.loads(_strip_fence(content), strict=False)
	except json.JSONDecodeError as e:
		raise OpenRouterMalformedResponseError(
			f"OpenRouter message content was not valid JSON: {e}"
		) from e
	if not isinstance(payload, dict):
		raise OpenRouterMalformedResponseError(
			f"OpenRouter payload must be a JSON object; got {type(payload).__name__}"
		)

	usage = raw.get("usage") or {}
	return LLMResponse(
		payload=payload,
		prompt_tokens=_int(usage.get("prompt_tokens")),
		completion_tokens=_int(usage.get("completion_tokens")),
		latency_ms=0,  # OpenRouter doesn't report server-side latency in the body
		model_version=raw.get("model") if isinstance(raw.get("model"), str) else model_id,
	)


def _int(value: Any) -> int:
	return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = (
	"API_KEY_ENV",
	"DEFAULT_BASE_URL",
	"OpenRouterAuthError",
	"OpenRouterBadStatusError",
	"OpenRouterConnectionError",
	"OpenRouterError",
	"OpenRouterLLMCaller",
	"OpenRouterMalformedResponseError",
	"OpenRouterRateLimitedError",
	"OpenRouterTimeoutError",
)
