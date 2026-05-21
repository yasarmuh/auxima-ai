"""Unit tests for OpenRouterLLMCaller — driven by httpx.MockTransport (no network)."""
from __future__ import annotations

import httpx
import pytest

from auxima_ai.assist.openrouter import (
	OpenRouterAuthError,
	OpenRouterBadStatusError,
	OpenRouterLLMCaller,
	OpenRouterMalformedResponseError,
	OpenRouterRateLimitedError,
)


def _caller(handler) -> OpenRouterLLMCaller:
	return OpenRouterLLMCaller(api_key="test-key", transport=httpx.MockTransport(handler))


def _ok_body(content: str) -> dict:
	return {
		"model": "google/gemma-4-31b-it:free",
		"choices": [{"message": {"content": content}}],
		"usage": {"prompt_tokens": 11, "completion_tokens": 22},
	}


def test_requires_api_key(monkeypatch):
	monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
	with pytest.raises(OpenRouterAuthError):
		OpenRouterLLMCaller(api_key=None)


def test_success_parses_json_content():
	def handler(request: httpx.Request) -> httpx.Response:
		assert request.headers["Authorization"] == "Bearer test-key"
		return httpx.Response(200, json=_ok_body('{"subject":"Hi","body":"Hello there."}'))

	resp = _caller(handler).call(model_id="google/gemma-4-31b-it:free", prompt="draft it")
	assert resp.payload == {"subject": "Hi", "body": "Hello there."}
	assert resp.prompt_tokens == 11
	assert resp.completion_tokens == 22


def test_strips_markdown_fence():
	fenced = "```json\n{\"subject\":\"S\",\"body\":\"B\"}\n```"
	resp = _caller(lambda r: httpx.Response(200, json=_ok_body(fenced))).call(
		model_id="m", prompt="p"
	)
	assert resp.payload == {"subject": "S", "body": "B"}


def test_429_is_rate_limited():
	with pytest.raises(OpenRouterRateLimitedError):
		_caller(lambda r: httpx.Response(429, text="slow down")).call(model_id="m", prompt="p")


def test_402_spend_limit_is_rate_limited():
	# A "free" model that routes to a paid provider returns 402 — treat as transient.
	with pytest.raises(OpenRouterRateLimitedError):
		_caller(lambda r: httpx.Response(402, text="spend limit")).call(model_id="m", prompt="p")


def test_401_is_auth_error():
	with pytest.raises(OpenRouterAuthError):
		_caller(lambda r: httpx.Response(401, text="bad key")).call(model_id="m", prompt="p")


def test_404_is_bad_status():
	with pytest.raises(OpenRouterBadStatusError):
		_caller(lambda r: httpx.Response(404, text="no endpoints")).call(model_id="m", prompt="p")


def test_non_json_content_is_malformed():
	with pytest.raises(OpenRouterMalformedResponseError):
		_caller(lambda r: httpx.Response(200, json=_ok_body("not json at all"))).call(
			model_id="m", prompt="p"
		)


def test_error_object_with_200_is_transient():
	# Some provider routes return an error object with HTTP 200.
	body = {"error": {"message": "provider busy"}}
	with pytest.raises(OpenRouterRateLimitedError):
		_caller(lambda r: httpx.Response(200, json=body)).call(model_id="m", prompt="p")
