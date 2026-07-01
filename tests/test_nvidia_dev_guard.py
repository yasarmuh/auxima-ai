"""Guards for the DEV-ONLY NVIDIA hosted caller (ADR-OD-NVIDIA-dev-llm).

The point of these tests is not to exercise NVIDIA — it is to PROVE the caller can never
reach production data:
  * it refuses to construct without the explicit ``AUXIMA_DEV_LLM_ENABLED=1`` opt-in;
  * it refuses without a key;
  * the prod bootstrap (``auxima_ai/bootstrap.py``) never references it or NVIDIA's base URL,
    so it cannot be wired into the Ollama-first provider chain by a later edit;
  * when explicitly enabled in a dev context it works (OpenAI-compatible, no network).
"""
from __future__ import annotations

import pathlib

import httpx
import pytest

from auxima_ai.assist.nvidia_dev import (
	DEFAULT_BASE_URL,
	DEV_ENABLE_ENV,
	NvidiaDevAuthError,
	NvidiaDevDisabledError,
	NvidiaDevLLMCaller,
)


def test_refuses_without_dev_flag(monkeypatch):
	monkeypatch.delenv(DEV_ENABLE_ENV, raising=False)
	monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
	with pytest.raises(NvidiaDevDisabledError):
		NvidiaDevLLMCaller()


def test_refuses_when_flag_not_exactly_one(monkeypatch):
	# A truthy-but-not-"1" value must NOT enable it — the guard is an exact match.
	monkeypatch.setenv(DEV_ENABLE_ENV, "true")
	monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
	with pytest.raises(NvidiaDevDisabledError):
		NvidiaDevLLMCaller()


def test_refuses_without_key(monkeypatch):
	monkeypatch.setenv(DEV_ENABLE_ENV, "1")
	monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
	with pytest.raises(NvidiaDevAuthError):
		NvidiaDevLLMCaller()


def test_prod_bootstrap_never_references_nvidia_dev():
	"""Static guard: the prod provider chain must not import or mention the dev caller /
	NVIDIA's hosted endpoint. If this fails, someone wired a US-cloud egress path into
	the Ollama-first chain — a residency violation (CLAUDE §2)."""
	bootstrap = pathlib.Path(__file__).resolve().parents[1] / "auxima_ai" / "bootstrap.py"
	src = bootstrap.read_text(encoding="utf-8")
	assert "nvidia_dev" not in src, "bootstrap.py must not reference the dev-only NVIDIA caller"
	assert "NvidiaDevLLMCaller" not in src
	assert "integrate.api.nvidia.com" not in src


def test_works_when_explicitly_enabled(monkeypatch):
	"""With the dev flag + key + a mock transport, the caller functions (OpenAI-compatible)."""
	monkeypatch.setenv(DEV_ENABLE_ENV, "1")

	def handler(request: httpx.Request) -> httpx.Response:
		# NVIDIA base URL is honoured, and the OpenAI-compatible body is parsed.
		assert str(request.url).startswith(DEFAULT_BASE_URL)
		return httpx.Response(
			200,
			json={
				"model": "meta/llama-3.1-8b-instruct",
				"choices": [{"message": {"content": '{"reply": "hi from a synthetic dev run"}'}}],
				"usage": {"prompt_tokens": 5, "completion_tokens": 7},
			},
		)

	caller = NvidiaDevLLMCaller(api_key="nvapi-test", transport=httpx.MockTransport(handler))
	resp = caller.call(model_id="meta/llama-3.1-8b-instruct", prompt="draft a reply")
	assert resp.payload == {"reply": "hi from a synthetic dev run"}
	assert resp.prompt_tokens == 5
	assert resp.model_version == "meta/llama-3.1-8b-instruct"
