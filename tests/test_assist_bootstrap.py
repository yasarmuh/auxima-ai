"""Tests for build_assist_service — the provider fallback chain composition."""
from __future__ import annotations

from auxima_ai.assist.fallback import FallbackLLMCaller
from auxima_ai.bootstrap import build_assist_service
from auxima_ai.config import Settings


def _settings() -> Settings:
	return Settings(_env_file=None)


def test_ollama_only_when_no_openrouter_key(monkeypatch):
	monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
	svc = build_assist_service(_settings())
	assert isinstance(svc.llm, FallbackLLMCaller)
	models = [m for _, m in svc.llm.steps]
	assert models == ["llama3.2:1b"]  # OpenRouter step skipped, Ollama remains


def test_openrouter_first_when_key_present(monkeypatch):
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
	svc = build_assist_service(_settings())
	models = [m for _, m in svc.llm.steps]
	assert models == ["google/gemma-4-31b-it:free", "llama3.2:1b"]


def test_custom_models_from_settings(monkeypatch):
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
	s = Settings(
		_env_file=None,
		assist_openrouter_model="meta-llama/llama-3.3-70b-instruct:free",
		assist_ollama_model="qwen2.5:7b",
	)
	svc = build_assist_service(s)
	models = [m for _, m in svc.llm.steps]
	assert models == ["meta-llama/llama-3.3-70b-instruct:free", "qwen2.5:7b"]
