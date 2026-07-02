"""Tests for build_assist_service — the Ollama-first, policy-gated chain (R1/R2)."""
from __future__ import annotations

from auxima_ai.bootstrap import build_assist_service
from auxima_ai.config import Settings


def _settings() -> Settings:
	return Settings(_env_file=None)


def _models(svc) -> list[str]:
	return [s.model_id for s in svc.steps]


def _classes(svc) -> list[str]:
	return [s.provider_class for s in svc.steps]


def test_ollama_only_when_no_openrouter_key(monkeypatch):
	monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
	svc = build_assist_service(_settings())
	assert _models(svc) == ["qwen2.5:0.5b"]  # cloud step skipped, Ollama only
	assert _classes(svc) == ["self-hosted"]
	assert svc.enforcer is not None  # policy gate wired


def test_ollama_FIRST_when_key_present(monkeypatch):
	"""CLAUDE §2: self-hosted is the default — Ollama precedes the cloud step."""
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
	svc = build_assist_service(_settings())
	assert _models(svc) == ["qwen2.5:0.5b", "google/gemma-4-31b-it:free"]
	assert _classes(svc) == ["self-hosted", "free-cloud"]


def test_custom_models_from_settings(monkeypatch):
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
	s = Settings(
		_env_file=None,
		assist_openrouter_model="meta-llama/llama-3.3-70b-instruct:free",
		assist_ollama_model="qwen2.5:7b",
	)
	svc = build_assist_service(s)
	assert _models(svc) == ["qwen2.5:7b", "meta-llama/llama-3.3-70b-instruct:free"]
	assert _classes(svc) == ["self-hosted", "free-cloud"]
