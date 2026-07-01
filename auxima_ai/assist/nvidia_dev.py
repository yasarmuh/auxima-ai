"""DEV / PROTOTYPING-ONLY LLM caller for NVIDIA's hosted API (build.nvidia.com).

⚠️  SYNTHETIC / NON-PERSONAL DATA ONLY. This talks to NVIDIA's hosted NIM endpoints,
which run on NVIDIA DGX Cloud (US) — a cross-border egress path. Under the locked
architecture (CLAUDE.md §2: self-hosted Ollama default, in-Kingdom inference; the
``local_only=True`` egress pin) real KSA customer PII / health free-text must NEVER be
sent here. This module exists to iterate on prompts and benchmark models during
development against SYNTHETIC data — nothing else. Decision: ADR-OD-NVIDIA-dev-llm.

Two hard guards keep it out of production:
  1. **Runtime:** construction RAISES unless ``AUXIMA_DEV_LLM_ENABLED=1`` is set. A prod
     process does not set it, so an accidental import can never issue a network call.
  2. **Wiring:** ``auxima_ai/bootstrap.py`` (the prod provider chain) never references this
     module — asserted by ``tests/test_nvidia_dev_guard.py`` so a future edit can't sneak
     it into the Ollama-first chain.

NVIDIA's API is OpenAI-compatible Chat Completions, identical in shape to OpenRouter, so
this composes :class:`OpenRouterLLMCaller` rather than duplicating the httpx client.

    # dev shell only — against synthetic data:
    #   export AUXIMA_DEV_LLM_ENABLED=1
    #   export NVIDIA_API_KEY=nvapi-...
    from auxima_ai.assist.nvidia_dev import NvidiaDevLLMCaller
    from auxima_ai.assist.service import AssistService
    svc = AssistService(llm=NvidiaDevLLMCaller())
    svc.draft_email(DraftEmailRequest(tenant_id="dev", purpose="test", recipient_name="Test Co"))
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from auxima_ai.assist.openrouter import OpenRouterLLMCaller
from auxima_ai.intake.llm import LLMResponse

#: NVIDIA's OpenAI-compatible base URL (build.nvidia.com hosted NIM).
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: The env var that MUST equal "1" for this caller to construct. The single runtime
#: guard against a hosted-cloud call ever happening in a non-dev process.
DEV_ENABLE_ENV = "AUXIMA_DEV_LLM_ENABLED"

#: Env var holding the NVIDIA key (prefixed ``nvapi-``).
API_KEY_ENV = "NVIDIA_API_KEY"


class NvidiaDevDisabledError(RuntimeError):
	"""Raised when the caller is constructed without the explicit dev opt-in flag."""


class NvidiaDevAuthError(RuntimeError):
	"""Raised when no NVIDIA API key is available."""


@dataclass
class NvidiaDevLLMCaller:
	"""OpenAI-compatible caller for NVIDIA's hosted API — DEV ONLY, synthetic data only.

	Refuses to construct unless ``AUXIMA_DEV_LLM_ENABLED=1``. ``api_key`` falls back to the
	``NVIDIA_API_KEY`` env var. ``transport`` is injectable for tests (no network).
	"""

	api_key: str | None = None
	base_url: str = DEFAULT_BASE_URL
	timeout_seconds: float = 45.0
	temperature: float = 0.4
	max_tokens: int = 700
	transport: httpx.BaseTransport | None = None
	_inner: OpenRouterLLMCaller = field(init=False)

	def __post_init__(self) -> None:
		if os.environ.get(DEV_ENABLE_ENV) != "1":
			raise NvidiaDevDisabledError(
				f"{type(self).__name__} is dev-only and refuses to run in production. "
				f"Set {DEV_ENABLE_ENV}=1 (dev shell, synthetic data only) to enable it."
			)
		key = self.api_key or os.environ.get(API_KEY_ENV)
		if not key:
			raise NvidiaDevAuthError(f"NVIDIA API key required (pass api_key= or set {API_KEY_ENV})")
		# Reuse the OpenRouter OpenAI-compatible client, pointed at NVIDIA's base URL.
		self._inner = OpenRouterLLMCaller(
			api_key=key,
			base_url=self.base_url,
			timeout_seconds=self.timeout_seconds,
			temperature=self.temperature,
			max_tokens=self.max_tokens,
			transport=self.transport,
		)

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		"""Issue one chat-completion against NVIDIA's hosted API (dev/synthetic only)."""
		return self._inner.call(model_id=model_id, prompt=prompt)

	def close(self) -> None:
		self._inner.close()


__all__ = (
	"API_KEY_ENV",
	"DEFAULT_BASE_URL",
	"DEV_ENABLE_ENV",
	"NvidiaDevAuthError",
	"NvidiaDevDisabledError",
	"NvidiaDevLLMCaller",
)
