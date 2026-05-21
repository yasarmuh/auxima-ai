"""Unit tests for the FallbackLLMCaller provider chain."""
from __future__ import annotations

import pytest

from auxima_ai.assist.fallback import AllProvidersUnavailable, FallbackLLMCaller
from auxima_ai.intake.llm import LLMResponse, StubLLMCaller


class _Boom:
	"""A caller that always raises — stands in for a rate-limited/down provider."""

	def __init__(self, exc: Exception) -> None:
		self.exc = exc
		self.calls = 0

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		self.calls += 1
		raise self.exc


def test_empty_chain_is_unavailable():
	with pytest.raises(AllProvidersUnavailable):
		FallbackLLMCaller(steps=[]).call(model_id="x", prompt="p")


def test_first_success_short_circuits():
	good = StubLLMCaller(payload={"subject": "S", "body": "B"})
	never = _Boom(RuntimeError("should not be called"))
	chain = FallbackLLMCaller(steps=[(good, "primary"), (never, "fallback")])
	resp = chain.call(model_id="ignored", prompt="p")
	assert resp.payload == {"subject": "S", "body": "B"}
	assert never.calls == 0  # second step never reached


def test_falls_through_to_next_on_failure():
	boom = _Boom(RuntimeError("429 rate limited"))
	good = StubLLMCaller(payload={"subject": "S2", "body": "B2"})
	chain = FallbackLLMCaller(steps=[(boom, "free-model"), (good, "ollama")])
	resp = chain.call(model_id="ignored", prompt="p")
	assert resp.payload == {"subject": "S2", "body": "B2"}
	assert boom.calls == 1  # tried first, failed, moved on


def test_all_fail_raises_with_all_errors():
	a = _Boom(RuntimeError("429"))
	b = _Boom(ConnectionError("ollama down"))
	chain = FallbackLLMCaller(steps=[(a, "m-a"), (b, "m-b")])
	with pytest.raises(AllProvidersUnavailable) as ei:
		chain.call(model_id="ignored", prompt="p")
	# Both per-step errors are recorded for diagnostics.
	models = [m for m, _ in ei.value.errors]
	assert models == ["m-a", "m-b"]
