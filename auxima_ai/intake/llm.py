"""LLM caller protocol + deterministic stub for tests / dev.

The real implementation (LiteLLM + Ollama / Gemini / OpenAI) plugs in
at deployment time by satisfying :class:`LLMCaller`. Keeping the
real client out of the import graph means the unit tests for the
intake pipeline have zero network dependence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """Normalised response from any provider after LiteLLM unification."""

    payload: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model_version: str = "stub"


@runtime_checkable
class LLMCaller(Protocol):
    """The boundary the intake service speaks to."""

    def call(self, *, model_id: str, prompt: str) -> LLMResponse: ...


@dataclass
class StubLLMCaller:
    """Deterministic stub — returns a fixed structured payload.

    The stub mirrors the *shape* the real intake.extract endpoint will
    return (lead name + email + phone slots) so the service / route
    contract tests don't depend on a live model.

    ``token_factor`` controls the synthetic token cost — tests bump it
    when they want to exercise the cost-ceiling rejection path.
    """

    payload: Mapping[str, Any] = ()  # type: ignore[assignment]
    prompt_tokens: int = 100
    completion_tokens: int = 50
    latency_ms: int = 42
    model_version: str = "stub-v1"

    def call(self, *, model_id: str, prompt: str) -> LLMResponse:
        # Mapping default = empty tuple by sentinel — convert at call time
        # so a mutable default isn't shared between instances. Default
        # payload matches IntakeExtractFields exactly so the validator
        # accepts it; tests that want failure cases override explicitly.
        body: dict[str, Any] = (
            dict(self.payload) if (isinstance(self.payload, Mapping) and self.payload) else {
                "lead_name": "Acme Brokers",
                "contact_email": "ops@acme.example",
                "contact_phone": "+966500000000",
                "line_of_business": "property",
                "urgency": "normal",
                "notes": None,
            }
        )
        return LLMResponse(
            payload=body,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=self.latency_ms,
            model_version=self.model_version,
        )


__all__ = ("LLMCaller", "LLMResponse", "StubLLMCaller")
