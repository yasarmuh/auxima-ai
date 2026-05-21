"""A :class:`LLMCaller` that tries an ordered chain of providers.

Why: the dev backend is OpenRouter's *free* tier, which is heavily
rate-limited and intermittently unavailable. A single-provider caller would
make the "draft my email" button fail constantly. This wraps an ordered list
of ``(caller, model_id)`` steps — e.g. [OpenRouter free, local Ollama] — and
returns the first success. If *every* step fails it raises
:class:`AllProvidersUnavailable`, which the assist service turns into a clean
"AI temporarily unavailable" degradation (never a 500, never a blocked UI).

Each step's own exceptions are caught and recorded; we move to the next step.
This is deliberately broad: a 429 from one model, a malformed reply from
another, and a connection error from a third should all just advance the chain
rather than surface provider internals to the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from auxima_ai.intake.llm import LLMCaller, LLMResponse

logger = logging.getLogger(__name__)


class AllProvidersUnavailable(RuntimeError):
	"""Every provider in the chain failed. Carries the per-step errors."""

	def __init__(self, errors: list[tuple[str, str]]) -> None:
		self.errors = errors  # [(model_id, "ExcType: message"), ...]
		detail = "; ".join(f"{m} -> {e}" for m, e in errors) or "no providers configured"
		super().__init__(f"all assist providers unavailable: {detail}")


@dataclass
class FallbackLLMCaller:
	"""Try each ``(caller, model_id)`` step in order; first success wins.

	The ``model_id`` passed to :meth:`call` is ignored — the chain defines its
	own per-step models. Kept in the signature to satisfy the
	:class:`LLMCaller` protocol so this composes anywhere a caller is expected.
	"""

	steps: list[tuple[LLMCaller, str]]

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:  # noqa: ARG002 - protocol shape
		if not self.steps:
			raise AllProvidersUnavailable([])
		errors: list[tuple[str, str]] = []
		for caller, step_model in self.steps:
			try:
				response = caller.call(model_id=step_model, prompt=prompt)
			except Exception as e:  # noqa: BLE001 - intentional: any failure advances the chain
				errors.append((step_model, f"{type(e).__name__}: {e}"))
				logger.warning("assist provider %s failed, trying next: %s", step_model, e)
				continue
			if errors:
				logger.info("assist recovered on fallback model %s after %d failure(s)",
				            step_model, len(errors))
			return response
		raise AllProvidersUnavailable(errors)


__all__ = ("AllProvidersUnavailable", "FallbackLLMCaller")
