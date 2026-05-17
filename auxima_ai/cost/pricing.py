"""Per-model token pricing → :class:`Decimal` cost for the ledger (CLAUDE §2).

The cost ledger expects a ``Decimal`` ``cost`` per :class:`LedgerEntry`,
but the LiteLLM router only emits ``(model_id, prompt_tokens,
completion_tokens, latency_ms)``. This module bridges them: it holds
the public per-million-token prices for every model the sidecar routes
to, and turns a usage triple into a Decimal cost.

Pricing model:
  * **Self-hosted via Ollama** — Tier-A / Tier-B / Tier-C defaults
    (qwen2.5:32b, jais:13b, llama3.1:8b) are priced at exactly zero.
    Hardware amortisation is tracked elsewhere; per-call cost from a
    billable perspective is $0, and the per-tenant monthly ceiling
    therefore doesn't deduct anything for Ollama traffic.
  * **Cloud providers** (Gemini, OpenAI, …) — billed per-token at the
    provider's published list price. Per-tenant policy decides whether
    a cloud path is even invoked (see CLAUDE §2 ``ollama_only`` /
    ``ollama_then_free_cloud`` / ``ollama_then_paid_cloud`` flags).

The pricing table is a process-level dict. ``register_pricing()``
mutates it for ops-driven updates (a new model alias, a quarterly
price change), but the seeded baseline is the safe default — callers
that import this module get a known-good table without further setup.

Money is :class:`Decimal` everywhere (CLAUDE.md §6). Cost output is
quantised to :data:`auxima_ai.cost.ledger.COST_QUANTUM` so a price-table
entry quoted as "$2.50 per 1M tokens" produces an exact ``Decimal`` per
single-token call without float rounding.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Final, Mapping

from auxima_ai.cost.ledger import COST_QUANTUM

logger = logging.getLogger(__name__)

ONE_MILLION: Final[Decimal] = Decimal("1000000")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PricingError(ValueError):
    """Base — every pricing failure raises a subclass of this."""


class UnknownModelError(PricingError):
    """Raised when ``cost_for`` is called with a model not in the table."""


class InvalidPricingError(PricingError):
    """Raised when a ``ModelPricing`` entry is malformed."""


# ---------------------------------------------------------------------------
# Pricing record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """Public list price for one model, both directions.

    ``prompt_per_million`` / ``completion_per_million`` are in
    **dollars per million tokens** — the format every major provider
    publishes (Gemini, OpenAI, Anthropic, etc). The runtime converts to
    per-token by dividing by 1_000_000.

    Self-hosted-via-Ollama models price at exactly zero.
    """

    model_id: str
    provider: str
    prompt_per_million: Decimal
    completion_per_million: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise InvalidPricingError("model_id must be a non-empty string")
        if not isinstance(self.provider, str) or not self.provider:
            raise InvalidPricingError("provider must be a non-empty string")
        for name in ("prompt_per_million", "completion_per_million"):
            v = getattr(self, name)
            if not isinstance(v, Decimal):
                raise InvalidPricingError(
                    f"{name} must be Decimal (not {type(v).__name__}) — "
                    "money invariant per CLAUDE.md §6"
                )
            if v.is_nan() or v.is_infinite():
                raise InvalidPricingError(f"{name} must be finite; got {v}")
            if v < 0:
                raise InvalidPricingError(f"{name} must be >= 0; got {v}")


# ---------------------------------------------------------------------------
# Seeded baseline — the CLAUDE §2 three tiers (all Ollama, $0)
# plus a small set of opt-in cloud reference prices.
# ---------------------------------------------------------------------------

_FREE: Final[Decimal] = Decimal("0")

_SEED_TABLE: Final[Mapping[str, ModelPricing]] = {
    # Tier A (general / reasoning) — self-hosted via Ollama.
    "ollama/qwen2.5:32b": ModelPricing(
        model_id="ollama/qwen2.5:32b",
        provider="ollama",
        prompt_per_million=_FREE,
        completion_per_million=_FREE,
    ),
    # Tier B (Arabic) — self-hosted via Ollama.
    "ollama/jais:13b": ModelPricing(
        model_id="ollama/jais:13b",
        provider="ollama",
        prompt_per_million=_FREE,
        completion_per_million=_FREE,
    ),
    # Tier C (high-volume parse) — self-hosted via Ollama.
    "ollama/llama3.1:8b": ModelPricing(
        model_id="ollama/llama3.1:8b",
        provider="ollama",
        prompt_per_million=_FREE,
        completion_per_million=_FREE,
    ),
    # --- Opt-in cloud (per-tenant policy decides whether these are used) ---
    # Reference list prices as of 2026-05. Re-verify quarterly and update
    # via register_pricing() when providers change their schedules.
    "gemini/gemini-2.0-flash": ModelPricing(
        model_id="gemini/gemini-2.0-flash",
        provider="gemini",
        prompt_per_million=Decimal("0.10"),
        completion_per_million=Decimal("0.40"),
    ),
    "openai/gpt-4o-mini": ModelPricing(
        model_id="openai/gpt-4o-mini",
        provider="openai",
        prompt_per_million=Decimal("0.15"),
        completion_per_million=Decimal("0.60"),
    ),
}


_table: dict[str, ModelPricing] = dict(_SEED_TABLE)
_table_lock: Final[threading.Lock] = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_pricing(pricing: ModelPricing) -> None:
    """Insert or replace a row in the process-level pricing table.

    Idempotent: registering the same model twice is fine — the second
    call's values win. The lock keeps concurrent writers consistent.
    """
    if not isinstance(pricing, ModelPricing):
        raise InvalidPricingError(
            f"pricing must be ModelPricing; got {type(pricing).__name__}"
        )
    with _table_lock:
        _table[pricing.model_id] = pricing
        logger.debug(
            "registered pricing: %s ($%s prompt / $%s completion per 1M)",
            pricing.model_id,
            pricing.prompt_per_million,
            pricing.completion_per_million,
        )


def pricing_for(model_id: str) -> ModelPricing:
    """Look up the pricing row for ``model_id`` or raise :class:`UnknownModelError`."""
    if not isinstance(model_id, str) or not model_id:
        raise UnknownModelError("model_id must be a non-empty string")
    with _table_lock:
        try:
            return _table[model_id]
        except KeyError as e:
            raise UnknownModelError(
                f"no pricing registered for model {model_id!r}; "
                f"known: {sorted(_table)}"
            ) from e


def cost_for(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """Return the Decimal cost for one LLM call.

    Cost = (prompt_tokens * prompt_per_million + completion_tokens *
    completion_per_million) / 1_000_000, quantised to
    :data:`COST_QUANTUM` (micro-dollars).

    Raises
    ------
    UnknownModelError
        ``model_id`` is not in the pricing table.
    PricingError
        Negative or non-int token counts; ``bool`` rejected explicitly
        as an ``int`` subclass.
    """
    for name, v in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise PricingError(f"{name} must be int; got {type(v).__name__}")
        if v < 0:
            raise PricingError(f"{name} must be >= 0; got {v}")

    p = pricing_for(model_id)
    raw = (
        Decimal(prompt_tokens) * p.prompt_per_million
        + Decimal(completion_tokens) * p.completion_per_million
    ) / ONE_MILLION
    return raw.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP)


def known_models() -> tuple[str, ...]:
    """Snapshot of currently registered model ids, sorted for determinism."""
    with _table_lock:
        return tuple(sorted(_table))


def reset_pricing_table() -> None:
    """Restore the seeded baseline. Test-only — never call in prod."""
    global _table
    with _table_lock:
        _table = dict(_SEED_TABLE)


__all__ = (
    "InvalidPricingError",
    "ModelPricing",
    "PricingError",
    "UnknownModelError",
    "cost_for",
    "known_models",
    "pricing_for",
    "register_pricing",
    "reset_pricing_table",
)
