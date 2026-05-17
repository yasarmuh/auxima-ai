"""Tests for ``auxima_ai.cost.pricing`` — per-model token-cost calculator.

Coverage:
  - Seeded baseline contains the three CLAUDE §2 Ollama tiers (zero cost).
  - cost_for returns Decimal(0) for any Ollama model regardless of tokens.
  - cost_for computes Gemini / OpenAI cloud cost exactly.
  - Cost is quantised to micro-dollars (COST_QUANTUM).
  - Quantisation is half-up (consistent with the ledger).
  - cost_for rejects unknown models with UnknownModelError.
  - cost_for rejects bad token counts (negative, non-int, bool).
  - register_pricing inserts AND replaces.
  - register_pricing rejects non-ModelPricing input.
  - ModelPricing rejects bad fields (empty string, float, negative, NaN, Inf).
  - known_models returns sorted snapshot.
  - reset_pricing_table restores the seeded baseline.
  - pricing_for raises on empty / non-string id.
  - cost_for output integrates with InMemoryCostLedger.try_spend round-trip.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from auxima_ai.cost.ledger import COST_QUANTUM, InMemoryCostLedger, LedgerEntry, Recorded
from auxima_ai.cost.pricing import (
    InvalidPricingError,
    ModelPricing,
    PricingError,
    UnknownModelError,
    cost_for,
    known_models,
    pricing_for,
    register_pricing,
    reset_pricing_table,
)

UTC = timezone.utc
TS = datetime(2026, 5, 17, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_table():
    """Each test starts with the seeded baseline."""
    reset_pricing_table()
    yield
    reset_pricing_table()


# ---------------------------------------------------------------------------
# Seeded baseline
# ---------------------------------------------------------------------------


def test_baseline_has_the_three_ollama_tiers() -> None:
    ids = known_models()
    assert "ollama/qwen2.5:32b" in ids
    assert "ollama/jais:13b" in ids
    assert "ollama/llama3.1:8b" in ids


def test_baseline_has_opt_in_cloud_examples() -> None:
    ids = known_models()
    assert "gemini/gemini-2.0-flash" in ids
    assert "openai/gpt-4o-mini" in ids


# ---------------------------------------------------------------------------
# cost_for — Ollama = 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["ollama/qwen2.5:32b", "ollama/jais:13b", "ollama/llama3.1:8b"],
)
def test_ollama_cost_is_zero(model: str) -> None:
    """Self-hosted Ollama models price at exactly zero regardless of tokens."""
    assert cost_for(model, 1_000_000, 1_000_000) == Decimal("0")
    assert cost_for(model, 0, 0) == Decimal("0")
    assert cost_for(model, 1, 0) == Decimal("0")


# ---------------------------------------------------------------------------
# cost_for — cloud math is exact
# ---------------------------------------------------------------------------


def test_gemini_cost_at_one_million_tokens_each() -> None:
    """1M prompt @ $0.10 + 1M completion @ $0.40 = $0.50 exactly."""
    cost = cost_for("gemini/gemini-2.0-flash", 1_000_000, 1_000_000)
    assert cost == Decimal("0.500000")


def test_openai_cost_at_one_million_tokens_each() -> None:
    """1M prompt @ $0.15 + 1M completion @ $0.60 = $0.75 exactly."""
    cost = cost_for("openai/gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == Decimal("0.750000")


def test_gemini_cost_for_typical_short_completion() -> None:
    """500 prompt + 200 completion at gemini-2.0-flash prices."""
    cost = cost_for("gemini/gemini-2.0-flash", 500, 200)
    expected = (
        Decimal(500) * Decimal("0.10") + Decimal(200) * Decimal("0.40")
    ) / Decimal("1000000")
    assert cost == expected.quantize(COST_QUANTUM)


def test_cost_is_decimal_not_float() -> None:
    cost = cost_for("gemini/gemini-2.0-flash", 100, 100)
    assert isinstance(cost, Decimal)


def test_cost_quantised_to_micro_dollars() -> None:
    cost = cost_for("gemini/gemini-2.0-flash", 1, 1)
    # The micro-dollar grid is 6 decimal places.
    assert -cost.as_tuple().exponent == 6


def test_quantisation_is_half_up() -> None:
    """A 7-decimal-place raw value rounds up at the 6th place."""
    # Construct a model where 1 token of prompt = $0.0000005000001 raw -> rounds to .000001.
    register_pricing(
        ModelPricing(
            model_id="custom/half-up-test",
            provider="custom",
            prompt_per_million=Decimal("0.5000001"),
            completion_per_million=Decimal("0"),
        ),
    )
    cost = cost_for("custom/half-up-test", 1, 0)
    assert cost == Decimal("0.000001")


# ---------------------------------------------------------------------------
# Validation — model id
# ---------------------------------------------------------------------------


def test_cost_for_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError, match="no pricing"):
        cost_for("anthropic/claude-sonnet-9", 100, 100)


@pytest.mark.parametrize("bad_id", ["", None])
def test_cost_for_bad_model_id_raises(bad_id: object) -> None:
    with pytest.raises(UnknownModelError):
        cost_for(bad_id, 1, 1)  # type: ignore[arg-type]


def test_pricing_for_unknown_raises() -> None:
    with pytest.raises(UnknownModelError):
        pricing_for("nope")


# ---------------------------------------------------------------------------
# Validation — token counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"prompt_tokens": -1}, {"completion_tokens": -1}])
def test_negative_token_count_rejected(kwargs: dict) -> None:
    args = {"model_id": "ollama/qwen2.5:32b", "prompt_tokens": 1, "completion_tokens": 1}
    args.update(kwargs)
    with pytest.raises(PricingError, match=">= 0"):
        cost_for(**args)


@pytest.mark.parametrize("bad", [True, False, 1.5, "1", None])
def test_non_int_token_count_rejected(bad: object) -> None:
    with pytest.raises(PricingError, match="int"):
        cost_for("ollama/qwen2.5:32b", bad, 0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModelPricing validation
# ---------------------------------------------------------------------------


def test_model_pricing_rejects_float() -> None:
    with pytest.raises(InvalidPricingError, match="Decimal"):
        ModelPricing(
            model_id="m", provider="p",
            prompt_per_million=0.10,  # type: ignore[arg-type]
            completion_per_million=Decimal("0.40"),
        )


def test_model_pricing_rejects_negative() -> None:
    with pytest.raises(InvalidPricingError, match=">= 0"):
        ModelPricing(
            model_id="m", provider="p",
            prompt_per_million=Decimal("-0.01"),
            completion_per_million=Decimal("0"),
        )


def test_model_pricing_rejects_nan() -> None:
    with pytest.raises(InvalidPricingError, match="finite"):
        ModelPricing(
            model_id="m", provider="p",
            prompt_per_million=Decimal("NaN"),
            completion_per_million=Decimal("0"),
        )


def test_model_pricing_rejects_inf() -> None:
    with pytest.raises(InvalidPricingError, match="finite"):
        ModelPricing(
            model_id="m", provider="p",
            prompt_per_million=Decimal("Infinity"),
            completion_per_million=Decimal("0"),
        )


@pytest.mark.parametrize("field", ["model_id", "provider"])
def test_model_pricing_rejects_empty_strings(field: str) -> None:
    kwargs = dict(
        model_id="m", provider="p",
        prompt_per_million=Decimal("0"), completion_per_million=Decimal("0"),
    )
    kwargs[field] = ""
    with pytest.raises(InvalidPricingError, match=field):
        ModelPricing(**kwargs)


def test_model_pricing_is_frozen() -> None:
    p = ModelPricing(
        model_id="m", provider="p",
        prompt_per_million=Decimal("0"), completion_per_million=Decimal("0"),
    )
    with pytest.raises((AttributeError, TypeError)):
        p.model_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# register_pricing
# ---------------------------------------------------------------------------


def test_register_pricing_inserts_new_model() -> None:
    register_pricing(
        ModelPricing(
            model_id="anthropic/claude-sonnet-9",
            provider="anthropic",
            prompt_per_million=Decimal("3.00"),
            completion_per_million=Decimal("15.00"),
        ),
    )
    cost = cost_for("anthropic/claude-sonnet-9", 1_000_000, 1_000_000)
    assert cost == Decimal("18.000000")


def test_register_pricing_replaces_existing() -> None:
    register_pricing(
        ModelPricing(
            model_id="ollama/qwen2.5:32b",
            provider="ollama",
            prompt_per_million=Decimal("99"),
            completion_per_million=Decimal("99"),
        ),
    )
    p = pricing_for("ollama/qwen2.5:32b")
    assert p.prompt_per_million == Decimal("99")


def test_register_pricing_rejects_non_pricing() -> None:
    with pytest.raises(InvalidPricingError, match="ModelPricing"):
        register_pricing("not-a-pricing")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Snapshot + reset
# ---------------------------------------------------------------------------


def test_known_models_returns_sorted_tuple() -> None:
    ids = known_models()
    assert isinstance(ids, tuple)
    assert list(ids) == sorted(ids)


def test_reset_pricing_table_restores_baseline() -> None:
    register_pricing(
        ModelPricing(
            model_id="x/y", provider="x",
            prompt_per_million=Decimal("1"), completion_per_million=Decimal("1"),
        ),
    )
    assert "x/y" in known_models()
    reset_pricing_table()
    assert "x/y" not in known_models()


# ---------------------------------------------------------------------------
# Ledger integration
# ---------------------------------------------------------------------------


def test_cost_for_round_trips_through_ledger() -> None:
    """A pricing-derived cost goes straight into a LedgerEntry without conversion."""
    cost = cost_for("gemini/gemini-2.0-flash", 500, 200)
    entry = LedgerEntry(
        tenant_id="tenant-acme",
        provider="gemini",
        model="gemini/gemini-2.0-flash",
        model_version="2026-05",
        prompt_tokens=500,
        completion_tokens=200,
        latency_ms=873,
        cost=cost,
        ts=TS,
    )
    ledger = InMemoryCostLedger()
    r = ledger.try_spend(entry)
    assert isinstance(r, Recorded)
    assert r.period_total == cost
