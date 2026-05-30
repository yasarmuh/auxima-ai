"""Perf characterisation for the quote-intake pipeline (P1-10 §4.3).

§4.3 sets an END-TO-END target of p99 < 10 s for "emailed PDF → Quote row".
That end-to-end number is dominated by the LLM inference latency, which depends
on the deployment backend (GPU-hosted Ollama / cloud), NOT on this CPU dev
bench — so it is a **deployment-gated measurement** taken against the real
model in the eval-harness runner, and is deliberately NOT asserted here (a CPU
bench with no model would either skip or mislead).

What this test DOES lock is the sidecar's OWN per-request overhead with the LLM
stubbed out (decode + classify + extract + confidence + redaction + activity).
That overhead must be a negligible slice of the 10 s budget; a regression that
made it large (e.g. an accidental O(n²) over the document, or re-reading the
PDF) would show up here. Bounds are generous so the test is not flaky on a
loaded runner — the goal is catching a gross regression, not microbenchmarking.
"""
from __future__ import annotations

import base64
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from auxima_ai.cost.ledger import InMemoryCostLedger
from auxima_ai.cost.pricing import reset_pricing_table
from auxima_ai.idempotency.store import InMemoryIdempotencyStore
from auxima_ai.intake.llm import StubLLMCaller
from auxima_ai.intake.pdf_text import StubPdfTextExtractor
from auxima_ai.intake.quote_service import QuoteIntakeService, QuoteSuccess
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter

UTC = timezone.utc
TS = datetime(2026, 5, 30, tzinfo=UTC)

_QUOTE_PAYLOAD = {
    "premium": "12500.00",
    "currency": "SAR",
    "coverage": ["Own damage", "Third-party liability"],
    "exclusions": ["War"],
    "model_confidence": 0.9,
}

#: Sidecar-overhead budget (LLM stubbed). Each call is realistically << 10 ms;
#: 0.25 s p99 catches a ~25x regression without flaking on a busy runner.
_P99_OVERHEAD_BUDGET_S = 0.25
_ITERATIONS = 30


@pytest.fixture(autouse=True)
def _reset_pricing():
    reset_pricing_table()
    yield
    reset_pricing_table()


def _service() -> QuoteIntakeService:
    enf = PolicyEnforcer(
        ledger=InMemoryCostLedger(),
        rate_limiter=PerTenantRateLimiter(capacity=100000.0, refill_per_second=100000.0),
    )
    enf.set_policy(
        TenantPolicy(
            tenant_id="tenant-acme", tier=TierPolicy.OLLAMA_THEN_PAID_CLOUD, region="INTL",
            monthly_ceiling=Decimal("100000"), rate_capacity=100000.0, rate_refill_per_second=100000.0,
        )
    )
    return QuoteIntakeService(
        enforcer=enf,
        idempotency=InMemoryIdempotencyStore(),
        llm=StubLLMCaller(payload=_QUOTE_PAYLOAD, latency_ms=0),
        pdf_extractor=StubPdfTextExtractor(text="Quote text. " * 60),  # ~720 chars
    )


def test_sidecar_overhead_is_a_negligible_slice_of_the_budget() -> None:
    svc = _service()
    doc_b64 = base64.b64encode(b"%PDF-1.4\n" + b"x" * 4000 + b"\n%%EOF\n").decode("ascii")
    samples: list[float] = []
    for i in range(_ITERATIONS):
        req = {"tenant_id": "tenant-acme", "document_b64": doc_b64, "model_id": "ollama/qwen2.5:32b"}
        from auxima_ai.intake.quote_schema import QuoteIntakeRequest

        t0 = time.perf_counter()
        out = svc.extract_quote(QuoteIntakeRequest(**req), idempotency_key=f"perf-{i}", now=TS)
        samples.append(time.perf_counter() - t0)
        assert isinstance(out, QuoteSuccess)

    samples.sort()
    p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
    median = statistics.median(samples)
    # Median is the steady-state signal; p99 guards the tail. Both must sit far
    # below the 10 s end-to-end budget — the LLM is what consumes that, not us.
    assert median < _P99_OVERHEAD_BUDGET_S, f"median overhead {median:.4f}s too high"
    assert p99 < _P99_OVERHEAD_BUDGET_S, f"p99 overhead {p99:.4f}s too high"
