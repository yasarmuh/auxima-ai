"""In-process RED metrics registry + SLO-breach predicate (S-19 §3.3 / R1, R6).

The RED triad per endpoint: Requests (count), Errors (count of error-class
outcomes), Duration (latency samples → p95/p99). Plus an SLO-breach predicate
over the per-endpoint SLO (p95/p99 latency, error budget).

Pure + in-process (no prometheus dependency — prod scrapes a real histogram;
this is the dev/test surface + the SLO predicate). Outcome vocabulary matches
S-19 R3: ok / error / provider_error / timeout / redaction_required.
"""
from __future__ import annotations

import pytest

from auxima_ai.observability.metrics import (
    EndpointSLO,
    Metrics,
    MetricsError,
)

_EP = "intake.extract"


def test_requests_and_errors_counted_by_outcome() -> None:
    m = Metrics()
    m.observe(_EP, "ok", 100)
    m.observe(_EP, "redaction_required", 120)  # not an error
    m.observe(_EP, "error", 200)
    m.observe(_EP, "provider_error", 300)
    m.observe(_EP, "timeout", 400)
    assert m.requests(_EP) == 5
    assert m.errors(_EP) == 3  # error + provider_error + timeout


def test_error_rate_zero_when_no_requests() -> None:
    assert Metrics().error_rate("never.seen") == 0.0


def test_error_rate_fraction() -> None:
    m = Metrics()
    m.observe(_EP, "ok", 10)
    m.observe(_EP, "error", 10)
    assert m.error_rate(_EP) == 0.5


def test_nearest_rank_quantiles() -> None:
    m = Metrics()
    for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        m.observe(_EP, "ok", ms)
    assert m.quantile_ms(_EP, 0.95) == 1000
    assert m.quantile_ms(_EP, 0.5) == 500


def test_quantile_none_without_samples() -> None:
    assert Metrics().quantile_ms("never.seen", 0.95) is None


def test_endpoints_are_isolated() -> None:
    m = Metrics()
    m.observe("a", "ok", 10)
    m.observe("b", "error", 20)
    assert m.requests("a") == 1 and m.errors("a") == 0
    assert m.requests("b") == 1 and m.errors("b") == 1


def test_slo_within_budget_is_ok() -> None:
    m = Metrics()
    for _ in range(99):
        m.observe(_EP, "ok", 500)
    m.observe(_EP, "error", 500)  # 1% errors
    slo = EndpointSLO(p95_ms=1000, p99_ms=2000, error_budget=0.02)
    status = m.slo_status(_EP, slo)
    assert status.ok is True
    assert status.reasons == ()


def test_slo_error_budget_breach() -> None:
    m = Metrics()
    m.observe(_EP, "ok", 100)
    m.observe(_EP, "error", 100)  # 50% errors
    slo = EndpointSLO(p95_ms=1000, p99_ms=2000, error_budget=0.02)
    status = m.slo_status(_EP, slo)
    assert status.ok is False
    assert any("error" in r for r in status.reasons)


def test_slo_latency_breach() -> None:
    m = Metrics()
    for _ in range(100):
        m.observe(_EP, "ok", 5000)  # all slow
    slo = EndpointSLO(p95_ms=1000, p99_ms=2000, error_budget=0.5)
    status = m.slo_status(_EP, slo)
    assert status.ok is False
    assert any("p95" in r for r in status.reasons)


@pytest.mark.parametrize("bad", ["weird", "OK", ""])
def test_unknown_outcome_rejected(bad: str) -> None:
    with pytest.raises(MetricsError):
        Metrics().observe(_EP, bad, 100)


@pytest.mark.parametrize("bad", [-1, -0.5])
def test_negative_latency_rejected(bad: float) -> None:
    with pytest.raises(MetricsError):
        Metrics().observe(_EP, "ok", bad)


def test_snapshot_shape() -> None:
    m = Metrics()
    m.observe(_EP, "ok", 100)
    m.observe(_EP, "error", 200)
    snap = m.snapshot()
    assert snap[_EP]["requests"] == 2
    assert snap[_EP]["errors"] == 1
