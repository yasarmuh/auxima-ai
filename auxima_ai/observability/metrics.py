"""In-process RED metrics registry + SLO-breach predicate (S-19 §3.3 / R1, R6).

The RED triad, per endpoint:
  - **R**equests: total observed requests.
  - **E**rrors: requests whose outcome is an error class
    (``error`` / ``provider_error`` / ``timeout`` per S-19 R3; ``ok`` and
    ``redaction_required`` are NOT errors).
  - **D**uration: latency samples → nearest-rank p95/p99.

Plus :meth:`Metrics.slo_status` — does an endpoint currently breach its
per-endpoint SLO (p95/p99 latency, error budget)?

Pure + in-process, stdlib only — NO prometheus dependency (prod scrapes a real
histogram; this registry is the dev/test surface and the SLO predicate). Bounded
memory: per-endpoint latency samples are a ring buffer (``maxlen``); over the
cap, p95/p99 are computed over the most recent window. Thread-safe.

NOT in scope (flagged): the ``@observed`` route decorator wiring (S-19 §3.3
example), Prometheus recording rules, and the Azure Monitor sink — those land
with the endpoint rollout. This ships the registry contract the wiring will use.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Final

#: S-19 R3 outcome vocabulary.
_VALID_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"ok", "error", "provider_error", "timeout", "redaction_required"}
)
#: Outcomes counted as errors for the RED 'E' (S-19 R1: 5xx + provider errors).
_ERROR_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"error", "provider_error", "timeout"}
)
#: Ring-buffer size for per-endpoint latency samples.
DEFAULT_SAMPLE_CAP: Final[int] = 10_000


class MetricsError(ValueError):
    """Invalid metric input (unknown outcome, negative latency, bad endpoint)."""


@dataclass(frozen=True)
class EndpointSLO:
    """Per-endpoint SLO (S-19 §3.2). Latencies in ms; budget as a fraction."""

    p95_ms: float
    p99_ms: float
    error_budget: float  # e.g. 0.02 = 2% errors allowed


@dataclass(frozen=True)
class SLOStatus:
    """Result of an SLO check. ``ok`` is True iff no dimension breached."""

    endpoint: str
    ok: bool
    reasons: tuple[str, ...]


@dataclass
class _EndpointStat:
    requests: int = 0
    errors: int = 0
    latencies_ms: deque = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = deque(maxlen=DEFAULT_SAMPLE_CAP)


class Metrics:
    """Thread-safe in-process RED registry keyed by endpoint."""

    def __init__(self, sample_cap: int = DEFAULT_SAMPLE_CAP) -> None:
        self._cap = sample_cap
        self._stats: dict[str, _EndpointStat] = {}
        self._lock = threading.Lock()

    def observe(self, endpoint: str, outcome: str, latency_ms: float) -> None:
        """Record one request's outcome + latency for ``endpoint``."""
        if not isinstance(endpoint, str) or not endpoint:
            raise MetricsError("endpoint must be a non-empty string")
        if outcome not in _VALID_OUTCOMES:
            raise MetricsError(
                f"outcome must be one of {sorted(_VALID_OUTCOMES)}; got {outcome!r}"
            )
        if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool):
            raise MetricsError("latency_ms must be a number")
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise MetricsError(f"latency_ms must be finite and >= 0; got {latency_ms!r}")

        with self._lock:
            stat = self._stats.get(endpoint)
            if stat is None:
                stat = _EndpointStat(latencies_ms=deque(maxlen=self._cap))
                self._stats[endpoint] = stat
            stat.requests += 1
            if outcome in _ERROR_OUTCOMES:
                stat.errors += 1
            stat.latencies_ms.append(float(latency_ms))

    def requests(self, endpoint: str) -> int:
        with self._lock:
            stat = self._stats.get(endpoint)
            return stat.requests if stat else 0

    def errors(self, endpoint: str) -> int:
        with self._lock:
            stat = self._stats.get(endpoint)
            return stat.errors if stat else 0

    def error_rate(self, endpoint: str) -> float:
        """Errors / requests; 0.0 when no requests (no requests = no breach)."""
        with self._lock:
            stat = self._stats.get(endpoint)
            if not stat or stat.requests == 0:
                return 0.0
            return stat.errors / stat.requests

    def quantile_ms(self, endpoint: str, q: float) -> float | None:
        """Nearest-rank percentile of recorded latencies; None if no samples.

        ``q`` is a fraction in (0, 1] (e.g. 0.95). Nearest-rank: the smallest
        sample at or above the ceil(q*n)-th position.
        """
        if not (0.0 < q <= 1.0):
            raise MetricsError(f"q must be in (0, 1]; got {q}")
        with self._lock:
            stat = self._stats.get(endpoint)
            if not stat or not stat.latencies_ms:
                return None
            ordered = sorted(stat.latencies_ms)
        rank = math.ceil(q * len(ordered))
        return ordered[max(rank - 1, 0)]

    def slo_status(self, endpoint: str, slo: EndpointSLO) -> SLOStatus:
        """Check the endpoint against its SLO; list every breached dimension."""
        reasons: list[str] = []

        rate = self.error_rate(endpoint)
        if rate > slo.error_budget:
            reasons.append(
                f"error_rate {rate:.4f} > budget {slo.error_budget:.4f}"
            )
        p95 = self.quantile_ms(endpoint, 0.95)
        if p95 is not None and p95 > slo.p95_ms:
            reasons.append(f"p95 {p95:.0f}ms > {slo.p95_ms:.0f}ms")
        p99 = self.quantile_ms(endpoint, 0.99)
        if p99 is not None and p99 > slo.p99_ms:
            reasons.append(f"p99 {p99:.0f}ms > {slo.p99_ms:.0f}ms")

        return SLOStatus(endpoint=endpoint, ok=not reasons, reasons=tuple(reasons))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """A read-only view of current counters per endpoint (diagnostics)."""
        with self._lock:
            return {
                ep: {
                    "requests": s.requests,
                    "errors": s.errors,
                    "samples": len(s.latencies_ms),
                }
                for ep, s in self._stats.items()
            }


__all__ = (
    "DEFAULT_SAMPLE_CAP",
    "EndpointSLO",
    "Metrics",
    "MetricsError",
    "SLOStatus",
)
