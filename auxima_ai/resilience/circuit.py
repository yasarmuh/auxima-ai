"""Three-state circuit breaker for outbound dependency calls.

States and transitions:

    CLOSED  -- failure_threshold consecutive failures -->  OPEN
    OPEN    -- cooldown_seconds elapsed              -->  HALF_OPEN
    HALF_OPEN -- success                             -->  CLOSED  (failure counter resets)
    HALF_OPEN -- any failure                         -->  OPEN    (cooldown re-armed)

In ``CLOSED`` the breaker passes traffic through and counts failures.
In ``OPEN`` it fails fast — :meth:`try_call` returns :class:`RejectOpen`
with the seconds-until-half-open hint so callers can return a clean
HTTP 503 + Retry-After.

``HALF_OPEN`` is the probe window: at most ``half_open_max_calls``
in-flight test calls are admitted; the next outcome decides whether the
dependency is healthy again. Bounding the probe traffic is the point —
without it, the moment the breaker half-opens we'd send the full thunder
of paused traffic through the still-fragile dependency and likely re-
break it. With it, a small number of canaries decide before the herd is
let through.

Pure stdlib (``threading`` + ``time``); thread-safe via per-breaker
Lock; injectable monotonic clock for deterministic tests.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CircuitError(ValueError):
    """Base — invalid configuration raises a subclass of this."""


# ---------------------------------------------------------------------------
# State + policy
# ---------------------------------------------------------------------------


class State(str, Enum):
    """The three legal breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerPolicy:
    """Tuning knobs for the circuit."""

    failure_threshold: int = 5
    """Consecutive failures in CLOSED before flipping to OPEN."""

    cooldown_seconds: float = 30.0
    """How long OPEN waits before flipping to HALF_OPEN to probe."""

    half_open_max_calls: int = 1
    """Concurrent probe calls allowed in HALF_OPEN. Keep small (1-3)."""

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise CircuitError(
                f"failure_threshold must be >= 1; got {self.failure_threshold}"
            )
        if self.cooldown_seconds <= 0:
            raise CircuitError(
                f"cooldown_seconds must be > 0; got {self.cooldown_seconds}"
            )
        if self.half_open_max_calls < 1:
            raise CircuitError(
                f"half_open_max_calls must be >= 1; got {self.half_open_max_calls}"
            )


# ---------------------------------------------------------------------------
# Admission decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Admit:
    """Caller may proceed with the call.

    ``probing`` is ``True`` iff the breaker is HALF_OPEN — the caller
    SHOULD prefer a single probe request (e.g. a health check) over a
    heavy real workload.
    """

    state: State
    probing: bool


@dataclass(frozen=True)
class RejectOpen:
    """OPEN circuit — fail fast; retry after ``retry_after_seconds``."""

    retry_after_seconds: float


@dataclass(frozen=True)
class RejectHalfOpenSaturated:
    """HALF_OPEN circuit and the probe slot is taken — wait briefly + retry."""

    retry_after_seconds: float


AdmissionResult = Admit | RejectOpen | RejectHalfOpenSaturated


# ---------------------------------------------------------------------------
# Breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Three-state breaker. One instance per (dependency, identity) pair.

    The breaker is mutable by design — its whole job is to evolve state
    based on outcomes. All mutation happens under ``_lock``.
    """

    policy: BreakerPolicy = field(default_factory=BreakerPolicy)
    clock: Callable[[], float] = field(default=time.monotonic)
    name: str = "default"
    _state: State = field(default=State.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_in_flight: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # -- introspection -----------------------------------------------------

    @property
    def state(self) -> State:
        """The current state, after any due OPEN -> HALF_OPEN transition."""
        with self._lock:
            self._maybe_transition_open_to_half_open_locked()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    # -- admission ---------------------------------------------------------

    def try_call(self) -> AdmissionResult:
        """Atomically check the state and reserve a slot in HALF_OPEN."""
        with self._lock:
            self._maybe_transition_open_to_half_open_locked()
            if self._state == State.CLOSED:
                return Admit(state=State.CLOSED, probing=False)
            if self._state == State.HALF_OPEN:
                if self._half_open_in_flight >= self.policy.half_open_max_calls:
                    # Probe slot taken — caller retries after a short backoff.
                    return RejectHalfOpenSaturated(
                        retry_after_seconds=self.policy.cooldown_seconds,
                    )
                self._half_open_in_flight += 1
                return Admit(state=State.HALF_OPEN, probing=True)
            # State.OPEN — caller is told how long until we'll probe again.
            remaining = self._seconds_until_half_open_locked()
            return RejectOpen(retry_after_seconds=remaining)

    # -- outcome recording -------------------------------------------------

    def record_success(self) -> None:
        """Record one successful outcome; transitions HALF_OPEN -> CLOSED."""
        with self._lock:
            if self._state == State.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                logger.info(
                    "breaker %r: HALF_OPEN probe succeeded -> CLOSED", self.name,
                )
                self._state = State.CLOSED
                self._consecutive_failures = 0
                return
            if self._state == State.CLOSED:
                self._consecutive_failures = 0
                return
            # OPEN: success reported (shouldn't happen if callers respect
            # the breaker, but be defensive) — treat as no-op.
            logger.warning(
                "breaker %r: record_success called while OPEN — ignored", self.name,
            )

    def record_failure(self) -> None:
        """Record one failed outcome; may transition CLOSED/HALF_OPEN -> OPEN."""
        with self._lock:
            if self._state == State.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                logger.warning(
                    "breaker %r: HALF_OPEN probe failed -> OPEN", self.name,
                )
                self._open_locked()
                return
            if self._state == State.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.policy.failure_threshold:
                    logger.warning(
                        "breaker %r: %d consecutive failures -> OPEN",
                        self.name, self._consecutive_failures,
                    )
                    self._open_locked()
                return
            # Already OPEN — no-op (extra failure doesn't extend cooldown
            # because the cooldown is measured from the original transition).

    # -- ops / test helpers ------------------------------------------------

    def reset(self) -> None:
        """Force the breaker back to CLOSED. Ops-only — for incident recovery."""
        with self._lock:
            self._state = State.CLOSED
            self._consecutive_failures = 0
            self._opened_at = 0.0
            self._half_open_in_flight = 0
            logger.info("breaker %r: manually reset to CLOSED", self.name)

    # -- internal ----------------------------------------------------------

    def _open_locked(self) -> None:
        self._state = State.OPEN
        self._opened_at = self.clock()
        self._half_open_in_flight = 0

    def _seconds_until_half_open_locked(self) -> float:
        elapsed = self.clock() - self._opened_at
        return max(0.0, self.policy.cooldown_seconds - elapsed)

    def _maybe_transition_open_to_half_open_locked(self) -> None:
        if self._state != State.OPEN:
            return
        if self._seconds_until_half_open_locked() <= 0:
            self._state = State.HALF_OPEN
            self._half_open_in_flight = 0
            logger.info(
                "breaker %r: cooldown elapsed -> HALF_OPEN", self.name,
            )


__all__ = (
    "Admit",
    "AdmissionResult",
    "BreakerPolicy",
    "CircuitBreaker",
    "CircuitError",
    "RejectHalfOpenSaturated",
    "RejectOpen",
    "State",
)
