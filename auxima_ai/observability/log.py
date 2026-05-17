"""Structured log-event emitter for the auxima-ai sidecar.

Per S-19 §3.4 (structured-log shape) + §3.5 (PII redaction on payloads):

  - Every event is a single JSON object with a fixed envelope:
      ``ts`` (UTC ISO-8601 with milliseconds + ``Z`` suffix)
      ``level`` (``debug`` / ``info`` / ``warn`` / ``error`` — lowercase)
      ``event`` (dotted name, e.g. ``intake.extract.completed``)
      ``trace_id`` / ``span_id`` (W3C Trace Context — may be ``None``)
      ``redaction_required`` (``bool`` — ``True`` iff the redactor fired
       on any field value)
      ``fields`` (nested dict, every string leaf already PII-redacted)
  - The serialised line is emitted through stdlib ``logging`` so the
    deployment can swap handlers (stdout / syslog / OTel) without
    touching call sites.
  - The function returns the dict it emitted so unit tests can assert
    on the structure without parsing stderr.

This module is pure-Python: no FastAPI, no Frappe, no third-party deps
beyond the in-package :mod:`auxima_ai.observability.redact` and stdlib
``logging`` / ``json`` / ``datetime``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Literal

from auxima_ai.observability.redact import redact_json

LogLevel = Literal["debug", "info", "warn", "error"]
_VALID_LEVELS: Final[frozenset[str]] = frozenset({"debug", "info", "warn", "error"})

_LEVEL_TO_STDLIB: Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}

# Dedicated logger so deployments can route structured events separately
# from arbitrary stdlib logging chatter (e.g. uvicorn access logs).
EVENT_LOGGER_NAME: Final[str] = "auxima_ai.events"
_event_logger: Final[logging.Logger] = logging.getLogger(EVENT_LOGGER_NAME)


class LogEventError(ValueError):
    """Raised when an event cannot be constructed (bad level, bad name, etc.)."""


@dataclass(frozen=True)
class LogEvent:
    """The in-memory shape of one emitted event.

    Frozen by intent — events are append-only facts; once emitted they must
    not be mutated. Callers that want to "decorate" an event should construct
    a fresh one with the additional fields.
    """

    ts: str
    level: LogLevel
    event: str
    trace_id: str | None
    span_id: str | None
    redaction_required: bool
    fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict matching the wire envelope."""
        return {
            "ts": self.ts,
            "level": self.level,
            "event": self.event,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "redaction_required": self.redaction_required,
            "fields": self.fields,
        }

    def to_json(self) -> str:
        """Compact JSON serialisation — no spaces, sorted keys for determinism."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True, default=str)


def _utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 millisecond-precision + Z."""
    # datetime.now(tz=UTC) is deterministic across platforms; isoformat() with
    # timespec="milliseconds" + .replace("+00:00", "Z") gives the wire shape.
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _validate_event_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise LogEventError(f"event must be a non-empty string; got {name!r}")
    # Dotted lowercase names — keeps the namespace tidy + greps cleanly.
    for ch in name:
        if not (ch.isalnum() or ch in "._-"):
            raise LogEventError(
                f"event must contain only [a-zA-Z0-9._-]; got {name!r}"
            )


def emit(
    level: LogLevel,
    event: str,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    fields: dict[str, Any] | None = None,
    clock: callable = _utc_now_iso,  # type: ignore[valid-type]
) -> LogEvent:
    """Build, redact, log, and return one structured event.

    Parameters
    ----------
    level
        One of ``"debug" | "info" | "warn" | "error"``.
    event
        Dotted lowercase event name; e.g. ``"intake.extract.completed"``.
    trace_id, span_id
        Optional W3C Trace Context identifiers; pass ``None`` when no
        trace is active. The caller is responsible for propagation —
        this function only records the IDs.
    fields
        Arbitrary JSON-shaped payload. Every string leaf is passed through
        :func:`auxima_ai.observability.redact.redact_json` before serialisation;
        the ``redaction_required`` flag is ``True`` iff any leaf was modified.
        ``None`` is treated as an empty dict.
    clock
        Injectable wall-clock — overridable in tests so timestamps are
        deterministic.

    Returns
    -------
    LogEvent
        The frozen event that was emitted. The same object is what gets
        serialised to the configured handler.

    Raises
    ------
    LogEventError
        On invalid level or invalid event name.

    Notes
    -----
    - The function never raises on PII content; the redactor's pass-through
      contract handles unexpected leaf types defensively.
    - Logging side-effects use stdlib :mod:`logging`; capture-via-``caplog``
      works for tests.
    """
    if level not in _VALID_LEVELS:
        raise LogEventError(
            f"level must be one of {sorted(_VALID_LEVELS)}; got {level!r}"
        )
    _validate_event_name(event)

    raw_fields = fields if fields is not None else {}
    if not isinstance(raw_fields, dict):
        raise LogEventError(
            f"fields must be a dict (or None); got {type(raw_fields).__name__}"
        )

    redacted_fields, fired = redact_json(raw_fields)

    evt = LogEvent(
        ts=clock(),
        level=level,
        event=event,
        trace_id=trace_id,
        span_id=span_id,
        redaction_required=bool(fired),
        fields=redacted_fields,
    )

    # Route through stdlib logging so handlers (stdout / OTel / syslog) plug in
    # at deployment time without touching call sites. We pass the JSON line as
    # the message body — handlers that want the structured dict can re-parse,
    # or we can attach `extra={"event": evt.to_dict()}` later.
    _event_logger.log(_LEVEL_TO_STDLIB[level], evt.to_json())
    return evt


__all__ = (
    "EVENT_LOGGER_NAME",
    "LogEvent",
    "LogEventError",
    "LogLevel",
    "emit",
)
