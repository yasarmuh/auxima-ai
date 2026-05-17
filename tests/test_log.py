"""Tests for ``auxima_ai.observability.log`` (structured-event emitter).

Coverage per S-19 §3.4:
  - Envelope shape: ts, level, event, trace_id, span_id, redaction_required, fields.
  - PII in fields is redacted; ``redaction_required`` flag is True iff fired.
  - Clean fields leave ``redaction_required`` False.
  - Timestamp is ISO-8601 UTC + millisecond precision + Z suffix.
  - JSON serialisation is deterministic (sorted keys, compact separators).
  - LogEvent is frozen / immutable.
  - The stdlib logger receives the serialised JSON line at the correct level.
  - Validation: bad level / bad event name / non-dict fields all raise.
  - Injectable clock for deterministic tests.
"""
from __future__ import annotations

import json
import logging
import re

import pytest

from auxima_ai.observability.log import (
    EVENT_LOGGER_NAME,
    LogEvent,
    LogEventError,
    emit,
)

# ---------------------------------------------------------------------------
# Happy path — envelope shape
# ---------------------------------------------------------------------------

FIXED_TS = "2026-05-17T20:00:00.000Z"


def fixed_clock() -> str:
    return FIXED_TS


def test_emit_returns_logevent_with_full_envelope() -> None:
    evt = emit(
        "info",
        "intake.extract.completed",
        trace_id="abc-trace",
        span_id="xyz-span",
        fields={"customer_id": "CUST-1", "tokens": 412},
        clock=fixed_clock,
    )
    assert isinstance(evt, LogEvent)
    assert evt.ts == FIXED_TS
    assert evt.level == "info"
    assert evt.event == "intake.extract.completed"
    assert evt.trace_id == "abc-trace"
    assert evt.span_id == "xyz-span"
    assert evt.redaction_required is False
    assert evt.fields == {"customer_id": "CUST-1", "tokens": 412}


def test_emit_with_no_fields_treats_as_empty() -> None:
    evt = emit("info", "ping", clock=fixed_clock)
    assert evt.fields == {}
    assert evt.redaction_required is False
    assert evt.trace_id is None
    assert evt.span_id is None


def test_emit_redacts_pii_in_fields() -> None:
    evt = emit(
        "warn",
        "lead.created",
        fields={"email": "lead@acme.sa", "phone": "0512345678", "ok": True},
        clock=fixed_clock,
    )
    assert evt.redaction_required is True
    assert evt.fields["email"] == "<redacted:email>"
    assert evt.fields["phone"] == "<redacted:phone_ksa_local>"
    assert evt.fields["ok"] is True


def test_emit_redacts_nested_pii() -> None:
    evt = emit(
        "info",
        "webhook.delivered",
        fields={"customer": {"email": "x@y.com", "phones": ["+966500000000"]}},
        clock=fixed_clock,
    )
    assert evt.redaction_required is True
    assert evt.fields["customer"]["email"] == "<redacted:email>"
    assert evt.fields["customer"]["phones"][0] == "<redacted:phone_e164>"


def test_emit_redaction_flag_false_when_no_pii() -> None:
    evt = emit("info", "ping", fields={"count": 1, "ok": True, "label": "hello"})
    assert evt.redaction_required is False


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_logevent_to_json_is_deterministic_compact_and_sorted() -> None:
    evt = emit(
        "info",
        "evt.a",
        fields={"b": 1, "a": 2},
        trace_id="t",
        span_id="s",
        clock=fixed_clock,
    )
    s = evt.to_json()
    # No whitespace between key-value separators
    assert " " not in s
    # Keys appear in sorted order at the top level
    parsed = json.loads(s)
    assert list(parsed.keys()) == sorted(parsed.keys())
    # Re-serialising the dict deterministically yields the same string
    assert evt.to_json() == evt.to_json()


def test_logevent_to_dict_matches_to_json_roundtrip() -> None:
    evt = emit("info", "evt.a", fields={"k": "v"}, clock=fixed_clock)
    assert json.loads(evt.to_json()) == evt.to_dict()


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_logevent_is_frozen() -> None:
    evt = emit("info", "evt", clock=fixed_clock)
    with pytest.raises((AttributeError, TypeError)):
        evt.level = "error"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Timestamp format
# ---------------------------------------------------------------------------

_ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_default_clock_returns_iso_utc_ms_z() -> None:
    """The default clock yields an ISO timestamp with ms precision + Z."""
    evt = emit("info", "ping")
    assert _ISO_Z_RE.match(evt.ts), f"unexpected ts format: {evt.ts!r}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_level", ["INFO", "trace", "fatal", "", None, 1])
def test_emit_rejects_bad_level(bad_level: object) -> None:
    with pytest.raises(LogEventError, match="level must be one of"):
        emit(bad_level, "evt", clock=fixed_clock)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_name",
    ["", "has space", "punct!", "with/slash", "tab\there"],
)
def test_emit_rejects_bad_event_name(bad_name: str) -> None:
    with pytest.raises(LogEventError, match="event"):
        emit("info", bad_name, clock=fixed_clock)


@pytest.mark.parametrize("non_str", [None, 42, []])
def test_emit_rejects_non_string_event(non_str: object) -> None:
    with pytest.raises(LogEventError, match="event"):
        emit("info", non_str, clock=fixed_clock)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_fields", [42, "not-a-dict", [1, 2, 3], ("a",)])
def test_emit_rejects_non_dict_fields(bad_fields: object) -> None:
    with pytest.raises(LogEventError, match="fields must be a dict"):
        emit("info", "evt", fields=bad_fields, clock=fixed_clock)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Logger integration — capture the emitted line
# ---------------------------------------------------------------------------


def test_emit_writes_serialised_json_to_event_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER_NAME):
        evt = emit(
            "info",
            "intake.extract.completed",
            fields={"tokens": 412},
            clock=fixed_clock,
        )
    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.levelno == logging.INFO
    assert rec.name == EVENT_LOGGER_NAME
    parsed = json.loads(rec.getMessage())
    assert parsed == evt.to_dict()


def test_emit_routes_warn_to_warning_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=EVENT_LOGGER_NAME):
        emit("warn", "evt", clock=fixed_clock)
    assert caplog.records[0].levelno == logging.WARNING


def test_emit_routes_error_to_error_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=EVENT_LOGGER_NAME):
        emit("error", "evt", clock=fixed_clock)
    assert caplog.records[0].levelno == logging.ERROR


def test_emit_routes_debug_to_debug_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=EVENT_LOGGER_NAME):
        emit("debug", "evt", clock=fixed_clock)
    assert caplog.records[0].levelno == logging.DEBUG


def test_redacted_payload_never_leaks_original_pii_in_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Security property: the serialised log line must NOT contain the original PII."""
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER_NAME):
        emit(
            "info",
            "lead.created",
            fields={"email": "leaked@bad.com", "phone": "0512345678"},
            clock=fixed_clock,
        )
    msg = caplog.records[0].getMessage()
    assert "leaked@bad.com" not in msg, f"SECURITY: PII leaked into log: {msg!r}"
    assert "0512345678" not in msg, f"SECURITY: PII leaked into log: {msg!r}"
    assert "<redacted:email>" in msg
    assert "<redacted:phone_ksa_local>" in msg
