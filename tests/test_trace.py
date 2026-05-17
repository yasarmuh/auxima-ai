"""Tests for ``auxima_ai.observability.trace`` — W3C traceparent parser.

Spec: https://www.w3.org/TR/trace-context/

Coverage:
  - Happy path parses a well-formed v00 header.
  - All-zero trace_id / span_id are rejected.
  - Uppercase hex is rejected (spec mandates lowercase).
  - Wrong lengths (trace_id != 32, span_id != 16, flags != 2) are rejected.
  - Non-hex characters are rejected.
  - Unsupported version is rejected (we accept only "00").
  - Missing fields (less than 4 parts) are rejected.
  - Extra fields (more than 4 parts) are rejected for v00.
  - None / empty / non-string input returns None (NEVER raises — spec §3.2).
  - Whitespace tolerated on both ends.
  - format_header round-trips with parse_traceparent.
  - flags=0x01 -> sampled is True; flags=0x00 -> sampled is False.
  - with_new_span_id preserves trace_id + flags.
  - new_trace_id / new_span_id length + hex shape.
  - new_context produces a parseable header.
"""
from __future__ import annotations

import pytest

from auxima_ai.observability.trace import (
    FLAG_SAMPLED,
    SPAN_ID_HEX_LEN,
    TRACE_ID_HEX_LEN,
    TraceContext,
    new_context,
    new_span_id,
    new_trace_id,
    parse_traceparent,
)

# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------

VALID_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"  # 32 hex
VALID_SPAN_ID = "b7ad6b7169203331"  # 16 hex
VALID_HEADER_SAMPLED = f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01"
VALID_HEADER_UNSAMPLED = f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parse_valid_sampled_header() -> None:
    ctx = parse_traceparent(VALID_HEADER_SAMPLED)
    assert isinstance(ctx, TraceContext)
    assert ctx.trace_id == VALID_TRACE_ID
    assert ctx.span_id == VALID_SPAN_ID
    assert ctx.flags == FLAG_SAMPLED
    assert ctx.sampled is True


def test_parse_valid_unsampled_header() -> None:
    ctx = parse_traceparent(VALID_HEADER_UNSAMPLED)
    assert ctx is not None
    assert ctx.flags == 0
    assert ctx.sampled is False


def test_parse_tolerates_surrounding_whitespace() -> None:
    ctx = parse_traceparent(f"  {VALID_HEADER_SAMPLED}\n")
    assert ctx is not None
    assert ctx.trace_id == VALID_TRACE_ID


# ---------------------------------------------------------------------------
# Spec §3.2: never raise on malformed input — return None instead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        # None / missing
        None,
        "",
        "   ",
        # Wrong number of fields
        "00",
        f"00-{VALID_TRACE_ID}",
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}",
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01-extra-field",
        # Unsupported version
        f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01",
        f"ff-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01",
        # All-zero trace_id (spec §3.2.2.2 — invalid)
        f"00-{'0' * 32}-{VALID_SPAN_ID}-01",
        # All-zero span_id (spec §3.2.2.3 — invalid)
        f"00-{VALID_TRACE_ID}-{'0' * 16}-01",
        # Trace_id wrong length
        f"00-{VALID_TRACE_ID[:-1]}-{VALID_SPAN_ID}-01",
        f"00-{VALID_TRACE_ID}ff-{VALID_SPAN_ID}-01",
        # Span_id wrong length
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID[:-1]}-01",
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}ff-01",
        # Flags wrong length
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-1",
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-001",
        # Uppercase hex (spec mandates lowercase)
        f"00-{VALID_TRACE_ID.upper()}-{VALID_SPAN_ID}-01",
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID.upper()}-01",
        # Non-hex characters
        f"00-{'g' * 32}-{VALID_SPAN_ID}-01",
        f"00-{VALID_TRACE_ID}-{'z' * 16}-01",
        f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-gg",
        # Wrong separator
        f"00.{VALID_TRACE_ID}.{VALID_SPAN_ID}.01",
        f"00 {VALID_TRACE_ID} {VALID_SPAN_ID} 01",
    ],
)
def test_parse_returns_none_on_malformed(header: object) -> None:
    """Malformed traceparents must not raise — they're silently discarded."""
    assert parse_traceparent(header) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("non_str", [42, [1, 2], b"00-..."])
def test_parse_returns_none_on_non_string_input(non_str: object) -> None:
    assert parse_traceparent(non_str) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Format + round-trip
# ---------------------------------------------------------------------------


def test_format_header_round_trip_sampled() -> None:
    ctx = TraceContext(VALID_TRACE_ID, VALID_SPAN_ID, FLAG_SAMPLED)
    s = ctx.format_header()
    assert s == VALID_HEADER_SAMPLED
    round_tripped = parse_traceparent(s)
    assert round_tripped == ctx


def test_format_header_round_trip_unsampled() -> None:
    ctx = TraceContext(VALID_TRACE_ID, VALID_SPAN_ID, 0)
    s = ctx.format_header()
    assert s == VALID_HEADER_UNSAMPLED
    assert parse_traceparent(s) == ctx


def test_format_pads_flags_to_two_hex_chars() -> None:
    ctx = TraceContext(VALID_TRACE_ID, VALID_SPAN_ID, 0)
    assert ctx.format_header().endswith("-00")


# ---------------------------------------------------------------------------
# with_new_span_id — propagation helper
# ---------------------------------------------------------------------------


def test_with_new_span_id_preserves_trace_id_and_flags() -> None:
    ctx = TraceContext(VALID_TRACE_ID, VALID_SPAN_ID, FLAG_SAMPLED)
    child = ctx.with_new_span_id()
    assert child.trace_id == ctx.trace_id
    assert child.flags == ctx.flags
    assert child.span_id != ctx.span_id
    # span_id is fresh + well-formed
    assert len(child.span_id) == SPAN_ID_HEX_LEN
    assert int(child.span_id, 16) >= 0


def test_with_new_span_id_uses_injected_rng() -> None:
    ctx = TraceContext(VALID_TRACE_ID, VALID_SPAN_ID, FLAG_SAMPLED)
    fake_bytes = bytes.fromhex("aaaaaaaaaaaaaaaa")  # 8 bytes -> 16 hex chars
    child = ctx.with_new_span_id(rng=lambda n: fake_bytes[:n])
    assert child.span_id == "aaaaaaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# Id generators
# ---------------------------------------------------------------------------


def test_new_trace_id_shape() -> None:
    tid = new_trace_id()
    assert len(tid) == TRACE_ID_HEX_LEN
    assert tid == tid.lower()
    int(tid, 16)  # parses as hex


def test_new_span_id_shape() -> None:
    sid = new_span_id()
    assert len(sid) == SPAN_ID_HEX_LEN
    assert sid == sid.lower()
    int(sid, 16)


def test_new_context_produces_parseable_header() -> None:
    ctx = new_context()
    parsed = parse_traceparent(ctx.format_header())
    assert parsed == ctx
    assert ctx.sampled is True


def test_new_context_unsampled() -> None:
    ctx = new_context(sampled=False)
    assert ctx.flags == 0
    assert ctx.sampled is False


def test_new_context_uses_injected_rng() -> None:
    """Inject a deterministic RNG and verify the trace + span ids match."""
    seq = iter([
        bytes.fromhex("11" * 16),  # trace_id source
        bytes.fromhex("22" * 8),   # span_id source
    ])
    ctx = new_context(rng=lambda n: next(seq))
    assert ctx.trace_id == "11" * 16
    assert ctx.span_id == "22" * 8


# ---------------------------------------------------------------------------
# Frozen invariant
# ---------------------------------------------------------------------------


def test_trace_context_is_frozen() -> None:
    ctx = parse_traceparent(VALID_HEADER_SAMPLED)
    assert ctx is not None
    with pytest.raises((AttributeError, TypeError)):
        ctx.trace_id = "tampered"  # type: ignore[misc]
