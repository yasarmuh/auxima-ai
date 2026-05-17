"""W3C Trace Context parser (S-19 trace propagation, S-43 §6).

Parses + formats the ``traceparent`` HTTP header per the W3C Trace
Context specification (https://www.w3.org/TR/trace-context/) so the
sidecar can:

  1. Accept inbound ``traceparent`` headers from Frappe server scripts
     and propagate the trace_id / span_id into structured log events.
  2. Inject a fresh ``traceparent`` header into outbound LiteLLM /
     webhook calls so downstream systems continue the trace.

Wire format (version 00 — the only version this implementation accepts):

    traceparent: 00-{trace_id}-{parent_id}-{flags}

      version    = 2 hex chars      (always "00")
      trace_id   = 32 hex chars     (16 bytes; all-zero is invalid)
      parent_id  = 16 hex chars     (8 bytes; all-zero is invalid; a.k.a. span_id)
      flags      = 2 hex chars      ("00" = not sampled, "01" = sampled)

Per the spec §3.2, **malformed traceparent headers must NOT propagate
nor cause an error** — the receiver simply discards them and starts a
new trace. We mirror that: :func:`parse_traceparent` returns ``None``
on any failure (header missing, version unknown, wrong field counts,
non-hex chars, invalid trace_id / span_id zeros, bad length).

This module is pure stdlib (``re`` + ``secrets``); no FastAPI / Frappe.
"""
from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from typing import Callable, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — pin field lengths from the spec
# ---------------------------------------------------------------------------

VERSION_SUPPORTED: Final[str] = "00"

TRACE_ID_HEX_LEN: Final[int] = 32  # 16 bytes
SPAN_ID_HEX_LEN: Final[int] = 16   # 8 bytes
FLAGS_HEX_LEN: Final[int] = 2      # 1 byte

# Per spec §3.2.2.3, sampled flag is the LSB.
FLAG_SAMPLED: Final[int] = 0x01

_INVALID_TRACE_ID: Final[str] = "0" * TRACE_ID_HEX_LEN
_INVALID_SPAN_ID: Final[str] = "0" * SPAN_ID_HEX_LEN

_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]+$")

TRACEPARENT_HEADER: Final[str] = "traceparent"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceContext:
    """The three fields the receiver needs from the traceparent header.

    All ids are stored as lowercase hex strings (the wire form), so
    structured log events can record them directly without conversion.
    """

    trace_id: str
    span_id: str
    flags: int

    @property
    def sampled(self) -> bool:
        """``True`` iff the sampled flag (0x01) is set."""
        return bool(self.flags & FLAG_SAMPLED)

    def format_header(self) -> str:
        """Render the v00 traceparent wire format."""
        return f"{VERSION_SUPPORTED}-{self.trace_id}-{self.span_id}-{self.flags:02x}"

    def with_new_span_id(self, *, rng: Callable[[int], bytes] = secrets.token_bytes) -> "TraceContext":
        """Return a copy with a freshly generated span_id (same trace_id + flags).

        Used when propagating the trace downstream — the caller becomes
        the parent span; the downstream call gets a new span id but the
        trace_id stays the same so the whole call tree links up.
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=new_span_id(rng=rng),
            flags=self.flags,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_lowercase_hex(s: str, expected_len: int) -> bool:
    if len(s) != expected_len:
        return False
    return bool(_HEX_RE.match(s))


def new_trace_id(*, rng: Callable[[int], bytes] = secrets.token_bytes) -> str:
    """Generate a fresh 16-byte trace_id as a 32-char lowercase hex string."""
    return rng(16).hex()


def new_span_id(*, rng: Callable[[int], bytes] = secrets.token_bytes) -> str:
    """Generate a fresh 8-byte span_id as a 16-char lowercase hex string."""
    return rng(8).hex()


def new_context(
    *,
    sampled: bool = True,
    rng: Callable[[int], bytes] = secrets.token_bytes,
) -> TraceContext:
    """Create a brand-new trace context (root span)."""
    return TraceContext(
        trace_id=new_trace_id(rng=rng),
        span_id=new_span_id(rng=rng),
        flags=FLAG_SAMPLED if sampled else 0,
    )


# ---------------------------------------------------------------------------
# Parsing — strict + spec-faithful (returns None on any malformedness)
# ---------------------------------------------------------------------------


def parse_traceparent(header_value: str | None) -> TraceContext | None:
    """Parse a ``traceparent`` header value into a :class:`TraceContext`.

    Returns ``None`` if the header is missing or any part of the v00 wire
    format is invalid — per the W3C spec §3.2 we must NOT raise on
    malformed input; the caller should start a fresh trace instead.

    Accepts only version ``00`` (the only version currently defined by
    the spec). Future versions can be added as separate parsers.
    """
    if header_value is None:
        return None
    if not isinstance(header_value, str):
        # Defensive — if a framework hands us bytes / a list, just discard.
        logger.debug("traceparent header is not a str: %r", type(header_value).__name__)
        return None

    value = header_value.strip()
    if not value:
        return None

    parts = value.split("-")
    if len(parts) != 4:
        logger.debug("traceparent has %d fields (want 4): %r", len(parts), value)
        return None

    version, trace_id, span_id, flags_hex = parts

    # Version must be exactly the supported one. (Spec §3.2.2.1 — unknown
    # versions are allowed to extend the format with extra trailing
    # fields, but until we *know* a future version we treat anything but
    # "00" as malformed.)
    if version != VERSION_SUPPORTED:
        logger.debug("traceparent version %r not supported", version)
        return None

    if not _is_lowercase_hex(trace_id, TRACE_ID_HEX_LEN):
        logger.debug("traceparent trace_id malformed: %r", trace_id)
        return None
    if trace_id == _INVALID_TRACE_ID:
        logger.debug("traceparent trace_id is all-zero (invalid per spec)")
        return None

    if not _is_lowercase_hex(span_id, SPAN_ID_HEX_LEN):
        logger.debug("traceparent span_id malformed: %r", span_id)
        return None
    if span_id == _INVALID_SPAN_ID:
        logger.debug("traceparent span_id is all-zero (invalid per spec)")
        return None

    if not _is_lowercase_hex(flags_hex, FLAGS_HEX_LEN):
        logger.debug("traceparent flags malformed: %r", flags_hex)
        return None
    flags = int(flags_hex, 16)

    return TraceContext(trace_id=trace_id, span_id=span_id, flags=flags)


__all__ = (
    "FLAG_SAMPLED",
    "SPAN_ID_HEX_LEN",
    "TRACE_ID_HEX_LEN",
    "TRACEPARENT_HEADER",
    "TraceContext",
    "VERSION_SUPPORTED",
    "new_context",
    "new_span_id",
    "new_trace_id",
    "parse_traceparent",
)
