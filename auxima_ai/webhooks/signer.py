"""HMAC-SHA256 webhook signer + verifier (S-34 §3.3).

Implements a Stripe-style ``v1`` signing scheme that protects outbound
webhook payloads against three concrete threats:

  1. **Payload tampering** — the receiver can verify that the body has
     not been modified in transit.
  2. **Replay attacks** — the signature covers a timestamp; receivers
     reject any payload older than ``max_age_seconds`` (default 5 min).
  3. **Forwarded-signature attacks** — the signature is bound to the
     body via a length-prefixed canonical string, so a signature from
     one payload cannot be re-attached to another payload with the same
     timestamp.

Wire format (HTTP headers attached to the outbound POST):

    X-Auxima-Signature-Version: v1
    X-Auxima-Timestamp:         1715986380
    X-Auxima-Signature:         v1=<hex(hmac_sha256(secret, "v1:{ts}:{body}"))>

Verification reproduces the canonical string from the received body +
header timestamp, recomputes the HMAC with the shared secret, and
compares in constant time. A skew window (``max_skew_seconds``, default
60 s) tolerates honest clock drift but rejects timestamps from the far
future (a defense against an attacker fixing a future ``ts`` to extend
the replay window).

This module is **pure stdlib** (``hmac`` + ``hashlib`` + ``time``); no
FastAPI / Frappe / third-party deps. The clock is injectable so tests
are deterministic.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Callable, Final, Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants — wire-format identifiers
# ---------------------------------------------------------------------------

SIGNATURE_VERSION: Final[str] = "v1"
SIGNATURE_PREFIX: Final[str] = f"{SIGNATURE_VERSION}="
HEADER_SIGNATURE: Final[str] = "X-Auxima-Signature"
HEADER_TIMESTAMP: Final[str] = "X-Auxima-Timestamp"
HEADER_VERSION: Final[str] = "X-Auxima-Signature-Version"

DEFAULT_MAX_AGE_SECONDS: Final[int] = 300  # 5 minutes — Stripe parity.
DEFAULT_MAX_SKEW_SECONDS: Final[int] = 60  # accept up to 1 minute clock skew forward.


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SignatureError(ValueError):
    """Base class — any signature failure raises a subclass of this."""


class InvalidSecretError(SignatureError):
    """Raised when the shared secret is empty / not provided (fail closed)."""


class MalformedSignatureError(SignatureError):
    """Raised when the signature header is missing or doesn't parse as ``v1=...``."""


class InvalidTimestampError(SignatureError):
    """Raised when the timestamp header is missing or not a valid integer."""


class ExpiredSignatureError(SignatureError):
    """Raised when the payload is older than ``max_age_seconds`` (replay protection)."""


class FutureTimestampError(SignatureError):
    """Raised when the timestamp is further in the future than ``max_skew_seconds`` allow."""


class SignatureMismatchError(SignatureError):
    """Raised when the recomputed signature doesn't match the provided one."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedHeaders:
    """The three headers a sender attaches to the outbound webhook."""

    signature: str
    timestamp: int
    version: str = SIGNATURE_VERSION

    def as_dict(self) -> dict[str, str]:
        """Return as a plain ``dict[str, str]`` ready to merge into request headers."""
        return {
            HEADER_SIGNATURE: self.signature,
            HEADER_TIMESTAMP: str(self.timestamp),
            HEADER_VERSION: self.version,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_body(body: bytes | str) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    raise SignatureError(
        f"body must be bytes / bytearray / str; got {type(body).__name__}"
    )


def _validate_secret(secret: str) -> bytes:
    if not isinstance(secret, str):
        raise InvalidSecretError(
            f"secret must be str; got {type(secret).__name__}"
        )
    stripped = secret.strip()
    if not stripped:
        raise InvalidSecretError("secret must be a non-empty string")
    return stripped.encode("utf-8")


def _canonical_string(timestamp: int, body: bytes) -> bytes:
    """``v1:{timestamp}:{body}`` — the HMAC payload.

    The version prefix lets us roll forward to ``v2`` (e.g. SHA3 or
    BLAKE3) without breaking deployed verifiers, by negotiating the
    version via the X-Auxima-Signature-Version header.
    """
    return f"{SIGNATURE_VERSION}:{timestamp}:".encode("utf-8") + body


def _compute_signature(secret_bytes: bytes, timestamp: int, body: bytes) -> str:
    digest = hmac.new(
        secret_bytes,
        _canonical_string(timestamp, body),
        hashlib.sha256,
    ).hexdigest()
    return f"{SIGNATURE_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sign(
    body: bytes | str,
    secret: str,
    *,
    timestamp: int | None = None,
    clock: Callable[[], float] = time.time,
) -> SignedHeaders:
    """Compute the v1 signature headers for an outbound webhook payload.

    Parameters
    ----------
    body
        The HTTP request body. ``str`` inputs are UTF-8 encoded; ``bytes``
        / ``bytearray`` are used as-is. The signature covers the exact
        bytes that will be sent on the wire.
    secret
        The shared HMAC secret. Empty / whitespace-only secrets raise
        :class:`InvalidSecretError` (fail closed — no silent unauth send).
    timestamp
        Override the wall-clock timestamp (seconds since epoch). Useful
        for tests; defaults to ``int(clock())``.
    clock
        Injectable clock; defaults to :func:`time.time`. Returns float
        seconds; coerced to ``int``.

    Returns
    -------
    SignedHeaders
        Frozen dataclass with the three header values. Use
        :meth:`SignedHeaders.as_dict` to merge into outbound request
        headers.

    Raises
    ------
    InvalidSecretError
        Secret is empty / not a string.
    SignatureError
        Body is not bytes/bytearray/str.
    """
    secret_bytes = _validate_secret(secret)
    body_bytes = _coerce_body(body)
    ts = int(timestamp) if timestamp is not None else int(clock())
    signature = _compute_signature(secret_bytes, ts, body_bytes)
    return SignedHeaders(signature=signature, timestamp=ts)


def verify(
    body: bytes | str,
    headers: Mapping[str, str],
    secret: str,
    *,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    clock: Callable[[], float] = time.time,
) -> None:
    """Verify a webhook signature; raise on any failure.

    Parameters
    ----------
    body
        The received body (bytes preferred; str is UTF-8 encoded). Must
        match the bytes that were signed — no whitespace / charset
        normalisation is applied.
    headers
        The received HTTP headers; the lookup is case-sensitive on the
        keys defined at the top of this module
        (``X-Auxima-Signature`` / ``X-Auxima-Timestamp``). Callers that
        receive case-insensitive headers should normalise before calling.
    secret
        The shared HMAC secret. Empty / whitespace-only secrets raise
        :class:`InvalidSecretError` (fail closed).
    max_age_seconds
        Reject signatures whose timestamp is more than this many seconds
        in the past. Default 300 s (5 min) — Stripe parity. Set 0 to
        disable the age check (NOT recommended).
    max_skew_seconds
        Reject signatures whose timestamp is more than this many seconds
        in the future. Default 60 s. Defends against an attacker fixing
        a far-future timestamp to extend the replay window arbitrarily.
    clock
        Injectable clock; defaults to :func:`time.time`.

    Returns
    -------
    None
        Returns silently on success. Use a try/except to handle failure.

    Raises
    ------
    InvalidSecretError
        Secret is empty / not a string.
    MalformedSignatureError
        Signature header is missing, empty, or doesn't start with ``v1=``.
    InvalidTimestampError
        Timestamp header is missing or not a valid integer.
    ExpiredSignatureError
        Payload timestamp is older than ``max_age_seconds``.
    FutureTimestampError
        Payload timestamp is more than ``max_skew_seconds`` in the future.
    SignatureMismatchError
        Recomputed signature does not match.
    """
    secret_bytes = _validate_secret(secret)
    body_bytes = _coerce_body(body)

    signature = headers.get(HEADER_SIGNATURE)
    if not signature or not signature.startswith(SIGNATURE_PREFIX):
        raise MalformedSignatureError(
            f"missing or malformed {HEADER_SIGNATURE} header "
            f"(must start with {SIGNATURE_PREFIX!r})"
        )

    ts_raw = headers.get(HEADER_TIMESTAMP)
    if ts_raw is None or ts_raw.strip() == "":
        raise InvalidTimestampError(
            f"missing {HEADER_TIMESTAMP} header"
        )
    try:
        ts = int(ts_raw)
    except (TypeError, ValueError) as e:
        raise InvalidTimestampError(
            f"{HEADER_TIMESTAMP} must be integer seconds; got {ts_raw!r}"
        ) from e

    now = int(clock())
    age = now - ts
    if max_age_seconds and age > max_age_seconds:
        raise ExpiredSignatureError(
            f"signature older than {max_age_seconds}s (age={age}s)"
        )
    if ts - now > max_skew_seconds:
        raise FutureTimestampError(
            f"signature timestamp more than {max_skew_seconds}s in the future "
            f"(skew={ts - now}s)"
        )

    expected = _compute_signature(secret_bytes, ts, body_bytes)
    if not hmac.compare_digest(expected, signature):
        # Don't log the secret or expected/actual digests — knowing the
        # expected digest defeats the signature for one payload. Log a
        # generic mismatch only.
        logger.debug("webhook signature mismatch (ts=%s, age=%ss)", ts, age)
        raise SignatureMismatchError("signature mismatch")


__all__ = (
    "DEFAULT_MAX_AGE_SECONDS",
    "DEFAULT_MAX_SKEW_SECONDS",
    "ExpiredSignatureError",
    "FutureTimestampError",
    "HEADER_SIGNATURE",
    "HEADER_TIMESTAMP",
    "HEADER_VERSION",
    "InvalidSecretError",
    "InvalidTimestampError",
    "MalformedSignatureError",
    "SIGNATURE_PREFIX",
    "SIGNATURE_VERSION",
    "SignatureError",
    "SignatureMismatchError",
    "SignedHeaders",
    "sign",
    "verify",
)
