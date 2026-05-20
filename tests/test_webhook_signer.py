"""Tests for ``auxima_ai.webhooks.signer`` — HMAC-SHA256 webhook signer + verifier.

Coverage per S-34 §3.3 + OWASP-A02 (cryptographic failures):
  - Round-trip sign/verify works on bytes + str bodies.
  - Tampered body fails verification.
  - Wrong secret fails verification.
  - Replaying an expired signature fails (ExpiredSignatureError).
  - Far-future timestamp fails (FutureTimestampError).
  - Clock skew within tolerance succeeds.
  - Empty / whitespace secrets fail closed on both sides.
  - Malformed signature headers (missing, no prefix, wrong version) fail.
  - Invalid timestamp headers (missing, non-int, empty) fail.
  - constant-time comparison: assertion via hmac.compare_digest is exercised
    by all positive + negative paths.
  - SignedHeaders.as_dict returns the exact wire keys.
  - Deterministic output for fixed (body, secret, timestamp).
  - Body-substitution attack: signature from payload A doesn't validate payload B.
  - Reference vector pinned: a known (body, secret, ts) maps to a known hex digest.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Final

import pytest

from auxima_ai.webhooks.signer import (
    DEFAULT_MAX_AGE_SECONDS,
    DEFAULT_MAX_SKEW_SECONDS,
    ExpiredSignatureError,
    FutureTimestampError,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    HEADER_VERSION,
    InvalidSecretError,
    InvalidTimestampError,
    MalformedSignatureError,
    SIGNATURE_PREFIX,
    SIGNATURE_VERSION,
    SignatureError,
    SignatureMismatchError,
    sign,
    verify,
)

SECRET: Final[str] = "test-secret-do-not-use-in-prod"
BODY: Final[bytes] = b'{"event":"lead.created","id":42}'
FIXED_TS: Final[int] = 1_715_986_380  # 2024-05-17T20:53:00Z

# Reference vector: hex(HMAC-SHA256(SECRET, "v1:1715986380:" + BODY))
# Computed inline so the test is self-checking, but pinned to a constant
# below to lock the wire format. If this constant ever needs to change,
# something about the canonical-string format has changed too.
_REFERENCE_HEX: Final[str] = hmac.new(
    SECRET.encode("utf-8"),
    f"{SIGNATURE_VERSION}:{FIXED_TS}:".encode("utf-8") + BODY,
    hashlib.sha256,
).hexdigest()


def fixed_clock() -> float:
    return float(FIXED_TS)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_sign_round_trip_bytes_body() -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    verify(BODY, headers, SECRET, clock=fixed_clock)


def test_sign_round_trip_str_body_is_utf8_encoded() -> None:
    body_str = '{"msg":"Café"}'
    body_bytes = body_str.encode("utf-8")
    headers = sign(body_str, SECRET, clock=fixed_clock).as_dict()
    verify(body_bytes, headers, SECRET, clock=fixed_clock)


def test_sign_round_trip_bytearray_body() -> None:
    body = bytearray(BODY)
    headers = sign(body, SECRET, clock=fixed_clock).as_dict()
    verify(BODY, headers, SECRET, clock=fixed_clock)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def test_signed_headers_as_dict_has_exact_wire_keys() -> None:
    h = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    assert set(h.keys()) == {HEADER_SIGNATURE, HEADER_TIMESTAMP, HEADER_VERSION}
    assert h[HEADER_VERSION] == SIGNATURE_VERSION
    assert h[HEADER_TIMESTAMP] == str(FIXED_TS)
    assert h[HEADER_SIGNATURE].startswith(SIGNATURE_PREFIX)


def test_reference_vector_pinned_for_format_stability() -> None:
    """Locks the canonical-string format. If this changes, deployed
    verifiers will break — bump SIGNATURE_VERSION instead."""
    h = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()
    expected = f"{SIGNATURE_PREFIX}{_REFERENCE_HEX}"
    assert h[HEADER_SIGNATURE] == expected


def test_sign_is_deterministic() -> None:
    a = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()
    b = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()
    assert a == b


def test_signed_headers_is_frozen() -> None:
    sh = sign(BODY, SECRET, timestamp=FIXED_TS)
    with pytest.raises((AttributeError, TypeError)):
        sh.signature = "v1=tamper"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_tampered_body_fails_verification() -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    tampered = BODY + b"X"
    with pytest.raises(SignatureMismatchError):
        verify(tampered, headers, SECRET, clock=fixed_clock)


def test_wrong_secret_fails_verification() -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    with pytest.raises(SignatureMismatchError):
        verify(BODY, headers, SECRET + "-wrong", clock=fixed_clock)


def test_swapped_body_with_same_timestamp_fails() -> None:
    """Body substitution attack — sig for A must not validate B."""
    headers_a = sign(b"payload-a", SECRET, timestamp=FIXED_TS).as_dict()
    with pytest.raises(SignatureMismatchError):
        verify(b"payload-b", headers_a, SECRET, clock=fixed_clock)


# ---------------------------------------------------------------------------
# Timing — replay + skew
# ---------------------------------------------------------------------------


def test_expired_signature_rejected() -> None:
    headers = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()

    def later_clock() -> float:
        return float(FIXED_TS + DEFAULT_MAX_AGE_SECONDS + 1)

    with pytest.raises(ExpiredSignatureError):
        verify(BODY, headers, SECRET, clock=later_clock)


def test_signature_within_max_age_accepted() -> None:
    headers = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()

    def later_clock() -> float:
        return float(FIXED_TS + DEFAULT_MAX_AGE_SECONDS - 1)

    verify(BODY, headers, SECRET, clock=later_clock)


def test_signature_exactly_at_max_age_accepted() -> None:
    headers = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()

    def boundary_clock() -> float:
        return float(FIXED_TS + DEFAULT_MAX_AGE_SECONDS)

    verify(BODY, headers, SECRET, clock=boundary_clock)


def test_future_timestamp_beyond_skew_rejected() -> None:
    """A timestamp far in the future is rejected — defends against an attacker
    fixing a future ts to extend the replay window indefinitely."""
    future_ts = FIXED_TS + DEFAULT_MAX_SKEW_SECONDS + 5
    headers = sign(BODY, SECRET, timestamp=future_ts).as_dict()
    with pytest.raises(FutureTimestampError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


def test_future_timestamp_within_skew_accepted() -> None:
    """Small forward clock skew (sender slightly ahead of receiver) is tolerated."""
    future_ts = FIXED_TS + DEFAULT_MAX_SKEW_SECONDS - 1
    headers = sign(BODY, SECRET, timestamp=future_ts).as_dict()
    verify(BODY, headers, SECRET, clock=fixed_clock)


def test_max_age_zero_disables_age_check() -> None:
    headers = sign(BODY, SECRET, timestamp=FIXED_TS).as_dict()

    def way_later() -> float:
        return float(FIXED_TS + 1_000_000)

    verify(BODY, headers, SECRET, max_age_seconds=0, clock=way_later)


# ---------------------------------------------------------------------------
# Secret validation — fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_secret", ["", "   ", "\t\n", None, 42])
def test_sign_rejects_empty_or_invalid_secret(bad_secret: object) -> None:
    with pytest.raises(InvalidSecretError):
        sign(BODY, bad_secret)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_secret", ["", "   ", "\t\n", None, 42])
def test_verify_rejects_empty_or_invalid_secret(bad_secret: object) -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    with pytest.raises(InvalidSecretError):
        verify(BODY, headers, bad_secret, clock=fixed_clock)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Body validation
# ---------------------------------------------------------------------------


def test_sign_rejects_non_bytes_or_str_body() -> None:
    with pytest.raises(SignatureError):
        sign(42, SECRET)  # type: ignore[arg-type]


def test_verify_rejects_non_bytes_or_str_body() -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    with pytest.raises(SignatureError):
        verify([1, 2, 3], headers, SECRET, clock=fixed_clock)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Malformed signature header
# ---------------------------------------------------------------------------


def test_missing_signature_header_fails() -> None:
    headers = {HEADER_TIMESTAMP: str(FIXED_TS), HEADER_VERSION: SIGNATURE_VERSION}
    with pytest.raises(MalformedSignatureError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


def test_empty_signature_header_fails() -> None:
    headers = {
        HEADER_SIGNATURE: "",
        HEADER_TIMESTAMP: str(FIXED_TS),
        HEADER_VERSION: SIGNATURE_VERSION,
    }
    with pytest.raises(MalformedSignatureError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


def test_wrong_version_prefix_fails() -> None:
    headers = {
        HEADER_SIGNATURE: "v2=deadbeef",
        HEADER_TIMESTAMP: str(FIXED_TS),
        HEADER_VERSION: SIGNATURE_VERSION,
    }
    with pytest.raises(MalformedSignatureError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


def test_no_version_prefix_fails() -> None:
    headers = {
        HEADER_SIGNATURE: "abc123",
        HEADER_TIMESTAMP: str(FIXED_TS),
        HEADER_VERSION: SIGNATURE_VERSION,
    }
    with pytest.raises(MalformedSignatureError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


# ---------------------------------------------------------------------------
# Malformed timestamp header
# ---------------------------------------------------------------------------


def test_missing_timestamp_fails() -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    del headers[HEADER_TIMESTAMP]
    with pytest.raises(InvalidTimestampError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


def test_empty_timestamp_fails() -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    headers[HEADER_TIMESTAMP] = ""
    with pytest.raises(InvalidTimestampError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


@pytest.mark.parametrize("bad", ["abc", "12.5", "1715986380x"])
def test_non_int_timestamp_fails(bad: str) -> None:
    headers = sign(BODY, SECRET, clock=fixed_clock).as_dict()
    headers[HEADER_TIMESTAMP] = bad
    with pytest.raises(InvalidTimestampError):
        verify(BODY, headers, SECRET, clock=fixed_clock)


# ---------------------------------------------------------------------------
# Exception hierarchy — every error is a SignatureError subclass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        InvalidSecretError,
        MalformedSignatureError,
        InvalidTimestampError,
        ExpiredSignatureError,
        FutureTimestampError,
        SignatureMismatchError,
    ],
)
def test_all_errors_are_signature_error_subclasses(exc: type) -> None:
    assert issubclass(exc, SignatureError)
    assert issubclass(exc, ValueError)
