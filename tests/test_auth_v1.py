"""Tests for the Auxima-v1 sidecar auth core (auxima_ai.auth_v1 / S-54 / GAP-16).

Covers the acceptance criteria the PURE core can satisfy:
  - AC-2 (timestamp skew → 401-equivalent)
  - AC-3 (body tamper → bad_hmac)
  - the S-54 §3.5 failure-mode rows that don't need Redis/FastAPI:
    missing header, bad scheme, bad format, unknown key.

Replay (AC-1), Redis-503 (AC-6), AI-Run-Log (AC-7), /whoami (AC-8) need the
middleware layer and are NOT covered here (separate GAP-16 follow-up iters).
"""
from __future__ import annotations

import base64

import pytest

from auxima_ai.auth_v1 import (
    DEFAULT_SKEW_SECONDS,
    SCHEME,
    BadHmacError,
    BadSchemeError,
    FutureTimestampError,
    InvalidKeyError,
    Keyring,
    MalformedTokenError,
    StaleTimestampError,
    UnknownKeyError,
    canonical_preimage,
    parse_authorization,
    sign_request,
    verify_request,
)

# A deterministic 32-byte (256-bit) key, base64-encoded (S-54 R11 shape).
_KEY_RAW = bytes(range(32))
_KEY_B64 = base64.b64encode(_KEY_RAW).decode("ascii")
_KEY_RAW_2 = bytes(range(32, 64))
_KEY_B64_2 = base64.b64encode(_KEY_RAW_2).decode("ascii")

_FIXED_NOW = 1_747_567_200  # a fixed epoch for deterministic skew tests.


def _clock(now: int = _FIXED_NOW):
    return lambda: float(now)


def _keyring():
    return Keyring(keys={"p2026q2": _KEY_B64, "s2026q3": _KEY_B64_2})


def _signed(method="POST", path="/v1/intake/extract", body=b'{"x":1}', *,
            key_id="p2026q2", secret_b64=_KEY_B64, nonce="bm9uY2UtMTIz", ts=_FIXED_NOW):
    return sign_request(
        key_id, secret_b64, method, path, body, nonce=nonce, timestamp=ts
    )


# ---------------------------------------------------------------------------
# Round-trip happy path
# ---------------------------------------------------------------------------


def test_sign_then_verify_roundtrip() -> None:
    header = _signed()
    token = verify_request(
        header, "POST", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock()
    )
    assert token.key_id == "p2026q2"
    assert token.timestamp == _FIXED_NOW
    assert token.nonce == "bm9uY2UtMTIz"


def test_verify_accepts_secondary_key_dual_key_window() -> None:
    # S-54 R6: both primary and secondary keys are accepted simultaneously.
    header = _signed(key_id="s2026q3", secret_b64=_KEY_B64_2)
    token = verify_request(
        header, "POST", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock()
    )
    assert token.key_id == "s2026q3"


def test_empty_body_roundtrip() -> None:
    header = _signed(body=b"", path="/v1/auth/whoami", method="GET")
    token = verify_request(
        header, "GET", "/v1/auth/whoami", b"", _keyring(), clock=_clock()
    )
    assert token.key_id == "p2026q2"


# ---------------------------------------------------------------------------
# AC-3 — body tamper → bad_hmac
# ---------------------------------------------------------------------------


def test_tampered_body_raises_bad_hmac() -> None:
    header = _signed(body=b'{"x":1}')
    with pytest.raises(BadHmacError) as ei:
        verify_request(
            header, "POST", "/v1/intake/extract", b'{"x":2}', _keyring(), clock=_clock()
        )
    assert ei.value.reason == "bad_hmac"


def test_tampered_method_raises_bad_hmac() -> None:
    header = _signed(method="POST")
    with pytest.raises(BadHmacError):
        verify_request(
            header, "GET", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock()
        )


def test_tampered_path_raises_bad_hmac() -> None:
    header = _signed(path="/v1/intake/extract")
    with pytest.raises(BadHmacError):
        verify_request(
            header, "POST", "/v1/intake/extract?x=1", b'{"x":1}', _keyring(), clock=_clock()
        )


def test_wrong_key_secret_raises_bad_hmac() -> None:
    # Signed with key_id p2026q2 but using the WRONG secret bytes.
    header = sign_request(
        "p2026q2", _KEY_B64_2, "POST", "/v1/x", b"{}", nonce="bm9uY2U", timestamp=_FIXED_NOW
    )
    with pytest.raises(BadHmacError):
        verify_request(header, "POST", "/v1/x", b"{}", _keyring(), clock=_clock())


# ---------------------------------------------------------------------------
# AC-2 — timestamp skew
# ---------------------------------------------------------------------------


def test_stale_timestamp_just_outside_window_raises() -> None:
    header = _signed(ts=_FIXED_NOW - (DEFAULT_SKEW_SECONDS + 1))
    with pytest.raises(StaleTimestampError) as ei:
        verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock())
    assert ei.value.reason == "stale_timestamp"


def test_stale_timestamp_at_window_edge_accepted() -> None:
    # Exactly -skew is still inside the window (inclusive edge).
    header = _signed(ts=_FIXED_NOW - DEFAULT_SKEW_SECONDS)
    token = verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock())
    assert token.key_id == "p2026q2"


def test_future_timestamp_outside_window_raises() -> None:
    header = _signed(ts=_FIXED_NOW + (DEFAULT_SKEW_SECONDS + 1))
    with pytest.raises(FutureTimestampError):
        verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock())


def test_future_timestamp_at_window_edge_accepted() -> None:
    header = _signed(ts=_FIXED_NOW + DEFAULT_SKEW_SECONDS)
    token = verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', _keyring(), clock=_clock())
    assert token.key_id == "p2026q2"


# ---------------------------------------------------------------------------
# Failure-mode matrix (S-54 §3.5) — parse + key resolution
# ---------------------------------------------------------------------------


def test_missing_header_raises_bad_scheme() -> None:
    for v in (None, "", "   "):
        with pytest.raises(BadSchemeError):
            verify_request(v, "POST", "/v1/x", b"{}", _keyring(), clock=_clock())


def test_wrong_scheme_raises_bad_scheme() -> None:
    with pytest.raises(BadSchemeError) as ei:
        verify_request("Bearer abc.def", "POST", "/v1/x", b"{}", _keyring(), clock=_clock())
    assert ei.value.reason == "bad_scheme"


def test_wrong_field_count_raises_malformed() -> None:
    # 3 fields instead of 4.
    with pytest.raises(MalformedTokenError):
        verify_request(f"{SCHEME} p2026q2:123:nonce", "POST", "/v1/x", b"{}", _keyring(), clock=_clock())


def test_empty_field_raises_malformed() -> None:
    with pytest.raises(MalformedTokenError):
        verify_request(f"{SCHEME} p2026q2::nonce:hmac", "POST", "/v1/x", b"{}", _keyring(), clock=_clock())


def test_non_integer_timestamp_raises_malformed() -> None:
    with pytest.raises(MalformedTokenError):
        verify_request(f"{SCHEME} p2026q2:notanint:nonce:hmac", "POST", "/v1/x", b"{}", _keyring(), clock=_clock())


def test_unknown_key_id_raises() -> None:
    header = sign_request(
        "x9999", _KEY_B64, "POST", "/v1/x", b"{}", nonce="bm9uY2U", timestamp=_FIXED_NOW
    )
    with pytest.raises(UnknownKeyError) as ei:
        verify_request(header, "POST", "/v1/x", b"{}", _keyring(), clock=_clock())
    assert ei.value.reason == "unknown_key"


def test_empty_keyring_rejects_everything() -> None:
    header = _signed()
    with pytest.raises(UnknownKeyError):
        verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', Keyring(keys={}), clock=_clock())


# ---------------------------------------------------------------------------
# Key-format validation (config errors, fail closed)
# ---------------------------------------------------------------------------


def test_non_base64_key_raises_invalid_key() -> None:
    bad_ring = Keyring(keys={"p2026q2": "!!!not base64!!!"})
    header = _signed()
    with pytest.raises(InvalidKeyError):
        verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', bad_ring, clock=_clock())


def test_wrong_length_key_raises_invalid_key() -> None:
    short_key = base64.b64encode(b"too short").decode("ascii")  # 9 bytes
    bad_ring = Keyring(keys={"p2026q2": short_key})
    header = _signed()
    with pytest.raises(InvalidKeyError) as ei:
        verify_request(header, "POST", "/v1/intake/extract", b'{"x":1}', bad_ring, clock=_clock())
    assert "32 bytes" in str(ei.value)


def test_sign_with_bad_key_raises_invalid_key() -> None:
    with pytest.raises(InvalidKeyError):
        sign_request("p", "not-base64!!", "POST", "/v1/x", b"{}", nonce="n", timestamp=_FIXED_NOW)


# ---------------------------------------------------------------------------
# Canonical preimage shape (S-54 §3.2)
# ---------------------------------------------------------------------------


def test_canonical_preimage_is_lf_separated_six_fields() -> None:
    pre = canonical_preimage("p2026q2", 1747567200, "abc", "post", "/v1/x", b"{}")
    parts = pre.decode("utf-8").split("\n")
    assert len(parts) == 6
    assert parts[0] == "p2026q2"
    assert parts[1] == "1747567200"
    assert parts[2] == "abc"
    assert parts[3] == "POST"  # method upper-cased
    assert parts[4] == "/v1/x"
    # sha256 of b"{}" hex
    assert parts[5] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def test_canonical_preimage_empty_body_uses_empty_sha256() -> None:
    pre = canonical_preimage("k", 1, "n", "GET", "/v1/whoami", b"")
    last = pre.decode("utf-8").split("\n")[-1]
    assert last == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_canonical_preimage_str_and_bytes_body_equivalent() -> None:
    a = canonical_preimage("k", 1, "n", "POST", "/p", '{"x":1}')
    b = canonical_preimage("k", 1, "n", "POST", "/p", b'{"x":1}')
    assert a == b


def test_parse_authorization_returns_fields() -> None:
    header = _signed()
    token = parse_authorization(header)
    assert token.key_id == "p2026q2"
    assert token.timestamp == _FIXED_NOW
    assert token.nonce == "bm9uY2UtMTIz"
    assert token.hmac_b64  # non-empty
