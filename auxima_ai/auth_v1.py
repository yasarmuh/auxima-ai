"""Inbound sidecar request auth — the ``Auxima-v1`` HMAC scheme (S-54 / GAP-16).

This module is the **pure verification core** for authenticating requests
from the ``auxima`` Frappe app to the ``auxima-ai`` sidecar. It supersedes
the static shared-secret in :mod:`auxima_ai.auth` (which remains for the
Phase-0 spike). The full scheme is specified in
``BMS/Docs/Planning/slices/S-54-sidecar-auth-spec.md``.

What this module DOES (pure, stdlib-only, fully unit-testable):
  - parse the ``Authorization: Auxima-v1 <key_id>:<ts>:<nonce>:<hmac>`` header
  - build the canonical preimage over key_id ‖ ts ‖ nonce ‖ method ‖ path ‖ sha256(body)
  - compute + constant-time-verify the HMAC-SHA256
  - enforce a ±skew timestamp window (default 300 s)
  - select the signing key from a dual-key :class:`Keyring` (primary + secondary)
  - sign a request (the producer side — used by tests and mirrored by the
    Frappe-side signer)

What this module does NOT do (deliberately out of scope — separate concerns
that need integration dependencies, tracked as GAP-16 follow-up iters):
  - **replay protection** (S-54 R5/§3.3) — needs Redis SETNX; lives in the
    FastAPI middleware, not here. Verification here is stateless.
  - **the FastAPI middleware wiring**, ``/v1/auth/whoami`` (R12), and the
    ``AI Run Log`` key_id audit (R7) — those compose this core.
  - **the Frappe-side signer** — lands in the ``auxima`` repo, mirroring
    :func:`sign_request`.

Security posture (mirrors :mod:`auxima_ai.webhooks.signer`):
  - pure ``hmac`` + ``hashlib`` + ``base64`` + ``time``; no third-party deps.
  - fail-closed: every error path raises a subclass of :class:`AuthError`;
    callers MUST translate any :class:`AuthError` to 401 with no body parse.
  - constant-time HMAC comparison via :func:`hmac.compare_digest`.
  - never logs the raw HMAC, nonce, secret, or timestamp (S-54 R10).
  - injectable clock for deterministic tests.

NOTE: this is the isolated crypto core. A ``security-reviewer`` pass is
recommended before the production middleware wiring (the integration —
Redis fail-closed, header forwarding, AI Run Log — is where most of the
residual risk lives).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Callable, Final, Mapping

SCHEME: Final[str] = "Auxima-v1"
DEFAULT_SKEW_SECONDS: Final[int] = 300  # ±5 min (S-54 R4).
#: Upper bound on the attacker-influenced token fields (key_id, nonce).
#: Aligned with ``auth_nonce.MAX_NONCE_LEN`` so a token that passes
#: verification cannot then be rejected by the replay store (which would
#: surface as a 500, not a clean 401 — see the iter-279 security review).
MAX_FIELD_LEN: Final[int] = 256
#: Characters that must never appear in key_id / nonce. ``\n`` and ``\r``
#: are the canonical-preimage field separators; ``:`` is the token
#: separator. Allowing any of them would let two distinct
#: (key_id, ts, nonce, method, path, body) tuples produce an IDENTICAL
#: canonical preimage (S-54 §3.2 "always exactly one separator"), breaking
#: the injective-preimage guarantee the whole scheme rests on.
_FORBIDDEN_FIELD_CHARS: Final[tuple[str, ...]] = ("\n", "\r", ":")
_EMPTY_BODY_SHA256: Final[str] = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


# ---------------------------------------------------------------------------
# Exceptions — every failure raises a subclass of AuthError (fail closed).
# ---------------------------------------------------------------------------


class AuthError(ValueError):
    """Base — any auth failure. Callers translate to 401 (no body parse)."""

    #: short machine-readable reason for the structured log (S-54 §3.5).
    reason: str = "auth_error"


class BadSchemeError(AuthError):
    """Authorization header missing or not the ``Auxima-v1`` scheme."""

    reason = "bad_scheme"


class MalformedTokenError(AuthError):
    """Token body does not parse into exactly key_id:timestamp:nonce:hmac."""

    reason = "bad_format"


class UnknownKeyError(AuthError):
    """The presented ``key_id`` is not in the keyring."""

    reason = "unknown_key"


class StaleTimestampError(AuthError):
    """Timestamp is more than ``skew_seconds`` in the past."""

    reason = "stale_timestamp"


class FutureTimestampError(AuthError):
    """Timestamp is more than ``skew_seconds`` in the future.

    Distinct reason from stale (iter-279 review M2): a far-future timestamp
    is a replay-window-extension attempt, not benign past clock skew, and
    audit/alerting should be able to tell them apart.
    """

    reason = "future_timestamp"


class BadHmacError(AuthError):
    """Recomputed HMAC does not match the presented one (tamper / key mismatch)."""

    reason = "bad_hmac"


class InvalidKeyError(AuthError):
    """A keyring secret is not valid base64 / not 32 bytes (config error)."""

    reason = "invalid_key"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedToken:
    """The four fields parsed out of an ``Auxima-v1`` Authorization header."""

    key_id: str
    timestamp: int
    nonce: str
    hmac_b64: str


@dataclass(frozen=True)
class Keyring:
    """A dual-key (primary + secondary) lookup of ``key_id -> base64 secret``.

    Holds the secrets as their base64 strings (as stored in the Frappe Site
    Config / sidecar env per S-54 R11). The 256-bit raw key is decoded on
    demand by :func:`_decode_key`. An empty keyring accepts nothing
    (fail-closed); a sidecar started with no keys rejects every request.
    """

    keys: Mapping[str, str]

    def secret_b64_for(self, key_id: str) -> str | None:
        """Return the base64 secret for ``key_id``, or None if unknown."""
        return self.keys.get(key_id)

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(self.keys.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_body(body: bytes | str) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    raise AuthError(f"body must be bytes / bytearray / str; got {type(body).__name__}")


def _decode_key(secret_b64: str) -> bytes:
    """Decode a base64 256-bit key to raw bytes. Raise InvalidKeyError otherwise.

    Enforces the S-54 R11 contract: keys are ``base64(32-random-bytes)``.
    A misconfigured key (bad base64, wrong length) is a config error, not a
    caller error — but it still fails closed (the request cannot be verified).
    """
    if not isinstance(secret_b64, str) or not secret_b64.strip():
        raise InvalidKeyError("keyring secret is empty / not a string")
    try:
        raw = base64.b64decode(secret_b64.strip(), validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidKeyError("keyring secret is not valid base64") from e
    if len(raw) != 32:
        raise InvalidKeyError(
            f"keyring secret must decode to 32 bytes (256-bit); got {len(raw)}"
        )
    return raw


def _sha256_hex(body: bytes) -> str:
    if not body:
        return _EMPTY_BODY_SHA256
    return hashlib.sha256(body).hexdigest()


def _validate_token_field(name: str, value: str) -> None:
    """Reject a token field that could break the LF-separated preimage.

    ``key_id`` and ``nonce`` are attacker-influenced. A field containing a
    field separator (``\\n`` / ``\\r``) or the token separator (``:``) would
    let two distinct field tuples produce the same canonical preimage
    (S-54 §3.2). Legitimate base64url nonces and short-ASCII key_ids never
    contain these characters. Raises :class:`MalformedTokenError`.
    """
    if not isinstance(value, str) or value == "":
        raise MalformedTokenError(f"{name} must be a non-empty string")
    if any(c in value for c in _FORBIDDEN_FIELD_CHARS):
        raise MalformedTokenError(
            f"{name} contains a forbidden separator character"
        )
    if len(value) > MAX_FIELD_LEN:
        raise MalformedTokenError(
            f"{name} exceeds maximum length {MAX_FIELD_LEN}"
        )


def canonical_preimage(
    key_id: str,
    timestamp: int,
    nonce: str,
    method: str,
    path: str,
    body: bytes | str,
) -> bytes:
    """Build the LF-separated canonical preimage (S-54 §3.2).

    ``key_id \\n timestamp \\n nonce \\n METHOD \\n path \\n sha256_hex(body)``

    - exactly one ``\\n`` (0x0A) between fields; no trailing newline.
    - ``method`` is upper-cased here (the sidecar canonicalises).
    - ``path`` is the raw URL path incl. query string; NOT host, NOT scheme.
    - body is hashed to its lowercase hex sha256 (empty body → the empty
      string digest).

    Defensively rejects a ``\\n`` / ``\\r`` / ``:`` in ``key_id`` or
    ``nonce`` (the attacker-influenced fields) so the sign path fails closed
    too — :func:`parse_authorization` enforces the same on the verify path.
    """
    _validate_token_field("key_id", key_id)
    _validate_token_field("nonce", nonce)
    body_bytes = _coerce_body(body)
    fields = (
        key_id,
        str(int(timestamp)),
        nonce,
        method.upper(),
        path,
        _sha256_hex(body_bytes),
    )
    return "\n".join(fields).encode("utf-8")


def _hmac_digest(raw_key: bytes, preimage: bytes) -> bytes:
    """Raw HMAC-SHA256 digest bytes (used for constant-time verify)."""
    return hmac.new(raw_key, preimage, hashlib.sha256).digest()


def _compute_hmac_b64(raw_key: bytes, preimage: bytes) -> str:
    """Base64 HMAC (the on-the-wire form produced by :func:`sign_request`)."""
    return base64.b64encode(_hmac_digest(raw_key, preimage)).decode("ascii")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_authorization(header_value: str | None) -> ParsedToken:
    """Parse an ``Authorization: Auxima-v1 key_id:ts:nonce:hmac`` header.

    Raises
    ------
    BadSchemeError
        Header missing/empty, or scheme is not ``Auxima-v1``.
    MalformedTokenError
        Token body is not exactly 4 colon-separated non-empty fields, or
        the timestamp is not an integer.
    """
    if not header_value or not header_value.strip():
        raise BadSchemeError("missing Authorization header")
    parts = header_value.strip().split(" ", 1)
    if len(parts) != 2 or parts[0] != SCHEME:
        raise BadSchemeError(
            f"Authorization scheme must be {SCHEME!r}"
        )
    token = parts[1].strip()
    # Exactly 4 fields. The hmac/nonce are base64 and never contain ':',
    # so a plain split on ':' with maxsplit=3 would mis-handle a stray ':'
    # in the hmac — use an exact 4-field split and reject anything else.
    fields = token.split(":")
    if len(fields) != 4 or any(f == "" for f in fields):
        raise MalformedTokenError(
            "token must be key_id:timestamp:nonce:hmac (4 non-empty fields)"
        )
    key_id, ts_raw, nonce, hmac_b64 = fields
    # Reject forbidden separator chars / overlong values in the attacker-
    # influenced fields BEFORE any further processing (iter-279 review C1).
    # (`:` cannot appear post-split, but `\n` / `\r` can, and would break
    # the canonical preimage's injectivity.)
    _validate_token_field("key_id", key_id)
    _validate_token_field("nonce", nonce)
    try:
        timestamp = int(ts_raw)
    except (TypeError, ValueError) as e:
        raise MalformedTokenError(
            f"timestamp must be integer seconds; got {ts_raw!r}"
        ) from e
    return ParsedToken(
        key_id=key_id, timestamp=timestamp, nonce=nonce, hmac_b64=hmac_b64
    )


# ---------------------------------------------------------------------------
# Sign (producer side — used by tests and mirrored by the Frappe signer)
# ---------------------------------------------------------------------------


def sign_request(
    key_id: str,
    secret_b64: str,
    method: str,
    path: str,
    body: bytes | str,
    *,
    nonce: str,
    timestamp: int | None = None,
    clock: Callable[[], float] = time.time,
) -> str:
    """Produce the ``Authorization`` header value for a request.

    The caller supplies the ``nonce`` (the producer is responsible for nonce
    generation — typically ``base64url(16 random bytes)``; this core does not
    generate randomness so it stays deterministic + pure). ``timestamp``
    defaults to ``int(clock())``.

    Returns the full header value: ``Auxima-v1 key_id:ts:nonce:hmac_b64``.
    """
    raw_key = _decode_key(secret_b64)
    ts = int(timestamp) if timestamp is not None else int(clock())
    preimage = canonical_preimage(key_id, ts, nonce, method, path, body)
    hmac_b64 = _compute_hmac_b64(raw_key, preimage)
    return f"{SCHEME} {key_id}:{ts}:{nonce}:{hmac_b64}"


# ---------------------------------------------------------------------------
# Verify (consumer side — stateless; replay protection is a separate layer)
# ---------------------------------------------------------------------------


def verify_request(
    header_value: str | None,
    method: str,
    path: str,
    body: bytes | str,
    keyring: Keyring,
    *,
    skew_seconds: int = DEFAULT_SKEW_SECONDS,
    clock: Callable[[], float] = time.time,
) -> ParsedToken:
    """Verify an inbound request's ``Auxima-v1`` Authorization header.

    Performs (in order, fail-closed at each step):
      1. parse the header (scheme + 4-field token)
      2. resolve the key_id against the keyring
      3. enforce the ±``skew_seconds`` timestamp window
      4. recompute the HMAC over the canonical preimage and constant-time
         compare it to the presented one

    Does NOT perform replay protection (the nonce uniqueness check) — that is
    a stateful concern handled by the Redis-backed middleware layer. A caller
    that needs replay protection MUST check ``returned.nonce`` against its
    nonce cache AFTER this returns.

    Returns
    -------
    ParsedToken
        The validated token (the caller logs ``key_id`` per S-54 R7 and
        passes ``nonce`` to the replay-cache check).

    Raises
    ------
    BadSchemeError / MalformedTokenError / UnknownKeyError /
    StaleTimestampError / FutureTimestampError / BadHmacError / InvalidKeyError
        Any of these is an auth failure → the caller returns 401 (no body
        parse, no downstream LLM call) per S-54 R9.
    """
    token = parse_authorization(header_value)

    secret_b64 = keyring.secret_b64_for(token.key_id)
    if secret_b64 is None:
        raise UnknownKeyError(f"unknown key_id {token.key_id!r}")
    raw_key = _decode_key(secret_b64)

    now = int(clock())
    delta = token.timestamp - now
    if delta < -skew_seconds:
        raise StaleTimestampError(
            f"timestamp {abs(delta)}s in the past (window ±{skew_seconds}s)"
        )
    if delta > skew_seconds:
        raise FutureTimestampError(
            f"timestamp {delta}s in the future (window ±{skew_seconds}s)"
        )

    preimage = canonical_preimage(
        token.key_id, token.timestamp, token.nonce, method, path, body
    )
    # Compare RAW digest bytes, not the base64 strings: type-stable (both
    # bytes), and an unparseable base64 hmac is an explicit BadHmacError
    # rather than a silent never-matches (iter-279 review H4).
    expected_digest = _hmac_digest(raw_key, preimage)
    try:
        presented_digest = base64.b64decode(token.hmac_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise BadHmacError("presented HMAC is not valid base64") from e
    if not hmac.compare_digest(expected_digest, presented_digest):
        # Never log expected/actual digests — knowing `expected` defeats the
        # signature for this one preimage (S-54 R10).
        raise BadHmacError("HMAC mismatch")

    return token


__all__ = (
    "DEFAULT_SKEW_SECONDS",
    "SCHEME",
    "AuthError",
    "BadHmacError",
    "BadSchemeError",
    "FutureTimestampError",
    "InvalidKeyError",
    "Keyring",
    "MalformedTokenError",
    "ParsedToken",
    "StaleTimestampError",
    "UnknownKeyError",
    "canonical_preimage",
    "parse_authorization",
    "sign_request",
    "verify_request",
)
