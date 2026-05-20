"""Config-selected inbound auth middleware (S-54 / GAP-16 cutover).

The sidecar supports two inbound auth schemes; ``Settings.sidecar_auth_mode``
selects which one guards ``/v1/*``:

  - ``"shared_secret"`` (default) — the Phase-0 ``X-Auxima-Sidecar-Token``
    scheme (:func:`auxima_ai.auth.shared_secret_middleware`). UNCHANGED live
    behavior; this is what runs until the Frappe-side Auxima-v1 signer ships.
  - ``"auxima_v1"`` — the HMAC + timestamp-skew + nonce-replay Auxima-v1
    scheme (:func:`auxima_ai.auth_v1_middleware.make_auth_v1_middleware`).

This module is the single, unit-testable seam between config and the two
middlewares, so :mod:`auxima_ai.main` stays a thin wiring layer. Selecting
``auxima_v1`` with no keys configured fails fast (:class:`AuthConfigError`)
rather than silently starting a sidecar whose empty keyring 401s every
request.
"""
from __future__ import annotations

from auxima_ai.auth import shared_secret_middleware
from auxima_ai.auth_nonce import InMemoryNonceStore, NonceStore
from auxima_ai.auth_v1 import Keyring
from auxima_ai.auth_v1_middleware import make_auth_v1_middleware
from auxima_ai.config import Settings


class AuthConfigError(ValueError):
    """The selected auth mode is misconfigured (e.g. auxima_v1 with no keys)."""


def _keyring_from_settings(settings: Settings) -> Keyring:
    """Build the dual-key :class:`Keyring` from config.

    A key VALUE without its key_id (or vice-versa) is a configuration error
    — both halves are required to name + verify a key. At least one complete
    key must be present, else there is nothing to authenticate against.
    """
    keys: dict[str, str] = {}
    for label, key_id, secret_b64 in (
        ("primary", settings.primary_key_id, settings.primary_key_b64),
        ("secondary", settings.secondary_key_id, settings.secondary_key_b64),
    ):
        if not key_id and not secret_b64:
            continue  # this slot is unset — fine
        if not (key_id and secret_b64):
            raise AuthConfigError(
                f"{label} key needs BOTH a key_id and a key value "
                f"(got key_id={key_id!r}, value_set={bool(secret_b64)})"
            )
        keys[key_id] = secret_b64

    if not keys:
        raise AuthConfigError(
            "sidecar_auth_mode='auxima_v1' requires at least one key "
            "(set primary_key_id + primary_key_b64)"
        )
    return Keyring(keys=keys)


def select_auth_middleware(settings: Settings, *, nonce_store: NonceStore | None = None):
    """Return the ``http`` middleware callable for the configured auth mode.

    Parameters
    ----------
    settings
        The loaded sidecar settings.
    nonce_store
        Replay store for ``auxima_v1`` mode. Defaults to an in-memory store
        (single-process). Production injects a ``RedisNonceStore`` so all
        replicas share one nonce namespace (S-54 §3.3); the in-memory default
        does NOT protect against cross-replica replay.

    Raises
    ------
    AuthConfigError
        If ``auxima_v1`` is selected but no valid key material is configured.
    """
    if settings.sidecar_auth_mode == "shared_secret":
        return shared_secret_middleware

    keyring = _keyring_from_settings(settings)
    store = nonce_store if nonce_store is not None else InMemoryNonceStore()
    return make_auth_v1_middleware(keyring, store)


__all__ = ("AuthConfigError", "select_auth_middleware")
