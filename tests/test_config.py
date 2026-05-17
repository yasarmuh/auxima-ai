"""Tests for ``auxima_ai.config`` — startup-time validation of sidecar settings.

Coverage:
  - Defaults load (env unset → empty secret, default model, INFO log level).
  - shared_secret: empty allowed; short non-empty rejected; >= 16 chars accepted.
  - log_level: canonical values accepted; case-normalised to upper; bad rejected.
  - frappe_base_url: http loopback allowed; http non-loopback rejected; https
    allowed; bad schemes rejected; missing host rejected.
  - shared_secret_configured property reflects whether the secret is set.
  - reset_settings_cache clears the module-level cache.
  - get_settings is cached (returns the same instance) until reset.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

import auxima_ai.config as cfg_mod
from auxima_ai.config import (
    MIN_SHARED_SECRET_LEN,
    Settings,
    get_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache_and_env(monkeypatch):
    """Reset the cache + scrub all AUXIMA_SIDECAR_* env vars between tests."""
    for key in list(__import__("os").environ):
        if key.startswith("AUXIMA_SIDECAR_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_load_with_no_env() -> None:
    s = Settings()
    assert s.shared_secret == ""
    assert s.shared_secret_configured is False
    assert s.default_model == "ollama/qwen2.5:32b"
    assert s.log_level == "INFO"
    assert s.frappe_base_url.startswith("http://")


# ---------------------------------------------------------------------------
# shared_secret validation
# ---------------------------------------------------------------------------


def test_empty_shared_secret_allowed() -> None:
    s = Settings(shared_secret="")
    assert s.shared_secret == ""
    assert s.shared_secret_configured is False


def test_short_shared_secret_rejected_at_construction(monkeypatch) -> None:
    """A 4-char "test" secret left in env is a config mistake — fail at startup."""
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "test")
    reset_settings_cache()
    with pytest.raises(ValidationError) as exc_info:
        get_settings()
    msg = str(exc_info.value)
    assert "shared_secret" in msg
    assert str(MIN_SHARED_SECRET_LEN) in msg


def test_shared_secret_at_minimum_length_accepted() -> None:
    s = Settings(shared_secret="a" * MIN_SHARED_SECRET_LEN)
    assert s.shared_secret_configured is True


def test_shared_secret_above_minimum_accepted() -> None:
    s = Settings(shared_secret="a" * (MIN_SHARED_SECRET_LEN + 16))
    assert len(s.shared_secret) == MIN_SHARED_SECRET_LEN + 16


# ---------------------------------------------------------------------------
# log_level validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_level_canonical_values_accepted(level: str) -> None:
    assert Settings(log_level=level).log_level == level


def test_log_level_lowercase_normalised_to_upper() -> None:
    assert Settings(log_level="info").log_level == "INFO"
    assert Settings(log_level="warning").log_level == "WARNING"


def test_log_level_with_whitespace_normalised() -> None:
    assert Settings(log_level="  debug  ").log_level == "DEBUG"


@pytest.mark.parametrize("bad", ["TRACE", "FATAL", "verbose", "", "9001"])
def test_log_level_invalid_rejected(bad: str) -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(log_level=bad)


# ---------------------------------------------------------------------------
# frappe_base_url validation
# ---------------------------------------------------------------------------


def test_http_loopback_url_accepted() -> None:
    Settings(frappe_base_url="http://localhost:8000")
    Settings(frappe_base_url="http://127.0.0.1:8000")
    Settings(frappe_base_url="http://[::1]:8000")
    # *.localhost per RFC 6761 §6.3 — also loopback
    Settings(frappe_base_url="http://demo.localhost:8000")
    Settings(frappe_base_url="http://test.localhost")


def test_https_any_host_accepted() -> None:
    Settings(frappe_base_url="https://auxima.production.internal")
    Settings(frappe_base_url="https://demo.localhost:8000")


def test_http_non_loopback_rejected() -> None:
    """Cleartext http to a real host would leak the shared secret on the wire."""
    with pytest.raises(ValidationError, match="https://"):
        Settings(frappe_base_url="http://auxima.production.internal")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not-a-url",
        "ftp://example.com",
        "ws://example.com",
        "file:///etc/passwd",
        "https://",  # missing host
    ],
)
def test_frappe_base_url_bad_inputs_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        Settings(frappe_base_url=bad)


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


def test_get_settings_caches_the_instance(monkeypatch) -> None:
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "a" * MIN_SHARED_SECRET_LEN)
    reset_settings_cache()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings_cache_clears(monkeypatch) -> None:
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "a" * MIN_SHARED_SECRET_LEN)
    reset_settings_cache()
    a = get_settings()
    reset_settings_cache()
    b = get_settings()
    assert a is not b
    assert cfg_mod._settings is b


# ---------------------------------------------------------------------------
# Env-driven construction end-to-end
# ---------------------------------------------------------------------------


def test_env_vars_drive_settings(monkeypatch) -> None:
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "x" * 32)
    monkeypatch.setenv("AUXIMA_SIDECAR_LOG_LEVEL", "warning")
    monkeypatch.setenv("AUXIMA_SIDECAR_FRAPPE_BASE_URL", "https://frappe.internal:8000")
    monkeypatch.setenv("AUXIMA_SIDECAR_DEFAULT_MODEL", "ollama/llama3.1:8b")
    reset_settings_cache()
    s = get_settings()
    assert s.shared_secret_configured is True
    assert s.log_level == "WARNING"
    assert s.frappe_base_url == "https://frappe.internal:8000"
    assert s.default_model == "ollama/llama3.1:8b"


def test_extra_env_vars_are_ignored(monkeypatch) -> None:
    """``extra="ignore"`` keeps unrelated env vars from poisoning Settings."""
    monkeypatch.setenv("AUXIMA_SIDECAR_TOTALLY_UNKNOWN", "yes")
    reset_settings_cache()
    s = get_settings()
    assert not hasattr(s, "totally_unknown")
