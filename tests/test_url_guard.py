"""Tests for ``auxima_ai.webhooks.url_guard`` — SSRF-safe outbound URL guard.

Coverage per OWASP A10 + S-34 §3.6:
  - Public URLs pass.
  - file:// / gopher:// / ftp:// / data: / javascript: schemes rejected.
  - URLs with user:pass@host credentials rejected.
  - Non-standard ports rejected unless explicitly allow-listed.
  - Hostname that resolves to 127.0.0.1 rejected (the canonical SSRF attack).
  - 169.254.169.254 (cloud metadata) rejected.
  - 10.x / 172.16-31.x / 192.168.x (private RFC1918) rejected.
  - ::1 / fc00::/7 / fe80::/10 (IPv6 loopback/private/link-local) rejected.
  - Multicast / reserved / unspecified rejected.
  - Multi-answer DNS where ANY answer is private -> rejection (DNS pinning).
  - Empty / None / whitespace URL rejected.
  - Missing host rejected.
  - DNS lookup failure raises HostUnresolvableError.
  - allow_private=True bypass works for dev/test.
  - Custom allowed_ports allow opt-in for partner integrations.
  - ValidatedURL is frozen and carries the parsed components + resolved IPs.
"""
from __future__ import annotations

import socket
from typing import Iterable

import pytest

from auxima_ai.webhooks.url_guard import (
    DEFAULT_ALLOWED_PORTS,
    DisallowedPortError,
    DisallowedSchemeError,
    HostUnresolvableError,
    MalformedURLError,
    PrivateAddressError,
    URLValidationError,
    UserInfoNotAllowedError,
    ValidatedURL,
    validate_webhook_url,
)


# ---------------------------------------------------------------------------
# Fake resolvers
# ---------------------------------------------------------------------------


def _resolver_returning(ips: Iterable[str]):
    def _resolve(host: str):
        return [(socket.AF_INET, ip) for ip in ips]
    return _resolve


_PUBLIC_RESOLVER = _resolver_returning(["93.184.216.34"])  # example.com


# ---------------------------------------------------------------------------
# Happy path — public URLs pass
# ---------------------------------------------------------------------------


def test_public_https_url_accepted() -> None:
    r = validate_webhook_url(
        "https://example.com/webhook",
        resolver=_PUBLIC_RESOLVER,
    )
    assert isinstance(r, ValidatedURL)
    assert r.scheme == "https"
    assert r.host == "example.com"
    assert r.port == 443
    assert r.path == "/webhook"
    assert r.resolved_ips == ("93.184.216.34",)


def test_public_http_url_accepted() -> None:
    r = validate_webhook_url(
        "http://example.com/wh",
        resolver=_PUBLIC_RESOLVER,
    )
    assert r.port == 80


def test_public_url_default_path() -> None:
    """Empty path normalises to "/" so downstream HTTP clients don't need to."""
    r = validate_webhook_url(
        "https://example.com",
        resolver=_PUBLIC_RESOLVER,
    )
    assert r.path == "/"


# ---------------------------------------------------------------------------
# Scheme rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "gopher://example.com/_",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "ws://example.com/sock",
        "ldap://example.com/dc=foo",
    ],
)
def test_non_http_schemes_rejected(url: str) -> None:
    with pytest.raises(DisallowedSchemeError):
        validate_webhook_url(url, resolver=_PUBLIC_RESOLVER)


# ---------------------------------------------------------------------------
# Credentials in URL — leak risk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.com/wh",
        "https://user@example.com/wh",
        "https://:pass@example.com/wh",
    ],
)
def test_userinfo_rejected(url: str) -> None:
    with pytest.raises(UserInfoNotAllowedError):
        validate_webhook_url(url, resolver=_PUBLIC_RESOLVER)


# ---------------------------------------------------------------------------
# Port allow-list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", [22, 25, 5432, 8000, 8080, 8443])
def test_non_standard_ports_rejected_by_default(port: int) -> None:
    with pytest.raises(DisallowedPortError):
        validate_webhook_url(
            f"https://example.com:{port}/wh",
            resolver=_PUBLIC_RESOLVER,
        )


def test_custom_allowed_ports_let_partner_integrations_through() -> None:
    r = validate_webhook_url(
        "https://example.com:8443/wh",
        allowed_ports={443, 8443},
        resolver=_PUBLIC_RESOLVER,
    )
    assert r.port == 8443


def test_default_allowed_ports_are_80_and_443() -> None:
    assert DEFAULT_ALLOWED_PORTS == frozenset({80, 443})


# ---------------------------------------------------------------------------
# Private / loopback / link-local rejection — the SSRF core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",        # IPv4 loopback
        "127.5.6.7",        # IPv4 loopback (full /8)
        "169.254.169.254",  # cloud metadata (AWS/Azure/GCP)
        "169.254.0.1",      # link-local
        "10.0.0.1",         # RFC1918
        "172.16.5.6",
        "172.31.255.254",
        "192.168.1.1",
        "0.0.0.0",          # unspecified
        "224.0.0.1",        # multicast
        "240.0.0.1",        # reserved
    ],
)
def test_attacker_controlled_dns_to_private_ipv4_rejected(ip: str) -> None:
    """Hostname textually clean; resolves to a private IP — must be refused."""
    with pytest.raises(PrivateAddressError):
        validate_webhook_url(
            "https://attacker-controlled.com/wh",
            resolver=_resolver_returning([ip]),
        )


@pytest.mark.parametrize(
    "ip",
    [
        "::1",             # IPv6 loopback
        "fc00::1",         # IPv6 unique-local
        "fe80::1",         # IPv6 link-local
        "ff00::1",         # IPv6 multicast
    ],
)
def test_private_ipv6_rejected(ip: str) -> None:
    def resolver(host: str):
        return [(socket.AF_INET6, ip)]
    with pytest.raises(PrivateAddressError):
        validate_webhook_url(
            "https://attacker-controlled.com/wh",
            resolver=resolver,
        )


def test_dns_pinning_attack_rejected_if_any_answer_private() -> None:
    """Multi-answer DNS: first IP public, second IP loopback.

    A naive validator that only checks the first answer is bypassable —
    the resolver returns 127.0.0.1 to the HTTP client later. We check
    EVERY answer so a single bad IP poisons the whole resolution.
    """
    mixed = _resolver_returning(["93.184.216.34", "127.0.0.1"])
    with pytest.raises(PrivateAddressError):
        validate_webhook_url("https://attacker.com/wh", resolver=mixed)


def test_direct_loopback_ip_literal_rejected() -> None:
    """No DNS involved — the URL bakes in 127.0.0.1 directly."""
    def resolver(host: str):
        return [(socket.AF_INET, host)]
    with pytest.raises(PrivateAddressError):
        validate_webhook_url("https://127.0.0.1/wh", resolver=resolver)


# ---------------------------------------------------------------------------
# allow_private bypass — dev / test only
# ---------------------------------------------------------------------------


def test_allow_private_lets_loopback_through_for_dev() -> None:
    def resolver(host: str):
        return [(socket.AF_INET, "127.0.0.1")]
    r = validate_webhook_url(
        "https://localhost/wh",
        allow_private=True,
        resolver=resolver,
    )
    assert r.host == "localhost"


# ---------------------------------------------------------------------------
# Malformed / missing parts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_or_none_url_rejected(bad: object) -> None:
    with pytest.raises(MalformedURLError):
        validate_webhook_url(bad, resolver=_PUBLIC_RESOLVER)  # type: ignore[arg-type]


def test_missing_host_rejected() -> None:
    with pytest.raises((MalformedURLError, DisallowedSchemeError)):
        validate_webhook_url("https:///path", resolver=_PUBLIC_RESOLVER)


def test_non_string_url_rejected() -> None:
    with pytest.raises(MalformedURLError):
        validate_webhook_url(42, resolver=_PUBLIC_RESOLVER)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DNS failures
# ---------------------------------------------------------------------------


def test_dns_resolution_failure_raises_host_unresolvable() -> None:
    def angry_resolver(host: str):
        raise HostUnresolvableError("simulated NXDOMAIN")
    with pytest.raises(HostUnresolvableError):
        validate_webhook_url("https://nxdomain.test/wh", resolver=angry_resolver)


def test_empty_resolution_set_raises() -> None:
    def empty_resolver(host: str):
        return []
    with pytest.raises(HostUnresolvableError):
        validate_webhook_url("https://example.com/wh", resolver=empty_resolver)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        MalformedURLError,
        DisallowedSchemeError,
        UserInfoNotAllowedError,
        DisallowedPortError,
        HostUnresolvableError,
        PrivateAddressError,
    ],
)
def test_every_failure_is_a_url_validation_error(exc: type) -> None:
    assert issubclass(exc, URLValidationError)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_validated_url_is_frozen() -> None:
    r = validate_webhook_url("https://example.com/wh", resolver=_PUBLIC_RESOLVER)
    with pytest.raises((AttributeError, TypeError)):
        r.url = "tampered"  # type: ignore[misc]
