"""SSRF-safe outbound URL validator (OWASP A10, S-34 §3.6).

Webhooks send tenant-controlled URLs out from the sidecar. Without
explicit safeguards, an attacker who can control the destination URL
can make our server fetch ANY host reachable from our network —
including:

  * Cloud metadata endpoints (169.254.169.254 — the AWS / Azure /
    GCP instance-metadata service that returns IAM credentials to
    whatever calls it from the instance).
  * Internal admin panels (10.x, 192.168.x, 172.16-31.x).
  * The Frappe app itself (localhost:8000), bypassing the shared-
    secret auth.

This module checks every outbound URL against a tight allow-list of
properties:

  1. Scheme is ``http`` or ``https`` (no ``file://``, ``gopher://``,
     ``ftp://``, ``data:`` etc).
  2. URL carries no userinfo (``user:pass@host`` — would leak creds
     into the destination's access log).
  3. Host parses as a public unicast address. Resolving the hostname
     to its A / AAAA records and checking EVERY answer keeps a host
     that resolves to 127.0.0.1 OR a public IP from sneaking through
     by serving the public address first.
  4. Port is in the standard web range (80, 443 by default — extra
     ports can be opted in for partner integrations).

The DNS resolution step is the key SSRF defence. An attacker who
registers ``attacker.com`` with an A record of ``127.0.0.1`` would
otherwise pass a textual check; the resolver lookup catches them.

Pure stdlib (``ipaddress`` + ``socket`` + ``urllib.parse``); the
resolver is injectable so tests don't need real DNS.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Callable, Final, Iterable
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
DEFAULT_ALLOWED_PORTS: Final[frozenset[int]] = frozenset({80, 443})

# The metadata IPs that, if forgotten, hand an attacker IAM credentials.
# Already covered by the link-local + IPv6 unique-local checks below,
# but listed explicitly so the deny rationale is grep-able.
_METADATA_IPS: Final[frozenset[str]] = frozenset(
    {
        "169.254.169.254",  # AWS / Azure / GCP IPv4 IMDS
        "fd00:ec2::254",    # AWS IMDS over IPv6
    },
)


# Resolver protocol: a function that takes a host and returns an iterable
# of (family, ip_str) tuples. Wraps socket.getaddrinfo so tests can pass
# a deterministic mapping.
Resolver = Callable[[str], Iterable[tuple[int, str]]]


def default_resolver(host: str) -> list[tuple[int, str]]:
    """Resolve ``host`` to all of its A / AAAA records.

    Returns ``(family, ip_str)`` tuples (no port). On resolution failure
    raises :class:`HostUnresolvableError` so the caller sees the cause.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise HostUnresolvableError(f"DNS lookup failed for {host!r}: {e}") from e
    out: list[tuple[int, str]] = []
    for family, _socktype, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        out.append((family, ip))
    return out


# ---------------------------------------------------------------------------
# Errors — every refusal raises a subclass of URLValidationError so callers
# can distinguish bad-URL-syntax (4xx to the client) from network-config
# problems (5xx with retry) cleanly.
# ---------------------------------------------------------------------------


class URLValidationError(ValueError):
    """Base — every URL refusal raises a subclass of this."""


class MalformedURLError(URLValidationError):
    """Parse failure / missing host / not-a-string input."""


class DisallowedSchemeError(URLValidationError):
    """Scheme is not http / https."""


class UserInfoNotAllowedError(URLValidationError):
    """URL contains ``user:pass@host`` — credentials would leak."""


class DisallowedPortError(URLValidationError):
    """Port is outside the allow-list."""


class HostUnresolvableError(URLValidationError):
    """DNS resolution failed — caller may retry or DLQ the payload."""


class PrivateAddressError(URLValidationError):
    """Host resolves to a private / loopback / link-local / multicast IP."""


# ---------------------------------------------------------------------------
# Result type — keep the parsed components handy for the caller (avoids
# re-parsing in the HTTP client).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedURL:
    """The result of a successful :func:`validate_webhook_url` call."""

    url: str
    scheme: str
    host: str
    port: int
    path: str
    resolved_ips: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_ip(ip_str: str) -> str:
    """Return a label for ``ip_str`` — ``"public"`` iff it's globally routable."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return "invalid"
    # ``is_global`` is the canonical "publicly routable" predicate; it
    # already excludes loopback, link-local, private, multicast, reserved,
    # and unspecified ranges for both v4 and v6.
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_private:
        return "private"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_unspecified:
        return "unspecified"
    if not ip.is_global:
        return "non-global"
    return "public"


def _default_port_for_scheme(scheme: str) -> int:
    return 443 if scheme == "https" else 80


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_webhook_url(
    url: str,
    *,
    allowed_ports: Iterable[int] | None = None,
    allow_private: bool = False,
    resolver: Resolver = default_resolver,
) -> ValidatedURL:
    """Validate an outbound webhook URL against the SSRF allow-list.

    Parameters
    ----------
    url
        The URL to validate.
    allowed_ports
        Port allow-list; defaults to :data:`DEFAULT_ALLOWED_PORTS`
        (80, 443). Pass a wider set for partner endpoints on custom
        ports.
    allow_private
        Set to ``True`` ONLY in test / dev environments to permit
        loopback / private targets (e.g. ``http://localhost:8000``).
        Production callers MUST leave this ``False``; the default
        refuses everything that isn't a globally routable unicast
        address.
    resolver
        Hostname-to-IP resolver. Defaults to :func:`default_resolver`
        which wraps :func:`socket.getaddrinfo`. Tests inject a
        deterministic mapping to avoid network dependence.

    Returns
    -------
    :class:`ValidatedURL`
        With the parsed components and the full list of resolved IPs.

    Raises
    ------
    MalformedURLError | DisallowedSchemeError | UserInfoNotAllowedError |
    DisallowedPortError | HostUnresolvableError | PrivateAddressError
        Every refusal is one of these :class:`URLValidationError`
        subclasses.
    """
    if not isinstance(url, str) or not url.strip():
        raise MalformedURLError(f"url must be a non-empty string; got {url!r}")

    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise MalformedURLError(f"failed to parse URL {url!r}: {e}") from e

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise DisallowedSchemeError(
            f"scheme {scheme!r} not allowed; must be one of {sorted(ALLOWED_SCHEMES)}"
        )

    if parts.username is not None or parts.password is not None:
        raise UserInfoNotAllowedError(
            "URL contains userinfo (user:pass@host); credentials would "
            "leak to the destination's access log"
        )

    host = parts.hostname
    if not host:
        raise MalformedURLError(f"URL missing host: {url!r}")

    port = parts.port if parts.port is not None else _default_port_for_scheme(scheme)
    allowed = frozenset(allowed_ports) if allowed_ports is not None else DEFAULT_ALLOWED_PORTS
    if port not in allowed:
        raise DisallowedPortError(
            f"port {port} not allowed; must be one of {sorted(allowed)}"
        )

    # If the host is itself an IP literal, urlsplit returns the bare
    # form (square brackets stripped for IPv6). Either way, resolve so
    # the metadata-IP / private-range checks run on EVERY answer; that
    # closes the "DNS pinning" + "first answer is public, second is
    # 127.0.0.1" attack class.
    resolved = list(resolver(host))
    if not resolved:
        raise HostUnresolvableError(f"host {host!r} resolved to no addresses")

    resolved_ips = tuple(ip for _family, ip in resolved)

    if not allow_private:
        for ip in resolved_ips:
            label = _classify_ip(ip)
            if label != "public":
                raise PrivateAddressError(
                    f"host {host!r} resolves to {ip} ({label}); "
                    f"outbound calls to non-public IPs are refused (SSRF defence)"
                )

    return ValidatedURL(
        url=url,
        scheme=scheme,
        host=host,
        port=port,
        path=parts.path or "/",
        resolved_ips=resolved_ips,
    )


__all__ = (
    "ALLOWED_SCHEMES",
    "DEFAULT_ALLOWED_PORTS",
    "DisallowedPortError",
    "DisallowedSchemeError",
    "HostUnresolvableError",
    "MalformedURLError",
    "PrivateAddressError",
    "Resolver",
    "URLValidationError",
    "UserInfoNotAllowedError",
    "ValidatedURL",
    "default_resolver",
    "validate_webhook_url",
)
