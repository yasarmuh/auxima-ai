"""Email normaliser — strips formatting, validates shape, lowercases domain.

The intake.extract endpoint returns ``contact_email`` as whatever the
LLM extracted. Before the address reaches the CRM doctype we want:
  - leading / trailing whitespace stripped,
  - a ``mailto:`` URI prefix dropped,
  - the local part preserved as-is (technically case-sensitive per
    RFC 5321 §2.4, even though most providers ignore case),
  - the domain part lowercased (case-insensitive per RFC 1035 §2.3.3),
  - a strict shape check that catches the common LLM mistakes
    (missing ``@``, multiple ``@``, no TLD, empty local/domain).

We intentionally DON'T implement full RFC 5322 — that grammar accepts
"foo@bar" + comments + quoted local parts + escaped chars and is a
known XSS / injection surface. Our regex is the conservative subset
that every real-world transactional email service accepts.

Pure stdlib (``re`` + dataclasses); no email-validator dep
(over-engineered for a Phase-1 single-broker scope).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

# RFC 5321 §4.5.3.1.1: local part max 64 octets.
LOCAL_PART_MAX: Final[int] = 64
# RFC 5321 §4.5.3.1.2: domain max 255 octets.
DOMAIN_MAX: Final[int] = 255
# RFC 5321 §4.5.3.1.3: forwarded path max 256 octets — apply same limit overall.
EMAIL_MAX: Final[int] = 254  # widely-deployed practical cap (incl. @)


# Conservative shape:
#   local : non-empty; ASCII letters/digits + . _ % + - ; no leading / trailing dot;
#           no consecutive dots
#   domain: one or more labels separated by dot; each label = letters/digits/hyphen
#           not at the boundaries; final TLD label is letters only and >= 2 chars
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"^"
    r"(?P<local>[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*)"
    r"@"
    r"(?P<domain>"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}"
    r")"
    r"$",
)


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Email:
    """A successfully normalised email address.

    ``address`` is the canonical form to persist (local part untouched;
    domain lowercased). ``local`` and ``domain`` are kept split for the
    rare consumer that needs to bucket by domain (e.g. routing leads
    from ``@acme.sa`` to a named handler).
    """

    address: str
    local: str
    domain: str

    def __post_init__(self) -> None:
        if not is_valid_email(self.address):
            raise ValueError(f"Email.address must be valid; got {self.address!r}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_email(text: str) -> bool:
    """Pure-shape check against the conservative RFC-5321-subset regex.

    Also enforces the overall, local, and domain length caps per
    RFC 5321 §4.5.3.1.
    """
    if not isinstance(text, str):
        return False
    if not text or len(text) > EMAIL_MAX:
        return False
    m = _EMAIL_RE.match(text)
    if m is None:
        return False
    local = m.group("local")
    domain = m.group("domain")
    if len(local) > LOCAL_PART_MAX or len(domain) > DOMAIN_MAX:
        return False
    return True


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


_MAILTO_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^mailto:", re.IGNORECASE)


def normalise_email(text: str | None) -> Email | None:
    """Try to normalise ``text`` into an :class:`Email`.

    Returns ``None`` if the input is missing / not a string / doesn't
    satisfy :func:`is_valid_email` after trimming. Never raises on
    bad input — callers see ``None`` and decide whether to drop the
    field or surface the original.

    Normalisation steps:
      1. Strip surrounding whitespace and any ``mailto:`` URI prefix.
      2. Lowercase the domain part (case-insensitive per RFC 1035).
      3. Preserve the local part exactly as it appears (some providers
         distinguish ``Foo@x.co`` from ``foo@x.co``).
    """
    if text is None or not isinstance(text, str):
        return None

    cleaned = _MAILTO_PREFIX_RE.sub("", text.strip())
    if not cleaned:
        return None

    if not is_valid_email(cleaned):
        return None

    # Split on the LAST '@' so a (rare-but-valid) quoted-style local
    # with an embedded '@' wouldn't crash here. Our regex already
    # rejects local parts with @, so rsplit and split give the same
    # result; rsplit just costs nothing extra.
    local, domain = cleaned.rsplit("@", 1)
    canonical = f"{local}@{domain.lower()}"
    return Email(address=canonical, local=local, domain=domain.lower())


__all__ = (
    "DOMAIN_MAX",
    "EMAIL_MAX",
    "Email",
    "LOCAL_PART_MAX",
    "is_valid_email",
    "normalise_email",
)
