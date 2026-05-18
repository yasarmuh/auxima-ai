"""KSA National ID + Commercial Registration shape validators.

Per CLAUDE §6 ("Cite when claiming a regulator fact"), this module
sticks to documented + verifiable rules:

  - **National ID / Iqama** (Saudi Authority for Statistics):
      10 digits exactly. Leading ``1`` = Saudi citizen. Leading ``2``
      = resident / Iqama holder. All other leading digits invalid.
  - **Commercial Registration (CR)** (Ministry of Commerce / SAGIA):
      10 digits exactly. Modern issuance starts with ``7``; legacy
      Riyadh-issued CRs start with ``1010``. No public checksum
      algorithm — validity is established by lookup against the MOC
      register, not by the number itself.

**Checksum status.** The widely-circulated "KSA National ID uses
Luhn" claim is undocumented in any primary source we can cite, so
this module ships SHAPE VALIDATION ONLY. When a citation lands (the
Saudi Authority for Statistics published algorithm spec, or an
official IA verifier endpoint), checksum validation goes behind a
new :func:`verify_checksum` hook without changing the parse API.

Pure stdlib; no third-party deps. Strings only — bytes / ints / etc
fail validation (callers normalise upstream).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

NATIONAL_ID_LEN: Final[int] = 10
CR_LEN: Final[int] = 10

_NATIONAL_ID_RE: Final[re.Pattern[str]] = re.compile(r"\A[12]\d{9}\Z")
# Modern (leading 7) OR legacy Riyadh (leading 1010).
# Use \A / \Z (not ^ / $) so a trailing newline isn't silently accepted —
# Python's $ matches just before a final \n by default.
_CR_RE: Final[re.Pattern[str]] = re.compile(r"\A(?:7\d{9}|10\d{8})\Z")


class IDType(str, Enum):
    """Whether the holder is a Saudi citizen or a resident (Iqama)."""

    CITIZEN = "citizen"
    RESIDENT = "resident"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NationalID:
    """A successfully parsed KSA National ID / Iqama (shape-validated)."""

    digits: str
    type: IDType

    def __post_init__(self) -> None:
        if not is_valid_national_id(self.digits):
            raise ValueError(f"NationalID.digits must be 10 digits with leading 1 or 2; got {self.digits!r}")


@dataclass(frozen=True)
class CommercialRegistration:
    """A successfully parsed KSA Commercial Registration (shape-validated)."""

    digits: str

    def __post_init__(self) -> None:
        if not is_valid_cr(self.digits):
            raise ValueError(f"CommercialRegistration.digits must be 10 digits with leading 7 or 1010; got {self.digits!r}")


# ---------------------------------------------------------------------------
# Shape predicates
# ---------------------------------------------------------------------------


def is_valid_national_id(text: str) -> bool:
    """``True`` iff ``text`` is 10 digits starting with 1 (citizen) or 2 (resident).

    Does NOT verify any checksum — see module docstring.
    """
    if not isinstance(text, str):
        return False
    return bool(_NATIONAL_ID_RE.match(text))


def is_valid_cr(text: str) -> bool:
    """``True`` iff ``text`` is 10 digits matching the modern (7…) or legacy (1010…) prefix."""
    if not isinstance(text, str):
        return False
    return bool(_CR_RE.match(text))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"[\s\-]")


def _clean(text: str) -> str:
    """Strip whitespace + dashes; leave everything else for validation to reject."""
    return _STRIP_RE.sub("", text)


def parse_national_id(text: str | None) -> NationalID | None:
    """Try to parse ``text`` into a :class:`NationalID`.

    Strips surrounding whitespace + internal dashes (common in human
    formats like ``"1234-567890"``) before validation. Returns ``None``
    on any input that fails the shape predicate — never raises.
    """
    if text is None or not isinstance(text, str):
        return None
    cleaned = _clean(text)
    if not is_valid_national_id(cleaned):
        return None
    id_type = IDType.CITIZEN if cleaned[0] == "1" else IDType.RESIDENT
    return NationalID(digits=cleaned, type=id_type)


def parse_cr(text: str | None) -> CommercialRegistration | None:
    """Try to parse ``text`` into a :class:`CommercialRegistration`.

    Same whitespace + dash handling as :func:`parse_national_id`.
    Returns ``None`` on any input that fails the shape predicate.
    """
    if text is None or not isinstance(text, str):
        return None
    cleaned = _clean(text)
    if not is_valid_cr(cleaned):
        return None
    return CommercialRegistration(digits=cleaned)


__all__ = (
    "CR_LEN",
    "CommercialRegistration",
    "IDType",
    "NATIONAL_ID_LEN",
    "NationalID",
    "is_valid_cr",
    "is_valid_national_id",
    "parse_cr",
    "parse_national_id",
)
