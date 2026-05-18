"""KSA IBAN validator + normaliser (ISO 13616 + ISO 7064 mod-97-10).

KSA IBAN format (per the Saudi Central Bank's IBAN registry):

    SA + 2 check digits + 2-digit bank code + 18-digit basic account number
    = 24 characters total

Validation steps (ISO 7064 mod-97-10, applied to any country IBAN):

    1. Strip whitespace + any IBAN: URI prefix; uppercase what remains.
    2. Confirm overall shape (country code + 2 digits + alphanumerics
       totalling the country-specific length).
    3. Move the first 4 characters (CC + check digits) to the END.
    4. Replace each letter with its position-derived digits:
         A = 10, B = 11, ..., Z = 35.
    5. Interpret the resulting decimal string as ONE big integer.
    6. The integer mod 97 must equal 1.

The mod-97 check catches all single-character substitutions and the
overwhelming majority of two-character transpositions — the actual
guarantee in the ISO 7064 paper. Any mistyped IBAN is rejected
before the broker'\''s payments engine sees it.

KSA pays for the storage of a 2-digit bank code separately from the
BBAN; accessors expose both. Other countries' IBANs are recognised
for the shape check + checksum but the country-specific BBAN
breakdown is left to a future module that knows each country's
schema.

Pure stdlib (``re`` + dataclasses); no ``schwifty`` / ``stdnum`` dep
(both pull country-data tables that are over-engineered for v1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

# IBAN per-country length registry — SA is the only one we extract bank
# code / BBAN for in v1; the others (UAE/QA/BH/KW/OM) are accepted only
# at shape-check level so a foreign IBAN doesn't crash the validator.
_IBAN_LENGTHS: Final[dict[str, int]] = {
    "SA": 24,
    "AE": 23,
    "QA": 29,
    "BH": 22,
    "KW": 30,
    "OM": 16,
    # Common reference country lengths so cross-border tests work.
    "GB": 22,
    "DE": 22,
    "FR": 27,
    "US": 0,  # USA has no IBAN scheme — keeps the lookup clean.
}

# KSA bank-code -> short-name lookup. Sourced from SAMA's IBAN registry
# (the old SAMA — now Insurance Authority for insurance; banking-side
# IBAN registry remains SAMA's). The mapping is informational only —
# validation works regardless of whether the bank code is known.
_KSA_BANK_CODES: Final[dict[str, str]] = {
    "10": "NCB",       # National Commercial Bank (now Saudi National Bank)
    "15": "ALINMA",
    "20": "ARNB",      # Arab National Bank
    "30": "STC PAY",
    "40": "ALAWWAL",
    "45": "SAIB",      # Saudi Investment Bank
    "50": "RIBL",      # Riyad Bank
    "55": "ALBILAD",
    "60": "ALJAZIRA",
    "65": "ALRAJHI",
    "80": "SABB",      # Saudi British Bank
}

# Country code (2 letters) + 2 check digits + 11-30 alphanumerics.
_IBAN_BBAN_CHARS_RE: Final[re.Pattern[str]] = re.compile(r"\A[A-Z]{2}\d{2}[A-Z0-9]+\Z")
_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"\s")
_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^iban:", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IBANError(ValueError):
    """Base — every IBAN failure raises a subclass of this."""


class MalformedIBANError(IBANError):
    """Wrong shape: bad chars, wrong length, missing country code, etc."""


class ChecksumError(IBANError):
    """Shape OK but the ISO 7064 mod-97 check failed."""


class UnsupportedCountryError(IBANError):
    """Country code is valid 2-letter ISO but we don't carry its length."""


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBAN:
    """A successfully validated IBAN."""

    value: str          # canonical: uppercase, no whitespace
    country: str        # 2-letter ISO country code
    check_digits: str   # 2 digits — the IBAN's own checksum
    bban: str           # everything after the check digits

    def __post_init__(self) -> None:
        if not is_valid_iban(self.value):
            raise IBANError(f"IBAN.value must pass mod-97 validation; got {self.value!r}")

    # -- KSA-specific accessors (return "" for non-SA IBANs) ---------------

    @property
    def ksa_bank_code(self) -> str:
        """First 2 BBAN digits — meaningful for SA IBANs; ``""`` otherwise."""
        if self.country != "SA":
            return ""
        return self.bban[:2]

    @property
    def ksa_bank_name(self) -> str:
        """Short name for the KSA bank, or ``""`` if unknown / non-SA."""
        return _KSA_BANK_CODES.get(self.ksa_bank_code, "") if self.country == "SA" else ""

    @property
    def ksa_account_number(self) -> str:
        """18-digit account number portion of a KSA IBAN; ``""`` otherwise."""
        if self.country != "SA":
            return ""
        return self.bban[2:]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Strip whitespace + the optional IBAN: URI prefix; uppercase."""
    stripped = _PREFIX_RE.sub("", text.strip())
    return _STRIP_RE.sub("", stripped).upper()


def _mod97_check(value: str) -> int:
    """Return ``value`` interpreted per ISO 7064 mod-97-10. Returns the mod."""
    # Move first 4 chars to the end (the country + check digits).
    rearranged = value[4:] + value[:4]
    # Replace each letter with its 2-digit code (A=10..Z=35).
    digits = "".join(
        ch if ch.isdigit() else str(ord(ch) - ord("A") + 10)
        for ch in rearranged
    )
    return int(digits) % 97


def is_valid_iban(text: str) -> bool:
    """``True`` iff ``text`` is a well-formed IBAN passing ISO 7064 mod-97.

    Shape rules:
      - 2-letter country code at start
      - 2 check digits next
      - Total length matches the country's registered IBAN length
        (returns False for unsupported countries — use :func:`parse_iban`
        if you need the typed UnsupportedCountryError instead).
    """
    if not isinstance(text, str):
        return False
    cleaned = _clean(text)
    if not _IBAN_BBAN_CHARS_RE.match(cleaned):
        return False
    country = cleaned[:2]
    expected_len = _IBAN_LENGTHS.get(country)
    if not expected_len or len(cleaned) != expected_len:
        return False
    return _mod97_check(cleaned) == 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_iban(text: str | None) -> IBAN | None:
    """Try to parse ``text`` into a validated :class:`IBAN`.

    Returns ``None`` on any input that fails shape OR checksum. Never
    raises on bad input — callers see ``None`` and decide what to log.

    For the typed exception API (distinguish malformed vs checksum vs
    unsupported-country), call :func:`parse_iban_strict` instead.
    """
    if text is None or not isinstance(text, str):
        return None
    try:
        return parse_iban_strict(text)
    except IBANError:
        return None


def parse_iban_strict(text: str) -> IBAN:
    """Parse ``text`` into a validated :class:`IBAN` or raise.

    Raises:
      :class:`MalformedIBANError` for bad chars / wrong shape /
        unknown country / wrong length.
      :class:`UnsupportedCountryError` for ISO-shaped country codes
        whose IBAN length isn't in the registry yet.
      :class:`ChecksumError` for shape-OK / checksum-bad input.
    """
    if not isinstance(text, str):
        raise MalformedIBANError(f"IBAN must be str; got {type(text).__name__}")
    cleaned = _clean(text)
    if not cleaned:
        raise MalformedIBANError("IBAN must be a non-empty string")
    if not _IBAN_BBAN_CHARS_RE.match(cleaned):
        raise MalformedIBANError(
            "IBAN must be country code + 2 check digits + alphanumerics; "
            f"got {cleaned!r}"
        )
    country = cleaned[:2]
    expected_len = _IBAN_LENGTHS.get(country)
    if expected_len is None:
        raise UnsupportedCountryError(
            f"IBAN country code {country!r} not in length registry; "
            f"supported: {sorted(k for k, v in _IBAN_LENGTHS.items() if v)}"
        )
    if expected_len == 0:
        raise UnsupportedCountryError(
            f"country {country!r} does not participate in IBAN"
        )
    if len(cleaned) != expected_len:
        raise MalformedIBANError(
            f"IBAN for country {country!r} must be {expected_len} chars; "
            f"got {len(cleaned)}"
        )
    if _mod97_check(cleaned) != 1:
        raise ChecksumError(f"IBAN {cleaned!r} failed ISO 7064 mod-97 checksum")
    return IBAN(
        value=cleaned,
        country=country,
        check_digits=cleaned[2:4],
        bban=cleaned[4:],
    )


__all__ = (
    "ChecksumError",
    "IBAN",
    "IBANError",
    "MalformedIBANError",
    "UnsupportedCountryError",
    "is_valid_iban",
    "parse_iban",
    "parse_iban_strict",
)
