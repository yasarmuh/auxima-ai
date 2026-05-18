"""Phone-number normaliser — converts human formats to E.164.

The intake.extract endpoint returns ``contact_phone`` as whatever the
LLM extracted, which means anything from ``"+966 50 000 0000"`` to
``"00966-512345678"`` to ``"0512345678"``. The CRM doctype expects
strict E.164 (``+`` + country code + national digits, 8-15 digits
total per ITU-T E.164). This module bridges the two.

Phase 1 scope (CLAUDE §1): KSA only. The design supports a registry
of :class:`CountryRules` keyed by ISO 3166-1 alpha-2 code, so adding
UAE / Qatar / Bahrain / Kuwait / Oman in Phase 3+ is a one-tuple
change rather than a rewrite. The default country is SA.

Behaviour:
  - ``normalise_phone(text)``  -> :class:`Phone` | ``None``.
    Returns ``None`` on unparseable input — callers decide whether to
    drop the field, leave the raw text, or log a normalisation miss.
  - ``is_valid_e164(text)``    -> ``bool``.
    Pure shape check against the ITU-T E.164 rules (no country
    knowledge — just ``+`` + 8-15 digits).

Pure stdlib (``re`` + dataclasses); no phonenumbers / libphonenumber
dep (3MB+ of locale data is over-engineered for a Phase-1 single-
country scope).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# E.164 shape — ITU-T spec
# ---------------------------------------------------------------------------

E164_MIN_DIGITS: Final[int] = 8
E164_MAX_DIGITS: Final[int] = 15
_E164_RE: Final[re.Pattern[str]] = re.compile(r"^\+[1-9]\d{7,14}$")


def is_valid_e164(text: str) -> bool:
    """Pure-shape check: ``+`` + non-zero country digit + 7-14 more digits."""
    if not isinstance(text, str):
        return False
    return bool(_E164_RE.match(text))


# ---------------------------------------------------------------------------
# Country rules — Phase 1 carries SA only; design is extensible
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountryRules:
    """Per-country dialing rules used by :func:`normalise_phone`."""

    iso2: str
    country_code: str  # without "+"
    national_prefix: str  # what users dial at the start of a domestic number
    mobile_national_length: int  # incl. the national_prefix
    mobile_prefixes: tuple[str, ...]  # first digit(s) AFTER national prefix


# KSA: country code 966; national prefix "0"; mobile = 10 digits incl.
# the leading 0 (e.g. "0512345678"); mobile range starts with "5".
RULES_SA: Final[CountryRules] = CountryRules(
    iso2="SA",
    country_code="966",
    national_prefix="0",
    mobile_national_length=10,
    mobile_prefixes=("5",),
)

# Registry keyed by ISO 3166-1 alpha-2 — extend as Phase 3+ markets land.
_COUNTRY_REGISTRY: Final[dict[str, CountryRules]] = {
    "SA": RULES_SA,
}


# ---------------------------------------------------------------------------
# Phone value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phone:
    """A successfully normalised phone number."""

    e164: str
    country_code: str
    national_number: str

    def __post_init__(self) -> None:
        if not is_valid_e164(self.e164):
            raise ValueError(f"Phone.e164 must be valid E.164; got {self.e164!r}")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


# Allow `tel:` URI scheme, whitespace, dashes, parens, dots — strip all of them.
_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"[\s\-\(\)\.]")
_TEL_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^tel:", re.IGNORECASE)
_LEADING_PLUS: Final[str] = "+"
_LEADING_DOUBLE_ZERO: Final[str] = "00"


def _strip_formatting(text: str) -> str:
    """Remove whitespace, dashes, parens, dots, and the optional tel: prefix."""
    s = _TEL_PREFIX_RE.sub("", text)
    return _STRIP_RE.sub("", s)


def normalise_phone(
    text: str | None,
    *,
    default_country: str = "SA",
) -> Phone | None:
    """Try to normalise ``text`` into an E.164 :class:`Phone`.

    Recognised input shapes (with optional whitespace / dashes / parens
    / dots / ``tel:`` URI prefix):

      - ``+966512345678``    (already E.164)
      - ``00966512345678``   (international-access prefix)
      - ``966512345678``     (international without ``+``)
      - ``0512345678``       (national format — default country applied)
      - ``512345678``        (national without the national prefix)

    Returns ``None`` if:
      - ``text`` is ``None`` / empty / not a string,
      - the cleaned text contains non-digit characters besides the
        leading ``+``,
      - the result doesn't satisfy E.164 (8-15 digits, starts with non-zero),
      - the country in question doesn't recognise the local form.

    Never raises on bad input; callers get ``None`` and decide what to
    log / drop / surface.
    """
    if text is None or not isinstance(text, str):
        return None
    cleaned = _strip_formatting(text)
    if not cleaned:
        return None

    # Variant 1: already starts with "+" -> assume E.164 form.
    if cleaned.startswith(_LEADING_PLUS):
        digits = cleaned[1:]
        if not digits.isdigit():
            return None
        return _build_if_valid_e164("+" + digits)

    # Variant 2: "00" international-access prefix -> swap for "+".
    if cleaned.startswith(_LEADING_DOUBLE_ZERO):
        digits = cleaned[2:]
        if not digits.isdigit():
            return None
        return _build_if_valid_e164("+" + digits)

    # Variant 3 / 4 / 5: national form (with or without national prefix)
    # OR a bare country-code-prefixed form (no leading + / 00).
    if not cleaned.isdigit():
        return None

    rules = _COUNTRY_REGISTRY.get(default_country)
    if rules is None:
        return None

    # If the cleaned string starts with the country code AND the total
    # length is plausible E.164, treat it as international.
    if cleaned.startswith(rules.country_code):
        candidate = "+" + cleaned
        result = _build_if_valid_e164(candidate)
        if result is not None:
            return result
        # Fall through — could be a national number that happens to
        # start with the country-code digits.

    # National form — apply per-country rule.
    if default_country == "SA":
        return _normalise_sa_national(cleaned, rules)

    return None


# ---------------------------------------------------------------------------
# Per-country helpers
# ---------------------------------------------------------------------------


def _normalise_sa_national(digits: str, rules: CountryRules) -> Phone | None:
    """Convert a KSA national-format string into E.164.

    Two accepted local shapes:
      - 10 digits starting with ``05``  (with the national prefix)
      - 9 digits starting with ``5``    (without the national prefix)
    """
    # With national prefix: 0 + 9 digits starting with 5.
    if (
        len(digits) == rules.mobile_national_length
        and digits.startswith(rules.national_prefix)
        and digits[len(rules.national_prefix):len(rules.national_prefix) + 1]
            in rules.mobile_prefixes
    ):
        national = digits[len(rules.national_prefix):]
        return _build_if_valid_e164(f"+{rules.country_code}{national}")

    # Without national prefix: 9 digits starting with 5.
    if (
        len(digits) == rules.mobile_national_length - len(rules.national_prefix)
        and digits[:1] in rules.mobile_prefixes
    ):
        return _build_if_valid_e164(f"+{rules.country_code}{digits}")

    return None


def _build_if_valid_e164(candidate: str) -> Phone | None:
    """Construct a :class:`Phone` iff ``candidate`` passes E.164 validation."""
    if not is_valid_e164(candidate):
        return None
    # Strip the leading "+"; split country from national portion.
    digits = candidate[1:]
    # Conservatively use the first 1-3 digits as country code based on
    # the registry — for SA that's "966". Falling back to "1" leaves
    # us no parse for unregistered countries, but Phase 1 is SA-only
    # so this is enough.
    cc = None
    for rules in _COUNTRY_REGISTRY.values():
        if digits.startswith(rules.country_code):
            cc = rules.country_code
            break
    if cc is None:
        # Unknown country — we still return a valid E.164 Phone but
        # leave the split blank-ish so callers don't pretend to know.
        return Phone(e164=candidate, country_code="", national_number=digits)
    national = digits[len(cc):]
    return Phone(e164=candidate, country_code=cc, national_number=national)


__all__ = (
    "CountryRules",
    "E164_MAX_DIGITS",
    "E164_MIN_DIGITS",
    "Phone",
    "RULES_SA",
    "is_valid_e164",
    "normalise_phone",
)
