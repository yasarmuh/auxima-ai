"""Tests for ``auxima_ai.util.phone`` — KSA phone normaliser.

Coverage:
  - is_valid_e164 strict shape (+, 8-15 digits, no leading 0 after +).
  - KSA mobile national format with the leading 0 round-trips.
  - KSA mobile national format without leading 0 still parses.
  - 00966xxx international-access prefix swaps for +966xxx.
  - 966xxx (no +) treated as international.
  - Already-E.164 +966xxx passes through.
  - Whitespace / dashes / parens / dots / "tel:" prefix stripped.
  - Invalid / non-mobile / wrong length / non-digit input returns None.
  - None / non-string input returns None (never raises).
  - Phone.e164 must always be valid (construction guard).
  - Phone result carries split country_code + national_number.
"""
from __future__ import annotations

import pytest

from auxima_ai.util.phone import (
    Phone,
    is_valid_e164,
    normalise_phone,
)


# ---------------------------------------------------------------------------
# is_valid_e164
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "+966512345678",     # 12 digits
        "+12025550123",      # US 11 digits
        "+447911123456",     # UK 12 digits
        "+9665123456",       # KSA-shape 10 digits (min for KSA, but valid E.164)
        "+1" + "0" * 14,     # 15 total digits — at the ceiling
    ],
)
def test_is_valid_e164_accepts_well_formed(good: str) -> None:
    assert is_valid_e164(good) is True


@pytest.mark.parametrize(
    "bad",
    [
        "",                  # empty
        "+",                 # no digits
        "+0123456789",       # leading 0 after + (E.164 forbids country code 0)
        "+966 50 0 0",       # spaces
        "966512345678",      # no leading +
        "+12345",            # too short (5 digits)
        "+" + "1" * 16,      # too long (16)
        "+966-512-345-678",  # dashes
        "tel:+966512345678", # URI prefix
    ],
)
def test_is_valid_e164_rejects_malformed(bad: str) -> None:
    assert is_valid_e164(bad) is False


@pytest.mark.parametrize("not_str", [None, 42, b"+966512345678", ["+966512345678"]])
def test_is_valid_e164_returns_false_for_non_string(not_str: object) -> None:
    assert is_valid_e164(not_str) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalise_phone — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+966512345678",       "+966512345678"),   # already E.164
        ("00966512345678",      "+966512345678"),   # 00-prefix
        ("966512345678",        "+966512345678"),   # no +
        ("0512345678",          "+966512345678"),   # national with leading 0
        ("512345678",           "+966512345678"),   # national without leading 0
        ("+966 50 000 0000",    "+966500000000"),   # spaces
        ("0 51 234 56 78",      "+966512345678"),   # spaces in national
        ("0(51)234-5678",       "+966512345678"),   # parens + dash
        ("0.51.234.56.78",      "+966512345678"),   # dots
        ("tel:+966512345678",   "+966512345678"),   # URI prefix
        ("TEL:+966512345678",   "+966512345678"),   # URI prefix case-insensitive
    ],
)
def test_ksa_mobile_normalises_to_e164(raw: str, expected: str) -> None:
    result = normalise_phone(raw)
    assert result is not None, f"expected normalisation to succeed for {raw!r}"
    assert result.e164 == expected
    assert is_valid_e164(result.e164)


def test_phone_object_carries_split_components() -> None:
    p = normalise_phone("0512345678")
    assert p is not None
    assert p.country_code == "966"
    assert p.national_number == "512345678"
    assert p.e164 == "+966512345678"


# ---------------------------------------------------------------------------
# normalise_phone — rejection paths (returns None, never raises)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "   ",
        "not-a-phone",
        "0212345678",      # KSA landline (starts with 0, not 05) — out of v1 scope
        "0512345",         # too short
        "05123456789012",  # too long
        "abcdefghij",      # all letters
        "++966512345678",  # double +
        "+966-abc-def",    # non-digit
        "0512abcd56",      # mixed digit + letters
    ],
)
def test_normalise_returns_none_for_bad_input(bad: object) -> None:
    assert normalise_phone(bad) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("non_str", [42, b"0512345678", ["0512345678"], {"phone": "0512345678"}])
def test_normalise_returns_none_for_non_string(non_str: object) -> None:
    assert normalise_phone(non_str) is None  # type: ignore[arg-type]


def test_unknown_default_country_returns_none() -> None:
    """A country not in the registry can'\''t parse a national-format string."""
    assert normalise_phone("0512345678", default_country="XX") is None


# ---------------------------------------------------------------------------
# Phone construction guard
# ---------------------------------------------------------------------------


def test_phone_construction_rejects_invalid_e164() -> None:
    with pytest.raises(ValueError, match="E.164"):
        Phone(e164="966512345678", country_code="966", national_number="512345678")


def test_phone_is_frozen() -> None:
    p = normalise_phone("0512345678")
    assert p is not None
    with pytest.raises((AttributeError, TypeError)):
        p.e164 = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip — output of normalise_phone is itself valid E.164
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "0512345678",
        "+966512345678",
        "00966512345678",
        "966512345678",
        "0 51 234 5678",
        "+966 50 000 0000",
    ],
)
def test_output_round_trips_through_is_valid_e164(raw: str) -> None:
    p = normalise_phone(raw)
    assert p is not None
    assert is_valid_e164(p.e164)


# ---------------------------------------------------------------------------
# International numbers (non-KSA) — pass-through when already E.164
# ---------------------------------------------------------------------------


def test_already_e164_non_ksa_passes_through() -> None:
    """A UK / US E.164 number is still recognised even though the default
    country is SA — it just won'\''t carry split country_code/national fields."""
    p = normalise_phone("+447911123456")
    assert p is not None
    assert p.e164 == "+447911123456"


def test_already_e164_through_00_prefix() -> None:
    p = normalise_phone("00447911123456")
    assert p is not None
    assert p.e164 == "+447911123456"
