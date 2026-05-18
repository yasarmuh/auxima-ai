"""Tests for ``auxima_ai.util.iban`` — ISO 7064 mod-97 + KSA accessors.

Coverage:
  - Known-valid SA IBAN (the SAMA reference) passes.
  - Known-valid UK / DE / FR IBANs pass.
  - A single-character substitution in the SA IBAN -> ChecksumError.
  - A two-character transposition in the SA IBAN -> ChecksumError.
  - Wrong length for country -> MalformedIBANError.
  - Lowercase input is uppercased before validation.
  - Whitespace + "IBAN:" URI prefix stripped.
  - Bad chars (e.g. punctuation) -> MalformedIBANError.
  - Unsupported country code -> UnsupportedCountryError.
  - parse_iban returns None on bad input; parse_iban_strict raises typed.
  - IBAN value object construction guard.
  - KSA bank-code / bank-name / account-number accessors.
  - Non-SA IBAN returns "" for KSA-specific accessors.
"""
from __future__ import annotations

import pytest

from auxima_ai.util.iban import (
    IBAN,
    ChecksumError,
    IBANError,
    MalformedIBANError,
    UnsupportedCountryError,
    is_valid_iban,
    parse_iban,
    parse_iban_strict,
)

# Public reference IBAN from the SAMA registry:
SA_VALID = "SA0380000000608010167519"
# Other well-known reference IBANs (from ISO 13616 examples + ECBS test set):
GB_VALID = "GB82WEST12345698765432"
DE_VALID = "DE89370400440532013000"
FR_VALID = "FR1420041010050500013M02606"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good", [SA_VALID, GB_VALID, DE_VALID, FR_VALID])
def test_is_valid_iban_accepts_known_good(good: str) -> None:
    assert is_valid_iban(good) is True


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sa0380000000608010167519",         SA_VALID),  # lowercase
        ("SA03 8000 0000 6080 1016 7519",    SA_VALID),  # spaced groups
        ("  SA0380000000608010167519  ",     SA_VALID),  # whitespace
        ("IBAN:SA0380000000608010167519",    SA_VALID),  # URI prefix
        ("iban:SA0380000000608010167519",    SA_VALID),  # URI prefix case-insens
    ],
)
def test_is_valid_iban_normalises_formatting(raw: str, expected: str) -> None:
    assert is_valid_iban(raw) is True


def test_parse_iban_returns_value_object() -> None:
    iban = parse_iban(SA_VALID)
    assert iban is not None
    assert iban.value == SA_VALID
    assert iban.country == "SA"
    assert iban.check_digits == "03"
    assert iban.bban == "80000000608010167519"


def test_parse_iban_normalises_lowercase_and_spaces() -> None:
    iban = parse_iban("sa03 8000 0000 6080 1016 7519")
    assert iban is not None
    assert iban.value == SA_VALID


# ---------------------------------------------------------------------------
# KSA accessors
# ---------------------------------------------------------------------------


def test_ksa_bank_code_for_sa_iban() -> None:
    iban = parse_iban(SA_VALID)
    assert iban is not None
    assert iban.ksa_bank_code == "80"  # 80 = SABB per the registry


def test_ksa_bank_name_for_known_code() -> None:
    iban = parse_iban(SA_VALID)
    assert iban is not None
    assert iban.ksa_bank_name == "SABB"


def test_ksa_account_number_is_18_digits() -> None:
    iban = parse_iban(SA_VALID)
    assert iban is not None
    assert len(iban.ksa_account_number) == 18


def test_ksa_accessors_empty_for_non_sa_iban() -> None:
    iban = parse_iban(GB_VALID)
    assert iban is not None
    assert iban.ksa_bank_code == ""
    assert iban.ksa_bank_name == ""
    assert iban.ksa_account_number == ""


# ---------------------------------------------------------------------------
# Checksum failure detection
# ---------------------------------------------------------------------------


def test_single_char_substitution_rejected() -> None:
    """Mutate a digit in the SA reference -> mod-97 must catch it."""
    tampered = SA_VALID[:5] + ("9" if SA_VALID[5] != "9" else "8") + SA_VALID[6:]
    assert is_valid_iban(tampered) is False
    with pytest.raises(ChecksumError):
        parse_iban_strict(tampered)


def test_two_char_transposition_rejected() -> None:
    """Swap two adjacent digits -> the dominant class of typing errors."""
    # Pick a pair where the two digits actually differ so the swap is real.
    pos = next(
        i for i in range(4, len(SA_VALID) - 1)
        if SA_VALID[i] != SA_VALID[i + 1]
    )
    tampered = (
        SA_VALID[:pos]
        + SA_VALID[pos + 1]
        + SA_VALID[pos]
        + SA_VALID[pos + 2:]
    )
    assert is_valid_iban(tampered) is False


def test_wrong_check_digits_rejected() -> None:
    tampered = "SA99" + SA_VALID[4:]
    assert is_valid_iban(tampered) is False


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not-an-iban",
        "SA0380000000608010167519-",      # trailing punctuation
        "SA0380000000608010167519!",
        "SA03 8000 0000 6080 1016 75",    # too short for SA (was 24, now 22)
        "SA0380000000608010167519000",    # too long for SA
        "10380000000608010167519",        # missing 'A' from country
        "**380000000608010167519",
    ],
)
def test_is_valid_iban_rejects_malformed(bad: str) -> None:
    assert is_valid_iban(bad) is False


@pytest.mark.parametrize("not_str", [None, 42, b"SA0380000000608010167519"])
def test_is_valid_iban_rejects_non_string(not_str: object) -> None:
    assert is_valid_iban(not_str) is False  # type: ignore[arg-type]


def test_parse_iban_strict_typed_errors() -> None:
    with pytest.raises(MalformedIBANError):
        parse_iban_strict("not-an-iban")
    with pytest.raises(MalformedIBANError):
        parse_iban_strict("SA038")  # too short / wrong shape
    with pytest.raises(ChecksumError):
        # Same length / chars but wrong checksum.
        parse_iban_strict("SA9980000000608010167519")
    with pytest.raises(UnsupportedCountryError):
        parse_iban_strict("ZZ820000000000000000000000")  # ZZ not in registry


def test_us_country_rejected_as_no_iban_scheme() -> None:
    with pytest.raises(UnsupportedCountryError):
        parse_iban_strict("US12345678901234567890")


def test_parse_iban_returns_none_for_all_bad_paths() -> None:
    """The non-strict parser must NEVER raise."""
    assert parse_iban(None) is None  # type: ignore[arg-type]
    assert parse_iban("") is None
    assert parse_iban("not-an-iban") is None
    assert parse_iban("ZZ820000000000000000000000") is None
    assert parse_iban("SA9980000000608010167519") is None  # bad checksum
    assert parse_iban(42) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Value object construction guards
# ---------------------------------------------------------------------------


def test_iban_construction_rejects_invalid_value() -> None:
    with pytest.raises(IBANError, match="mod-97"):
        IBAN(value="SA99" + SA_VALID[4:], country="SA", check_digits="99", bban=SA_VALID[4:])


def test_iban_is_frozen() -> None:
    iban = parse_iban(SA_VALID)
    assert iban is not None
    with pytest.raises((AttributeError, TypeError)):
        iban.value = "tampered"  # type: ignore[misc]
