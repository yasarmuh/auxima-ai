"""Tests for ``auxima_ai.util.ksa_id`` — National ID + CR shape validators.

Coverage:
  - is_valid_national_id: leading 1 OR 2 + 10 digits exactly.
  - is_valid_cr: leading 7 OR 1010 + 10 digits exactly.
  - Both reject wrong length, non-digit chars, wrong leading digit.
  - parse_national_id: citizen vs resident classification via leading digit.
  - parse_cr: returns frozen value object.
  - Whitespace + dashes stripped before validation.
  - None / non-string inputs return None (never raise).
  - Value-object construction guards against invalid inputs.
  - Frozen dataclasses: assignment after construction raises.
"""
from __future__ import annotations

import pytest

from auxima_ai.util.ksa_id import (
    CommercialRegistration,
    IDType,
    NationalID,
    is_valid_cr,
    is_valid_national_id,
    parse_cr,
    parse_national_id,
)


# ---------------------------------------------------------------------------
# is_valid_national_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "1234567890",          # citizen
        "1000000000",          # citizen, all-zero-after-leading
        "1999999999",          # citizen, all-nine-after-leading
        "2987654321",          # resident
        "2000000000",          # resident, all-zero-after-leading
    ],
)
def test_is_valid_national_id_accepts(good: str) -> None:
    assert is_valid_national_id(good) is True


@pytest.mark.parametrize(
    "bad",
    [
        "",                    # empty
        "123",                 # too short
        "12345678901",         # too long
        "0234567890",          # leading 0 (neither citizen nor resident)
        "3234567890",          # leading 3
        "9234567890",          # leading 9
        "12a4567890",          # non-digit
        " 1234567890",         # whitespace inside the raw string
        "1234567890\n",        # trailing newline
        "-1234567890",         # leading dash
    ],
)
def test_is_valid_national_id_rejects(bad: str) -> None:
    assert is_valid_national_id(bad) is False


@pytest.mark.parametrize("not_str", [None, 1234567890, b"1234567890", ["1234567890"]])
def test_is_valid_national_id_rejects_non_string(not_str: object) -> None:
    assert is_valid_national_id(not_str) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# is_valid_cr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "7012345678",          # modern leading-7
        "7999999999",
        "1012345678",          # legacy Riyadh leading-1010
        "1010000000",
        "1010999999",
    ],
)
def test_is_valid_cr_accepts(good: str) -> None:
    assert is_valid_cr(good) is True


@pytest.mark.parametrize(
    "bad",
    [
        "",                    # empty
        "7012345",             # too short
        "70123456789",         # too long
        "8012345678",          # leading 8 — not in spec
        "2012345678",          # leading 2 — that's a national ID, not CR
        "1110000000",          # legacy must start with 1010, not 1110
        "70a2345678",          # non-digit
        "7012-345-678",        # dashes inside raw string (parser strips, validator doesn'\''t)
    ],
)
def test_is_valid_cr_rejects(bad: str) -> None:
    assert is_valid_cr(bad) is False


@pytest.mark.parametrize("not_str", [None, 7012345678, b"7012345678", ["7012345678"]])
def test_is_valid_cr_rejects_non_string(not_str: object) -> None:
    assert is_valid_cr(not_str) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_national_id
# ---------------------------------------------------------------------------


def test_parse_national_id_citizen() -> None:
    nid = parse_national_id("1234567890")
    assert nid is not None
    assert nid.digits == "1234567890"
    assert nid.type == IDType.CITIZEN


def test_parse_national_id_resident() -> None:
    nid = parse_national_id("2987654321")
    assert nid is not None
    assert nid.type == IDType.RESIDENT


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1234567890",      "1234567890"),
        ("  1234567890  ",  "1234567890"),       # whitespace stripped
        ("1234-567890",     "1234567890"),       # dash stripped
        ("12-34-56-78-90",  "1234567890"),       # multiple dashes
    ],
)
def test_parse_national_id_strips_formatting(raw: str, expected: str) -> None:
    nid = parse_national_id(raw)
    assert nid is not None
    assert nid.digits == expected


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "abc",
        "0234567890",
        "3234567890",
        "123",
        "12345678901",
    ],
)
def test_parse_national_id_returns_none_on_bad_input(bad: object) -> None:
    assert parse_national_id(bad) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("not_str", [42, 1234567890, b"1234567890", ["1234567890"]])
def test_parse_national_id_returns_none_on_non_string(not_str: object) -> None:
    assert parse_national_id(not_str) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_cr
# ---------------------------------------------------------------------------


def test_parse_cr_modern_format() -> None:
    cr = parse_cr("7012345678")
    assert cr is not None
    assert cr.digits == "7012345678"


def test_parse_cr_legacy_riyadh_format() -> None:
    cr = parse_cr("1010123456")
    assert cr is not None


def test_parse_cr_strips_formatting() -> None:
    cr = parse_cr("  7012-345-678  ")
    assert cr is not None
    assert cr.digits == "7012345678"


@pytest.mark.parametrize(
    "bad",
    [
        None, "", "abc", "8012345678", "1234567890",  # 1234... is a citizen ID, not legacy CR
        "7012345", "70123456789",
    ],
)
def test_parse_cr_returns_none_on_bad_input(bad: object) -> None:
    assert parse_cr(bad) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Value object construction guards
# ---------------------------------------------------------------------------


def test_national_id_construction_rejects_invalid_digits() -> None:
    with pytest.raises(ValueError, match="10 digits"):
        NationalID(digits="abc", type=IDType.CITIZEN)


def test_cr_construction_rejects_invalid_digits() -> None:
    with pytest.raises(ValueError, match="10 digits"):
        CommercialRegistration(digits="abc")


def test_national_id_is_frozen() -> None:
    nid = parse_national_id("1234567890")
    assert nid is not None
    with pytest.raises((AttributeError, TypeError)):
        nid.digits = "9999999999"  # type: ignore[misc]


def test_cr_is_frozen() -> None:
    cr = parse_cr("7012345678")
    assert cr is not None
    with pytest.raises((AttributeError, TypeError)):
        cr.digits = "7999999999"  # type: ignore[misc]
