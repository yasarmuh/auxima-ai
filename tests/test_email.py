"""Tests for ``auxima_ai.util.email`` — email shape check + normaliser.

Coverage:
  - is_valid_email accepts the common-case shapes.
  - Length caps enforced (local 64, domain 255, overall 254).
  - Conservative regex rejects multi-@, missing TLD, leading/trailing dots,
    consecutive dots, IP-literal domains (out of v1 scope).
  - normalise_email lowercases the domain but preserves the local case.
  - "mailto:" URI prefix stripped, case-insensitive.
  - Surrounding whitespace stripped.
  - None / non-string / empty / over-length input returns None.
  - Email value object construction guards against invalid addresses.
  - Email.local + Email.domain reflect the split correctly.
  - Round-trip property: normalise(x).address is_valid_email always True.
"""
from __future__ import annotations

import pytest

from auxima_ai.util.email import (
    EMAIL_MAX,
    Email,
    LOCAL_PART_MAX,
    is_valid_email,
    normalise_email,
)


# ---------------------------------------------------------------------------
# is_valid_email — happy + rejected shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "foo@example.com",
        "foo.bar@example.com",
        "foo+tag@example.com",          # gmail-style aliasing preserved
        "foo_bar@example.com",
        "foo-bar@example.com",
        "FOO@EXAMPLE.COM",              # uppercase ok at the shape level
        "ops@al-mansour.sa",            # KSA TLD with hyphen in label
        "x@y.co",                       # minimal valid (TLD = 2 chars)
        "1234567890@example.com",
        "user@sub.domain.example.com",  # multi-label domain
    ],
)
def test_is_valid_email_accepts_well_formed(good: str) -> None:
    assert is_valid_email(good) is True


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no-at-sign.com",
        "@no-local.com",
        "no-domain@",
        "two@signs@example.com",
        ".leading-dot@example.com",
        "trailing-dot.@example.com",
        "consec..dots@example.com",
        "spaces in@example.com",
        "user@example",                   # no TLD
        "user@example.c",                 # 1-char TLD
        "user@-leading-hyphen.com",
        "user@trailing-hyphen-.com",
        "user@.no-label.com",
        "user@example.com.",              # trailing root dot
        "user@127.0.0.1",                 # IP-literal domain (v1 doesn'\''t accept)
        "user@[127.0.0.1]",               # bracketed IP literal
    ],
)
def test_is_valid_email_rejects_malformed(bad: str) -> None:
    assert is_valid_email(bad) is False


@pytest.mark.parametrize("not_str", [None, 42, b"foo@bar.com", ["foo@bar.com"]])
def test_is_valid_email_returns_false_for_non_string(not_str: object) -> None:
    assert is_valid_email(not_str) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Length caps
# ---------------------------------------------------------------------------


def test_local_at_max_length_accepted() -> None:
    addr = ("a" * LOCAL_PART_MAX) + "@example.com"
    assert is_valid_email(addr) is True


def test_local_over_max_length_rejected() -> None:
    addr = ("a" * (LOCAL_PART_MAX + 1)) + "@example.com"
    assert is_valid_email(addr) is False


def test_overall_over_email_max_rejected() -> None:
    """Construct an address well over the practical 254-char cap."""
    long = "a" * 200 + "@" + "b" * 200 + ".com"
    assert len(long) > EMAIL_MAX
    assert is_valid_email(long) is False


# ---------------------------------------------------------------------------
# normalise_email — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Foo@Example.COM",          "Foo@example.com"),         # domain lowered, local kept
        ("foo@example.com",          "foo@example.com"),         # already canonical
        ("  foo@example.com  ",      "foo@example.com"),         # whitespace stripped
        ("mailto:foo@example.com",   "foo@example.com"),         # URI prefix stripped
        ("MAILTO:Foo@Example.com",   "Foo@example.com"),         # case-insensitive
        ("foo+tag@example.com",      "foo+tag@example.com"),     # aliasing preserved
        ("ops@al-mansour.SA",        "ops@al-mansour.sa"),
    ],
)
def test_normalise_lowercases_domain_only(raw: str, expected: str) -> None:
    result = normalise_email(raw)
    assert result is not None
    assert result.address == expected


def test_normalise_splits_local_and_domain() -> None:
    e = normalise_email("Foo@Example.COM")
    assert e is not None
    assert e.local == "Foo"
    assert e.domain == "example.com"


# ---------------------------------------------------------------------------
# normalise_email — rejection paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "    ",
        "not-an-email",
        "two@@signs.com",
        "@no-local.com",
        "no-at-sign",
        "spaces in@example.com",
    ],
)
def test_normalise_returns_none_for_bad_input(bad: object) -> None:
    assert normalise_email(bad) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("not_str", [42, b"foo@bar.com", ["foo@bar.com"], {"email": "foo@bar.com"}])
def test_normalise_returns_none_for_non_string(not_str: object) -> None:
    assert normalise_email(not_str) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Email value object
# ---------------------------------------------------------------------------


def test_email_construction_rejects_invalid_address() -> None:
    with pytest.raises(ValueError, match="valid"):
        Email(address="not-an-email", local="not-an-email", domain="")


def test_email_is_frozen() -> None:
    e = normalise_email("foo@example.com")
    assert e is not None
    with pytest.raises((AttributeError, TypeError)):
        e.address = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "foo@example.com",
        "Foo@Example.COM",
        "  ops@al-mansour.sa  ",
        "mailto:foo+tag@sub.example.co",
        "FOO@EXAMPLE.COM",
    ],
)
def test_normalise_output_round_trips_through_validator(raw: str) -> None:
    e = normalise_email(raw)
    assert e is not None
    assert is_valid_email(e.address)
