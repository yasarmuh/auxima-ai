"""Tests for ``auxima_ai.observability.redact``.

Coverage targets per S-19 §3.5 + S-43 §6 + S-44 R6:
  - one positive + one negative per pattern (email / phone E.164 / phone KSA
    local / KSA national ID / KSA CR / KSA IBAN).
  - empty / non-string input is handled defensively.
  - ``fired`` flag is True iff at least one replacement happened.
  - ``is_clean`` matches the negation of ``redact(...)[1]`` for the same input.
  - mixed-class strings have ALL classes replaced in one pass.
  - the redacted output never contains the original PII substring (the security
    property we actually care about; CI grep also asserts this).
"""
from __future__ import annotations

import pytest

from auxima_ai.observability.redact import is_clean, redact


# ---------------------------------------------------------------------------
# Positive cases — each pattern fires + is replaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_token, original_substring",
    [
        # Email
        ("Contact procurement@al-mansour.sa for quotes.",
         "<redacted:email>",
         "procurement@al-mansour.sa"),

        # E.164 phone
        ("Call +966500000000 today.",
         "<redacted:phone_e164>",
         "+966500000000"),

        # KSA local mobile
        ("Mobile 0512345678 on file.",
         "<redacted:phone_ksa_local>",
         "0512345678"),

        # KSA national ID — citizen (leading 1)
        ("ID 1234567890 in record.",
         "<redacted:ksa_national_id>",
         "1234567890"),

        # KSA national ID — resident (leading 2)
        ("Iqama 2987654321 verified.",
         "<redacted:ksa_national_id>",
         "2987654321"),

        # KSA CR — modern leading 7
        ("CR number 7012345678 active.",
         "<redacted:ksa_cr>",
         "7012345678"),

        # KSA CR — legacy leading 10
        ("Legacy CR 1012345678 expired.",
         "<redacted:ksa_cr>",
         "1012345678"),

        # KSA IBAN
        ("Bank SA0380000000608010167519 on file.",
         "<redacted:iban_sa>",
         "SA0380000000608010167519"),
    ],
)
def test_redact_positive_replaces_and_strips_original(
    raw: str, expected_token: str, original_substring: str
) -> None:
    """For each pattern, the redacted token appears AND the original substring is gone."""
    out, fired = redact(raw)
    assert fired is True, f"expected redact to fire on: {raw!r}"
    assert expected_token in out, f"expected token {expected_token!r} in: {out!r}"
    assert original_substring not in out, (
        f"SECURITY: original substring {original_substring!r} still present in: {out!r}"
    )


# ---------------------------------------------------------------------------
# Negative cases — benign strings that look adjacent must NOT fire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        # Plain text
        "Acme Brokers — annual review meeting next Tuesday.",
        # Numbers that aren't long enough to match
        "Year 2026; 50 quotes received.",
        # Phone-shaped but too short for either pattern
        "Call 12345.",
        # Email-shaped but missing the TLD requirement (>=2 chars)
        "Reach out at user@x.a — but not really.",
        # 9-digit number (one short of national ID / CR)
        "Reference 123456789 closed.",
        # 11-digit number (one over national ID / CR)
        "Reference 12345678901 invalid.",
        # IBAN-shaped but wrong country code
        "AE070331234567890123456 placeholder.",
        # Number starting with 3 (neither national ID nor CR)
        "Token 3987654321 not PII.",
        # Empty + whitespace
        "",
        "   ",
    ],
)
def test_redact_negative_does_not_fire_on_benign_inputs(raw: str) -> None:
    out, fired = redact(raw)
    assert fired is False, f"redact should NOT fire on: {raw!r} (output: {out!r})"
    assert out == raw, f"benign input should round-trip unchanged: {raw!r} -> {out!r}"


# ---------------------------------------------------------------------------
# Mixed-class — all classes replaced in a single pass
# ---------------------------------------------------------------------------


def test_redact_mixed_classes_in_one_pass() -> None:
    raw = (
        "Hi, this is admin@example.com, my mobile is 0512345678, ID 1234567890, "
        "CR 7012345678, IBAN SA0380000000608010167519, intl +447911123456."
    )
    out, fired = redact(raw)
    assert fired is True
    # All 6 placeholder kinds present
    for token in (
        "<redacted:email>",
        "<redacted:phone_ksa_local>",
        "<redacted:ksa_national_id>",
        "<redacted:ksa_cr>",
        "<redacted:iban_sa>",
        "<redacted:phone_e164>",
    ):
        assert token in out, f"missing {token} in: {out!r}"
    # The originals must be gone
    for original in (
        "admin@example.com",
        "0512345678",
        "1234567890",
        "7012345678",
        "SA0380000000608010167519",
        "+447911123456",
    ):
        assert original not in out, f"SECURITY: {original!r} survived in: {out!r}"


# ---------------------------------------------------------------------------
# Idempotency + determinism
# ---------------------------------------------------------------------------


def test_redact_is_idempotent_when_already_clean() -> None:
    """Running redact twice on a clean input yields the same (unchanged) output."""
    raw = "Just a regular sentence with no PII."
    once, fired1 = redact(raw)
    twice, fired2 = redact(once)
    assert fired1 is False
    assert fired2 is False
    assert raw == once == twice


def test_redact_is_deterministic_across_calls() -> None:
    raw = "Reach 0512345678 or +966500000000."
    a, _ = redact(raw)
    b, _ = redact(raw)
    assert a == b


# ---------------------------------------------------------------------------
# Defensive input handling
# ---------------------------------------------------------------------------


def test_redact_empty_string_returns_empty_and_false() -> None:
    out, fired = redact("")
    assert out == ""
    assert fired is False


@pytest.mark.parametrize("falsy", [None, 0, [], {}])
def test_redact_handles_falsy_non_strings_gracefully(falsy: object) -> None:
    """``redact`` short-circuits on falsy inputs without raising."""
    out, fired = redact(falsy)  # type: ignore[arg-type]
    assert fired is False
    assert out == falsy


# ---------------------------------------------------------------------------
# is_clean helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Plain text.", True),
        ("", True),
        ("Has email user@example.com.", False),
        ("Has phone +966500000000.", False),
        ("Has CR 7012345678.", False),
    ],
)
def test_is_clean(raw: str, expected: bool) -> None:
    assert is_clean(raw) is expected
