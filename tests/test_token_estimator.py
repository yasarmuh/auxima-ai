"""Tests for ``auxima_ai.tokens.estimator``.

Coverage:
  - Empty string -> 0; any non-empty -> >= 1.
  - English/Latin text uses ~4 chars/token.
  - Arabic uses ~2 chars/token (denser).
  - Mixed-script text blends proportionally.
  - estimate_tokens output is always an int (no fractional).
  - detect_script returns LATIN for English, DENSE for Arabic, LATIN for empty.
  - estimate_chat_tokens adds per-message overhead.
  - estimate_chat_tokens adds conversation priming overhead.
  - Chat: empty list returns just the priming overhead.
  - Chat: malformed message raises (missing/empty role, non-string content).
  - Non-string text raises TypeError.
  - Conservative property: estimate >= actual tokens for known inputs.
"""
from __future__ import annotations

import pytest

from auxima_ai.tokens.estimator import (
    Script,
    detect_script,
    estimate_chat_tokens,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Empty / boundary
# ---------------------------------------------------------------------------


def test_empty_string_is_zero_tokens() -> None:
    assert estimate_tokens("") == 0


def test_single_char_is_at_least_one_token() -> None:
    assert estimate_tokens("a") >= 1
    assert estimate_tokens(" ") >= 1
    assert estimate_tokens("ا") >= 1  # single Arabic letter


def test_whitespace_only_is_at_least_one_token() -> None:
    """Defends against `'    '` slipping in for free under the ceiling."""
    assert estimate_tokens("    ") >= 1


# ---------------------------------------------------------------------------
# Type safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["list"]])
def test_estimate_tokens_rejects_non_string(bad: object) -> None:
    with pytest.raises(TypeError, match="str"):
        estimate_tokens(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Latin / English heuristic
# ---------------------------------------------------------------------------


def test_short_english_phrase() -> None:
    """`"hello world"` is 11 chars; 11/3.8 ≈ 2.89 -> ceil = 3."""
    assert estimate_tokens("hello world") == 3


def test_longer_english_paragraph_scales_linearly() -> None:
    text = "Acme Brokers' renewal proposal arrived this morning."
    n = estimate_tokens(text)
    # Conservative bounds: between len/5 (under-estimate) and len/2 (over).
    assert len(text) / 5 <= n <= len(text) / 2


# ---------------------------------------------------------------------------
# Arabic / dense heuristic
# ---------------------------------------------------------------------------


def test_arabic_uses_dense_ratio() -> None:
    """`"السلام عليكم"` is 11 chars; 11/1.8 ≈ 6.1 -> ceil = 7."""
    ar = "السلام عليكم"
    assert estimate_tokens(ar) >= 6


def test_arabic_denser_than_english_per_char() -> None:
    ar = "ا" * 20
    en = "a" * 20
    assert estimate_tokens(ar) > estimate_tokens(en)


# ---------------------------------------------------------------------------
# Mixed-script
# ---------------------------------------------------------------------------


def test_mixed_script_blends() -> None:
    """English + Arabic together should fall between pure-English and pure-Arabic."""
    en_only = estimate_tokens("a" * 40)
    ar_only = estimate_tokens("ا" * 40)
    mixed = estimate_tokens("a" * 20 + "ا" * 20)
    assert en_only <= mixed <= ar_only


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_estimate_returns_int() -> None:
    """Token counts are always exact integers (no fractional tokens)."""
    n = estimate_tokens("any text here")
    assert isinstance(n, int)
    assert not isinstance(n, bool)


def test_estimate_is_non_negative() -> None:
    assert estimate_tokens("") >= 0
    assert estimate_tokens("x") >= 0


# ---------------------------------------------------------------------------
# detect_script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", Script.LATIN),
        ("Hello world", Script.LATIN),
        ("12345", Script.LATIN),
        ("السلام عليكم", Script.DENSE),
        ("اب12cd", Script.LATIN),  # 4 latin/digits vs 2 arabic
        ("اب", Script.DENSE),
        ("こんにちは", Script.DENSE),  # CJK
        ("안녕하세요", Script.DENSE),  # Hangul
        ("שלום", Script.DENSE),  # Hebrew
    ],
)
def test_detect_script(text: str, expected: Script) -> None:
    assert detect_script(text) == expected


def test_detect_script_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        detect_script(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------


def test_chat_with_empty_list_returns_conversation_overhead() -> None:
    """Even an empty conversation has a small priming overhead."""
    n = estimate_chat_tokens([])
    assert n >= 1


def test_chat_per_message_overhead_added() -> None:
    one = estimate_chat_tokens([{"role": "user", "content": "hi"}])
    two = estimate_chat_tokens([
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "hi"},
    ])
    # Second message adds at least the per-message overhead.
    assert two - one >= 4


def test_chat_content_contributes_to_total() -> None:
    short = estimate_chat_tokens([{"role": "user", "content": "hi"}])
    long = estimate_chat_tokens(
        [{"role": "user", "content": "hi " * 50}],
    )
    assert long > short


@pytest.mark.parametrize(
    "bad_msg",
    [
        {"role": "", "content": "hi"},
        {"role": None, "content": "hi"},
        {"content": "hi"},  # missing role
    ],
)
def test_chat_rejects_bad_role(bad_msg: dict) -> None:
    with pytest.raises(ValueError, match="role"):
        estimate_chat_tokens([bad_msg])


@pytest.mark.parametrize(
    "bad_msg",
    [
        {"role": "user", "content": None},
        {"role": "user", "content": 42},
        {"role": "user"},  # missing content
    ],
)
def test_chat_rejects_bad_content(bad_msg: dict) -> None:
    with pytest.raises(ValueError, match="content"):
        estimate_chat_tokens([bad_msg])


def test_chat_rejects_non_mapping_message() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        estimate_chat_tokens(["not-a-dict"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Conservatism property — over-estimating is OK; under-estimating is not.
# ---------------------------------------------------------------------------


def test_estimate_meets_or_exceeds_known_token_floors() -> None:
    """For a few hand-curated samples, estimator must not under-count.

    These floors come from spot-checks against tiktoken; the estimator
    is allowed to be higher (conservative is safe) but must never dip
    below the real count, since the policy enforcer trusts this number
    to gate the monthly ceiling.
    """
    # `"hello world"` -> tiktoken=2; our estimator should return >= 2.
    assert estimate_tokens("hello world") >= 2
    # `"The quick brown fox jumps over the lazy dog"` -> tiktoken=9.
    assert estimate_tokens("The quick brown fox jumps over the lazy dog") >= 9
    # A short Arabic greeting tokenises to roughly 5-7 BPE pieces;
    # estimator should never report fewer than 5.
    assert estimate_tokens("السلام عليكم") >= 5
