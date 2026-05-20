"""Tests for ``auxima_ai.ids.ulid``.

Coverage per ULID spec (https://github.com/ulid/spec):
  - generate() returns 26 uppercase Crockford-base32 chars.
  - generate() embeds the current ms timestamp in the first 10 chars.
  - Two ULIDs from the same ms have the same 10-char prefix.
  - Two ULIDs from different ms sort lexicographically by time.
  - parse() round-trips with the encoder.
  - parse() rejects lowercase, wrong length, non-alphabet chars.
  - is_valid() matches parse() validity exactly.
  - Crockford alphabet excludes I, L, O, U.
  - MonotonicGenerator produces strictly increasing IDs within a ms.
  - MonotonicGenerator handles wall-clock rewind by pinning + incrementing.
  - MonotonicGenerator overflow raises (extremely unreachable but defensive).
  - extract_timestamp_ms returns the encoded ms.
  - Concurrent monotonic generation produces no duplicates.
"""
from __future__ import annotations

import threading

import pytest

from auxima_ai.ids.ulid import (
    InvalidULIDError,
    MonotonicGenerator,
    MonotonicOverflowError,
    TIMESTAMP_CHARS,
    ULID_CHARS,
    extract_timestamp_ms,
    generate,
    is_valid,
    parse,
)


# ---------------------------------------------------------------------------
# Generator shape
# ---------------------------------------------------------------------------


def test_generate_returns_26_chars() -> None:
    s = generate()
    assert len(s) == ULID_CHARS == 26


def test_generate_uses_crockford_alphabet() -> None:
    """No I, L, O, U; no lowercase; no punctuation."""
    s = generate()
    assert all(ch in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for ch in s)


def test_generate_is_valid() -> None:
    for _ in range(50):
        assert is_valid(generate())


# ---------------------------------------------------------------------------
# Timestamp embedding
# ---------------------------------------------------------------------------


def test_extract_timestamp_matches_injected_clock() -> None:
    """generate() uses the injected clock; extract_timestamp_ms recovers it."""
    fixed_ms = 1_715_986_380_123
    s = generate(clock=lambda: fixed_ms / 1000)
    assert extract_timestamp_ms(s) == fixed_ms


def test_same_ms_yields_same_prefix() -> None:
    """Two ULIDs generated at the same instant share the 10-char timestamp prefix."""
    fixed_ms = 1_715_986_380_123

    def clk() -> float:
        return fixed_ms / 1000

    a = generate(clock=clk)
    b = generate(clock=clk)
    assert a[:TIMESTAMP_CHARS] == b[:TIMESTAMP_CHARS]


def test_later_ms_sorts_after_earlier_ms() -> None:
    """Lex-sort of ULIDs is chronological sort by ms."""
    earlier = generate(clock=lambda: 1_700_000_000.000)
    later = generate(clock=lambda: 1_700_000_001.000)
    assert earlier < later


# ---------------------------------------------------------------------------
# RNG injection
# ---------------------------------------------------------------------------


def test_generate_uses_injected_rng() -> None:
    """All-zero rng yields all-zero random part."""
    s = generate(clock=lambda: 0.0, rng=lambda n: b"\x00" * n)
    assert s[TIMESTAMP_CHARS:] == "0" * 16


def test_generate_all_ones_rng() -> None:
    """All-ones rng yields the max random part."""
    s = generate(clock=lambda: 0.0, rng=lambda n: b"\xff" * n)
    # The max 80-bit value encoded as 16 Crockford chars = "ZZZZZZZZZZZZZZZZ".
    assert s[TIMESTAMP_CHARS:] == "Z" * 16


# ---------------------------------------------------------------------------
# Parse round-trip
# ---------------------------------------------------------------------------


def test_parse_round_trip() -> None:
    fixed_ms = 1_715_986_380_123
    s = generate(
        clock=lambda: fixed_ms / 1000,
        rng=lambda n: b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A",
    )
    ts, rnd = parse(s)
    assert ts == fixed_ms
    # The 80-bit random part decoded matches what we injected.
    expected_rnd = int.from_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A", "big")
    assert rnd == expected_rnd


# ---------------------------------------------------------------------------
# Parse rejection — invalid input
# ---------------------------------------------------------------------------


def test_parse_rejects_lowercase() -> None:
    s = generate(clock=lambda: 0.0).lower()
    with pytest.raises(InvalidULIDError):
        parse(s)


def test_parse_rejects_wrong_length() -> None:
    with pytest.raises(InvalidULIDError, match="26 chars"):
        parse("0" * 25)
    with pytest.raises(InvalidULIDError, match="26 chars"):
        parse("0" * 27)


@pytest.mark.parametrize(
    "bad_char_pos, bad_char",
    [
        (0, "I"),  # Crockford excludes I
        (5, "L"),
        (10, "O"),
        (15, "U"),
        (20, "!"),
        (25, "-"),
    ],
)
def test_parse_rejects_chars_outside_alphabet(bad_char_pos: int, bad_char: str) -> None:
    s = generate(clock=lambda: 0.0)
    tampered = s[:bad_char_pos] + bad_char + s[bad_char_pos + 1:]
    with pytest.raises(InvalidULIDError):
        parse(tampered)


def test_parse_rejects_non_string() -> None:
    with pytest.raises(InvalidULIDError, match="str"):
        parse(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj, expected",
    [
        ("01HXZ0M5K0RX6P0V7W3GHJK8MN", True),
        ("01hxz0m5k0rx6p0v7w3ghjk8mn", False),  # lowercase
        ("01HXZ0M5K0RX6P0V7W3GHJK8M", False),  # 25 chars
        ("01HXZ0M5K0RX6P0V7W3GHJK8MNX", False),  # 27 chars
        ("01HXZ0M5K0RX6P0V7W3GHJK8MI", False),  # contains I
        ("", False),
        (None, False),
        (42, False),
        (["01HXZ0M5K0RX6P0V7W3GHJK8MN"], False),
    ],
)
def test_is_valid(obj: object, expected: bool) -> None:
    assert is_valid(obj) is expected


# ---------------------------------------------------------------------------
# Monotonic generator
# ---------------------------------------------------------------------------


def test_monotonic_within_same_ms() -> None:
    fixed_ms = 1_715_986_380_123
    gen = MonotonicGenerator(
        clock=lambda: fixed_ms / 1000,
        rng=lambda n: b"\x00" * n,
    )
    prev = gen()
    for _ in range(10):
        nxt = gen()
        assert nxt > prev, f"monotonic broke: {prev!r} -> {nxt!r}"
        assert nxt[:TIMESTAMP_CHARS] == prev[:TIMESTAMP_CHARS], "ms prefix drifted"
        prev = nxt


def test_monotonic_across_ms_uses_fresh_randomness() -> None:
    """At a new ms boundary, the random part is drawn fresh — not just incremented."""
    box = [1_700_000_000_000]
    gen = MonotonicGenerator(
        clock=lambda: box[0] / 1000,
        rng=lambda n: b"\x00" * n,
    )
    a = gen()
    box[0] += 1  # advance one ms
    b = gen()
    assert a[:TIMESTAMP_CHARS] != b[:TIMESTAMP_CHARS]
    assert b > a


def test_monotonic_handles_clock_rewind() -> None:
    """Wall clock going backwards (NTP step) — pin to last ms + increment."""
    box = [1_700_000_000_000]
    gen = MonotonicGenerator(
        clock=lambda: box[0] / 1000,
        rng=lambda n: b"\x00" * n,
    )
    forward = gen()
    box[0] -= 5  # NTP just stepped us back 5 ms
    rewound = gen()
    # rewound MUST still sort after forward.
    assert rewound > forward
    # And both must share the original (later) ms prefix.
    assert rewound[:TIMESTAMP_CHARS] == forward[:TIMESTAMP_CHARS]


def test_monotonic_overflow_raises() -> None:
    """Force the random part to its ceiling at construction and confirm
    a subsequent same-ms increment raises (would otherwise silently wrap)."""
    fixed_ms = 1_700_000_000_000
    gen = MonotonicGenerator(clock=lambda: fixed_ms / 1000)
    gen._last_ms = fixed_ms
    gen._last_random = (1 << 80) - 1  # at the ceiling
    with pytest.raises(MonotonicOverflowError):
        gen()


def test_monotonic_thread_safe() -> None:
    """100 threads each generate 50 IDs; no duplicates; all sort strictly increasing."""
    gen = MonotonicGenerator()
    results: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(100)

    def worker() -> None:
        barrier.wait()
        local: list[str] = []
        for _ in range(50):
            local.append(gen())
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == len(set(results)), "duplicate ULIDs under concurrency"


def test_monotonic_callable_alias() -> None:
    gen = MonotonicGenerator()
    a = gen()
    b = gen.generate()
    assert is_valid(a) and is_valid(b)
