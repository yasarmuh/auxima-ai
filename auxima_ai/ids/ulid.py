"""ULID — Universally Unique Lexicographically Sortable Identifier.

Spec: https://github.com/ulid/spec

Wire format: 26-character Crockford-base32 string.
  ttttttttttrrrrrrrrrrrrrrrr
  ^^^^^^^^^^                  10 chars = 48-bit milliseconds-since-epoch
            ^^^^^^^^^^^^^^^^  16 chars = 80-bit randomness

Why ULID over UUIDv4:
  - **Lex-sortable.** Sorting ULIDs alphabetically gives chronological
    order — b-tree indexes on primary keys keep insertion-order
    locality. UUIDv4 randomness destroys that locality and causes page
    splits at scale.
  - **URL-safe + case-insensitive.** Crockford base32 excludes
    visually-similar characters (I/L/O/U) so a human can read a ULID
    off a screen without ambiguity.
  - **128-bit.** Same collision-resistance as UUIDv4; the 80 random
    bits per ms give >2^80 collision-free draws within any single ms.

Two factories:
  - :func:`generate`  — independent randomness per call. Two calls in
    the same ms can sort in either order.
  - :class:`MonotonicGenerator` — strict monotonicity within a ms by
    incrementing the random part on collision. Use for activity-log
    rows where the lock-step order matters.

Pure stdlib (``os`` + ``time`` + ``threading``); no third-party deps.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final

# Crockford base32 alphabet — excludes I, L, O, U to avoid visual
# ambiguity. Documented in the ULID spec §3.
_CROCKFORD: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CROCKFORD_LOOKUP: Final[dict[str, int]] = {ch: i for i, ch in enumerate(_CROCKFORD)}

# Spec constants — pinning the wire shape.
TIMESTAMP_BITS: Final[int] = 48
RANDOMNESS_BITS: Final[int] = 80
TIMESTAMP_CHARS: Final[int] = 10  # ceil(48 / 5)
RANDOMNESS_CHARS: Final[int] = 16  # 80 / 5
ULID_CHARS: Final[int] = TIMESTAMP_CHARS + RANDOMNESS_CHARS  # 26

_TIMESTAMP_MAX: Final[int] = (1 << TIMESTAMP_BITS) - 1
_RANDOMNESS_MAX: Final[int] = (1 << RANDOMNESS_BITS) - 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ULIDError(ValueError):
    """Base — any invalid input raises a subclass of this."""


class InvalidULIDError(ULIDError):
    """Raised when a string is not a valid 26-char Crockford-base32 ULID."""


class TimestampOverflowError(ULIDError):
    """Raised when the timestamp would not fit in 48 bits (year ~10895)."""


class MonotonicOverflowError(ULIDError):
    """Raised when monotonic increment would overflow the random part.

    Practically unreachable — 80 bits of headroom per ms is enormous —
    but caught explicitly so the error path doesn't silently wrap.
    """


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_base32_fixed(value: int, length: int) -> str:
    """Encode ``value`` into a Crockford-base32 string of exactly ``length`` chars.

    Higher-order chars come first (big-endian). Pads with leading 0s if
    the value's natural representation is shorter than ``length``.
    """
    if value < 0:
        raise ULIDError(f"cannot encode negative value: {value}")
    out: list[str] = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    if value != 0:
        # The caller asked for fewer chars than the value needs — bug.
        raise ULIDError(f"value too large for {length}-char base32 field")
    return "".join(reversed(out))


def _decode_base32(s: str) -> int:
    """Decode a Crockford-base32 string to int (uppercase-only here)."""
    value = 0
    for ch in s:
        try:
            value = (value << 5) | _CROCKFORD_LOOKUP[ch]
        except KeyError as e:
            raise InvalidULIDError(
                f"character {ch!r} is not in the Crockford base32 alphabet"
            ) from e
    return value


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _ms_now(clock: Callable[[], float]) -> int:
    """Current wall-clock time in milliseconds since epoch (int)."""
    return int(clock() * 1000)


def generate(
    *,
    clock: Callable[[], float] = time.time,
    rng: Callable[[int], bytes] = os.urandom,
) -> str:
    """Generate a fresh ULID. Independent randomness; not monotonic.

    Two ULIDs from this function in the same millisecond will sort in
    arbitrary order. Use :class:`MonotonicGenerator` when strict
    same-ms ordering matters.

    Parameters
    ----------
    clock
        Injectable wall-clock seconds. Defaults to :func:`time.time`.
    rng
        Injectable CSPRNG (``n -> bytes``). Defaults to
        :func:`os.urandom` — DO NOT swap in :func:`random.random` for
        anything user-facing; ULID guessability is the same security
        property as UUIDv4 guessability.
    """
    ts_ms = _ms_now(clock)
    if ts_ms < 0 or ts_ms > _TIMESTAMP_MAX:
        raise TimestampOverflowError(
            f"timestamp {ts_ms} does not fit in {TIMESTAMP_BITS} bits"
        )
    randomness = int.from_bytes(rng(10), "big")  # 80 bits = 10 bytes
    if randomness > _RANDOMNESS_MAX:
        # Defensive: rng misbehaving would only matter if it returned
        # more than 10 bytes, which can't happen — keep the check anyway.
        randomness &= _RANDOMNESS_MAX
    return (
        _encode_base32_fixed(ts_ms, TIMESTAMP_CHARS)
        + _encode_base32_fixed(randomness, RANDOMNESS_CHARS)
    )


@dataclass
class MonotonicGenerator:
    """Generator that guarantees strict monotonic ordering within a millisecond.

    The first ULID in a new ms gets fresh randomness. Subsequent ULIDs
    in the SAME ms reuse the timestamp and increment the random part
    by 1. This preserves lex-sort order even at burst rates well past
    a million IDs/sec.

    Thread-safe via :class:`threading.Lock`.
    """

    clock: Callable[[], float] = field(default=time.time)
    rng: Callable[[int], bytes] = field(default=os.urandom)
    _last_ms: int = field(default=-1, init=False)
    _last_random: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __call__(self) -> str:
        return self.generate()

    def generate(self) -> str:
        with self._lock:
            ts_ms = _ms_now(self.clock)
            if ts_ms < 0 or ts_ms > _TIMESTAMP_MAX:
                raise TimestampOverflowError(
                    f"timestamp {ts_ms} does not fit in {TIMESTAMP_BITS} bits"
                )
            if ts_ms == self._last_ms:
                # Same ms — increment.
                nxt = self._last_random + 1
                if nxt > _RANDOMNESS_MAX:
                    raise MonotonicOverflowError(
                        f"monotonic increment overflowed {RANDOMNESS_BITS}-bit random part "
                        f"at ts_ms={ts_ms}"
                    )
                randomness = nxt
            elif ts_ms < self._last_ms:
                # Wall clock went backwards (NTP step). Pin to the
                # previous ms and increment so ordering doesn't break.
                ts_ms = self._last_ms
                nxt = self._last_random + 1
                if nxt > _RANDOMNESS_MAX:
                    raise MonotonicOverflowError(
                        "monotonic increment overflowed during clock-rewind handling"
                    )
                randomness = nxt
            else:
                randomness = int.from_bytes(self.rng(10), "big") & _RANDOMNESS_MAX
            self._last_ms = ts_ms
            self._last_random = randomness
            return (
                _encode_base32_fixed(ts_ms, TIMESTAMP_CHARS)
                + _encode_base32_fixed(randomness, RANDOMNESS_CHARS)
            )


# ---------------------------------------------------------------------------
# Validation + parse
# ---------------------------------------------------------------------------


def is_valid(s: object) -> bool:
    """``True`` iff ``s`` is a 26-char uppercase Crockford-base32 string."""
    if not isinstance(s, str) or len(s) != ULID_CHARS:
        return False
    for ch in s:
        if ch not in _CROCKFORD_LOOKUP:
            return False
    return True


def parse(s: str) -> tuple[int, int]:
    """Parse a ULID string into ``(timestamp_ms, randomness)``.

    Raises :class:`InvalidULIDError` on wrong length, lowercase input,
    or any character outside the Crockford alphabet (case-sensitive —
    callers MUST normalise to upper before parsing).
    """
    if not isinstance(s, str):
        raise InvalidULIDError(f"ULID must be str; got {type(s).__name__}")
    if len(s) != ULID_CHARS:
        raise InvalidULIDError(
            f"ULID must be {ULID_CHARS} chars; got {len(s)}"
        )
    ts = _decode_base32(s[:TIMESTAMP_CHARS])
    rnd = _decode_base32(s[TIMESTAMP_CHARS:])
    return ts, rnd


def extract_timestamp_ms(s: str) -> int:
    """Return the millisecond timestamp encoded in a ULID."""
    return parse(s)[0]


__all__ = (
    "InvalidULIDError",
    "MonotonicGenerator",
    "MonotonicOverflowError",
    "RANDOMNESS_BITS",
    "RANDOMNESS_CHARS",
    "TIMESTAMP_BITS",
    "TIMESTAMP_CHARS",
    "TimestampOverflowError",
    "ULID_CHARS",
    "ULIDError",
    "extract_timestamp_ms",
    "generate",
    "is_valid",
    "parse",
)
