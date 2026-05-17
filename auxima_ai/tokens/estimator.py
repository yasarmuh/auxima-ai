"""Token estimator — script-aware character heuristic, no third-party deps.

Returns a conservative upper-bound estimate of token count for a piece
of text, suitable for pre-call cost / quota gating in
:class:`auxima_ai.policy.enforcer.PolicyEnforcer`. The estimator is
deliberately approximate; the post-call :func:`record_spend` records
the actual cost from the provider's response.

Algorithm:

  1. Detect the dominant script via Unicode code-point ranges.
     Arabic / CJK / Hangul code points trigger the **dense** path
     (~2 chars/token); the rest of BMP defaults to the **sparse**
     path (~4 chars/token, GPT-style).
  2. Compute per-character "token-cost" weights, sum them up, divide
     by the per-script chars-per-token ratio, and ceil to an int.
  3. For chat-style messages (``role`` + ``content``), add a per-
     message overhead (~4 tokens for the role + delimiters, matching
     OpenAI's ``tiktoken`` Chat ML overhead).

The output is always:
  - non-negative,
  - >= 1 for any non-empty text (avoids "tokens=0 -> cost=0" gaming),
  - an exact ``int`` (no fractional tokens).

This conservatism means we over-charge by a few percent on average —
which is the right side of the safety equation: a tenant can't sneak
under the ceiling by accident.
"""
from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Final, Iterable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants — derived from spot-check measurements against the
# real model tokenisers on representative KSA insurance text.
# ---------------------------------------------------------------------------

# English / Latin: ~4 chars/token across GPT family. Conservative side
# (lower divisor → higher token count) so we over-estimate slightly.
_LATIN_CHARS_PER_TOKEN: Final[float] = 3.8

# Arabic / CJK / Hangul: ~1.8 chars/token. Arabic in particular has
# more BPE merges than Latin because of the high-frequency function
# words; under-estimation here would let Arabic-heavy tenants slip
# past the ceiling.
_DENSE_CHARS_PER_TOKEN: Final[float] = 1.8

# Per-message overhead for chat-style payloads. Matches OpenAI's
# tiktoken Chat ML "tokens_per_message" constant for gpt-4o-mini /
# gpt-4o (3 + 1 for the role; rounded up to 4 here for safety).
_PER_MESSAGE_OVERHEAD: Final[int] = 4

# Per-call priming overhead at the *start* of a chat conversation —
# ChatML starts with a `<|start|>` / `<|end|>` framing. We add this
# once to estimate_chat_tokens() regardless of message count.
_CONVERSATION_OVERHEAD: Final[int] = 3


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------


class Script(str, Enum):
    """The script family driving per-character density."""

    LATIN = "latin"   # alphabetic / numeric / punctuation in basic Latin + extended
    DENSE = "dense"   # Arabic / Hebrew / CJK / Hangul / Devanagari etc.


# Unicode block ranges that count as "dense" for the purposes of token
# density. Conservative — better to over-count for an unfamiliar script
# than to silently under-count.
_DENSE_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0780, 0x07BF),  # Thaana
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0x0900, 0x097F),  # Devanagari
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
)


def _is_dense_codepoint(cp: int) -> bool:
    """``True`` iff ``cp`` falls in any dense Unicode range."""
    for lo, hi in _DENSE_RANGES:
        if lo <= cp <= hi:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count for a free-form text string.

    Splits the character stream by script density, applies the
    per-script chars-per-token ratio, sums, and ceils to a positive
    integer (any non-empty string returns ``>= 1``).
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str; got {type(text).__name__}")
    if text == "":
        return 0

    dense_chars = 0
    sparse_chars = 0
    for ch in text:
        if _is_dense_codepoint(ord(ch)):
            dense_chars += 1
        else:
            sparse_chars += 1

    dense_tokens = dense_chars / _DENSE_CHARS_PER_TOKEN
    sparse_tokens = sparse_chars / _LATIN_CHARS_PER_TOKEN
    total = math.ceil(dense_tokens + sparse_tokens)
    # Any non-empty text counts as at least one token — defends against
    # whitespace-only payloads gaming the ceiling.
    return max(1, total)


def estimate_chat_tokens(messages: Iterable[Mapping[str, str]]) -> int:
    """Estimate token count for a chat-style message list.

    Each message is a ``Mapping`` with ``"role"`` and ``"content"`` keys.
    Token count = sum of (per-message overhead + tokens(content) +
    tokens(role)) + a conversation-level priming overhead.

    Raises ``TypeError`` / ``ValueError`` on malformed messages so a
    bad request fails before the LLM call instead of being silently
    under-estimated.
    """
    msg_list = list(messages)
    total = _CONVERSATION_OVERHEAD
    for i, msg in enumerate(msg_list):
        if not isinstance(msg, Mapping):
            raise TypeError(
                f"messages[{i}] must be a Mapping; got {type(msg).__name__}"
            )
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{i}].role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError(
                f"messages[{i}].content must be a string; got {type(content).__name__}"
            )
        total += _PER_MESSAGE_OVERHEAD
        total += estimate_tokens(role)
        total += estimate_tokens(content)
    return total


def detect_script(text: str) -> Script:
    """Return the dominant :class:`Script` for ``text`` (LATIN if empty)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str; got {type(text).__name__}")
    if not text:
        return Script.LATIN
    dense = sum(1 for ch in text if _is_dense_codepoint(ord(ch)))
    sparse = len(text) - dense
    return Script.DENSE if dense > sparse else Script.LATIN


__all__ = (
    "Script",
    "detect_script",
    "estimate_chat_tokens",
    "estimate_tokens",
)
