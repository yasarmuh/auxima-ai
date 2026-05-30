"""Prompt-injection delimiting for untrusted extraction input (P1-10 §5.1).

Both the lead-intake prompt (``prompts.py``) and the quote-intake prompt
(``quote_prompt.py``) wrap untrusted, externally-supplied text (an email body,
or text OCR'd out of an insurer's quote PDF) in fixed sentinel markers so the
LLM treats it strictly as data, never as instructions.

The breakout defence is **stripping** the markers from the untrusted text
before wrapping it: an attacker who embeds the exact closing marker would
otherwise be able to end the data block early and have following text read as
instructions. Removing both markers means the text can never contain the
delimiter that closes its own block. The markers are fixed (not random) so the
rendered prompt stays byte-stable for LLM-cache parity; stripping — not random
sentinels — is what makes that safe.

Replaced with a visible redaction token (not silently deleted) so the model
still sees that something was removed.
"""
from __future__ import annotations

from typing import Final

#: What a stripped marker is replaced with — visible, so the model sees a gap.
REMOVED_TOKEN: Final[str] = "[removed-delimiter]"


def markers_for(label: str) -> tuple[str, str]:
    """Return the (open, close) sentinel pair for an untrusted-text *label*.

    e.g. ``markers_for("LEAD_TEXT")`` →
    ``("<<<UNTRUSTED_LEAD_TEXT>>>", "<<<END_UNTRUSTED_LEAD_TEXT>>>")``.
    """
    return (f"<<<UNTRUSTED_{label}>>>", f"<<<END_UNTRUSTED_{label}>>>")


def neutralise_untrusted(text: str, *, label: str) -> str:
    """Strip both sentinels for *label* from *text*, replacing with a token.

    After this, *text* cannot contain either marker, so it cannot break out of
    the data block it will be wrapped in.
    """
    open_marker, close_marker = markers_for(label)
    for marker in (open_marker, close_marker):
        text = text.replace(marker, REMOVED_TOKEN)
    return text


def wrap_untrusted_block(text: str, *, label: str) -> str:
    """Neutralise *text* and wrap it between the *label* sentinels.

    Returns the marker-delimited block (open line, safe text, close line) ready
    to embed in a prompt. The caller is responsible for any surrounding
    instructions / schema.
    """
    open_marker, close_marker = markers_for(label)
    safe_text = neutralise_untrusted(text, label=label)
    return f"{open_marker}\n{safe_text}\n{close_marker}"


__all__ = (
    "REMOVED_TOKEN",
    "markers_for",
    "neutralise_untrusted",
    "wrap_untrusted_block",
)
