"""PII redaction filter — applied to log payloads + cloud-bound LLM prompts.

Per the S-19 §3.5 + S-33 §3.4 specs:
  - Redacts 6 PII pattern classes in plaintext: email, E.164 / KSA-local phone,
    KSA national ID, KSA Commercial Registration (CR), KSA IBAN.
  - Replaces each hit with ``<redacted:<kind>>`` token (round-trip-stable per call).
  - Returns ``(redacted_text, fired: bool)``; the bool drives the
    ``redaction_required`` flag on each log/metric event.

This is a regex-based v1 redactor. **Documented v1 ceilings** (S-19 Q-Sec-1):
  - Novel patterns are missed (e.g. spelt-out numbers).
  - False positives possible on look-alike strings.
  - Acceptable for log redaction, not as a security boundary.

Pair with structured-input contracts on each endpoint (S-19 R5) so PII enters
via named fields rather than free-text, and only the field *names* (not values)
are logged for known-sensitive fields.
"""
from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Compiled patterns — order matters when a string could match more than one
# class (e.g. an IBAN starts with two letters then digits; an Email contains
# an "@" so it can't collide with the others). The patterns are mutually
# exclusive for KSA data classes in practice.
# ---------------------------------------------------------------------------

_EMAIL: Final = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# E.164 international form: leading + then 8-15 digits.
_PHONE_E164: Final = re.compile(r"\+\d{8,15}\b")

# KSA local mobile form: 10 digits starting with 05 (e.g. 0500123456).
_PHONE_KSA_LOCAL: Final = re.compile(r"\b05\d{8}\b")

# KSA national ID: exactly 10 digits, leading 1 (citizen) or 2 (resident).
_KSA_NATIONAL_ID: Final = re.compile(r"\b[12]\d{9}\b")

# KSA Commercial Registration: 10 digits, leading 7 (modern) or starting 10 (legacy).
_KSA_CR: Final = re.compile(r"\b(?:7\d{9}|10\d{8})\b")

# KSA IBAN: ISO 13616 layout for SA — SA + 2 check digits + 20 alphanumerics.
_IBAN_SA: Final = re.compile(r"\bSA\d{2}[A-Z0-9]{20}\b")

# Pattern application order: most specific / longest first so a generic 10-digit
# does not consume what could have been a CR. (Email is unambiguous; placed first
# anyway since it strips long substrings that could otherwise contain a phone.)
_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("email", _EMAIL),
    ("phone_e164", _PHONE_E164),
    ("phone_ksa_local", _PHONE_KSA_LOCAL),
    ("ksa_cr", _KSA_CR),
    ("ksa_national_id", _KSA_NATIONAL_ID),
    ("iban_sa", _IBAN_SA),
)


def redact(text: str) -> tuple[str, bool]:
    """Replace every PII match with a typed placeholder.

    Returns
    -------
    (redacted_text, fired)
        ``redacted_text`` is the input with every match replaced by
        ``<redacted:<kind>>``. ``fired`` is ``True`` iff at least one pattern
        produced at least one replacement.

    Notes
    -----
    The function is deterministic and side-effect-free: identical input always
    produces identical output. Empty / non-string inputs are handled
    defensively — an empty string returns ``("", False)``.
    """
    if not text:
        return text, False

    fired = False
    out = text
    for kind, pat in _PATTERNS:
        replacement = f"<redacted:{kind}>"

        def _sub(_match: re.Match[str], _r: str = replacement) -> str:
            nonlocal fired
            fired = True
            return _r

        out = pat.sub(_sub, out)
    return out, fired


def is_clean(text: str) -> bool:
    """``True`` iff no PII pattern matches the input.

    Convenience helper for tests + assertions; equivalent to ``not redact(text)[1]``
    but avoids the cost of building the replaced string when only the boolean is
    needed.
    """
    if not text:
        return True
    return not any(pat.search(text) for _kind, pat in _PATTERNS)


__all__ = ("redact", "is_clean")
