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
from typing import Any, Final

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


def redact_json(obj: Any) -> tuple[Any, bool]:
    """Recursively redact PII inside a JSON-like structure.

    Walks ``dict`` / ``list`` / ``tuple`` containers and applies :func:`redact`
    to every ``str`` leaf. All other leaf types (``int``, ``float``, ``bool``,
    ``None``) pass through unchanged. The traversal is purely functional —
    new containers are constructed; the input is never mutated (matches the
    repo-wide immutability rule).

    Per S-19 §3.4 + S-34 §3.4: needed by the structured-log emitter and the
    outbound webhook payload signer. The redactor is applied to the payload
    *before* HMAC signing so the signature covers the redacted body that
    actually leaves the boundary.

    Parameters
    ----------
    obj :
        Any JSON-decodable Python value (dict / list / tuple / str / int /
        float / bool / None) or a nested combination thereof. Tuples are
        preserved as tuples; lists as lists; dicts as dicts. Dict *keys* are
        not redacted (they're typically field names, not PII).

    Returns
    -------
    (redacted_obj, fired)
        ``redacted_obj`` is a new structure mirroring the input shape with
        string leaves replaced where PII matched. ``fired`` is ``True`` iff
        at least one leaf was modified anywhere in the tree.

    Notes
    -----
    - Cycle protection is intentionally **not** implemented — JSON payloads
      are acyclic by definition; a cyclic input is a programmer error.
    - Unsupported leaf types (e.g. ``bytes``, custom objects) pass through
      unchanged. Callers that want strict-mode behaviour should validate
      the payload shape upstream (Pydantic) before passing it here.
    """
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        fired = False
        new_dict: dict[Any, Any] = {}
        for k, v in obj.items():
            new_v, v_fired = redact_json(v)
            new_dict[k] = new_v
            fired = fired or v_fired
        return new_dict, fired
    if isinstance(obj, list):
        fired = False
        new_list: list[Any] = []
        for item in obj:
            new_item, item_fired = redact_json(item)
            new_list.append(new_item)
            fired = fired or item_fired
        return new_list, fired
    if isinstance(obj, tuple):
        fired = False
        new_items: list[Any] = []
        for item in obj:
            new_item, item_fired = redact_json(item)
            new_items.append(new_item)
            fired = fired or item_fired
        return tuple(new_items), fired
    # int / float / bool / None / unsupported leaf — pass through unchanged.
    return obj, False


__all__ = ("redact", "is_clean", "redact_json")
