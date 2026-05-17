"""Tests for ``auxima_ai.observability.redact.redact_json``.

Coverage per S-19 §3.4 + S-34 §3.4:
  - Recursive redaction across nested dicts / lists / tuples.
  - Non-string leaves (int / float / bool / None) pass through unchanged.
  - Input is never mutated (immutability invariant per CLAUDE coding-style).
  - ``fired`` flag is True iff at least one leaf was modified anywhere.
  - Dict keys are not redacted (keys are field names, not PII).
  - Tuple-ness is preserved (tuples stay tuples; lists stay lists).
  - Mixed nesting + multiple PII classes in one tree.
  - Empty containers round-trip cleanly.
"""
from __future__ import annotations

import copy

import pytest

from auxima_ai.observability.redact import redact_json


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def test_string_leaf_with_pii_is_redacted() -> None:
    out, fired = redact_json("Email me at user@example.com")
    assert fired is True
    assert "user@example.com" not in out
    assert "<redacted:email>" in out


def test_string_leaf_without_pii_passes_through() -> None:
    out, fired = redact_json("just a plain string")
    assert fired is False
    assert out == "just a plain string"


@pytest.mark.parametrize("scalar", [0, 1, -1, 3.14, True, False, None])
def test_non_string_scalars_pass_through_unchanged(scalar: object) -> None:
    out, fired = redact_json(scalar)
    assert fired is False
    assert out == scalar
    assert type(out) is type(scalar)


# ---------------------------------------------------------------------------
# Dicts
# ---------------------------------------------------------------------------


def test_dict_with_pii_value_is_redacted() -> None:
    payload = {"contact_email": "a@b.co", "mobile": "0512345678", "count": 3}
    out, fired = redact_json(payload)
    assert fired is True
    assert out["contact_email"] == "<redacted:email>"
    assert out["mobile"] == "<redacted:phone_ksa_local>"
    assert out["count"] == 3
    # Original unchanged (immutability)
    assert payload["contact_email"] == "a@b.co"
    assert payload["mobile"] == "0512345678"


def test_dict_keys_are_not_redacted() -> None:
    """Keys are field names, not PII — they pass through even if PII-shaped."""
    payload = {"user@example.com": "value"}
    out, fired = redact_json(payload)
    # Key preserved even though it looks like an email
    assert "user@example.com" in out
    assert out["user@example.com"] == "value"
    assert fired is False


def test_empty_dict_passes_through() -> None:
    out, fired = redact_json({})
    assert out == {}
    assert fired is False


def test_dict_without_pii_does_not_fire() -> None:
    payload = {"a": 1, "b": "hello", "c": [1, 2, 3]}
    out, fired = redact_json(payload)
    assert fired is False
    assert out == payload


# ---------------------------------------------------------------------------
# Lists + tuples
# ---------------------------------------------------------------------------


def test_list_redacts_each_element() -> None:
    payload = ["clean", "email me at a@b.co", "phone 0512345678", 42]
    out, fired = redact_json(payload)
    assert fired is True
    assert out[0] == "clean"
    assert "<redacted:email>" in out[1]
    assert "<redacted:phone_ksa_local>" in out[2]
    assert out[3] == 42


def test_empty_list_passes_through() -> None:
    out, fired = redact_json([])
    assert out == []
    assert fired is False


def test_tuple_preserves_tuple_type() -> None:
    """A tuple input returns a tuple output (not a list)."""
    payload = ("clean", "email a@b.co")
    out, fired = redact_json(payload)
    assert isinstance(out, tuple)
    assert fired is True
    assert out[0] == "clean"
    assert "<redacted:email>" in out[1]


def test_empty_tuple_passes_through() -> None:
    out, fired = redact_json(())
    assert out == ()
    assert isinstance(out, tuple)
    assert fired is False


# ---------------------------------------------------------------------------
# Nested structures — the actual webhook / log payload shape
# ---------------------------------------------------------------------------


def test_nested_dict_with_pii_at_depth() -> None:
    payload = {
        "event": "lead.created",
        "data": {
            "customer": {
                "name": "Acme Brokers LLC",
                "email": "ops@acme.sa",
                "phones": ["0512345678", "+966500000000"],
            },
            "cr": "7012345678",
        },
        "metadata": {"trace_id": "abc-123"},
    }
    out, fired = redact_json(payload)
    assert fired is True
    # Deep PII redacted
    assert out["data"]["customer"]["email"] == "<redacted:email>"
    assert out["data"]["customer"]["phones"][0] == "<redacted:phone_ksa_local>"
    assert out["data"]["customer"]["phones"][1] == "<redacted:phone_e164>"
    assert out["data"]["cr"] == "<redacted:ksa_cr>"
    # Non-PII strings untouched
    assert out["event"] == "lead.created"
    assert out["data"]["customer"]["name"] == "Acme Brokers LLC"
    assert out["metadata"]["trace_id"] == "abc-123"


def test_nested_does_not_mutate_input() -> None:
    """The input structure must be byte-for-byte identical after the call."""
    payload = {
        "user": {"email": "x@y.com", "ids": ["1234567890", "7012345678"]},
        "items": [{"phone": "+966500000000"}, {"phone": "0512345678"}],
    }
    snapshot = copy.deepcopy(payload)
    out, fired = redact_json(payload)
    assert fired is True
    assert payload == snapshot, "redact_json must NOT mutate the input structure"


def test_nested_no_pii_anywhere_returns_false_fired() -> None:
    payload = {
        "a": [1, 2, 3],
        "b": {"c": "hello", "d": [True, None, 42]},
        "e": ("tuple", "of", "strings"),
    }
    out, fired = redact_json(payload)
    assert fired is False
    assert out == payload


# ---------------------------------------------------------------------------
# Fired-flag propagation
# ---------------------------------------------------------------------------


def test_fired_true_when_pii_only_at_leaf() -> None:
    """A single PII string at any depth flips the top-level fired bit."""
    payload = {"a": {"b": {"c": [{"d": "buried@deep.com"}]}}}
    out, fired = redact_json(payload)
    assert fired is True
    assert out["a"]["b"]["c"][0]["d"] == "<redacted:email>"


def test_fired_false_when_only_keys_look_like_pii() -> None:
    """Keys are never redacted, so a PII-shaped key alone doesn't flip fired."""
    payload = {"7012345678": "value"}
    out, fired = redact_json(payload)
    assert fired is False
    assert out == payload


# ---------------------------------------------------------------------------
# Unsupported leaf types — pass-through, not crash
# ---------------------------------------------------------------------------


def test_unsupported_leaf_type_passes_through() -> None:
    """``bytes`` is not a supported JSON type — it must pass through unchanged."""
    payload = {"blob": b"raw bytes", "name": "x"}
    out, fired = redact_json(payload)
    assert fired is False
    assert out["blob"] == b"raw bytes"


def test_set_leaf_passes_through_unchanged() -> None:
    """``set`` is not in the JSON family — pass-through to avoid silent rewrites."""
    payload = {"tags": {"a", "b"}}
    out, fired = redact_json(payload)
    assert fired is False
    assert out["tags"] == {"a", "b"}
