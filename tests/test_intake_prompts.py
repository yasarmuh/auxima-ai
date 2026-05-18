"""Tests for ``auxima_ai.intake.prompts``.

Coverage:
  - Prompt builder embeds the system instructions, JSON schema, and lead text.
  - Prompt is byte-stable across calls (deterministic for cache parity).
  - Empty / whitespace lead_text rejected.
  - Non-string lead_text rejected.
  - Validator accepts a well-formed payload with all fields populated.
  - Validator accepts payload with optional fields omitted (defaults applied).
  - Validator rejects unknown extra keys (extra="forbid").
  - Validator rejects missing required field (lead_name).
  - Validator rejects bad enum value with the literal accepted values listed.
  - Validator rejects non-string lead_name / wrong type.
  - Validator rejects oversized fields (max_length limits).
  - SchemaViolationError carries a flat error list with loc/msg/type.
  - Non-dict input fails fast with SchemaViolationError.
  - Whitespace stripped from string fields.
"""
from __future__ import annotations

import json

import pytest

from auxima_ai.intake.prompts import (
    IntakeExtractFields,
    LineOfBusiness,
    PromptError,
    SchemaViolationError,
    Urgency,
    build_intake_extract_prompt,
    validate_intake_extract_response,
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_contains_system_instructions() -> None:
    p = build_intake_extract_prompt("Some lead text from a broker")
    assert "extract structured fields" in p.lower()
    assert "json schema" in p.lower()


def test_prompt_embeds_full_field_list() -> None:
    p = build_intake_extract_prompt("Lead text")
    for field in ("lead_name", "contact_email", "contact_phone", "line_of_business", "urgency", "notes"):
        assert field in p


def test_prompt_embeds_lead_text_verbatim() -> None:
    text = "Acme Brokers — renewal needs P&C cover by Friday."
    p = build_intake_extract_prompt(text)
    assert text in p


def test_prompt_strips_lead_whitespace() -> None:
    p = build_intake_extract_prompt("  hello  \n\n")
    assert "  hello  " not in p
    assert "hello" in p


def test_prompt_is_byte_stable_across_calls() -> None:
    """Cache parity: same input -> identical prompt (sorted-key JSON schema)."""
    a = build_intake_extract_prompt("same lead text")
    b = build_intake_extract_prompt("same lead text")
    assert a == b


def test_prompt_includes_enum_values() -> None:
    """Schema embedding must surface the enum values so the LLM sees the choices."""
    p = build_intake_extract_prompt("x")
    for v in ("motor", "property", "marine", "medical", "liability", "life", "energy"):
        assert v in p
    for v in ("low", "normal", "high"):
        assert v in p


@pytest.mark.parametrize("bad", ["", "   ", "\n\t  "])
def test_prompt_rejects_empty_or_whitespace_lead(bad: str) -> None:
    with pytest.raises(PromptError, match="empty"):
        build_intake_extract_prompt(bad)


@pytest.mark.parametrize("bad", [None, 42, [], {}, b"bytes"])
def test_prompt_rejects_non_string_lead(bad: object) -> None:
    with pytest.raises(PromptError, match="str"):
        build_intake_extract_prompt(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Validator — happy paths
# ---------------------------------------------------------------------------


def test_validator_accepts_full_payload() -> None:
    payload = {
        "lead_name": "Acme Brokers LLC",
        "contact_email": "ops@acme.example",
        "contact_phone": "+966 50 000 0000",
        "line_of_business": "property",
        "urgency": "high",
        "notes": "Renewal due Friday; needs all-risk cover.",
    }
    result = validate_intake_extract_response(payload)
    assert isinstance(result, IntakeExtractFields)
    assert result.lead_name == "Acme Brokers LLC"
    assert result.line_of_business == LineOfBusiness.PROPERTY
    assert result.urgency == Urgency.HIGH


def test_validator_accepts_only_required_field() -> None:
    """Optional fields default cleanly when omitted."""
    result = validate_intake_extract_response({"lead_name": "Acme"})
    assert result.lead_name == "Acme"
    assert result.contact_email is None
    assert result.contact_phone is None
    assert result.notes is None
    assert result.line_of_business == LineOfBusiness.UNKNOWN
    assert result.urgency == Urgency.UNKNOWN


def test_validator_accepts_unknown_enum_value() -> None:
    """The explicit "unknown" enum value is the LLM's "I couldn't tell" signal."""
    result = validate_intake_extract_response({
        "lead_name": "Acme",
        "line_of_business": "unknown",
        "urgency": "unknown",
    })
    assert result.line_of_business == LineOfBusiness.UNKNOWN
    assert result.urgency == Urgency.UNKNOWN


def test_validator_strips_whitespace_from_strings() -> None:
    result = validate_intake_extract_response({"lead_name": "  Acme Brokers  "})
    assert result.lead_name == "Acme Brokers"


# ---------------------------------------------------------------------------
# Validator — rejection paths
# ---------------------------------------------------------------------------


def test_validator_rejects_extra_keys() -> None:
    with pytest.raises(SchemaViolationError) as exc:
        validate_intake_extract_response({
            "lead_name": "Acme",
            "rogue_field": "should fail",
        })
    assert any("rogue_field" in e["loc"] or "extra" in e["type"] for e in exc.value.errors)


def test_validator_rejects_missing_required_lead_name() -> None:
    with pytest.raises(SchemaViolationError) as exc:
        validate_intake_extract_response({"contact_email": "x@y.co"})
    assert any("lead_name" in e["loc"] for e in exc.value.errors)


def test_validator_rejects_bad_enum_value() -> None:
    with pytest.raises(SchemaViolationError) as exc:
        validate_intake_extract_response({
            "lead_name": "Acme",
            "line_of_business": "spaceship-insurance",
        })
    assert any("line_of_business" in e["loc"] for e in exc.value.errors)


def test_validator_rejects_wrong_type_on_lead_name() -> None:
    with pytest.raises(SchemaViolationError):
        validate_intake_extract_response({"lead_name": 42})


def test_validator_rejects_empty_lead_name() -> None:
    with pytest.raises(SchemaViolationError):
        validate_intake_extract_response({"lead_name": ""})


def test_validator_rejects_oversized_notes() -> None:
    with pytest.raises(SchemaViolationError):
        validate_intake_extract_response({
            "lead_name": "Acme",
            "notes": "x" * 2001,
        })


def test_validator_rejects_non_dict() -> None:
    with pytest.raises(SchemaViolationError, match="JSON object"):
        validate_intake_extract_response([{"lead_name": "Acme"}])


def test_validator_rejects_string_payload() -> None:
    with pytest.raises(SchemaViolationError):
        validate_intake_extract_response("Acme")


def test_validator_rejects_none() -> None:
    with pytest.raises(SchemaViolationError):
        validate_intake_extract_response(None)


# ---------------------------------------------------------------------------
# Error metadata
# ---------------------------------------------------------------------------


def test_schema_violation_carries_flat_error_list() -> None:
    with pytest.raises(SchemaViolationError) as exc:
        validate_intake_extract_response({"line_of_business": "alien"})
    err = exc.value
    assert isinstance(err.errors, list)
    assert err.errors  # non-empty
    for e in err.errors:
        assert "loc" in e and "msg" in e and "type" in e


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


def test_validator_populates_email_canonical_when_normalisable() -> None:
    """contact_email "Foo@Example.COM" -> contact_email_canonical "Foo@example.com"."""
    result = validate_intake_extract_response({
        "lead_name": "Acme",
        "contact_email": "Foo@Example.COM",
    })
    assert result.contact_email == "Foo@Example.COM"
    assert result.contact_email_canonical == "Foo@example.com"


def test_validator_leaves_email_canonical_none_when_unparseable() -> None:
    """contact_email "not-an-email" -> contact_email_canonical stays None."""
    result = validate_intake_extract_response({
        "lead_name": "Acme",
        "contact_email": "not-an-email",
    })
    assert result.contact_email == "not-an-email"
    assert result.contact_email_canonical is None


@pytest.mark.parametrize(
    "raw, expected_e164",
    [
        ("0512345678",         "+966512345678"),
        ("+966 50 000 0000",   "+966500000000"),
        ("00966512345678",     "+966512345678"),
        ("tel:+966512345678",  "+966512345678"),
    ],
)
def test_validator_populates_phone_e164_when_normalisable(raw: str, expected_e164: str) -> None:
    result = validate_intake_extract_response({
        "lead_name": "Acme",
        "contact_phone": raw,
    })
    assert result.contact_phone == raw.strip()
    assert result.contact_phone_e164 == expected_e164


def test_validator_leaves_phone_e164_none_when_unparseable() -> None:
    result = validate_intake_extract_response({
        "lead_name": "Acme",
        "contact_phone": "not-a-phone",
    })
    assert result.contact_phone == "not-a-phone"
    assert result.contact_phone_e164 is None


def test_validator_honours_caller_supplied_canonical_forms() -> None:
    """If the LLM/test stub already filled the _canonical / _e164 fields,
    the post-validator must NOT overwrite them."""
    result = validate_intake_extract_response({
        "lead_name": "Acme",
        "contact_email": "raw@example.com",
        "contact_email_canonical": "explicit@override.com",
        "contact_phone": "0512345678",
        "contact_phone_e164": "+966999999999",
    })
    assert result.contact_email_canonical == "explicit@override.com"
    assert result.contact_phone_e164 == "+966999999999"


def test_validator_canonical_fields_none_when_raw_fields_none() -> None:
    result = validate_intake_extract_response({"lead_name": "Acme"})
    assert result.contact_email_canonical is None
    assert result.contact_phone_e164 is None


def test_schema_emits_all_required_fields() -> None:
    """The JSON schema embedded in the prompt must list lead_name as required."""
    schema = IntakeExtractFields.model_json_schema()
    assert "lead_name" in schema.get("required", [])


def test_schema_is_json_serialisable() -> None:
    """If this round-trips, the prompt embedding is safe."""
    schema = IntakeExtractFields.model_json_schema()
    s = json.dumps(schema)
    assert json.loads(s) == schema
