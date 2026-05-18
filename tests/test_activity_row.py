"""Tests for ``auxima_ai.activity.row`` — Auxima Activity row builder.

Coverage per CRM §4 + S-25 + CLAUDE §6:
  - build_activity_row fills ULID + UTC timestamp defaults.
  - All three retention classes accepted; non-enum value rejected.
  - PII in payload is redacted before construction; redaction_applied set.
  - Clean payload leaves redaction_applied=False.
  - Validation rejects: empty tenant_id, oversized fields, bad ULID,
    non-dict payload, non-JSON-serialisable payload, oversized payload,
    naive datetime, non-dotted-lowercase kind.
  - customer_id + idempotency_key optional and validated when present.
  - ActivityRow is frozen.
  - Monotonic builder produces lex-sortable ids across rapid calls.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auxima_ai.activity.row import (
    ActivityRow,
    InvalidActivityFieldError,
    MAX_KIND_LEN,
    MAX_PAYLOAD_BYTES,
    MAX_TENANT_ID_LEN,
    PayloadTooLargeError,
    RetentionClass,
    build_activity_row,
)
from auxima_ai.ids.ulid import is_valid

UTC = timezone.utc
TS = datetime(2026, 5, 18, 7, 0, tzinfo=UTC)


def _build(**kwargs) -> ActivityRow:
    defaults = dict(
        tenant_id="tenant-acme",
        kind="intake.extract.completed",
        payload={"activity_id": "x", "tokens": 412},
        retention=RetentionClass.OPERATIONAL,
        source="sidecar.intake.extract",
    )
    defaults.update(kwargs)
    return build_activity_row(**defaults)


# ---------------------------------------------------------------------------
# build_activity_row defaults + happy path
# ---------------------------------------------------------------------------


def test_build_fills_ulid_default() -> None:
    row = _build()
    assert is_valid(row.id)


def test_build_fills_utc_now_default() -> None:
    row = _build()
    assert row.ts.tzinfo is not None
    assert row.ts.utcoffset() == UTC.utcoffset(TS)


def test_build_accepts_explicit_id_and_ts() -> None:
    row = _build(row_id="01HXZ0M5K0RX6P0V7W3GHJK8MN", ts=TS)
    assert row.id == "01HXZ0M5K0RX6P0V7W3GHJK8MN"
    assert row.ts == TS


# ---------------------------------------------------------------------------
# Retention classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retention",
    [RetentionClass.WORM_AUDIT, RetentionClass.OPERATIONAL, RetentionClass.EPHEMERAL],
)
def test_all_three_retention_classes_accepted(retention: RetentionClass) -> None:
    row = _build(retention=retention)
    assert row.retention == retention


def test_retention_must_be_enum() -> None:
    with pytest.raises(InvalidActivityFieldError, match="retention"):
        _build(retention="worm_audit")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------


def test_pii_in_payload_is_redacted_and_flag_set() -> None:
    row = _build(payload={
        "lead_name": "Acme",
        "contact_email": "leak@bad.com",
        "phone": "0512345678",
    })
    assert row.redaction_applied is True
    assert row.payload["contact_email"] == "<redacted:email>"
    assert row.payload["phone"] == "<redacted:phone_ksa_local>"


def test_clean_payload_keeps_flag_false() -> None:
    row = _build(payload={"status": "ok", "count": 3})
    assert row.redaction_applied is False
    assert row.payload == {"status": "ok", "count": 3}


def test_empty_payload_allowed() -> None:
    row = _build(payload={})
    assert row.payload == {}
    assert row.redaction_applied is False


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", None, 42])
def test_empty_tenant_id_rejected(bad: object) -> None:
    with pytest.raises(InvalidActivityFieldError, match="tenant_id"):
        _build(tenant_id=bad)  # type: ignore[arg-type]


def test_oversized_tenant_id_rejected() -> None:
    with pytest.raises(InvalidActivityFieldError, match="tenant_id"):
        _build(tenant_id="x" * (MAX_TENANT_ID_LEN + 1))


@pytest.mark.parametrize(
    "bad_kind",
    ["", "has space", "Upper.Case", "punct!", "with/slash"],
)
def test_bad_kind_rejected(bad_kind: str) -> None:
    with pytest.raises(InvalidActivityFieldError, match="kind"):
        _build(kind=bad_kind)


def test_oversized_kind_rejected() -> None:
    with pytest.raises(InvalidActivityFieldError, match="kind"):
        _build(kind="a." * (MAX_KIND_LEN // 2 + 5))


def test_dotted_lowercase_kind_accepted() -> None:
    row = _build(kind="intake.extract.completed")
    assert row.kind == "intake.extract.completed"


@pytest.mark.parametrize("bad", [42, "not-a-dict", [1, 2]])
def test_non_mapping_payload_rejected(bad: object) -> None:
    with pytest.raises(InvalidActivityFieldError, match="payload"):
        _build(payload=bad)  # type: ignore[arg-type]


def test_non_json_serialisable_payload_rejected() -> None:
    class WeirdObject:
        pass

    # default=str in build_activity_row coerces most weird things to repr
    # strings; a circular dict actually blows json.dumps up.
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(InvalidActivityFieldError, match="JSON-serialisable"):
        _build(payload=cyclic)


def test_oversized_payload_rejected() -> None:
    big_str = "x" * (MAX_PAYLOAD_BYTES + 1)
    with pytest.raises(PayloadTooLargeError):
        _build(payload={"big": big_str})


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(InvalidActivityFieldError, match="timezone-aware"):
        _build(ts=datetime(2026, 5, 18, 7, 0))


def test_bad_ulid_rejected_at_construction() -> None:
    with pytest.raises(InvalidActivityFieldError, match="ULID"):
        ActivityRow(
            id="not-a-ulid",
            tenant_id="t", kind="x.y",
            payload={}, retention=RetentionClass.OPERATIONAL,
            source="src", ts=TS,
        )


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_customer_id_optional_when_omitted() -> None:
    row = _build(customer_id=None)
    assert row.customer_id is None


def test_customer_id_validated_when_present() -> None:
    with pytest.raises(InvalidActivityFieldError, match="customer_id"):
        _build(customer_id="")


def test_idempotency_key_optional_when_omitted() -> None:
    row = _build(idempotency_key=None)
    assert row.idempotency_key is None


def test_idempotency_key_validated_when_present() -> None:
    with pytest.raises(InvalidActivityFieldError, match="idempotency_key"):
        _build(idempotency_key="")


# ---------------------------------------------------------------------------
# Frozen invariant
# ---------------------------------------------------------------------------


def test_activity_row_is_frozen() -> None:
    row = _build()
    with pytest.raises((AttributeError, TypeError)):
        row.tenant_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Monotonic ULID across rapid calls
# ---------------------------------------------------------------------------


def test_consecutive_builds_produce_monotonic_ids() -> None:
    rows = [_build() for _ in range(10)]
    ids = [r.id for r in rows]
    assert ids == sorted(ids), "activity-row ids must be lex-sortable across calls"


# ---------------------------------------------------------------------------
# source validation
# ---------------------------------------------------------------------------


def test_source_must_be_non_empty() -> None:
    with pytest.raises(InvalidActivityFieldError, match="source"):
        _build(source="")
