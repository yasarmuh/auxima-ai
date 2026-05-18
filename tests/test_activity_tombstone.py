"""Tests for ``auxima_ai.activity.tombstone`` — PDPL erasure helper.

Coverage per CLAUDE §6 "PDPL erasure = tombstone, not delete":
  - tombstone() preserves the row skeleton (id / tenant / kind /
    retention / source / customer_id / idempotency_key / ts).
  - tombstone() replaces the payload with a typed tombstone object.
  - tombstone() is pure — original row unchanged.
  - is_tombstone() recognises tombstoned + raw rows correctly.
  - Tombstoning a tombstone preserves the FIRST original_kind.
  - WORM-class rows can still be tombstoned (skeleton stays).
  - erasure_audit_row builds a companion row with kind="activity.tombstoned",
    retention=WORM_AUDIT, payload carrying the target id + reason +
    optional operator + note.
  - Validation: bad row type / naive erased_at / bad reason / oversized
    note all raise.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auxima_ai.activity.row import (
    ActivityRow,
    RetentionClass,
    build_activity_row,
)
from auxima_ai.activity.tombstone import (
    ErasureReason,
    TOMBSTONE_KIND_KEY,
    TOMBSTONE_PAYLOAD_SHAPE,
    TombstoneError,
    erasure_audit_row,
    is_tombstone,
    tombstone,
)

UTC = timezone.utc
ORIGINAL_TS = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
ERASED_AT = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)


def _row(**kwargs) -> ActivityRow:
    defaults = dict(
        tenant_id="tenant-acme",
        kind="lead.created",
        payload={
            "lead_name": "Acme",
            "contact_email": "ops@acme.example",
            "extra_data": {"deep": "value"},
        },
        retention=RetentionClass.OPERATIONAL,
        source="sidecar.intake.extract",
        customer_id="cust-123",
        idempotency_key="k-original",
        ts=ORIGINAL_TS,
    )
    defaults.update(kwargs)
    return build_activity_row(**defaults)


# ---------------------------------------------------------------------------
# is_tombstone
# ---------------------------------------------------------------------------


def test_fresh_row_is_not_tombstone() -> None:
    row = _row()
    assert is_tombstone(row) is False


def test_tombstoned_row_recognised() -> None:
    row = _row()
    tomb = tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    assert is_tombstone(tomb) is True


def test_is_tombstone_rejects_non_row_input() -> None:
    assert is_tombstone("not-a-row") is False  # type: ignore[arg-type]
    assert is_tombstone(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# tombstone() — skeleton preserved
# ---------------------------------------------------------------------------


def test_tombstone_preserves_all_skeleton_fields() -> None:
    row = _row()
    tomb = tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    assert tomb.id == row.id
    assert tomb.tenant_id == row.tenant_id
    assert tomb.kind == row.kind
    assert tomb.retention == row.retention
    assert tomb.source == row.source
    assert tomb.customer_id == row.customer_id
    assert tomb.idempotency_key == row.idempotency_key
    assert tomb.ts == row.ts


def test_tombstone_replaces_payload_with_typed_object() -> None:
    row = _row()
    tomb = tombstone(
        row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST,
        note="subject SR-2026-001",
    )
    p = tomb.payload
    assert p[TOMBSTONE_KIND_KEY] is True
    assert p["shape"] == TOMBSTONE_PAYLOAD_SHAPE
    assert p["original_kind"] == row.kind
    assert p["erased_at"] == ERASED_AT.isoformat()
    assert p["reason"] == "pdpl_request"
    assert p["note"] == "subject SR-2026-001"


def test_tombstone_strips_all_original_payload_fields() -> None:
    row = _row()
    tomb = tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    for original_key in ("lead_name", "contact_email", "extra_data"):
        assert original_key not in tomb.payload


def test_tombstone_is_pure_original_unchanged() -> None:
    row = _row()
    original_payload = dict(row.payload)
    tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    # row is frozen so this is structural — assert payload is byte-equal.
    assert dict(row.payload) == original_payload


def test_tombstone_worm_audit_row_keeps_worm_retention() -> None:
    """A WORM row CAN be tombstoned — skeleton stays for the 10-year window."""
    row = _row(retention=RetentionClass.WORM_AUDIT)
    tomb = tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    assert tomb.retention == RetentionClass.WORM_AUDIT


def test_tombstoning_a_tombstone_preserves_first_original_kind() -> None:
    """Two sequential erasures: original_kind tracks the FIRST origin, not "tombstoned"."""
    row = _row(kind="lead.created")
    first = tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    later = datetime(2026, 12, 31, tzinfo=UTC)
    second = tombstone(first, erased_at=later, reason=ErasureReason.RETENTION_POLICY)
    assert second.payload["original_kind"] == "lead.created"
    assert second.payload["reason"] == "retention_policy"
    assert second.payload["erased_at"] == later.isoformat()


def test_tombstone_without_note_omits_field() -> None:
    row = _row()
    tomb = tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    assert "note" not in tomb.payload


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_tombstone_rejects_non_row_input() -> None:
    with pytest.raises(TombstoneError, match="ActivityRow"):
        tombstone("not-a-row", erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)  # type: ignore[arg-type]


def test_tombstone_rejects_naive_erased_at() -> None:
    row = _row()
    with pytest.raises(TombstoneError, match="timezone-aware"):
        tombstone(row, erased_at=datetime(2026, 5, 18, 12, 0), reason=ErasureReason.PDPL_REQUEST)


def test_tombstone_rejects_non_enum_reason() -> None:
    row = _row()
    with pytest.raises(TombstoneError, match="ErasureReason"):
        tombstone(row, erased_at=ERASED_AT, reason="pdpl_request")  # type: ignore[arg-type]


def test_tombstone_rejects_oversized_note() -> None:
    row = _row()
    with pytest.raises(TombstoneError, match="note"):
        tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST,
                  note="x" * 513)


def test_tombstone_rejects_non_string_note() -> None:
    row = _row()
    with pytest.raises(TombstoneError, match="note"):
        tombstone(row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST,
                  note=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# erasure_audit_row
# ---------------------------------------------------------------------------


def test_erasure_audit_row_basic_shape() -> None:
    row = _row()
    audit = erasure_audit_row(
        target=row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST,
    )
    assert audit.kind == "activity.tombstoned"
    assert audit.retention == RetentionClass.WORM_AUDIT
    assert audit.source == "sidecar.activity.tombstone"
    assert audit.tenant_id == row.tenant_id
    assert audit.customer_id == row.customer_id
    assert audit.ts == ERASED_AT
    assert audit.id != row.id  # new ULID for the companion row


def test_erasure_audit_row_payload_carries_target_id_and_reason() -> None:
    """Audit row goes through the PII redactor — an email operator is
    redacted to a placeholder. Use a non-PII operator identifier so the
    presence assertion is meaningful."""
    row = _row()
    audit = erasure_audit_row(
        target=row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST,
        operator="ops-team-bot", note="ticket SR-2026-001",
    )
    p = audit.payload
    assert p["target_id"] == row.id
    assert p["target_kind"] == row.kind
    assert p["erased_at"] == ERASED_AT.isoformat()
    assert p["reason"] == "pdpl_request"
    assert p["operator"] == "ops-team-bot"
    assert p["note"] == "ticket SR-2026-001"


def test_erasure_audit_row_redacts_pii_operator() -> None:
    """If ops mistakenly passes an email as operator, the redactor catches it."""
    row = _row()
    audit = erasure_audit_row(
        target=row, erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST,
        operator="ops@auxima.example",
    )
    assert audit.payload["operator"] == "<redacted:email>"
    assert audit.redaction_applied is True


def test_erasure_audit_row_validates_inputs() -> None:
    row = _row()
    with pytest.raises(TombstoneError, match="ActivityRow"):
        erasure_audit_row(target="not-a-row",  # type: ignore[arg-type]
                          erased_at=ERASED_AT, reason=ErasureReason.PDPL_REQUEST)
    with pytest.raises(TombstoneError, match="timezone-aware"):
        erasure_audit_row(target=row,
                          erased_at=datetime(2026, 5, 18, 12, 0),
                          reason=ErasureReason.PDPL_REQUEST)
    with pytest.raises(TombstoneError, match="ErasureReason"):
        erasure_audit_row(target=row, erased_at=ERASED_AT, reason="x")  # type: ignore[arg-type]
    with pytest.raises(TombstoneError, match="operator"):
        erasure_audit_row(target=row, erased_at=ERASED_AT,
                          reason=ErasureReason.PDPL_REQUEST,
                          operator=42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "reason",
    [
        ErasureReason.PDPL_REQUEST,
        ErasureReason.RETENTION_POLICY,
        ErasureReason.REGULATORY_ORDER,
        ErasureReason.OPERATOR,
    ],
)
def test_all_erasure_reasons_supported(reason: ErasureReason) -> None:
    row = _row()
    tomb = tombstone(row, erased_at=ERASED_AT, reason=reason)
    assert tomb.payload["reason"] == reason.value
