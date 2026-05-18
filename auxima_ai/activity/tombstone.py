"""PDPL erasure helper for Auxima Activity rows.

CLAUDE.md §6 invariant:
    PDPL erasure = tombstone, not delete. Right-to-erasure redacts /
    anonymises PII but preserves the audit-bearing skeleton of policy
    / claim / complaint records for the 10-year retention window;
    writes an activity row recording the erasure.

This module implements that contract for any :class:`ActivityRow`:

  - :func:`tombstone(row, *, erased_at, reason)` returns a new
    :class:`ActivityRow` with the SAME id / tenant / kind / retention
    / source / customer_id / idempotency_key / ts but with the
    payload replaced by a typed tombstone object. The original
    skeleton stays in the audit log; PII is gone.
  - :func:`erasure_audit_row(...)` builds the COMPANION row that
    records the erasure event itself (the "we tombstoned X at Y
    because Z" trail per CLAUDE §6).

Both functions are pure: they return new frozen rows, never mutate.
"""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Final

from auxima_ai.activity.row import (
    ActivityRow,
    ActivityRowError,
    RetentionClass,
    build_activity_row,
)
from auxima_ai.ids.ulid import is_valid

logger = logging.getLogger(__name__)


class ErasureReason(str, Enum):
    """Why a row was tombstoned. Drives the companion audit record."""

    PDPL_REQUEST = "pdpl_request"             # data subject right-to-erasure
    RETENTION_POLICY = "retention_policy"     # scheduled expiry of a non-audit row
    REGULATORY_ORDER = "regulatory_order"     # IA / court / SDAIA directive
    OPERATOR = "operator"                     # manual ops action — auditable


# The tombstone payload is intentionally minimal. The skeleton fields
# (id, tenant, kind, retention, source, customer_id, idempotency_key,
# ts) stay on the ActivityRow itself; the payload is the only thing
# that contained PII.
TOMBSTONE_KIND_KEY: Final[str] = "tombstoned"
TOMBSTONE_PAYLOAD_SHAPE: Final[str] = "v1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TombstoneError(ActivityRowError):
    """Raised on invalid input to the tombstone helpers."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_erased_at(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TombstoneError(
            f"erased_at must be datetime; got {type(value).__name__}"
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise TombstoneError(
            "erased_at must be timezone-aware (UTC strongly recommended)"
        )
    return value


def _validate_reason(value: object) -> ErasureReason:
    if not isinstance(value, ErasureReason):
        raise TombstoneError(
            f"reason must be ErasureReason; got {type(value).__name__}"
        )
    return value


def _build_tombstone_payload(
    *,
    original_kind: str,
    erased_at: datetime,
    reason: ErasureReason,
    note: str | None,
) -> dict[str, Any]:
    """Build the typed tombstone payload that replaces the original."""
    payload: dict[str, Any] = {
        TOMBSTONE_KIND_KEY: True,
        "shape": TOMBSTONE_PAYLOAD_SHAPE,
        "original_kind": original_kind,
        "erased_at": erased_at.isoformat(),
        "reason": reason.value,
    }
    if note is not None:
        if not isinstance(note, str):
            raise TombstoneError(f"note must be str; got {type(note).__name__}")
        # Bounded so a single erasure can't break the activity-row payload cap.
        if len(note) > 512:
            raise TombstoneError(f"note length {len(note)} exceeds 512")
        payload["note"] = note
    return payload


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_tombstone(row: ActivityRow) -> bool:
    """``True`` iff ``row.payload`` is a tombstone object."""
    if not isinstance(row, ActivityRow):
        return False
    payload = row.payload
    if not isinstance(payload, dict):
        return False
    return payload.get(TOMBSTONE_KIND_KEY) is True


def tombstone(
    row: ActivityRow,
    *,
    erased_at: datetime,
    reason: ErasureReason,
    note: str | None = None,
) -> ActivityRow:
    """Return a new :class:`ActivityRow` with the payload tombstoned.

    The skeleton (id / tenant_id / kind / retention / source /
    customer_id / idempotency_key / ts) is preserved exactly — the
    audit log keeps its shape. Only the payload changes.

    Idempotent: tombstoning an already-tombstoned row returns a new
    row with the LATER ``erased_at`` and the new ``reason`` so
    sequential erasures (e.g. PDPL_REQUEST then later RETENTION_POLICY)
    are visible in the audit trail.
    """
    if not isinstance(row, ActivityRow):
        raise TombstoneError(
            f"row must be ActivityRow; got {type(row).__name__}"
        )
    erased_at_dt = _validate_erased_at(erased_at)
    reason_enum = _validate_reason(reason)

    original_kind = row.kind
    if is_tombstone(row):
        # If we're tombstoning a tombstone, preserve the FIRST
        # original_kind so the audit trail still points at the
        # logical source kind, not "tombstoned".
        original_kind = row.payload.get("original_kind", row.kind)

    new_payload = _build_tombstone_payload(
        original_kind=original_kind,
        erased_at=erased_at_dt,
        reason=reason_enum,
        note=note,
    )

    return ActivityRow(
        id=row.id,
        tenant_id=row.tenant_id,
        kind=row.kind,
        payload=new_payload,
        retention=row.retention,
        source=row.source,
        ts=row.ts,
        customer_id=row.customer_id,
        idempotency_key=row.idempotency_key,
        redaction_applied=row.redaction_applied,
    )


def erasure_audit_row(
    *,
    target: ActivityRow,
    erased_at: datetime,
    reason: ErasureReason,
    operator: str | None = None,
    note: str | None = None,
) -> ActivityRow:
    """Build the COMPANION audit row that records the erasure event.

    The pair (tombstoned row + companion erasure row) gives auditors a
    complete picture: the original row is now a skeleton with no PII,
    and a separate row carries the "we erased X at Y because Z by W"
    fact. The companion row itself is :class:`RetentionClass.WORM_AUDIT`
    so it cannot be erased without leaving a trail of its own.
    """
    if not isinstance(target, ActivityRow):
        raise TombstoneError(
            f"target must be ActivityRow; got {type(target).__name__}"
        )
    erased_at_dt = _validate_erased_at(erased_at)
    reason_enum = _validate_reason(reason)
    if operator is not None and not isinstance(operator, str):
        raise TombstoneError(
            f"operator must be str; got {type(operator).__name__}"
        )

    audit_payload: dict[str, Any] = {
        "target_id": target.id,
        "target_kind": target.kind,
        "erased_at": erased_at_dt.isoformat(),
        "reason": reason_enum.value,
    }
    if operator is not None:
        audit_payload["operator"] = operator
    if note is not None:
        if not isinstance(note, str):
            raise TombstoneError(f"note must be str; got {type(note).__name__}")
        if len(note) > 512:
            raise TombstoneError(f"note length {len(note)} exceeds 512")
        audit_payload["note"] = note

    return build_activity_row(
        tenant_id=target.tenant_id,
        kind="activity.tombstoned",
        payload=audit_payload,
        retention=RetentionClass.WORM_AUDIT,
        source="sidecar.activity.tombstone",
        customer_id=target.customer_id,
        ts=erased_at_dt,
    )


__all__ = (
    "ErasureReason",
    "TOMBSTONE_KIND_KEY",
    "TOMBSTONE_PAYLOAD_SHAPE",
    "TombstoneError",
    "erasure_audit_row",
    "is_tombstone",
    "tombstone",
)
