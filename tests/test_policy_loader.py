"""Tests for ``auxima_ai.policy.loader`` — tenants.yaml validator + applier.

Coverage:
  - Valid manifest loads + applies; PolicyEnforcer holds each tenant.
  - apply_manifest is idempotent (re-applying same manifest is a no-op).
  - Duplicate tenant_id raises (loud failure, not silent overwrite).
  - Unknown enum tier raises.
  - monthly_ceiling as YAML float rejected (must be quoted string).
  - monthly_ceiling as quoted string parses to exact Decimal.
  - monthly_ceiling negative / NaN / Inf / unparseable rejected.
  - Missing required field raises.
  - Unknown top-level key rejected.
  - Unknown per-tenant key rejected.
  - rate_capacity / rate_refill non-positive rejected.
  - File-not-found / malformed YAML / non-mapping root raise.
  - load_and_apply convenience helper round-trips end-to-end.
  - apply_manifest validates its inputs (rejects non-enforcer / non-manifest).
  - Empty tenants list is allowed (deployment-day-zero with no tenants yet).
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from textwrap import dedent

import pytest

from auxima_ai.policy.enforcer import PolicyEnforcer, TierPolicy
from auxima_ai.policy.loader import (
    TenantManifestError,
    TenantsManifest,
    apply_manifest,
    load_and_apply,
    load_manifest,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "tenants.yaml"
    p.write_text(dedent(body), encoding="utf-8")
    return p


def _valid_body() -> str:
    return """
        version: 1
        tenants:
          - tenant_id: tenant-acme
            tier: ollama_then_paid_cloud
            monthly_ceiling: "100.00"
            rate_capacity: 1000
            rate_refill_per_second: 100
          - tenant_id: tenant-bma
            tier: ollama_only
            monthly_ceiling: "0"
            rate_capacity: 50
            rate_refill_per_second: 5
    """


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_load_valid_manifest(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_body())
    m = load_manifest(p)
    assert isinstance(m, TenantsManifest)
    assert m.version == 1
    assert len(m.tenants) == 2
    assert m.tenants[0].tenant_id == "tenant-acme"
    assert m.tenants[0].tier == TierPolicy.OLLAMA_THEN_PAID_CLOUD


def test_apply_manifest_registers_each_tenant(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_body())
    enf = PolicyEnforcer()
    count = apply_manifest(enf, load_manifest(p))
    assert count == 2
    assert enf.policy_for("tenant-acme").monthly_ceiling == Decimal("100.00")
    assert enf.policy_for("tenant-bma").tier == TierPolicy.OLLAMA_ONLY


def test_load_and_apply_round_trip(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_body())
    enf = PolicyEnforcer()
    n = load_and_apply(enf, p)
    assert n == 2


def test_apply_manifest_is_idempotent(tmp_path: Path) -> None:
    """Re-applying same manifest replaces with identical values."""
    p = _write(tmp_path, _valid_body())
    enf = PolicyEnforcer()
    apply_manifest(enf, load_manifest(p))
    apply_manifest(enf, load_manifest(p))
    pol = enf.policy_for("tenant-acme")
    assert pol.monthly_ceiling == Decimal("100.00")


def test_empty_tenants_list_allowed(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants: []
    """)
    m = load_manifest(p)
    assert m.tenants == []


def test_monthly_ceiling_parses_to_exact_decimal(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_only
            monthly_ceiling: "0.1"
            rate_capacity: 1
            rate_refill_per_second: 1
    """)
    m = load_manifest(p)
    assert m.tenants[0].to_policy().monthly_ceiling == Decimal("0.1")


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_duplicate_tenant_id_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants:
          - tenant_id: dup
            tier: ollama_only
            monthly_ceiling: "0"
            rate_capacity: 1
            rate_refill_per_second: 1
          - tenant_id: dup
            tier: ollama_only
            monthly_ceiling: "0"
            rate_capacity: 1
            rate_refill_per_second: 1
    """)
    with pytest.raises(TenantManifestError, match="duplicate"):
        load_manifest(p)


def test_unknown_tier_value_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_then_secret_provider
            monthly_ceiling: "0"
            rate_capacity: 1
            rate_refill_per_second: 1
    """)
    with pytest.raises(TenantManifestError):
        load_manifest(p)


def test_monthly_ceiling_as_yaml_float_rejected(tmp_path: Path) -> None:
    """Unquoted floats are rejected — the CLAUDE §6 money invariant."""
    p = _write(tmp_path, """
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_only
            monthly_ceiling: 100.50
            rate_capacity: 1
            rate_refill_per_second: 1
    """)
    with pytest.raises(TenantManifestError, match="float"):
        load_manifest(p)


@pytest.mark.parametrize("bad", ["-1", "NaN", "Infinity", "not-a-number"])
def test_monthly_ceiling_invalid_decimal_rejected(tmp_path: Path, bad: str) -> None:
    p = _write(tmp_path, f"""
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_only
            monthly_ceiling: "{bad}"
            rate_capacity: 1
            rate_refill_per_second: 1
    """)
    with pytest.raises(TenantManifestError):
        load_manifest(p)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_only
            # monthly_ceiling missing
            rate_capacity: 1
            rate_refill_per_second: 1
    """)
    with pytest.raises(TenantManifestError):
        load_manifest(p)


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants: []
        rogue_key: true
    """)
    with pytest.raises(TenantManifestError):
        load_manifest(p)


def test_unknown_per_tenant_key_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_only
            monthly_ceiling: "0"
            rate_capacity: 1
            rate_refill_per_second: 1
            mystery_field: nope
    """)
    with pytest.raises(TenantManifestError):
        load_manifest(p)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("rate_capacity", 0),
        ("rate_capacity", -1),
        ("rate_refill_per_second", 0),
        ("rate_refill_per_second", -0.5),
    ],
)
def test_non_positive_rate_params_rejected(
    tmp_path: Path, field: str, bad_value: float,
) -> None:
    body = f"""
        version: 1
        tenants:
          - tenant_id: t1
            tier: ollama_only
            monthly_ceiling: "0"
            rate_capacity: {1 if field != "rate_capacity" else bad_value}
            rate_refill_per_second: {1 if field != "rate_refill_per_second" else bad_value}
    """
    p = _write(tmp_path, body)
    with pytest.raises(TenantManifestError):
        load_manifest(p)


# ---------------------------------------------------------------------------
# I/O errors
# ---------------------------------------------------------------------------


def test_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(TenantManifestError, match="not found"):
        load_manifest(tmp_path / "missing.yaml")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: [unclosed", encoding="utf-8")
    with pytest.raises(TenantManifestError, match="parse"):
        load_manifest(p)


def test_root_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TenantManifestError, match="mapping"):
        load_manifest(p)


# ---------------------------------------------------------------------------
# apply_manifest input validation
# ---------------------------------------------------------------------------


def test_apply_manifest_rejects_non_enforcer() -> None:
    m = TenantsManifest(version=1, tenants=[])
    with pytest.raises(TenantManifestError, match="enforcer"):
        apply_manifest("not-an-enforcer", m)  # type: ignore[arg-type]


def test_apply_manifest_rejects_non_manifest() -> None:
    enf = PolicyEnforcer()
    with pytest.raises(TenantManifestError, match="manifest"):
        apply_manifest(enf, {"version": 1})  # type: ignore[arg-type]
