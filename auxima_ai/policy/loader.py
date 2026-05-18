"""YAML loader for the per-tenant policy registry.

The sidecar reads ``tenants.yaml`` at startup and bulk-registers every
:class:`TenantPolicy` into a :class:`PolicyEnforcer`. Operators edit
the YAML file; nobody has to write Python to onboard a new tenant.

File shape (validated by :class:`TenantsManifest`):

    version: 1
    tenants:
      - tenant_id: tenant-acme
        tier: ollama_then_paid_cloud
        monthly_ceiling: "100.00"        # quoted so YAML doesn't lose precision
        rate_capacity: 1000
        rate_refill_per_second: 100

Loud-failure design (CLAUDE §6 + the schemas.py rationale):
  - Unknown keys raise — a typo in ``ceiling`` vs ``monthly_ceiling``
    becomes a startup error, never a silent zero-cap.
  - ``monthly_ceiling`` MUST be a string in the YAML; floats would
    invite float-shaped money drift (CLAUDE §6 "Money is Decimal,
    never float"). The loader parses the string into Decimal.
  - Duplicate ``tenant_id`` raises — silent overwrite is the bug
    that produced "tenant-a got tenant-b's ceiling" in prior
    incidents.
  - YAML date-shaped values are rejected with an explicit message
    (the schemas.py + manifest.py modules already pinned this).
"""
from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TenantManifestError(ValueError):
    """Raised for any tenants.yaml validation / parse failure."""


# ---------------------------------------------------------------------------
# Pydantic shapes — strict, no silent extras
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TenantEntry(_Strict):
    """One row of ``tenants.yaml``. Mirrors :class:`TenantPolicy`'s fields."""

    tenant_id: str = Field(..., min_length=1, max_length=128)
    tier: TierPolicy
    monthly_ceiling: str = Field(
        ...,
        description=(
            "Decimal amount, QUOTED in YAML so precision is preserved. "
            "Examples: '0', '100.00', '5000.50'. Floats are rejected."
        ),
    )
    rate_capacity: float = Field(..., gt=0)
    rate_refill_per_second: float = Field(..., gt=0)

    @field_validator("monthly_ceiling", mode="before")
    @classmethod
    def _ceiling_must_be_string(cls, v: object) -> str:
        # YAML 1.1 parses unquoted "1.5" as float; rejecting non-strings
        # forces operators to quote the value, which preserves the exact
        # Decimal representation they typed.
        if isinstance(v, float):
            raise TenantManifestError(
                f"monthly_ceiling must be a QUOTED string in YAML (got float {v!r}); "
                "this preserves Decimal precision per CLAUDE §6"
            )
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            raise TenantManifestError(
                f"monthly_ceiling must be a string (or int); "
                f"got {type(v).__name__}"
            )
        return str(v)

    @field_validator("monthly_ceiling")
    @classmethod
    def _ceiling_parses_as_decimal(cls, v: str) -> str:
        try:
            d = Decimal(v)
        except InvalidOperation as e:
            raise TenantManifestError(
                f"monthly_ceiling {v!r} is not a valid Decimal"
            ) from e
        if d.is_nan() or d.is_infinite():
            raise TenantManifestError(
                f"monthly_ceiling must be finite; got {v!r}"
            )
        if d < 0:
            raise TenantManifestError(
                f"monthly_ceiling must be >= 0; got {v!r}"
            )
        return v

    def to_policy(self) -> TenantPolicy:
        return TenantPolicy(
            tenant_id=self.tenant_id,
            tier=self.tier,
            monthly_ceiling=Decimal(self.monthly_ceiling),
            rate_capacity=self.rate_capacity,
            rate_refill_per_second=self.rate_refill_per_second,
        )


class TenantsManifest(_Strict):
    """The whole ``tenants.yaml`` document."""

    version: int = Field(..., ge=1)
    tenants: list[TenantEntry] = Field(..., min_length=0)

    @model_validator(mode="after")
    def _check_unique_tenant_ids(self) -> "TenantsManifest":
        seen: set[str] = set()
        dups: list[str] = []
        for t in self.tenants:
            if t.tenant_id in seen:
                dups.append(t.tenant_id)
            seen.add(t.tenant_id)
        if dups:
            raise TenantManifestError(
                f"duplicate tenant_id(s) in manifest: {sorted(set(dups))}"
            )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_manifest(path: Path | str) -> TenantsManifest:
    """Read + validate ``tenants.yaml`` from disk.

    Raises :class:`TenantManifestError` on file-not-found, malformed
    YAML, root-not-a-mapping, schema violation, or duplicate tenant id.
    """
    p = Path(path)
    if not p.is_file():
        raise TenantManifestError(f"tenants manifest not found: {p}")
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise TenantManifestError(f"failed to read tenants manifest {p}: {e}") from e
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise TenantManifestError(
            f"tenants manifest YAML parse error in {p}: {e}"
        ) from e
    if not isinstance(data, dict):
        raise TenantManifestError(
            f"tenants manifest root must be a mapping; got {type(data).__name__}"
        )
    try:
        return TenantsManifest.model_validate(data)
    except TenantManifestError:
        raise
    except Exception as e:  # noqa: BLE001 - re-raised as our typed error
        raise TenantManifestError(
            f"tenants manifest schema violation in {p}: {e}"
        ) from e


def apply_manifest(
    enforcer: PolicyEnforcer,
    manifest: TenantsManifest,
) -> int:
    """Bulk-register every tenant from ``manifest`` into ``enforcer``.

    Returns the number of policies registered (== ``len(manifest.tenants)``).
    Idempotent: re-applying the same manifest replaces existing policies
    with identical values (handy for hot-reload).
    """
    if not isinstance(enforcer, PolicyEnforcer):
        raise TenantManifestError(
            f"enforcer must be PolicyEnforcer; got {type(enforcer).__name__}"
        )
    if not isinstance(manifest, TenantsManifest):
        raise TenantManifestError(
            f"manifest must be TenantsManifest; got {type(manifest).__name__}"
        )
    count = 0
    for entry in manifest.tenants:
        enforcer.set_policy(entry.to_policy())
        count += 1
    logger.info("registered %d tenant policies from manifest", count)
    return count


def load_and_apply(
    enforcer: PolicyEnforcer,
    path: Path | str,
) -> int:
    """Convenience: :func:`load_manifest` then :func:`apply_manifest`."""
    return apply_manifest(enforcer, load_manifest(path))


__all__ = (
    "TenantEntry",
    "TenantManifestError",
    "TenantsManifest",
    "apply_manifest",
    "load_and_apply",
    "load_manifest",
)
