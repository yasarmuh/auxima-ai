"""Tests for ``auxima_ai.bootstrap``.

Coverage:
  - build_intake_service with empty tenants_path -> service has zero policies.
  - build_intake_service with valid tenants.yaml -> tenants registered.
  - build_intake_service with missing tenants file -> BootstrapError.
  - build_intake_service with malformed tenants.yaml -> BootstrapError.
  - bootstrap_app installs the service into the router singleton (idempotent).
  - Built service uses OllamaLLMCaller with the configured base_url.
  - Settings ollama_base_url validation: http loopback OK, http non-loopback rejected.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from textwrap import dedent

import pytest

from auxima_ai.bootstrap import BootstrapError, bootstrap_app, build_intake_service
from auxima_ai.config import Settings, reset_settings_cache
from auxima_ai.intake.ollama import OllamaLLMCaller
from auxima_ai.intake.router import get_intake_service, reset_intake_service
from auxima_ai.intake.service import IntakeService
from auxima_ai.policy.enforcer import TierPolicy, UnknownTenantError


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Scrub env + cached singletons between tests."""
    for key in list(__import__("os").environ):
        if key.startswith("AUXIMA_SIDECAR_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    reset_intake_service()
    yield
    reset_settings_cache()
    reset_intake_service()


def _write_tenants(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "tenants.yaml"
    p.write_text(dedent(body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# build_intake_service — composition
# ---------------------------------------------------------------------------


def test_build_with_empty_tenants_path_yields_empty_policy_registry() -> None:
    s = Settings(shared_secret="a" * 32, tenants_path="")
    svc = build_intake_service(s)
    assert isinstance(svc, IntakeService)
    # Every tenant lookup should refuse — no policies registered.
    with pytest.raises(UnknownTenantError):
        svc.enforcer.policy_for("any-tenant")


def test_build_with_valid_tenants_registers_policies(tmp_path: Path) -> None:
    p = _write_tenants(tmp_path, """
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
    """)
    s = Settings(shared_secret="a" * 32, tenants_path=str(p))
    svc = build_intake_service(s)
    pol_a = svc.enforcer.policy_for("tenant-acme")
    pol_b = svc.enforcer.policy_for("tenant-bma")
    assert pol_a.tier == TierPolicy.OLLAMA_THEN_PAID_CLOUD
    assert pol_a.monthly_ceiling == Decimal("100.00")
    assert pol_b.tier == TierPolicy.OLLAMA_ONLY


def test_build_with_missing_tenants_file_raises_bootstrap_error() -> None:
    s = Settings(shared_secret="a" * 32, tenants_path="/path/does/not/exist.yaml")
    with pytest.raises(BootstrapError, match="failed to load tenants"):
        build_intake_service(s)


def test_build_with_malformed_tenants_raises_bootstrap_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: [unclosed", encoding="utf-8")
    s = Settings(shared_secret="a" * 32, tenants_path=str(p))
    with pytest.raises(BootstrapError):
        build_intake_service(s)


def test_built_service_uses_ollama_caller_with_configured_url() -> None:
    s = Settings(
        shared_secret="a" * 32,
        ollama_base_url="http://localhost:11434",
    )
    svc = build_intake_service(s)
    assert isinstance(svc.llm, OllamaLLMCaller)
    assert svc.llm.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# bootstrap_app — singleton installation
# ---------------------------------------------------------------------------


def test_bootstrap_app_installs_into_router_singleton() -> None:
    s = Settings(shared_secret="a" * 32, tenants_path="")
    built = bootstrap_app(s)
    fetched = get_intake_service()
    assert fetched is built, "router singleton should be the built service"


def test_bootstrap_app_is_idempotent_replaces_singleton() -> None:
    s1 = Settings(shared_secret="a" * 32, tenants_path="")
    s2 = Settings(shared_secret="a" * 32, tenants_path="")
    a = bootstrap_app(s1)
    b = bootstrap_app(s2)
    assert a is not b
    assert get_intake_service() is b


def test_bootstrap_app_uses_module_settings_when_called_without_args(monkeypatch) -> None:
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "a" * 32)
    reset_settings_cache()
    svc = bootstrap_app()
    assert isinstance(svc, IntakeService)


# ---------------------------------------------------------------------------
# ollama_base_url validation (CLAUDE §6 + config.py URL validator)
# ---------------------------------------------------------------------------


def test_ollama_base_url_http_loopback_accepted() -> None:
    Settings(shared_secret="a" * 32, ollama_base_url="http://localhost:11434")
    Settings(shared_secret="a" * 32, ollama_base_url="http://127.0.0.1:11434")
    Settings(shared_secret="a" * 32, ollama_base_url="http://test.localhost:11434")


def test_ollama_base_url_http_non_loopback_rejected() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="https://"):
        Settings(shared_secret="a" * 32, ollama_base_url="http://gpu.production.internal")


def test_ollama_base_url_https_any_host_accepted() -> None:
    Settings(shared_secret="a" * 32, ollama_base_url="https://gpu.production.internal")


def test_ollama_base_url_bad_scheme_rejected() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(shared_secret="a" * 32, ollama_base_url="ftp://localhost:11434")
