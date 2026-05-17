"""Observability surface for the auxima-ai sidecar.

Per `BMS/Docs/Planning/slices/S-19-sidecar-observability.md`:
  - PII redaction filter (`redact` module) — runs before any log emission.
  - Structured-log shape (`log` module — future).
  - Metrics + trace propagation (future).

This sub-package is independent of FastAPI; the redactor in particular is a
pure-Python utility that can be unit-tested without any web framework.
"""
