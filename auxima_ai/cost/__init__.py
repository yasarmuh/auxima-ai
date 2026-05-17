"""Cost ledger — per-tenant AI spend tracking + monthly ceiling enforcement.

Per CLAUDE.md §2:
  - Every LLM call is logged to ``AI Run Log`` (provider, model, version,
    tokens, latency, cost).
  - Each tenant has a configurable per-month cost ceiling.
  - Money is always :class:`decimal.Decimal`, never ``float``.

This package provides:
  - :mod:`.ledger` — :class:`LedgerEntry`, the :class:`CostLedger`
    Protocol, and an in-memory implementation suitable for single-
    process deployments and tests. The Frappe / Postgres-backed
    implementation (the real AI Run Log doctype) satisfies the same
    Protocol.
"""
