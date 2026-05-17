"""Idempotency support for sidecar write endpoints (CLAUDE.md §6 + S-19).

The CRM activity invariant requires every state change to emit exactly
one activity row. Network retries are reality (Frappe → sidecar can
double-fire on transient failures), so write endpoints MUST be
idempotent: a re-submission with the same key + same body returns the
original response; a re-submission with the same key + different body
is a hard conflict (the client's bug, not ours).

This package implements that contract:

  - :mod:`.store` — the abstract :class:`IdempotencyStore` Protocol and
    an in-memory TTL-bounded implementation suitable for single-process
    deployments and unit tests. The Redis-backed implementation (for
    multi-replica prod) is a later module that satisfies the same
    Protocol.
"""
