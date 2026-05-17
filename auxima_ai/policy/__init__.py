"""Per-tenant policy enforcement — provider selection + rate limit + cost ceiling.

Composes the per-tenant primitives from the other packages into a
single ``try_authorize`` admission decision:

  - :mod:`auxima_ai.cost.ledger`   — monthly cost ceiling
  - :mod:`auxima_ai.cost.pricing`  — token → Decimal cost
  - :mod:`auxima_ai.ratelimit`     — per-tenant token bucket

Per CLAUDE.md §2: each tenant carries a ``tier_policy`` flag
(``ollama_only`` / ``ollama_then_free_cloud`` / ``ollama_then_paid_cloud``)
that gates which providers the sidecar may call on their behalf. The
enforcer in :mod:`.enforcer` evaluates that flag together with the rate
limit and cost ledger BEFORE the LLM call leaves the box, so a tenant
never burns spend or quota beyond their policy.
"""
