"""Per-tenant rate limiting for sidecar endpoints (S-19 backpressure).

Token-bucket primitive in :mod:`.bucket`. Backs the per-tenant monthly
cost ceiling (CLAUDE.md §2) and protects Ollama / cloud LLM providers
from accidental thundering-herd traffic.

The implementation is in-process; multi-replica deployments swap in
the Redis-backed bucket (future module) that satisfies the same
:class:`RateLimiter` Protocol.
"""
