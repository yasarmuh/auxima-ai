"""Lead-intake vertical slice — the first real endpoint composing the primitives.

Wires together:
  - :mod:`auxima_ai.policy.enforcer`   — tier + cost + rate authorisation
  - :mod:`auxima_ai.idempotency.store` — idempotency-key contract
  - :mod:`auxima_ai.tokens.estimator`  — pre-call token estimate for cost gate
  - :mod:`auxima_ai.observability.log` — structured event emission
  - :mod:`auxima_ai.observability.trace` — W3C traceparent propagation
  - :mod:`auxima_ai.ids.ulid`          — monotonic activity-row IDs

Two layers:

  * :mod:`.service` — pure-Python orchestration (the actual pipeline,
    framework-agnostic). 100% unit-testable without FastAPI.
  * :mod:`.router`  — thin FastAPI wrapper that maps service results
    to HTTP responses (200 / 401 / 409 / 422 / 429 / 503).

The real LLM call is hidden behind the :class:`LLMCaller` Protocol so
tests inject a deterministic stub. A real Ollama / LiteLLM caller
plugs in at deployment time without touching the service.
"""
