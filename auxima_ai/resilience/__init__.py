"""Resilience primitives — circuit breaker, bulkhead, etc.

:mod:`.circuit` implements a 3-state circuit breaker suitable for any
outbound dependency where serial failure storms should be replaced
with a single fast-fail until the dependency recovers (LiteLLM calls,
webhook receivers, the Frappe REST surface).
"""
