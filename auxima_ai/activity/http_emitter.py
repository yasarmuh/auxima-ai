"""HTTP-backed :class:`ActivityEmitter` that POSTs to the Frappe app.

Production bridge between the sidecar's intake.service and the
Frappe-side ``Auxima Activity`` doctype. Posts each :class:`ActivityRow`
as JSON to the Frappe REST endpoint over the configured base URL,
attaching the configured callback token in a header.

Error policy — emit() must NEVER raise:
  The intake.service calls emit() AFTER the LLM call has succeeded
  and the response is already shaped. Raising here would discard a
  successful run because of a transient downstream issue, which is
  worse than losing the audit row (the structured log event already
  captured the same facts; ops can re-emit from the log stream).
  Every failure path is logged and swallowed.

The HTTP transport is injectable (:class:`httpx.MockTransport` in
tests) so the unit-test suite has zero network dependence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import httpx

from auxima_ai.activity.row import ActivityRow

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
DEFAULT_PATH: Final[str] = "/api/method/auxima.api.activity.create"
TOKEN_HEADER: Final[str] = "X-Auxima-Frappe-Token"


@dataclass
class HTTPActivityEmitter:
    """POSTs each :class:`ActivityRow` to the Frappe `auxima.activity.create` endpoint.

    Construct once at app startup; reuse the underlying ``httpx.Client``
    so the connection pool warms up. Usable as a context manager for
    clean shutdown.
    """

    base_url: str
    token: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    path: str = DEFAULT_PATH
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("token must be a non-empty string")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0; got {self.timeout_seconds}"
            )
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
            headers={TOKEN_HEADER: self.token},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HTTPActivityEmitter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- ActivityEmitter Protocol -----------------------------------------

    def emit(self, row: ActivityRow) -> None:
        """Post one row. Logs + swallows every error — never raises."""
        if not isinstance(row, ActivityRow):
            logger.error(
                "HTTPActivityEmitter.emit called with non-ActivityRow: %s",
                type(row).__name__,
            )
            return
        body = _row_to_wire(row)
        try:
            resp = self._client.post(self.path, json=body)
        except httpx.TimeoutException:
            logger.warning(
                "activity emit timed out after %ss for row %s (kind=%s)",
                self.timeout_seconds, row.id, row.kind,
            )
            return
        except httpx.HTTPError as e:
            logger.warning(
                "activity emit network error for row %s (kind=%s): %s",
                row.id, row.kind, e,
            )
            return

        if resp.status_code >= 500:
            logger.error(
                "activity emit upstream %d for row %s (kind=%s); body=%s",
                resp.status_code, row.id, row.kind, resp.text[:200],
            )
            return
        if resp.status_code >= 400:
            logger.warning(
                "activity emit rejected %d for row %s (kind=%s); body=%s",
                resp.status_code, row.id, row.kind, resp.text[:200],
            )
            return
        logger.debug("activity emit ok %d for row %s", resp.status_code, row.id)


def _row_to_wire(row: ActivityRow) -> dict:
    """Convert an :class:`ActivityRow` into the Frappe-expected JSON body.

    Frappe's whitelisted REST handlers expect plain dicts; the activity
    row is broken out field-by-field so the doctype handler can validate
    each one against its own Field constraints. ``ts`` and ``retention``
    are serialised to their wire forms.
    """
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "customer_id": row.customer_id,
        "kind": row.kind,
        "payload": dict(row.payload),  # ensure plain dict, not Mapping
        "retention": row.retention.value,
        "source": row.source,
        "idempotency_key": row.idempotency_key,
        "redaction_applied": row.redaction_applied,
        "ts": row.ts.isoformat(),
    }


__all__ = (
    "DEFAULT_PATH",
    "DEFAULT_TIMEOUT_SECONDS",
    "HTTPActivityEmitter",
    "TOKEN_HEADER",
)
