"""Intake-extract orchestration — composes the primitives end-to-end.

Pure-Python. No FastAPI. Returns a sum-type :class:`IntakeOutcome` so
the FastAPI router maps each outcome to its HTTP status without
business logic leaking into the route layer.

Pipeline (in order, fail-fast at each step):

  1. Estimate prompt + completion tokens.
  2. Policy enforcer ``try_authorize`` (tier / rate / ceiling gates).
  3. Idempotency ``try_begin`` (replay / in-flight / conflict).
  4. LLM call (delegated to injected :class:`LLMCaller`).
  5. Policy enforcer ``record_spend`` (persist actual cost; may still
     reject if actual >> estimate).
  6. Idempotency ``complete`` with the final response.
  7. Emit structured ``intake.extract.completed`` event.

Every step that refuses returns a typed outcome the caller can map
1-to-1 to an HTTP response. No silent failures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from auxima_ai.activity.row import ActivityRow, RetentionClass, build_activity_row
from auxima_ai.cost.ledger import CeilingExceeded as LedgerCeilingExceeded, Recorded
from auxima_ai.idempotency.store import (
    BeginAccepted,
    BeginConflict,
    BeginInFlight,
    BeginReplay,
    IdempotencyKey,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    fingerprint_payload,
)
from auxima_ai.ids.ulid import MonotonicGenerator
from auxima_ai.intake.llm import LLMCaller, StubLLMCaller
from auxima_ai.intake.prompts import (
    SchemaViolationError,
    build_intake_extract_prompt,
    validate_intake_extract_response,
)
from auxima_ai.intake.schema import IntakeRequest, IntakeResponse
from auxima_ai.observability.log import emit
from auxima_ai.observability.redact import redact_json
from auxima_ai.observability.trace import TraceContext
from auxima_ai.policy.enforcer import (
    Authorized,
    CeilingWouldExceed,
    PolicyEnforcer,
    ProviderNotAllowed,
    RateLimited,
    UnknownProvider,
)
from auxima_ai.tokens.estimator import estimate_tokens


# ---------------------------------------------------------------------------
# Outcomes — one per terminal state of the pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntakeSuccess:
    response: IntakeResponse


@dataclass(frozen=True)
class IntakeReplay:
    """Idempotency cache hit; the original response is returned."""

    response: IntakeResponse


@dataclass(frozen=True)
class IntakeInFlight:
    """Another worker is processing this exact key+body — client retries."""

    key: str


@dataclass(frozen=True)
class IntakeConflict:
    """Same idempotency key, DIFFERENT body — client bug; HTTP 422."""

    key: str
    seen_fingerprint: str
    new_fingerprint: str


@dataclass(frozen=True)
class IntakeProviderDenied:
    provider: str
    provider_class: str


@dataclass(frozen=True)
class IntakeRateLimited:
    retry_after_seconds: float


@dataclass(frozen=True)
class IntakeCeilingExceeded:
    estimated_cost: str
    current_total: str
    ceiling: str


@dataclass(frozen=True)
class IntakeUnknownProvider:
    provider: str


@dataclass(frozen=True)
class IntakeSchemaInvalid:
    """LLM responded but the payload failed schema validation — upstream bug.

    Maps to HTTP 502 (Bad Gateway): not the client's fault, not the
    sidecar's bug, but the upstream model didn't honour the schema we
    asked for. ``errors`` is the flat error list from Pydantic.
    """

    errors: tuple[dict, ...]


IntakeOutcome = (
    IntakeSuccess
    | IntakeReplay
    | IntakeInFlight
    | IntakeConflict
    | IntakeProviderDenied
    | IntakeRateLimited
    | IntakeCeilingExceeded
    | IntakeUnknownProvider
    | IntakeSchemaInvalid
)


# ---------------------------------------------------------------------------
# ActivityEmitter — the CRM §4 invariant sink
# ---------------------------------------------------------------------------


@runtime_checkable
class ActivityEmitter(Protocol):
    """Where successful pipeline runs send their Auxima Activity rows.

    Production wires a Frappe-side HTTP POST satisfying this Protocol;
    tests use ``CapturingActivityEmitter`` to assert on the rows the
    service produced.
    """

    def emit(self, row: ActivityRow) -> None: ...


@dataclass
class CapturingActivityEmitter:
    """Test / dev helper — keeps every emitted row in memory."""

    rows: list[ActivityRow] = field(default_factory=list)

    def emit(self, row: ActivityRow) -> None:
        self.rows.append(row)


@dataclass
class NullActivityEmitter:
    """Drops every row on the floor. Default for un-wired test services."""

    def emit(self, row: ActivityRow) -> None:  # noqa: ARG002 - protocol shape
        return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class IntakeService:
    """Bundles the per-deployment singletons the pipeline needs.

    Constructed once at sidecar startup and shared by every request.
    All members are independently injectable so tests can compose
    minimal stubs.
    """

    enforcer: PolicyEnforcer
    idempotency: IdempotencyStore = field(default_factory=InMemoryIdempotencyStore)
    llm: LLMCaller = field(default_factory=StubLLMCaller)
    activity_ids: MonotonicGenerator = field(default_factory=MonotonicGenerator)
    activity_emitter: ActivityEmitter = field(default_factory=lambda: NullActivityEmitter())

    def extract(
        self,
        request: IntakeRequest,
        *,
        idempotency_key: str,
        now: datetime,
        trace: TraceContext | None = None,
    ) -> IntakeOutcome:
        """Run the intake pipeline; return a typed outcome.

        ``now`` is injected (rather than read inside) so callers can
        drive deterministic tests and so the same timestamp threads
        through the policy ceiling, the ledger entry, and the log
        event for traceability.
        """
        trace_id = trace.trace_id if trace is not None else None
        span_id = trace.span_id if trace is not None else None

        # 1. Token estimate (conservative upper-bound).
        prompt_estimate = estimate_tokens(request.lead_text)
        # Completion estimate: rough rule-of-thumb 25% of prompt; matches
        # what most extract-style endpoints actually return.
        completion_estimate = max(1, prompt_estimate // 4)

        # 2. Policy gate.
        auth = self.enforcer.try_authorize(
            request.tenant_id,
            request.model_id,
            estimated_prompt_tokens=prompt_estimate,
            estimated_completion_tokens=completion_estimate,
            now=now,
        )
        if isinstance(auth, ProviderNotAllowed):
            return IntakeProviderDenied(
                provider=auth.provider, provider_class=auth.provider_class,
            )
        if isinstance(auth, RateLimited):
            return IntakeRateLimited(retry_after_seconds=auth.retry_after_seconds)
        if isinstance(auth, CeilingWouldExceed):
            return IntakeCeilingExceeded(
                estimated_cost=str(auth.estimated_cost),
                current_total=str(auth.current_total),
                ceiling=str(auth.ceiling),
            )
        if isinstance(auth, UnknownProvider):
            return IntakeUnknownProvider(provider=auth.provider)
        assert isinstance(auth, Authorized)

        # 3. Idempotency reserve.
        body_fingerprint = fingerprint_payload(
            {"lead_text": request.lead_text, "model_id": request.model_id},
        )
        key = IdempotencyKey(tenant_id=request.tenant_id, key=idempotency_key)
        idem = self.idempotency.try_begin(key, body_fingerprint)
        if isinstance(idem, BeginReplay):
            # Cached response is a dict (already-validated IntakeResponse model dump);
            # re-hydrate so the route layer doesn't have to know the difference.
            return IntakeReplay(response=IntakeResponse.model_validate(idem.response))
        if isinstance(idem, BeginInFlight):
            return IntakeInFlight(key=idempotency_key)
        if isinstance(idem, BeginConflict):
            return IntakeConflict(
                key=idempotency_key,
                seen_fingerprint=idem.seen_fingerprint,
                new_fingerprint=idem.new_fingerprint,
            )
        assert isinstance(idem, BeginAccepted)

        # 4. LLM call — wrapped in the schema-shaped prompt template so
        #    the model knows exactly which fields to emit.
        prompt = build_intake_extract_prompt(request.lead_text)
        llm_response = self.llm.call(model_id=request.model_id, prompt=prompt)

        # 4a. Strict response validation against IntakeExtractFields.
        #    A model that returns the wrong shape is an upstream bug,
        #    NOT a client error — we surface IntakeSchemaInvalid (502)
        #    and refuse to write a malformed activity row.
        try:
            validated_fields = validate_intake_extract_response(llm_response.payload)
        except SchemaViolationError as e:
            emit(
                "warn",
                "intake.extract.schema_violation",
                trace_id=trace_id, span_id=span_id,
                fields={
                    "tenant_id": request.tenant_id,
                    "model_id": request.model_id,
                    "error_count": len(e.errors),
                },
            )
            return IntakeSchemaInvalid(errors=tuple(e.errors))

        # 5. Record actual spend (ledger may still reject on actual >> estimate).
        spend = self.enforcer.record_spend(
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=llm_response.latency_ms,
            ts=now,
            model_version=llm_response.model_version,
        )
        if isinstance(spend, LedgerCeilingExceeded):
            # Actual cost overshot the ceiling — the ledger refused the
            # entry; the LLM bytes already came back but we treat this
            # as the same "exceeded" outcome.
            return IntakeCeilingExceeded(
                estimated_cost=str(auth.estimated_cost),
                current_total=str(spend.current_total),
                ceiling=str(spend.ceiling),
            )
        assert isinstance(spend, Recorded)

        # 6. Build response. Use the validated model dump (not the raw
        #    payload) so optional fields normalise to their declared
        #    defaults and unknown extras are already stripped.
        redacted_fields, fired = redact_json(validated_fields.model_dump(mode="json"))
        response = IntakeResponse(
            activity_id=self.activity_ids.generate(),
            model_id=request.model_id,
            provider=auth.provider,
            fields=redacted_fields,
            cost=str(spend.entry.quantised_cost),
            period_total=str(spend.period_total),
            redaction_applied=fired,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=llm_response.latency_ms,
        )

        # 7. Complete idempotency reservation with the validated response.
        self.idempotency.complete(key, response.model_dump())

        # 8. Emit structured log event.
        emit(
            "info",
            "intake.extract.completed",
            trace_id=trace_id,
            span_id=span_id,
            fields={
                "tenant_id": request.tenant_id,
                "model_id": request.model_id,
                "activity_id": response.activity_id,
                "tokens": response.prompt_tokens + response.completion_tokens,
                "cost": response.cost,
                "redaction_applied": response.redaction_applied,
            },
        )

        # 9. Emit the canonical Auxima Activity row (CRM §4 invariant).
        #    Reuses the response.activity_id as the row ULID so the row
        #    in the audit log and the activity_id in the HTTP response
        #    are the same identifier — clients can follow the link
        #    without joining on a separate field.
        activity_row = build_activity_row(
            tenant_id=request.tenant_id,
            kind="intake.extract.completed",
            payload={
                "model_id": response.model_id,
                "provider": response.provider,
                "fields": response.fields,
                "cost": response.cost,
                "period_total": response.period_total,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
            },
            retention=RetentionClass.OPERATIONAL,
            source="sidecar.intake.extract",
            idempotency_key=idempotency_key,
            ts=now,
            row_id=response.activity_id,
        )
        self.activity_emitter.emit(activity_row)

        return IntakeSuccess(response=response)


__all__ = (
    "ActivityEmitter",
    "CapturingActivityEmitter",
    "IntakeCeilingExceeded",
    "IntakeConflict",
    "IntakeInFlight",
    "IntakeOutcome",
    "IntakeProviderDenied",
    "IntakeRateLimited",
    "IntakeReplay",
    "IntakeSchemaInvalid",
    "IntakeService",
    "NullActivityEmitter",
    "IntakeSuccess",
    "IntakeUnknownProvider",
)
