"""Quote-intake orchestration — PDF bytes → extracted Quote fields (P1-10).

Mirrors ``service.py`` (the lead path) but for the insurer-quote slice, adding
the two things the quote demo needs that the lead path doesn't:

  * **Document ingestion** — base64 PDF → document-class routing (§4.2) → native
    text (pypdf) or OCR (injected seam) → a stable ``QuoteDocumentFailed`` for
    every non-extractable class instead of a misleading 200.
  * **Length-aware confidence** — the model's self-confidence is combined with
    the extracted-text length so garbage-but-high-confidence can't auto-accept;
    the decision (auto_accept / hold_for_review) rides on the response.

Pure-Python, no FastAPI. Returns a sum-type the router maps 1:1 to HTTP. The
lead ``IntakeService`` is deliberately left untouched — this is a sibling
service sharing the same injected primitives (enforcer, idempotency, llm,
activity emitter), so neither path can regress the other.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from datetime import datetime

from auxima_ai.activity.row import RetentionClass, build_activity_row
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
from auxima_ai.intake.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_RELIABLE_TEXT_CHARS,
    AutoAccept,
    compute_confidence,
    decide,
)
from auxima_ai.intake.doc_class import classify_document, failure_reason, is_extractable
from auxima_ai.intake.llm import LLMCaller, StubLLMCaller
from auxima_ai.intake.pdf_text import (
    MIN_NATIVE_TEXT_CHARS,
    ExtractedText,
    NullOcrEngine,
    OcrEngine,
    OcrUnavailableError,
    PdfExtractionError,
    PdfTextExtractor,
    PypdfExtractor,
)
from auxima_ai.intake.prompts import SchemaViolationError
from auxima_ai.intake.quote_prompt import (
    build_quote_extract_prompt,
    validate_quote_extract_response,
)
from auxima_ai.intake.quote_schema import QuoteIntakeRequest, QuoteIntakeResponse
from auxima_ai.intake.service import ActivityEmitter, NullActivityEmitter
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
# Outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuoteSuccess:
    response: QuoteIntakeResponse


@dataclass(frozen=True)
class QuoteReplay:
    response: QuoteIntakeResponse


@dataclass(frozen=True)
class QuoteInFlight:
    key: str


@dataclass(frozen=True)
class QuoteConflict:
    key: str
    seen_fingerprint: str
    new_fingerprint: str


@dataclass(frozen=True)
class QuoteProviderDenied:
    provider: str
    provider_class: str


@dataclass(frozen=True)
class QuoteRateLimited:
    retry_after_seconds: float


@dataclass(frozen=True)
class QuoteCeilingExceeded:
    estimated_cost: str
    current_total: str
    ceiling: str


@dataclass(frozen=True)
class QuoteUnknownProvider:
    provider: str


@dataclass(frozen=True)
class QuoteSchemaInvalid:
    """LLM responded but the payload failed schema validation (→ 502)."""

    errors: tuple[dict, ...]


@dataclass(frozen=True)
class QuoteDocumentFailed:
    """The document could not be turned into text (→ 422 / Frappe routes Failed).

    ``reason`` is a stable machine-readable code (encrypted_pdf, corrupt_document,
    oversized_document, unsupported_document_type, no_text_layer_ocr_required,
    invalid_base64). The Frappe side records it on the Placement → Failed
    transition and writes an ``intake_failed`` activity row — never a silent drop.
    """

    reason: str
    doc_class: str


QuoteOutcome = (
    QuoteSuccess
    | QuoteReplay
    | QuoteInFlight
    | QuoteConflict
    | QuoteProviderDenied
    | QuoteRateLimited
    | QuoteCeilingExceeded
    | QuoteUnknownProvider
    | QuoteSchemaInvalid
    | QuoteDocumentFailed
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class QuoteIntakeService:
    """Per-deployment singleton for the quote-intake pipeline.

    Shares the lead path's enforcer / idempotency / llm / activity emitter, and
    adds the PDF text extractor + OCR engine (both injected so tests need no
    pypdf / Tesseract). ``confidence_threshold`` is the per-tenant auto-accept
    bar T — an engineering default here; the real bar is GAP-1 (user-owned).
    """

    enforcer: PolicyEnforcer
    idempotency: IdempotencyStore = field(default_factory=InMemoryIdempotencyStore)
    llm: LLMCaller = field(default_factory=StubLLMCaller)
    pdf_extractor: PdfTextExtractor = field(default_factory=PypdfExtractor)
    ocr_engine: OcrEngine = field(default_factory=NullOcrEngine)
    activity_ids: MonotonicGenerator = field(default_factory=MonotonicGenerator)
    activity_emitter: ActivityEmitter = field(default_factory=lambda: NullActivityEmitter())
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    def extract_quote(
        self,
        request: QuoteIntakeRequest,
        *,
        idempotency_key: str,
        now: datetime,
        trace: TraceContext | None = None,
    ) -> QuoteOutcome:
        """Run the quote-intake pipeline; return a typed outcome."""
        trace_id = trace.trace_id if trace is not None else None
        span_id = trace.span_id if trace is not None else None

        # 1. Decode + classify the document (no LLM bytes before this passes).
        try:
            data = base64.b64decode(request.document_b64, validate=True)
        except (ValueError, base64.binascii.Error):
            return self._doc_failed(request, "invalid_base64", "unsupported_type", trace_id, span_id)

        doc_class = classify_document(data)
        if not is_extractable(doc_class):
            return self._doc_failed(
                request, failure_reason(doc_class), doc_class.value, trace_id, span_id
            )

        # 2. Native text, falling back to OCR for a no-text-layer (scanned) PDF.
        extracted = self._extract_text(request, data, doc_class, trace_id, span_id)
        if isinstance(extracted, QuoteDocumentFailed):
            return extracted
        text = extracted.text

        # 3. Token estimate + policy gate (tier / rate / ceiling / residency).
        prompt_estimate = estimate_tokens(text)
        completion_estimate = max(1, prompt_estimate // 4)
        auth = self.enforcer.try_authorize(
            request.tenant_id,
            request.model_id,
            estimated_prompt_tokens=prompt_estimate,
            estimated_completion_tokens=completion_estimate,
            now=now,
        )
        if isinstance(auth, ProviderNotAllowed):
            return QuoteProviderDenied(provider=auth.provider, provider_class=auth.provider_class)
        if isinstance(auth, RateLimited):
            return QuoteRateLimited(retry_after_seconds=auth.retry_after_seconds)
        if isinstance(auth, CeilingWouldExceed):
            return QuoteCeilingExceeded(
                estimated_cost=str(auth.estimated_cost),
                current_total=str(auth.current_total),
                ceiling=str(auth.ceiling),
            )
        if isinstance(auth, UnknownProvider):
            return QuoteUnknownProvider(provider=auth.provider)
        assert isinstance(auth, Authorized)

        # 4. Idempotency reserve (keyed on the document hash + model, namespaced
        #    by tenant inside the store).
        doc_sha = hashlib.sha256(data).hexdigest()
        body_fingerprint = fingerprint_payload({"doc_sha256": doc_sha, "model_id": request.model_id})
        key = IdempotencyKey(tenant_id=request.tenant_id, key=idempotency_key)
        idem = self.idempotency.try_begin(key, body_fingerprint)
        if isinstance(idem, BeginReplay):
            return QuoteReplay(response=QuoteIntakeResponse.model_validate(idem.response))
        if isinstance(idem, BeginInFlight):
            return QuoteInFlight(key=idempotency_key)
        if isinstance(idem, BeginConflict):
            return QuoteConflict(
                key=idempotency_key,
                seen_fingerprint=idem.seen_fingerprint,
                new_fingerprint=idem.new_fingerprint,
            )
        assert isinstance(idem, BeginAccepted)

        # 5. LLM extraction (untrusted text wrapped in injection delimiters).
        prompt = build_quote_extract_prompt(text)
        llm_response = self.llm.call(model_id=request.model_id, prompt=prompt)
        try:
            fields = validate_quote_extract_response(llm_response.payload)
        except SchemaViolationError as e:
            emit(
                "warn",
                "intake.quote.schema_violation",
                trace_id=trace_id,
                span_id=span_id,
                fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)},
            )
            return QuoteSchemaInvalid(errors=tuple(e.errors))

        # 6. Record actual spend.
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
            return QuoteCeilingExceeded(
                estimated_cost=str(auth.estimated_cost),
                current_total=str(spend.current_total),
                ceiling=str(spend.ceiling),
            )
        assert isinstance(spend, Recorded)

        # 7. Length-aware confidence + auto-accept/hold decision.
        final_confidence = compute_confidence(
            fields.model_confidence,
            extracted.char_count,
            min_reliable_chars=MIN_RELIABLE_TEXT_CHARS,
        )
        decision = decide(final_confidence, threshold=self.confidence_threshold)
        decision_str = "auto_accept" if isinstance(decision, AutoAccept) else "hold_for_review"

        # 8. Build the response. Drop the model's self-confidence from the
        #    written fields (the final length-aware confidence supersedes it),
        #    then redact PII before it leaves the process.
        quote_fields = fields.model_dump(mode="json")
        quote_fields.pop("model_confidence", None)
        redacted_fields, fired = redact_json(quote_fields)
        response = QuoteIntakeResponse(
            activity_id=self.activity_ids.generate(),
            model_id=request.model_id,
            provider=auth.provider,
            fields=redacted_fields,
            confidence=final_confidence,
            decision=decision_str,
            doc_class=doc_class.value,
            text_source=extracted.source.value,
            page_count=extracted.page_count,
            char_count=extracted.char_count,
            model_version=llm_response.model_version,
            cost=str(spend.entry.quantised_cost),
            period_total=str(spend.period_total),
            redaction_applied=fired,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=llm_response.latency_ms,
        )

        # 9. Complete idempotency + emit log + the canonical Activity row.
        self.idempotency.complete(key, response.model_dump())
        emit(
            "info",
            "intake.quote.extracted",
            trace_id=trace_id,
            span_id=span_id,
            fields={
                "tenant_id": request.tenant_id,
                "activity_id": response.activity_id,
                "confidence": final_confidence,
                "decision": decision_str,
                "doc_class": doc_class.value,
                "text_source": extracted.source.value,
            },
        )
        activity_row = build_activity_row(
            tenant_id=request.tenant_id,
            kind="intake.quote.extracted",
            payload={
                "model_id": response.model_id,
                "provider": response.provider,
                "fields": response.fields,
                "confidence": final_confidence,
                "decision": decision_str,
                "doc_class": doc_class.value,
                "text_source": extracted.source.value,
                "page_count": extracted.page_count,
                "char_count": extracted.char_count,
            },
            retention=RetentionClass.OPERATIONAL,
            source="sidecar.intake.extract_quote",
            idempotency_key=idempotency_key,
            ts=now,
            row_id=response.activity_id,
        )
        self.activity_emitter.emit(activity_row)
        return QuoteSuccess(response=response)

    # -- helpers ----------------------------------------------------------

    def _extract_text(
        self,
        request: QuoteIntakeRequest,
        data: bytes,
        doc_class,
        trace_id: str | None,
        span_id: str | None,
    ) -> ExtractedText | QuoteDocumentFailed:
        """Native extraction, falling back to OCR; failures are typed, not silent."""
        try:
            native = self.pdf_extractor.extract(data)
        except PdfExtractionError:
            return self._doc_failed(request, "corrupt_document", "corrupt", trace_id, span_id)

        if native.char_count >= MIN_NATIVE_TEXT_CHARS:
            return native

        # No usable text layer → scanned image. OCR if a real engine is wired,
        # else fail loudly (never return an empty extraction as success).
        if self.ocr_engine.available:
            try:
                ocr_text = self.ocr_engine.ocr(data)
            except OcrUnavailableError:
                return self._doc_failed(
                    request, "no_text_layer_ocr_required", doc_class.value, trace_id, span_id
                )
            except Exception:
                # A real OCR engine (Tesseract/PaddleOCR) can fail many ways (binary
                # crash, OOM). Fail closed to a typed doc failure rather than a 500
                # that skips the structured failure path (M-3).
                return self._doc_failed(
                    request, "ocr_failed", doc_class.value, trace_id, span_id
                )
            if ocr_text.char_count == 0:
                return self._doc_failed(
                    request, "no_text_extracted", doc_class.value, trace_id, span_id
                )
            return ocr_text

        return self._doc_failed(
            request, "no_text_layer_ocr_required", doc_class.value, trace_id, span_id
        )

    def _doc_failed(
        self,
        request: QuoteIntakeRequest,
        reason: str,
        doc_class: str,
        trace_id: str | None,
        span_id: str | None,
    ) -> QuoteDocumentFailed:
        emit(
            "warn",
            "intake.quote.document_failed",
            trace_id=trace_id,
            span_id=span_id,
            fields={"tenant_id": request.tenant_id, "reason": reason, "doc_class": doc_class},
        )
        return QuoteDocumentFailed(reason=reason, doc_class=doc_class)


__all__ = (
    "QuoteCeilingExceeded",
    "QuoteConflict",
    "QuoteDocumentFailed",
    "QuoteInFlight",
    "QuoteIntakeService",
    "QuoteOutcome",
    "QuoteProviderDenied",
    "QuoteRateLimited",
    "QuoteReplay",
    "QuoteSchemaInvalid",
    "QuoteSuccess",
    "QuoteUnknownProvider",
)
