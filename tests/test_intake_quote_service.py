"""Tests for ``auxima_ai.intake.quote_service`` — the quote-intake pipeline.

Covers the quote-specific behaviour on top of the shared primitives:
  - happy path: valid PDF → extracted Quote fields + length-aware confidence;
  - auto_accept vs hold_for_review by extracted-text length;
  - §4.2 document-class failures (encrypted / corrupt / unsupported / no-text);
  - OCR seam: scanned PDF + available OCR engine → success; none → Failed;
  - schema-invalid (wrong shape) → QuoteSchemaInvalid (502);
  - replay / provider-denied reuse of the shared gates;
  - money serialised as string (Decimal, not float);
  - the canonical Activity row is emitted with confidence + decision.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from auxima_ai.cost.ledger import InMemoryCostLedger
from auxima_ai.cost.pricing import reset_pricing_table
from auxima_ai.idempotency.store import InMemoryIdempotencyStore
from auxima_ai.intake.llm import StubLLMCaller
from auxima_ai.intake.pdf_text import StubOcrEngine, StubPdfTextExtractor
from auxima_ai.intake.quote_schema import QuoteIntakeRequest
from auxima_ai.intake.quote_service import (
    QuoteConflict,
    QuoteDocumentFailed,
    QuoteIntakeService,
    QuoteProviderDenied,
    QuoteReplay,
    QuoteSchemaInvalid,
    QuoteSuccess,
)
from auxima_ai.intake.service import CapturingActivityEmitter
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter

UTC = timezone.utc
TS = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)

_QUOTE_PAYLOAD = {
    "insurer_name": "Tawuniya",
    "premium": "12500.00",
    "currency": "SAR",
    "sum_insured": "1000000.00",
    "deductible": "500.00",
    "coverage": ["Own damage", "Third-party liability"],
    "exclusions": ["War", "Nuclear"],
    "valid_until": "2026-12-31",
    "model_confidence": 0.95,
}


@pytest.fixture(autouse=True)
def _reset_pricing():
    reset_pricing_table()
    yield
    reset_pricing_table()


def _policy(*, tenant="tenant-acme", tier=TierPolicy.OLLAMA_THEN_PAID_CLOUD, region="INTL"):
    return TenantPolicy(
        tenant_id=tenant, tier=tier, region=region,
        monthly_ceiling=Decimal("100"), rate_capacity=1000.0, rate_refill_per_second=100.0,
    )


def _service(*, policy=None, llm=None, pdf_extractor=None, ocr_engine=None,
             activity_emitter=None, threshold=0.8) -> QuoteIntakeService:
    enf = PolicyEnforcer(
        ledger=InMemoryCostLedger(),
        rate_limiter=PerTenantRateLimiter(capacity=1000.0, refill_per_second=100.0),
    )
    enf.set_policy(policy or _policy())
    kwargs = dict(
        enforcer=enf,
        idempotency=InMemoryIdempotencyStore(),
        llm=llm or StubLLMCaller(payload=_QUOTE_PAYLOAD),
        pdf_extractor=pdf_extractor or StubPdfTextExtractor(text="Q" * 300),
        confidence_threshold=threshold,
    )
    if ocr_engine is not None:
        kwargs["ocr_engine"] = ocr_engine
    if activity_emitter is not None:
        kwargs["activity_emitter"] = activity_emitter
    return QuoteIntakeService(**kwargs)


def _pdf_bytes(body: bytes = b"quote body") -> bytes:
    return b"%PDF-1.4\n" + body + b"\n%%EOF\n"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _req(*, tenant="tenant-acme", data: bytes | None = None, model_id="ollama/qwen2.5:32b"):
    return QuoteIntakeRequest(
        tenant_id=tenant,
        document_b64=_b64(data if data is not None else _pdf_bytes()),
        model_id=model_id,
    )


# --- happy path -------------------------------------------------------------


def test_happy_path_extracts_quote_fields() -> None:
    svc = _service()
    out = svc.extract_quote(_req(), idempotency_key="q-1", now=TS)
    assert isinstance(out, QuoteSuccess)
    resp = out.response
    assert resp.fields["premium"] == "12500.00"  # Decimal serialised as string
    assert isinstance(resp.fields["premium"], str)
    assert "model_confidence" not in resp.fields  # superseded by final confidence
    assert resp.doc_class == "pdf_valid"
    assert resp.text_source == "native"
    assert resp.provider == "ollama"


def test_full_length_text_auto_accepts() -> None:
    # 300 chars >= MIN_RELIABLE (200) → factor 1.0 → final 0.95 → auto_accept
    out = _service().extract_quote(_req(), idempotency_key="q-aa", now=TS)
    assert isinstance(out, QuoteSuccess)
    assert out.response.decision == "auto_accept"
    assert out.response.confidence == pytest.approx(0.95)


def test_short_text_holds_for_review() -> None:
    # 100 chars (>= 20 native floor, < 200 reliable) → factor 0.5 → 0.475 → hold
    svc = _service(pdf_extractor=StubPdfTextExtractor(text="Q" * 100))
    out = svc.extract_quote(_req(), idempotency_key="q-hold", now=TS)
    assert isinstance(out, QuoteSuccess)
    assert out.response.decision == "hold_for_review"
    assert out.response.confidence == pytest.approx(0.475)


# --- §4.2 document-class failures (never silent) ---------------------------


def test_encrypted_pdf_routes_to_failed() -> None:
    data = b"%PDF-1.4\n/Encrypt 1 0 R\n%%EOF"
    out = _service().extract_quote(_req(data=data), idempotency_key="q-enc", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "encrypted_pdf"
    assert out.doc_class == "pdf_encrypted"


def test_corrupt_pdf_routes_to_failed() -> None:
    data = b"%PDF-1.4\nno eof marker here"  # has magic, missing %%EOF
    out = _service().extract_quote(_req(data=data), idempotency_key="q-cor", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "corrupt_document"


def test_unsupported_type_routes_to_failed() -> None:
    data = b"PK\x03\x04 this is a zip not a pdf"
    out = _service().extract_quote(_req(data=data), idempotency_key="q-uns", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "unsupported_document_type"


def test_invalid_base64_routes_to_failed() -> None:
    req = QuoteIntakeRequest(tenant_id="tenant-acme", document_b64="!!!not base64!!!")
    out = _service().extract_quote(req, idempotency_key="q-b64", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "invalid_base64"


def test_extractor_parse_failure_routes_to_failed() -> None:
    svc = _service(pdf_extractor=StubPdfTextExtractor(raise_error=True))
    out = svc.extract_quote(_req(), idempotency_key="q-parse", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "corrupt_document"


def test_no_text_layer_without_ocr_routes_to_failed() -> None:
    # native yields < 20 chars and no OCR engine → no_text_layer_ocr_required
    svc = _service(pdf_extractor=StubPdfTextExtractor(text="x"))
    out = svc.extract_quote(_req(), idempotency_key="q-not", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "no_text_layer_ocr_required"


# --- OCR seam ---------------------------------------------------------------


class _ExplodingOcrEngine:
    """An 'available' OCR engine whose ocr() raises a non-OcrUnavailable error."""

    @property
    def available(self) -> bool:
        return True

    def ocr(self, data: bytes):  # noqa: ARG002
        raise RuntimeError("tesseract segfault")


def test_ocr_engine_crash_fails_closed() -> None:
    # A real OCR engine crashing must route to a typed doc failure, not a 500 (M-3).
    svc = _service(pdf_extractor=StubPdfTextExtractor(text="x"), ocr_engine=_ExplodingOcrEngine())
    out = svc.extract_quote(_req(), idempotency_key="q-ocrboom", now=TS)
    assert isinstance(out, QuoteDocumentFailed)
    assert out.reason == "ocr_failed"


def test_scanned_pdf_with_ocr_engine_succeeds() -> None:
    svc = _service(
        pdf_extractor=StubPdfTextExtractor(text="x"),  # no native text layer
        ocr_engine=StubOcrEngine(text="O" * 300),  # OCR yields full text
    )
    out = svc.extract_quote(_req(), idempotency_key="q-ocr", now=TS)
    assert isinstance(out, QuoteSuccess)
    assert out.response.text_source == "ocr"
    assert out.response.decision == "auto_accept"


# --- schema invalid / shared gates -----------------------------------------


def test_wrong_shape_payload_is_schema_invalid() -> None:
    # default lead-shaped stub payload doesn't match the quote schema → 502
    svc = _service(llm=StubLLMCaller())
    out = svc.extract_quote(_req(), idempotency_key="q-bad", now=TS)
    assert isinstance(out, QuoteSchemaInvalid)


def test_provider_denied_reuses_tier_gate() -> None:
    svc = _service(policy=_policy(tier=TierPolicy.OLLAMA_ONLY))
    out = svc.extract_quote(_req(model_id="openai/gpt-4o-mini"), idempotency_key="q-deny", now=TS)
    assert isinstance(out, QuoteProviderDenied)


def test_replay_returns_cached_response() -> None:
    svc = _service()
    first = svc.extract_quote(_req(), idempotency_key="q-rep", now=TS)
    second = svc.extract_quote(_req(), idempotency_key="q-rep", now=TS)
    assert isinstance(first, QuoteSuccess)
    assert isinstance(second, QuoteReplay)
    assert second.response.activity_id == first.response.activity_id


def test_same_key_different_doc_is_conflict() -> None:
    svc = _service()
    svc.extract_quote(_req(), idempotency_key="q-cf", now=TS)
    out = svc.extract_quote(_req(data=_pdf_bytes(b"a different quote")), idempotency_key="q-cf", now=TS)
    assert isinstance(out, QuoteConflict)


# --- activity invariant -----------------------------------------------------


def test_success_emits_one_activity_row_with_confidence() -> None:
    cap = CapturingActivityEmitter()
    svc = _service(activity_emitter=cap)
    out = svc.extract_quote(_req(), idempotency_key="q-act", now=TS)
    assert isinstance(out, QuoteSuccess)
    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert row.kind == "intake.quote.extracted"
    assert row.payload["decision"] == "auto_accept"
    assert row.payload["confidence"] == pytest.approx(0.95)
