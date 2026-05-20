"""Document-class routing for intake (P1-10 §4.2 — fail, don't silently drop).

P1-10 requires a corrupt / encrypted / oversized / non-PDF document to route
to a Failed outcome, never a silent drop or a 200. This is the pure
classifier that decides extractable-vs-Failed BEFORE any OCR/LLM work.

Scope flag: the live intake request currently carries already-extracted
``lead_text`` (not raw bytes), so this classifier is not yet wired into the
request path — that's the larger slice (bytes ingestion + schema change +
the Frappe side sending bytes). This ships the decision logic + its typed
outcomes so the Failed routing is correct when the bytes path lands.
"""
from __future__ import annotations

from auxima_ai.intake.doc_class import (
    DocumentClass,
    classify_document,
    failure_reason,
    is_extractable,
)

_PDF_HEAD = b"%PDF-1.7\n"
_EOF = b"\n%%EOF\n"


def test_valid_pdf_is_extractable() -> None:
    data = _PDF_HEAD + b"1 0 obj<<>>endobj\n" + _EOF
    assert classify_document(data) is DocumentClass.PDF_VALID
    assert is_extractable(DocumentClass.PDF_VALID) is True


def test_encrypted_pdf_routes_to_failed() -> None:
    data = _PDF_HEAD + b"trailer<</Encrypt 5 0 R>>" + _EOF
    cls = classify_document(data)
    assert cls is DocumentClass.PDF_ENCRYPTED
    assert is_extractable(cls) is False
    assert failure_reason(cls) == "encrypted_pdf"


def test_truncated_pdf_is_corrupt() -> None:
    data = _PDF_HEAD + b"1 0 obj<<>>endobj\n"  # no %%EOF
    assert classify_document(data) is DocumentClass.CORRUPT


def test_empty_input_is_corrupt() -> None:
    assert classify_document(b"") is DocumentClass.CORRUPT


def test_oversized_is_rejected_before_parsing() -> None:
    big = _PDF_HEAD + b"x" * 100 + _EOF
    assert classify_document(big, max_bytes=10) is DocumentClass.OVERSIZED


def test_non_pdf_bytes_are_unsupported() -> None:
    assert classify_document(b"<html><body>not a pdf</body></html>") is DocumentClass.UNSUPPORTED_TYPE


def test_only_valid_pdf_is_extractable() -> None:
    failed = [c for c in DocumentClass if c is not DocumentClass.PDF_VALID]
    assert all(is_extractable(c) is False for c in failed)


def test_failure_reason_is_stable_for_each_failed_class() -> None:
    assert failure_reason(DocumentClass.CORRUPT) == "corrupt_document"
    assert failure_reason(DocumentClass.OVERSIZED) == "oversized_document"
    assert failure_reason(DocumentClass.UNSUPPORTED_TYPE) == "unsupported_document_type"
    assert failure_reason(DocumentClass.PDF_ENCRYPTED) == "encrypted_pdf"


def test_failure_reason_on_valid_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        failure_reason(DocumentClass.PDF_VALID)
