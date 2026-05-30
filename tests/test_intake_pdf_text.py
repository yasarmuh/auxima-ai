"""Tests for PDF text extraction + the OCR seam (P1-10 §4.2)."""
from __future__ import annotations

import importlib.util

import pytest

from auxima_ai.intake.pdf_text import (
    ExtractedText,
    NullOcrEngine,
    OcrUnavailableError,
    PdfExtractionError,
    PypdfExtractor,
    StubOcrEngine,
    StubPdfTextExtractor,
    TextSource,
)

_HAS_PYPDF = importlib.util.find_spec("pypdf") is not None


def test_char_count_ignores_whitespace() -> None:
    et = ExtractedText(text="  ab\n c d \t", page_count=1, source=TextSource.NATIVE)
    assert et.char_count == 4  # a b c d


def test_null_ocr_engine_is_unavailable_and_raises() -> None:
    eng = NullOcrEngine()
    assert eng.available is False
    with pytest.raises(OcrUnavailableError):
        eng.ocr(b"%PDF-1.4 ...")


def test_stub_extractor_returns_configured_text() -> None:
    ex = StubPdfTextExtractor(text="hello world", page_count=2)
    out = ex.extract(b"anything")
    assert out.text == "hello world"
    assert out.page_count == 2
    assert out.source is TextSource.NATIVE


def test_stub_extractor_can_simulate_corruption() -> None:
    with pytest.raises(PdfExtractionError):
        StubPdfTextExtractor(raise_error=True).extract(b"bad")


def test_stub_ocr_engine_available_path() -> None:
    eng = StubOcrEngine(text="scanned text")
    assert eng.available is True
    out = eng.ocr(b"%PDF-")
    assert out.text == "scanned text"
    assert out.source is TextSource.OCR


def test_pypdf_extractor_raises_on_garbage_bytes() -> None:
    # Non-PDF bytes must raise PdfExtractionError (→ Failed), never return ""
    # silently. Skips cleanly if pypdf isn't installed in this environment.
    if not _HAS_PYPDF:
        pytest.skip("pypdf not installed")
    with pytest.raises(PdfExtractionError):
        PypdfExtractor().extract(b"this is not a pdf at all")


def test_pypdf_extractor_missing_dep_message() -> None:
    # When pypdf is absent the extractor must fail loudly with a clear message.
    if _HAS_PYPDF:
        pytest.skip("pypdf is installed; the missing-dep path can't be exercised")
    with pytest.raises(PdfExtractionError, match="pypdf is not installed"):
        PypdfExtractor().extract(b"%PDF-1.4")
