"""Tests for TesseractOcrEngine (P2-09) — rasterise/join logic with an injected fake backend
(no binary needed), fail-closed behaviour, OcrEngine-protocol conformance, and a real-binary
integration test gated on availability."""
from __future__ import annotations

import io

import pytest

from auxima_ai.intake.ocr_tesseract import TesseractOcrEngine
from auxima_ai.intake.pdf_text import (
    ExtractedText,
    OcrEngine,
    OcrUnavailableError,
    PdfExtractionError,
    TextSource,
)

PIL = pytest.importorskip("PIL", reason="Pillow needed to synthesise image-only PDFs")
pytest.importorskip("pypdfium2", reason="pypdfium2 needed to rasterise")

from PIL import Image, ImageDraw  # noqa: E402


def _image_pdf(texts: list[str]) -> bytes:
    """An image-only (no text layer) PDF with one page per string — the OCR input case."""
    pages = []
    for line in texts:
        img = Image.new("RGB", (600, 120), "white")
        ImageDraw.Draw(img).text((20, 45), line, fill="black")
        pages.append(img)
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


def _fake_backend(pages: list[str]):
    """A backend that returns successive canned strings — one per rendered page."""
    seq = iter(pages)

    def backend(_image) -> str:
        return next(seq)

    return backend


def test_conforms_to_ocr_engine_protocol():
    assert isinstance(TesseractOcrEngine(ocr_backend=lambda _i: "x"), OcrEngine)


def test_injected_backend_is_available():
    # pypdfium2 is present + a backend is injected → engine reports available without a binary.
    assert TesseractOcrEngine(ocr_backend=lambda _i: "x").available is True


def test_single_page_ocr_with_injected_backend():
    pdf = _image_pdf(["POLICY NO 12345"])
    engine = TesseractOcrEngine(ocr_backend=_fake_backend(["POLICY NO 12345"]))
    result = engine.ocr(pdf)
    assert isinstance(result, ExtractedText)
    assert result.source is TextSource.OCR
    assert result.page_count == 1
    assert "POLICY NO 12345" in result.text


def test_multi_page_join_with_injected_backend():
    pdf = _image_pdf(["PAGE ONE", "PAGE TWO", "PAGE THREE"])
    engine = TesseractOcrEngine(ocr_backend=_fake_backend(["PAGE ONE", "PAGE TWO", "PAGE THREE"]))
    result = engine.ocr(pdf)
    assert result.page_count == 3
    assert result.text.splitlines() == ["PAGE ONE", "PAGE TWO", "PAGE THREE"]


def test_empty_ocr_yields_low_char_count_not_error():
    # An engine that recognises nothing returns an empty extraction (caller's confidence gate
    # routes to Failed) — it must NOT fabricate text or raise.
    pdf = _image_pdf(["anything"])
    engine = TesseractOcrEngine(ocr_backend=_fake_backend([""]))
    result = engine.ocr(pdf)
    assert result.source is TextSource.OCR
    assert result.char_count == 0


def test_malformed_pdf_fails_closed():
    engine = TesseractOcrEngine(ocr_backend=lambda _i: "x")
    with pytest.raises(PdfExtractionError):
        engine.ocr(b"%PDF-not-actually-a-pdf")


def test_backend_explosion_fails_closed_not_silent():
    def boom(_image) -> str:
        raise RuntimeError("tesseract subprocess died")

    pdf = _image_pdf(["text"])
    with pytest.raises(PdfExtractionError):
        TesseractOcrEngine(ocr_backend=boom).ocr(pdf)


def test_unavailable_engine_raises_on_ocr():
    # Force unavailability by faking the property via a subclass — ocr() must refuse to run.
    class _Down(TesseractOcrEngine):
        @property
        def available(self) -> bool:
            return False

    with pytest.raises(OcrUnavailableError):
        _Down(ocr_backend=lambda _i: "x").ocr(_image_pdf(["x"]))


# --- Real-binary integration: only runs where Tesseract is actually installed -------------

def test_real_tesseract_reads_english_text():
    engine = TesseractOcrEngine(lang="eng")  # real pytesseract + binary, English data
    if not engine.available:
        pytest.skip("Tesseract binary / pytesseract not installed")
    pdf = _image_pdf(["INVOICE 2026"])
    result = engine.ocr(pdf)
    assert result.source is TextSource.OCR
    # Tesseract is imperfect on synthetic default-font glyphs; assert it found the digits.
    assert "2026" in result.text.replace(" ", "")


def test_real_tesseract_arabic_data_loads():
    # The production default lang is 'ara+eng'; verify the Arabic traineddata is installed
    # and loads without error. Skips (not fails) where ara data isn't deployed yet.
    engine = TesseractOcrEngine(lang="ara+eng")
    if not engine.available:
        pytest.skip("Tesseract binary not installed")
    import pytesseract

    if "ara" not in pytesseract.get_languages(config=""):
        pytest.skip("Arabic traineddata (ara) not installed in tessdata")
    result = engine.ocr(_image_pdf(["INVOICE 2026"]))  # ara+eng must load, not error
    assert "2026" in result.text.replace(" ", "")
