"""PDF → text extraction for quote intake (P1-10 §4.2).

The intake-quote pipeline turns raw insurer-quote PDF bytes into the text the
LLM extracts fields from. Two extraction strategies, chosen by what the PDF
actually contains:

  * **Native** — the PDF has a real text layer (typed/exported PDF). Extracted
    directly with :class:`PypdfExtractor` (pypdf — BSD-3, pure-Python, no system
    binary). This is the common, fast, no-extra-dependency path.
  * **OCR** — the PDF is a scanned image with no text layer (native extraction
    yields ~nothing). Per §4.2 the spec wants OCR here. OCR needs a heavy
    engine (Tesseract/PaddleOCR — a system binary), so it is an **injected
    seam**: an :class:`OcrEngine` is passed in, defaulting to
    :class:`NullOcrEngine` (OCR unavailable). When no OCR engine is wired, a
    no-text-layer PDF routes to a stable Failed reason rather than silently
    producing an empty extraction — never "garbage but high confidence".

Both extractors are Protocols so the quote service can be unit-tested with
stubs that need neither pypdf nor a Tesseract binary installed — the same
pattern the codebase uses for ``LLMCaller`` and the Redis nonce store.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol, runtime_checkable

#: Below this many non-whitespace characters, a "valid" PDF is treated as
#: having no usable text layer (a scanned image) → OCR or Failed. Conservative;
#: a one-line quote still clears it, an image-only scan does not.
MIN_NATIVE_TEXT_CHARS: Final[int] = 20


class TextSource(Enum):
    """How the text was obtained — recorded for observability + confidence."""

    NATIVE = "native"
    OCR = "ocr"


@dataclass(frozen=True)
class ExtractedText:
    """The text pulled out of a document plus provenance for confidence scoring."""

    text: str
    page_count: int
    source: TextSource

    @property
    def char_count(self) -> int:
        """Non-whitespace character count — the confidence length input."""
        return len("".join(self.text.split()))


class PdfExtractionError(Exception):
    """The PDF could not be parsed (corrupt / malformed) — routes to Failed."""


class OcrUnavailableError(Exception):
    """OCR was needed (no text layer) but no OCR engine is wired."""


@runtime_checkable
class PdfTextExtractor(Protocol):
    """Pulls the native text layer out of PDF bytes."""

    def extract(self, data: bytes) -> ExtractedText: ...


@runtime_checkable
class OcrEngine(Protocol):
    """Optical-character-recognises image-only PDF bytes into text."""

    @property
    def available(self) -> bool: ...

    def ocr(self, data: bytes) -> ExtractedText: ...


class PypdfExtractor:
    """Native text extraction via pypdf (the production default).

    pypdf is imported lazily so that (a) unit tests injecting a stub need not
    have it installed, and (b) a deployment without it fails loudly *at call
    time* with a clear message rather than at import time.
    """

    def extract(self, data: bytes) -> ExtractedText:
        try:
            import pypdf  # noqa: PLC0415 - lazy by design (optional prod dep)
        except ImportError as e:  # pragma: no cover - exercised only when pypdf absent
            raise PdfExtractionError(
                "pypdf is not installed; the native PDF extractor is unavailable"
            ) from e

        import io

        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = reader.pages
            parts = [(page.extract_text() or "") for page in pages]
        except Exception as e:  # pypdf raises many error types on malformed input
            raise PdfExtractionError(f"pypdf failed to parse the document: {e}") from e

        return ExtractedText(
            text="\n".join(parts).strip(),
            page_count=len(parts),
            source=TextSource.NATIVE,
        )


class NullOcrEngine:
    """The default OCR engine: no OCR available.

    Keeps OCR a clean optional seam — a no-text-layer PDF routes to a stable
    Failed reason instead of a silently-empty extraction. Wire a real Tesseract
    /PaddleOCR engine here to enable scanned-image support.
    """

    @property
    def available(self) -> bool:
        return False

    def ocr(self, data: bytes) -> ExtractedText:  # noqa: ARG002 - protocol shape
        raise OcrUnavailableError("no OCR engine is configured for scanned PDFs")


@dataclass
class StubPdfTextExtractor:
    """Test double — returns configured text, or raises to simulate corruption."""

    text: str = "Sample quote text"
    page_count: int = 1
    raise_error: bool = False

    def extract(self, data: bytes) -> ExtractedText:  # noqa: ARG002 - protocol shape
        if self.raise_error:
            raise PdfExtractionError("stub: simulated parse failure")
        return ExtractedText(
            text=self.text, page_count=self.page_count, source=TextSource.NATIVE
        )


@dataclass
class StubOcrEngine:
    """Test double — an available OCR engine returning configured text."""

    text: str = "OCR'd quote text"
    page_count: int = 1
    is_available: bool = True

    @property
    def available(self) -> bool:
        return self.is_available

    def ocr(self, data: bytes) -> ExtractedText:  # noqa: ARG002 - protocol shape
        if not self.is_available:
            raise OcrUnavailableError("stub: OCR marked unavailable")
        return ExtractedText(text=self.text, page_count=self.page_count, source=TextSource.OCR)


__all__ = (
    "MIN_NATIVE_TEXT_CHARS",
    "ExtractedText",
    "NullOcrEngine",
    "OcrEngine",
    "OcrUnavailableError",
    "PdfExtractionError",
    "PdfTextExtractor",
    "PypdfExtractor",
    "StubOcrEngine",
    "StubPdfTextExtractor",
    "TextSource",
)
