"""Tesseract-backed OCR engine for scanned (image-only) quote PDFs (P2-09 / P1-10 §4.2).

This is the production implementation of the :class:`~auxima_ai.intake.pdf_text.OcrEngine`
seam. When native extraction finds no text layer (a scanned image), the quote pipeline
routes here. The flow is:

  PDF bytes ──pypdfium2──▶ one raster image per page ──pytesseract──▶ text ──▶ ExtractedText

Both heavy dependencies are imported lazily so that (a) unit tests that inject a fake OCR
callable need neither installed, and (b) a deployment missing the Tesseract *binary* reports
``available == False`` (→ a clean Failed route) instead of crashing at import.

Why Tesseract, not PaddleOCR: PaddleOCR 2.7.x pins numpy-1.x-era opencv/imgaug, which is
incompatible with the sidecar's numpy 2.x (required by scipy/pandas/fhir.resources). Tesseract
is a standalone system binary with no numpy coupling and first-class Arabic support — the
Arabic-first requirement (CLAUDE.md §5). No frappe/auxima import (sidecar Rule 2 — isolation).
"""
from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

from auxima_ai.intake.pdf_text import (
    ExtractedText,
    OcrUnavailableError,
    PdfExtractionError,
    TextSource,
)

#: Standard Windows install locations for the UB-Mannheim Tesseract build. The installer
#: does not always prepend these to PATH, so we probe them as a fallback — otherwise a
#: correctly-installed binary would read as "unavailable" purely for a PATH quirk.
_WINDOWS_DEFAULT_CMDS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _resolve_tesseract_cmd(explicit: Optional[str]) -> Optional[str]:
    """The Tesseract binary path: an explicit override, else PATH, else a known install dir.
    Returns None if no binary can be located (→ engine reports unavailable, fail-closed)."""
    if explicit:
        return explicit
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_DEFAULT_CMDS:
        if os.path.isfile(candidate):
            return candidate
    return None

#: Default recognition languages. Arabic first (CLAUDE.md §5 Arabic-first), then English —
#: Tesseract accepts multiple scripts joined with '+', trying each per region.
DEFAULT_OCR_LANG: str = "ara+eng"

#: Rasterisation scale passed to pypdfium2. 2.0 ≈ 144 DPI — enough for clean OCR of typed
#: scans without ballooning memory. Pure-image scans below this lose small glyphs.
_RENDER_SCALE: float = 2.0

#: An OCR backend: takes a single page image, returns its recognised text.
OcrBackend = Callable[[object], str]


class TesseractOcrEngine:
    """OCR via the Tesseract binary (through ``pytesseract``), rasterising with ``pypdfium2``.

    The OCR backend is injectable (``ocr_backend``) purely so the rasterise-and-join logic
    can be unit-tested without the Tesseract binary installed. In production it defaults to
    ``pytesseract.image_to_string`` with :data:`DEFAULT_OCR_LANG`.
    """

    def __init__(
        self,
        lang: str = DEFAULT_OCR_LANG,
        ocr_backend: Optional[OcrBackend] = None,
        render_scale: float = _RENDER_SCALE,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        self._lang = lang
        self._render_scale = render_scale
        self._ocr_backend = ocr_backend
        self._tesseract_cmd = tesseract_cmd

    def _ensure_cmd(self):
        """Import pytesseract and point it at a resolvable binary (PATH or known install
        dir). Raises if the wrapper isn't installed; sets nothing if no binary is found
        (the subsequent ``get_tesseract_version`` then fails → unavailable, fail-closed)."""
        import pytesseract  # noqa: PLC0415 - lazy by design (heavy optional dep)

        cmd = _resolve_tesseract_cmd(self._tesseract_cmd)
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        return pytesseract

    @property
    def available(self) -> bool:
        """True only if both the rasteriser (pypdfium2) and a working Tesseract are present.

        An injected backend is trusted as available (tests). The default path resolves the
        binary (PATH or standard install dir) and probes ``get_tesseract_version`` — a missing
        binary reports False (→ Failed route), never an exception.
        """
        try:
            import pypdfium2  # noqa: F401, PLC0415 - lazy by design (heavy optional dep)
        except ImportError:
            return False

        if self._ocr_backend is not None:
            return True

        try:
            self._ensure_cmd().get_tesseract_version()
        except Exception:
            # ImportError (no wrapper) or TesseractNotFoundError (no binary) → unavailable.
            return False
        return True

    def _backend(self) -> OcrBackend:
        if self._ocr_backend is not None:
            return self._ocr_backend
        pytesseract = self._ensure_cmd()
        lang = self._lang
        return lambda image: pytesseract.image_to_string(image, lang=lang)

    def ocr(self, data: bytes) -> ExtractedText:
        """Rasterise every page and OCR it. Fail-closed: a corrupt/unreadable PDF raises
        :class:`PdfExtractionError`; a missing engine raises :class:`OcrUnavailableError`.
        Empty recognised text is returned as-is (low ``char_count``) so the caller's confidence
        gate routes it to Failed — never an empty extraction dressed up as success."""
        if not self.available:
            raise OcrUnavailableError(
                "Tesseract OCR engine is unavailable (binary or pypdfium2 not installed)"
            )

        try:
            import pypdfium2  # noqa: PLC0415 - lazy by design
        except ImportError as e:  # pragma: no cover - guarded by available
            raise OcrUnavailableError("pypdfium2 is not installed") from e

        backend = self._backend()
        try:
            pdf = pypdfium2.PdfDocument(data)
        except Exception as e:  # pypdfium2 raises on malformed input
            raise PdfExtractionError(f"pypdfium2 failed to open the document: {e}") from e

        parts: list[str] = []
        try:
            page_count = len(pdf)
            for index in range(page_count):
                page = pdf[index]
                bitmap = page.render(scale=self._render_scale)
                image = bitmap.to_pil()
                try:
                    parts.append((backend(image) or "").strip())
                finally:
                    image.close()
                    bitmap.close()
                    page.close()
        except PdfExtractionError:
            raise
        except Exception as e:
            raise PdfExtractionError(f"OCR failed while rendering/recognising: {e}") from e
        finally:
            pdf.close()

        return ExtractedText(
            text="\n".join(p for p in parts if p).strip(),
            page_count=page_count,
            source=TextSource.OCR,
        )


__all__ = ("DEFAULT_OCR_LANG", "OcrBackend", "TesseractOcrEngine")
