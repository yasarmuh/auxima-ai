"""Document-class routing for intake extraction (P1-10 §4.2).

Before any OCR/LLM work, a document must be classified so a corrupt,
encrypted, oversized, or non-PDF input routes to a **Failed** outcome rather
than being silently dropped or returning a misleading 200. This module is the
pure, dependency-free decision layer.

Heuristics (stdlib only — no PDF parsing library, by design):
  - **Oversized** is checked FIRST (before any content inspection) so a
    pathological large input can't force expensive work.
  - **PDF magic** is the literal ``%PDF-`` file header.
  - **Encrypted** is detected by the presence of an ``/Encrypt`` reference —
    a standard PDF encryption-dictionary marker. This is a heuristic: it can
    in principle false-positive if the literal bytes appear in a content
    stream, but for the routing purpose (don't try to extract an encrypted
    PDF) erring toward "treat as encrypted → Failed" is the safe direction.
  - **Corrupt** = empty, or claims to be a PDF (has the magic) but is missing
    the ``%%EOF`` end marker (truncated/incomplete upload).
  - **Unsupported** = non-empty, within the size cap, but not a PDF.

NOT in scope: real PDF parsing, OCR, password-protected-PDF unlock, or
wiring this into the live request path (the intake request currently carries
already-extracted ``lead_text``, not raw bytes — the bytes-ingestion slice is
larger and partly cross-repo).
"""
from __future__ import annotations

from enum import Enum
from typing import Final

#: Default maximum document size. Mirrors the S-46 Documents 25 MB cap.
DEFAULT_MAX_BYTES: Final[int] = 25 * 1024 * 1024

_PDF_MAGIC: Final[bytes] = b"%PDF-"
_PDF_EOF: Final[bytes] = b"%%EOF"
_ENCRYPT_MARKER: Final[bytes] = b"/Encrypt"


class DocumentClass(Enum):
    """The routable class of an ingested document."""

    PDF_VALID = "pdf_valid"
    PDF_ENCRYPTED = "pdf_encrypted"
    CORRUPT = "corrupt"
    OVERSIZED = "oversized"
    UNSUPPORTED_TYPE = "unsupported_type"


#: The only class that proceeds to extraction. Everything else → Failed.
_EXTRACTABLE: Final[frozenset[DocumentClass]] = frozenset({DocumentClass.PDF_VALID})

#: Stable, machine-readable Failed reasons (for the activity row / observability
#: event the caller emits). Mirrors the S-54 §3.5-style reason convention.
_FAILURE_REASONS: Final[dict[DocumentClass, str]] = {
    DocumentClass.PDF_ENCRYPTED: "encrypted_pdf",
    DocumentClass.CORRUPT: "corrupt_document",
    DocumentClass.OVERSIZED: "oversized_document",
    DocumentClass.UNSUPPORTED_TYPE: "unsupported_document_type",
}


def classify_document(data: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> DocumentClass:
    """Classify raw document bytes into a routable :class:`DocumentClass`.

    The order is deliberate: size is checked before any content inspection,
    then empties, then PDF-specific checks.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"data must be bytes; got {type(data).__name__}")
    if len(data) > max_bytes:
        return DocumentClass.OVERSIZED
    if len(data) == 0:
        return DocumentClass.CORRUPT
    if not bytes(data).startswith(_PDF_MAGIC):
        return DocumentClass.UNSUPPORTED_TYPE
    if _ENCRYPT_MARKER in data:
        return DocumentClass.PDF_ENCRYPTED
    if _PDF_EOF not in data:
        return DocumentClass.CORRUPT
    return DocumentClass.PDF_VALID


def is_extractable(doc_class: DocumentClass) -> bool:
    """``True`` iff the document should proceed to extraction."""
    return doc_class in _EXTRACTABLE


def failure_reason(doc_class: DocumentClass) -> str:
    """Stable Failed reason for a non-extractable class.

    Raises :class:`ValueError` for :data:`DocumentClass.PDF_VALID` — a valid
    document has no failure reason, and asking for one is a caller bug.
    """
    try:
        return _FAILURE_REASONS[doc_class]
    except KeyError as e:
        raise ValueError(f"{doc_class} is extractable; it has no failure reason") from e


__all__ = (
    "DEFAULT_MAX_BYTES",
    "DocumentClass",
    "classify_document",
    "failure_reason",
    "is_extractable",
)
