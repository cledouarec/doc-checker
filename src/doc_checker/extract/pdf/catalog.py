# Copyright 2026 Christophe Le Douarec
"""Read, through PDFium, what the catalog of a PDF document declares."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pypdfium2.raw as pdfium_c

if TYPE_CHECKING:
    import pypdfium2 as pdfium


@dataclass(frozen=True, slots=True)
class PdfCatalog:
    """What the catalog of a PDF document declares about it.

    Attributes:
        lang: Declared language, such as ``en-US``, if any.
        tagged: Whether the document declares that it has a tag structure.
    """

    lang: str | None
    tagged: bool


def _declared_lang(pdf: pdfium.PdfDocument) -> str | None:
    """Return the language that the catalog declares, if not empty."""
    size = pdfium_c.FPDFCatalog_GetLanguage(pdf, None, 0)
    # The language is UTF-16, and the size counts its terminator, in bytes.
    buffer = (pdfium_c.FPDF_WCHAR * (size // 2))()
    pdfium_c.FPDFCatalog_GetLanguage(pdf, buffer, size)
    # A damaged file can hold half of a surrogate pair, which must not raise.
    text = bytes(buffer).decode("utf-16-le", errors="replace")
    return text.rstrip("\x00").strip() or None


def read_catalog(pdf: pdfium.PdfDocument) -> PdfCatalog:
    """Read the language and the tagged flag that a PDF declares.

    PDFium has already opened and decrypted the document, so the catalog is
    read through it, not by a second parse of the file.

    Args:
        pdf: The document, opened by `open_pdf`.

    Returns:
        The declared language, and whether the document is tagged.
    """
    return PdfCatalog(
        _declared_lang(pdf),
        tagged=bool(pdfium_c.FPDFCatalog_IsTagged(pdf)),
    )
