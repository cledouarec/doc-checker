# Copyright 2026 Christophe Le Douarec
"""Detect the format of an input and load it."""

from typing import TYPE_CHECKING, Final, Literal

from doc_checker.errors import InputError
from doc_checker.extract.html.loader import load_html
from doc_checker.extract.pdf.loader import load_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.config import Config
    from doc_checker.model.document import Document

_PDF_MAGIC: Final = b"%PDF-"
_PDF_HEAD: Final = 1024  # Bytes in which the PDF header is looked for.
_HTML_SUFFIXES: Final = frozenset({".html", ".htm"})


def detect_format(path: Path) -> Literal["html", "pdf"]:
    """Return the format of an input.

    The first of these rules that applies decides: a directory is HTML; a file
    that starts with the PDF header is a PDF; a file with an HTML suffix is
    HTML, even if its text quotes the header; a file that holds the header
    within its first 1,024 bytes is a PDF, as PDFium opens such files.

    Args:
        path: File or directory given by the user.

    Returns:
        ``"html"`` for a directory or an HTML file, ``"pdf"`` for a PDF.

    Raises:
        InputError: If the file cannot be read or has another format.
    """
    if path.is_dir():
        return "html"
    try:
        with path.open("rb") as stream:
            head = stream.read(_PDF_HEAD)
    except OSError as exc:
        msg = f"cannot read {path}: {exc.strerror}"
        raise InputError(msg) from None
    if head.startswith(_PDF_MAGIC):
        return "pdf"
    if path.suffix.lower() in _HTML_SUFFIXES:
        return "html"
    if _PDF_MAGIC in head:
        return "pdf"
    msg = f"unsupported input: {path}"
    raise InputError(msg)


def load_document(path: Path, cfg: Config) -> Document:
    """Load a documentation build.

    An input that cannot be read, or that is neither HTML nor PDF, raises an
    `InputError`.

    Args:
        path: File or directory given by the user.
        cfg: Run configuration.

    Returns:
        The normalized document.
    """
    if detect_format(path) == "pdf":
        return load_pdf(path, cfg.pdf, lang=cfg.lang, images=cfg.images)
    return load_html(path, cfg.html, lang=cfg.lang, images=cfg.images)
