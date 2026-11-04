# Copyright 2026 Christophe Le Douarec
"""Detect the format of an input and load it."""

from typing import TYPE_CHECKING, Final, Literal

from doc_checker.errors import InputError
from doc_checker.extract.html.loader import load_html

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.config import Config
    from doc_checker.model.document import Document

_PDF_MAGIC: Final = b"%PDF-"
_HTML_SUFFIXES: Final = frozenset({".html", ".htm"})


def detect_format(path: Path) -> Literal["html", "pdf"]:
    """Return the format of an input.

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
            head = stream.read(len(_PDF_MAGIC))
    except OSError as exc:
        msg = f"cannot read {path}: {exc.strerror}"
        raise InputError(msg) from None
    if head == _PDF_MAGIC:
        return "pdf"
    if path.suffix.lower() in _HTML_SUFFIXES:
        return "html"
    msg = f"unsupported input: {path}"
    raise InputError(msg)


def load_document(path: Path, cfg: Config) -> Document:
    """Load a documentation build.

    Args:
        path: File or directory given by the user.
        cfg: Run configuration.

    Returns:
        The normalized document.

    Raises:
        InputError: If the input cannot be read or is not supported yet.
    """
    if detect_format(path) == "pdf":
        msg = f"PDF input is not supported yet: {path}"
        raise InputError(msg)
    return load_html(path, cfg.html, lang=cfg.lang)
