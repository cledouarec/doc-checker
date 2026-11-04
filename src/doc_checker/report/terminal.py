# Copyright 2026 Christophe Le Douarec
"""Text rendered in the terminal."""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from doc_checker.model.document import Document

_MAX_TEXT: Final = 200
_ELLIPSIS: Final = "\U00002026"
_NEWLINE_MARK: Final = " \U000023ce "


def _shorten(text: str, limit: int) -> str:
    """Return the text on one line, cut to ``limit`` characters."""
    text = text.replace("\n", _NEWLINE_MARK)
    return text if len(text) <= limit else text[:limit] + _ELLIPSIS


def render_blocks(doc: Document) -> str:
    """Return a listing of the normalized blocks of a document.

    Args:
        doc: Loaded document.

    Returns:
        A header line followed by one line per block.
    """
    lines = [
        (
            f"{doc.path}: {doc.format}, {len(doc.parts)} page(s), "
            f"profile {doc.generator}, lang {doc.lang}"
        ),
    ]
    lines.extend(
        f"{index:>5}  {block.kind.value:<10}  {block.source.href}  "
        f"{_shorten(block.text, _MAX_TEXT)}"
        for index, block in enumerate(doc.blocks)
    )
    return "\n".join(lines)
