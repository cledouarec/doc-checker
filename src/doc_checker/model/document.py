# Copyright 2026 Christophe Le Douarec
"""Normalized document model shared by every extractor and analyzer."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections import Counter
    from pathlib import Path

# Key of `Block.attrs` that tells how a PDF found a heading.
HEADING_SOURCE: Final = "heading"


class BlockKind(StrEnum):
    """Kind of a content block."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    IMAGE = "image"
    CAPTION = "caption"
    QUOTE = "quote"
    ADMONITION = "admonition"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    TOC = "toc"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Location of a block in its source document.

    Attributes:
        file: Path of the page, relative to the document root.
        page: 1-based page number, for paginated formats.
        line: Source line number, for markup formats.
        anchor: Identifier of the nearest anchor.
    """

    file: str
    page: int | None = None
    line: int | None = None
    anchor: str | None = None

    @property
    def fragment(self) -> str | None:
        """Anchor, else ``page=N`` for a paginated format, else ``None``."""
        if self.anchor:
            return self.anchor
        if self.page is not None:
            return f"page={self.page}"
        return None

    @property
    def href(self) -> str:
        """File followed by its fragment, if there is one."""
        fragment = self.fragment
        return f"{self.file}#{fragment}" if fragment else self.file


@dataclass(slots=True)
class Block:
    """A unit of content: heading, paragraph, list item, table, code...

    Attributes:
        kind: Kind of the block.
        text: Display text, after display normalization.
        raw_text: Text as extracted, before any normalization.
        source: Where the block comes from.
        level: Heading level, for headings.
        section_path: Titles of the enclosing headings, outermost first.
        attrs: Extra extractor attributes, such as ``role``. The heading of a
            PDF has `HEADING_SOURCE`, which is ``"outline"`` if a bookmark
            found it and ``"style"`` if its look did. The heading of an HTML
            page does not have it.
        key_strict: Comparison key used to diff texts.
        key_loose: Case, accent and punctuation insensitive key used to
            align blocks.
    """

    kind: BlockKind
    text: str
    raw_text: str
    source: SourceRef
    level: int | None = None
    section_path: tuple[str, ...] = ()
    attrs: dict[str, str] = field(default_factory=dict)
    key_strict: str = ""
    key_loose: str = ""


@dataclass(slots=True)
class Document:
    """A documentation build reduced to its blocks in reading order.

    Attributes:
        path: Input path, either a file or a site directory.
        format: Input format.
        blocks: Blocks in global reading order.
        parts: Pages in reading order, relative to the document root.
        lang: Language used for language dependent rules.
        declared_lang: Language declared by the document, if any.
        generator: Detected documentation generator profile.
        normalization: Number of characters changed by each normalization.
        chrome: Texts removed as repeated page chrome.
        chrome_keys: Strict keys of the blocks removed as page chrome.
        warnings: Non fatal extraction warnings.
        tagged: Whether a PDF declares a tag structure, from its
            ``/MarkInfo``; ``None`` for HTML.
    """

    path: Path
    format: Literal["html", "pdf"]
    blocks: list[Block]
    parts: list[str]
    lang: str | None
    declared_lang: str | None
    generator: str | None
    normalization: Counter[str]
    chrome: list[str]
    chrome_keys: frozenset[str]
    warnings: list[str]
    tagged: bool | None = None

    def resolve(self, source: SourceRef) -> Path:
        """Return the filesystem path of a source reference.

        Args:
            source: Reference to a page of this document.

        Returns:
            The input file itself, or the page inside the site directory.
        """
        if self.path.is_file():
            return self.path
        return self.path / source.file
