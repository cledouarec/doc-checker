# Copyright 2026 Christophe Le Douarec
"""Normalized document model shared by every extractor and analyzer."""

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from doc_checker.model.facts import (
        HtmlPageFacts,
        OutlineEntry,
        PdfPageFacts,
        TagTree,
    )

# Key of `Block.attrs` that tells how a PDF found a heading, and its values:
# a bookmark of the outline, or the look of the block.
HEADING_SOURCE: Final = "heading"
type HeadingSource = Literal["outline", "style"]
FROM_OUTLINE: Final[HeadingSource] = "outline"
FROM_STYLE: Final[HeadingSource] = "style"
# Key of `Block.attrs` that keeps the number set apart from a heading or a
# caption.
NUMBER_ATTR: Final = "number"
# What separates the cells of a row in the text of a table.
CELL_SEPARATOR: Final = " | "


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


@dataclass(frozen=True, slots=True)
class Link:
    """A link of a block, resolved to a page of the site or to a URL.

    Attributes:
        text: Text of the link, made printable: its blanks collapsed.
        href: Target as written in the page, made printable.
        url: Normalized URL of an external link, as `normalize_url` of
            `doc_checker.extract.html.links` gives it; ``None`` for an
            internal link.
        page: Page of an internal link, a POSIX path relative to the site
            root, after the redirections; ``None`` for an external link and
            for a link out of the site.
        fragment: Anchor of an internal link, if any, decoded and made
            printable as the identifiers of the blocks are.
        found: Whether the page of an internal link exists: a page that was
            read, or a file under the site root.
    """

    text: str
    href: str
    url: str | None = None
    page: str | None = None
    fragment: str | None = None
    found: bool = False

    @property
    def target(self) -> str:
        """Where the link leads, as a report shows it.

        ``href`` for an external link or a link out of the site, else the
        page followed by its fragment, if there is one.
        """
        if self.url is not None or self.page is None:
            return self.href
        return f"{self.page}#{self.fragment}" if self.fragment else self.page


@dataclass(frozen=True, slots=True)
class PixelSource:
    """Where the pixels of an image are read again, on demand.

    Attributes:
        file: Image file of an HTML site, already checked to be under it.
        data: Bytes of an image written in a data URL.
        page: Page of a PDF image or figure, from 0.
        index: Position of a PDF image among the image objects of its page,
            in the order of ``get_objects(filter=[IMAGE], max_depth=2)``.
        box: Left, bottom, right and top of a PDF vector figure, in the
            coordinates of PDFium for its page, to render it again.
    """

    file: Path | None = None
    data: bytes | None = field(default=None, repr=False)
    page: int | None = None
    index: int | None = None
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class ImageRef:
    """An image of a document, or a vector figure of a PDF.

    Attributes:
        name: Site relative path of an HTML image file, normalized URL of
            an external image, or "" (PDF image, data URL, inline SVG).
        format: Lower case encoding: png, jpeg, gif, webp, bmp, tiff, svg,
            jpx, jbig2, ccitt, flate, raw, vector (a PDF figure), or "".
        digest: Content digest, 32 hexadecimal characters.
        width: Width in pixels, if the pixels were read.
        height: Height in pixels, if the pixels were read.
        shown: Displayed width and height: CSS pixels of the HTML
            attributes, or points of the box of a PDF image.
        dpi: Effective resolution of a PDF raster image.
        alt: Alternative text; None when absent, "" when decorative.
        vector: Whether the image is a drawing (SVG, PDF figure).
        phash: Perceptual hash, 64 bits, if the pixels were read.
        dhash: Difference hash, 64 bits, if the pixels were read.
        quality: Estimated JPEG quality, for a JPEG.
        alpha: Whether the image uses transparency; None when unknown.
        size: Size of the encoded image, in bytes.
        texts: Texts of an SVG, in document order.
        source: Where the pixels are read again; None without pixels.
        problem: Why the pixels cannot be read, such as "too large".
    """

    name: str
    format: str
    digest: str
    width: int | None = None
    height: int | None = None
    shown: tuple[float, float] | None = None
    dpi: float | None = None
    alt: str | None = None
    vector: bool = False
    phash: int | None = None
    dhash: int | None = None
    quality: int | None = None
    alpha: bool | None = None
    size: int | None = None
    texts: tuple[str, ...] = ()
    source: PixelSource | None = None
    problem: str | None = None

    @property
    def readable(self) -> bool:
        """Whether the pixels were read: the image has its hashes."""
        return self.phash is not None


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
            PDF has `HEADING_SOURCE`, which is `FROM_OUTLINE` if a bookmark
            found it and `FROM_STYLE` if its look did. The heading of an HTML
            page does not have it. A heading of a document that numbers its
            headings has `NUMBER_ATTR`, its number set apart from its text,
            and so does a caption whose label and number were set apart.
        key_strict: Comparison key used to diff texts.
        key_loose: Case, accent and punctuation insensitive key used to
            align blocks.
        links: Links of the block, in document order.
        anchors: Identifiers that designate the block, in document order.
        image: The image of an `IMAGE` block.
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
    links: tuple[Link, ...] = ()
    anchors: tuple[str, ...] = ()
    image: ImageRef | None = None


def carry_anchors(
    blocks: Sequence[Block],
    keep: Callable[[Block], bool],
) -> list[Block]:
    """Return the blocks to keep, with the anchors of the others.

    A link to an anchor of a block that is left out, such as page chrome,
    has to land somewhere: the anchors of each block left out go, in order
    and without duplicates, in front of the anchors of the next kept block
    of the same page. They are lost if the page keeps no block after it.

    Args:
        blocks: Blocks in reading order, left unchanged.
        keep: Tells whether a block is kept.

    Returns:
        The kept blocks, in order. A block that receives anchors is a copy.
    """
    kept: list[Block] = []
    waiting: dict[str, list[str]] = {}
    for block in blocks:
        file = block.source.file
        if not keep(block):
            waiting.setdefault(file, []).extend(block.anchors)
            continue
        carried = waiting.pop(file, None)
        if carried:
            anchors = tuple(dict.fromkeys((*carried, *block.anchors)))
            kept.append(replace(block, anchors=anchors))
        else:
            kept.append(block)
    return kept


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
        chrome_images: Digests of the images removed as page chrome.
        title: Title of the metadata of a PDF.
        outline: Bookmarks of a PDF, in order.
        html_pages: Facts of each HTML page, in reading order.
        pdf_pages: Facts of each PDF page, in order.
        tags: Structure tree of a PDF. The outline, the pages and the tree
            are read only for the audit.
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
    chrome_images: frozenset[str] = frozenset()
    title: str | None = None
    outline: tuple[OutlineEntry, ...] = ()
    html_pages: tuple[HtmlPageFacts, ...] = ()
    pdf_pages: tuple[PdfPageFacts, ...] = ()
    tags: TagTree | None = None

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
