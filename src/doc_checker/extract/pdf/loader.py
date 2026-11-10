# Copyright 2026 Christophe Le Douarec
"""Load a PDF file into a document.

The pages are read one at a time, and only the line segments and the images of
each page are kept. The running headers, footers and page numbers are removed,
the segments of each page are ordered into lines, and the lines are grouped
into blocks across page and column breaks. Then the text of each block is
joined, and the block gets its kind: contents or index, code, heading, list
item, caption or paragraph. The images repeated on many pages are removed, and
each other image becomes a block, before the first block that starts under its
top.

With the pdf-tables extra, asked for by the settings, the ruled tables that
pdfplumber finds become blocks too, placed as images are. The segments of a
table leave the flow before the running headers are looked for, so that no
row of a table is taken for one. A table continued on the next page is joined
to its start, and the header row that it repeats is removed.
"""

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Final

from doc_checker.errors import InputError
from doc_checker.extract.common import (
    IMAGE_CHROME,
    TINY_IMAGE,
    assign_section_paths,
    image_warnings,
    is_tiny_image,
    normalized_block,
    repeated_images,
    strip_caption_numbers,
    strip_heading_numbers,
)
from doc_checker.extract.pdf import reader, tables
from doc_checker.extract.pdf.blocks import (
    group_lines,
    merge_flows,
    strip_list_marker,
)
from doc_checker.extract.pdf.catalog import read_catalog
from doc_checker.extract.pdf.chrome import find_chrome, title_pages
from doc_checker.extract.pdf.headings import heading_levels
from doc_checker.extract.pdf.joining import code_text, prose_text, vocabulary
from doc_checker.extract.pdf.kinds import figure_list, is_caption, toc_pages
from doc_checker.extract.pdf.layout import page_lines
from doc_checker.extract.pdf.reader import (
    MAX_PAGE_FIGURES,
    METADATA_LIMIT,
    open_pdf,
    read_bookmarks,
    read_labels,
    read_metadata,
    read_page,
)
from doc_checker.extract.pdf.segments import body_size, build_segments
from doc_checker.extract.pdf.table_blocks import (
    KeptTables,
    Placed,
    flow,
    place_order,
    read_kept_tables,
    table_breaks,
    text_segments,
    with_tables,
)
from doc_checker.model.document import (
    HEADING_SOURCE,
    Block,
    BlockKind,
    Document,
    SourceRef,
)
from doc_checker.model.facts import (
    NO_STRUCTURE_TREE,
    OutlineEntry,
    PdfPageFacts,
)
from doc_checker.text.lang import (
    LANG_SAMPLE_BLOCKS,
    declared_language,
    resolve_language,
)
from doc_checker.text.normalize import ELLIPSIS, printable

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    import pypdfium2 as pdfium

    from doc_checker.config import PdfConfig
    from doc_checker.extract.pdf.blocks import Apart, DraftBlock
    from doc_checker.extract.pdf.catalog import PdfCatalog
    from doc_checker.extract.pdf.layout import Line
    from doc_checker.extract.pdf.reader import (
        Bookmark,
        LinkTargets,
        OutlineCut,
        PdfImage,
        PdfPageText,
    )
    from doc_checker.extract.pdf.segments import Segment
    from doc_checker.extract.pdf.tables import Box
    from doc_checker.model.document import HeadingSource
    from doc_checker.model.facts import PdfLink, TagTree

SCANNED_MIN_COVERAGE: Final = 0.5  # Smallest image share of a scanned page.
SCANNED_LABELS_MAX: Final = 10  # Most page labels that a warning lists.
MAX_DOCUMENT_LINKS: Final = 1_000_000  # Most links read in a document.

_DIGITS: Final = re.compile(r"\d+")
_MORE_LABELS: Final = f", {ELLIPSIS}"  # Ends a list of labels that goes on.


@dataclass(frozen=True, slots=True)
class _Page:
    """A page, once its characters are grouped into segments.

    Attributes:
        label: Page label, such as ``iv``, or the page number from 1.
        width: Width of the crop box of the page turned upright, in points.
        height: Height of the crop box of the page turned upright, in points.
        image_coverage: Share of the page covered by images, from 0 to 1.
        segments: The segments of the page, in the order of their first
            character.
        angled: Number of characters left out because they do not follow the
            direction of the page.
        images: The raster images and the vector figures of the page,
            unless they were not read.
        unrendered: Number of vector figures not rendered because they come
            after the first `MAX_PAGE_FIGURES` of the page.
        chars: Number of characters that the page keeps.
        unmapped: Number of them without Unicode mapping.
        links: The link annotations of the page, if they were read.
        more_links: Whether the page has more link annotations than
            `MAX_PAGE_LINKS` of the reader, those after them not read.
        untexted_links: Number of its links whose text was not read, for
            `LINK_TEXT_WORK` of the reader.
        code_words: Its words set in a monospaced font, as
            `monospaced_words` of the reader finds them, if they were read.
        figures: The boxes of its vector figures, whether the images were
            read or not, from the lower left corner of its crop box.
        turned: Whether the reader turned the page upright.
        crowded: Whether the page has too many paths for its drawings to be
            looked for, as `PdfPageText.crowded` of the reader tells.
    """

    label: str
    width: float
    height: float
    image_coverage: float
    segments: tuple[Segment, ...]
    angled: int
    images: tuple[PdfImage, ...]
    unrendered: int = 0
    chars: int = 0
    unmapped: int = 0
    links: tuple[PdfLink, ...] = ()
    more_links: bool = False
    untexted_links: int = 0
    code_words: frozenset[str] = frozenset()
    figures: tuple[Box, ...] = ()
    turned: bool = False
    crowded: bool = False


@dataclass(frozen=True, slots=True)
class _PdfFile:
    """What is read from a PDF file.

    Attributes:
        catalog: The language and the tagging that the catalog declares.
        bookmarks: The outline, in the order of the document.
        title: The title of the metadata, if any.
        generator: The producer of the metadata, or else its creator, if any.
        pages: Every page of the document, in order.
        links_cut: Whether links were left unread for `MAX_DOCUMENT_LINKS`.
        outline_cut: Why bookmarks were left unread, if they were: for
            `MAX_BOOKMARKS` or `OUTLINE_TITLE_WORK` of the reader.
        labels_cut: Whether pages were numbered rather than labelled for
            `PAGE_LABEL_WORK` of the reader.
    """

    catalog: PdfCatalog
    bookmarks: tuple[Bookmark, ...]
    title: str | None
    generator: str | None
    pages: tuple[_Page, ...]
    links_cut: bool = False
    outline_cut: OutlineCut = None
    labels_cut: bool = False


@dataclass(slots=True)
class _LinkBudget:
    """How the links of a document are read, page after page, for the audit.

    Attributes:
        room: How many more links are read before `MAX_DOCUMENT_LINKS`.
        cut: Whether a link was left unread for that limit, after which no
            link is read.
        targets: The targets of the link annotations already resolved, as
            `read_links` of the reader takes them, while the document is
            open.
    """

    room: int
    cut: bool = False
    targets: LinkTargets = field(default_factory=dict)

    @property
    def limit(self) -> int | None:
        """The most links to read on the next page; None to read none."""
        return None if self.cut else self.room

    def spend(self, page: PdfPageText) -> bool:
        """Count the links read on a page.

        The page was read with the lower of the room and `MAX_PAGE_LINKS`
        of the reader as its limit: the links that it left unread were left
        for the lower of the two.

        Args:
            page: The page, read with `limit`.

        Returns:
            Whether links of the page were left unread for `MAX_PAGE_LINKS`,
            rather than for the limit of the document.
        """
        for_document = page.more_links and self.room < reader.MAX_PAGE_LINKS
        self.cut = self.cut or for_document
        self.room -= len(page.links)
        return page.more_links and not for_document


def _count(counter: Counter[str], key: str, amount: int) -> None:
    """Add an amount to a count of normalizations, unless it is zero."""
    if amount:
        counter[key] += amount


def _read_page(  # ruff: ignore[too-many-arguments] - one keyword per part of the page that is read
    pdf: pdfium.PdfDocument,
    index: int,
    path: Path,
    *,
    images: bool,
    links: _LinkBudget | None,
    label: str,
) -> PdfPageText:
    """Read the characters, the images and the links of a page.

    Args:
        pdf: The document, opened by `open_pdf`.
        index: Position of the page, from 0.
        path: The PDF file, which the error names, and from whose folder the
            files of the links are checked.
        images: Whether the images are read.
        links: How the links of the document are read, or ``None`` not to
            read them.
        label: The label of the page, as `read_labels` reads it.

    Returns:
        The characters, the images and the links of the page, as
        `read_page` returns them.

    Raises:
        InputError: If the page cannot be read.
    """
    limit = None if links is None else links.limit
    try:
        return read_page(
            pdf,
            index,
            images=images,
            link_folder=None if limit is None else path.parent,
            link_limit=limit,
            link_targets=None if links is None else links.targets,
            label=label,
        )
    except InputError as exc:
        msg = f"cannot read {path}: {exc}"
        raise InputError(msg) from None


def _read(
    path: Path,
    counter: Counter[str],
    *,
    images: bool,
    audit: bool,
) -> _PdfFile:
    """Read the catalog, the outline, the metadata and the pages of a PDF.

    The pages are read one at a time, and each one keeps its segments, its
    images and its links only, not its characters. The links and the words
    set in a monospaced font are read for the audit only, the links
    `MAX_DOCUMENT_LINKS` at most in the whole document.

    Args:
        path: The PDF file.
        counter: The counts of the normalizations, in which `artifact` goes
            up by the characters dropped as artifacts, `unicode_map` by the
            characters without Unicode mapping, `angled_text` by the
            characters left out for their angle, and `figure_text` by the
            characters left out as drawn in a vector figure.
        images: Whether the images are read.
        audit: Whether the links and the monospaced words are read.

    Returns:
        What is read from the file.
    """
    with open_pdf(path) as pdf:
        budget = _LinkBudget(MAX_DOCUMENT_LINKS) if audit else None
        catalog = read_catalog(pdf)
        bookmarks, outline_cut = read_bookmarks(pdf)
        title, generator = read_metadata(pdf)
        labels, labels_cut = read_labels(pdf)
        pages = []
        for index, label in enumerate(labels):
            page = _read_page(
                pdf,
                index,
                path,
                images=images,
                links=budget,
                label=label,
            )
            _count(counter, "artifact", page.artifacts)
            _count(counter, "unicode_map", page.unmapped)
            _count(counter, "angled_text", page.angled)
            _count(counter, "figure_text", page.figure_chars)
            more_links = budget is not None and budget.spend(page)
            segments = tuple(build_segments(page))
            pages.append(
                _Page(
                    page.label,
                    page.width,
                    page.height,
                    page.image_coverage,
                    segments,
                    page.angled,
                    page.images,
                    page.unrendered,
                    chars=len(page.chars),
                    unmapped=page.unmapped,
                    links=page.links,
                    more_links=more_links,
                    untexted_links=page.untexted_links,
                    code_words=(
                        reader.monospaced_words(page.chars)
                        if audit
                        else frozenset()
                    ),
                    figures=page.figure_boxes,
                    turned=page.turn != 0,
                    crowded=page.crowded,
                ),
            )
    return _PdfFile(
        catalog,
        tuple(bookmarks),
        title,
        generator,
        tuple(pages),
        links_cut=budget is not None and budget.cut,
        outline_cut=outline_cut,
        labels_cut=labels_cut,
    )


def _strip_chrome(
    pdf: _PdfFile,
    flow: Sequence[Sequence[Segment]],
    *,
    body: float,
    cfg: PdfConfig,
    counter: Counter[str],
) -> tuple[list[list[Segment]], list[str]]:
    """Remove the running headers, footers and page numbers of every page.

    Args:
        pdf: What is read from the file.
        flow: The segments of each page that are looked at: all of them but
            those of the tables.
        body: The body size of the document, in points.
        cfg: The PDF settings, which give the bands and the share of pages.
        counter: The counts of the normalizations, in which `header_footer`
            goes up by one for each segment removed.

    Returns:
        The segments that each page keeps, one list per page, and the texts
        of the removed segments, in page order and without duplicates, with
        each run of digits written ``#``.
    """
    # The chrome steps take the position of a list as the page index.
    pages = flow
    heights = [page.height for page in pdf.pages]
    titles = title_pages(
        pages,
        heights,
        pdf.bookmarks,
        pdf.title,
        body=body,
        cfg=cfg,
    )
    chrome = find_chrome(
        pages,
        heights,
        titles,
        labels=[page.label for page in pdf.pages],
        body=body,
        cfg=cfg,
    )
    kept: list[list[Segment]] = []
    removed: list[str] = []
    for segments in pages:
        kept.append([s for s in segments if s not in chrome])
        removed.extend(s.text for s in segments if s in chrome)
    _count(counter, "header_footer", len(removed))
    texts = (_DIGITS.sub("#", text) for text in removed)
    return kept, list(dict.fromkeys(texts))


def _drafts(
    kept: Sequence[Sequence[Segment]],
    *,
    body: float,
    counter: Counter[str],
    apart: Apart | None = None,
    hidden: Sequence[Collection[Segment]] = (),
) -> tuple[dict[int, BlockKind], list[DraftBlock]]:
    """Order the lines of each page and group them into draft blocks.

    Args:
        kept: The segments of each page, chrome removed, one item for each
            page of the document. The count of the pages bounds the numbers
            of the entries of a page that has no title of contents.
        body: The body size of the document, in points.
        counter: The counts of the normalizations, in which `page_break`
            goes up by one for each block joined across a break.
        apart: Whether a table lies between two lines, given the line before
            and the line after, which are then never in one block; None if
            the document has no table.
        hidden: The segments of each page, among `kept`, that the cuts of
            the page take, as `page_lines` tells, but no line: those of its
            tables; none if omitted.

    Returns:
        The kind of each contents or index page, by position of the page, and
        the draft blocks of the document, in reading order.
    """
    lines = [
        line
        for index, segments in enumerate(kept)
        for line in page_lines(
            segments,
            body=body,
            hidden=hidden[index] if hidden else (),
        )
    ]
    kinds = toc_pages(lines, page_count=len(kept))
    return kinds, merge_flows(
        group_lines(lines, apart=apart),
        counter,
        page_kinds=kinds,
        apart=apart,
    )


def _texts(drafts: Sequence[DraftBlock], counter: Counter[str]) -> list[str]:
    """Join the lines of each draft block into its text.

    Args:
        drafts: The draft blocks of the document, in reading order.
        counter: The counts of the normalizations.

    Returns:
        The text of each block: its lines joined as code for a code sample,
        and as prose otherwise.
    """
    known = vocabulary(drafts)
    return [
        code_text(draft.lines, counter)
        if draft.code
        else prose_text(draft.lines, known, counter)
        for draft in drafts
    ]


def _levels(
    drafts: Sequence[DraftBlock],
    texts: Sequence[str],
    bookmarks: Sequence[Bookmark],
    *,
    body: float,
) -> dict[int, tuple[int, HeadingSource]]:
    """Find the headings of a document, their levels and their sources.

    The outline makes the headings that its bookmarks match, and the styles of
    the blocks complete it. Without bookmarks, or with an outline whose
    titles are found in no block, the styles tell all the headings.

    Args:
        drafts: The draft blocks of the document, in reading order.
        texts: The text of each block, in the same order.
        bookmarks: The outline of the document.
        body: The body size of the document, in points.

    Returns:
        The level and the source of each heading, by position of its block.
    """
    return heading_levels(drafts, texts, bookmarks, body=body)


def _kind(
    draft: DraftBlock,
    text: str,
    *,
    page_kind: BlockKind | None,
    heading: bool,
    counter: Counter[str],
) -> tuple[BlockKind, str]:
    """Tell the kind of a draft block, the first rule that applies winning.

    Args:
        draft: The draft block.
        text: Its text.
        page_kind: The kind of the page of its first line, if that page is a
            contents or an index page.
        heading: Whether the block is a heading.
        counter: The counts of the normalizations, in which `toc` goes up by
            one for a block of a contents or an index page, and `bullet` or
            `list_number` for a list item.

    Returns:
        The kind and the text of the block, which is the kind of its page, or
        else `BlockKind.CODE`, `BlockKind.HEADING`, `BlockKind.LIST_ITEM`
        with the text without its marker, `BlockKind.CAPTION` or
        `BlockKind.PARAGRAPH`.
    """
    if page_kind is not None:
        counter["toc"] += 1
        return page_kind, text
    if draft.code:
        return BlockKind.CODE, text
    if heading:
        return BlockKind.HEADING, text
    item, marker = strip_list_marker(text)
    if marker is not None:
        counter[marker] += 1
        return BlockKind.LIST_ITEM, item
    if is_caption(text):
        return BlockKind.CAPTION, text
    return BlockKind.PARAGRAPH, text


def _placed_images(
    pdf: _PdfFile,
    *,
    file: str,
    share: float,
    counter: Counter[str],
) -> tuple[list[Placed], frozenset[str]]:
    """Make the image blocks of a document, without its image chrome.

    An image whose content comes back on enough pages, as `repeated_images`
    tells with the share of the pages, is chrome, such as a logo: it is
    removed. So is a tiny image, as `is_tiny_image` tells, which is counted
    apart.

    Args:
        pdf: What is read from the file.
        file: The name of the PDF file, which the blocks point to.
        share: Least share of the pages on which an image is chrome.
        counter: The counts of the normalizations, in which `image_chrome`
            goes up by one for each image removed as chrome, and
            `tiny_image` for each tiny image.

    Returns:
        The image blocks that are kept, with where their images lie, by
        page, then from the top of the page down, then from left to right;
        and the digests of the images of the chrome.
    """
    per_page = [
        [
            Block(
                kind=BlockKind.IMAGE,
                text="",
                raw_text="",
                source=SourceRef(file, page=index + 1),
                image=image.ref,
            )
            for image in page.images
        ]
        for index, page in enumerate(pdf.pages)
    ]
    sized = [
        [b for b in blocks if not is_tiny_image(b)] for blocks in per_page
    ]
    chrome = repeated_images(sized, share=share)
    kept = [
        Placed(index, image.top, image.x0, block, image.bottom)
        for index, (page, blocks) in enumerate(
            zip(pdf.pages, per_page, strict=True),
        )
        for image, block in zip(page.images, blocks, strict=True)
        if image.ref.digest not in chrome and not is_tiny_image(block)
    ]
    found = sum(len(blocks) for blocks in sized)
    _count(counter, IMAGE_CHROME, found - len(kept))
    _count(counter, TINY_IMAGE, sum(map(len, per_page)) - found)
    return sorted(kept, key=place_order), chrome


def _with_placed(
    blocks: Sequence[Block],
    starts: Sequence[Line],
    placed: Sequence[Placed],
) -> list[Block]:
    """Put each image or table block before the first block under its top.

    An image or a table goes before the first block whose first line is on a
    later page, or on its page with its baseline under its top. Those that
    no block starts under follow the last block.

    Args:
        blocks: The blocks of text, in reading order.
        starts: The first line of each block, in the same order.
        placed: The image and table blocks, in the order of `place_order`.

    Returns:
        All the blocks, in reading order.
    """
    pending = deque(placed)
    ordered: list[Block] = []
    for block, start in zip(blocks, starts, strict=True):
        while pending and (
            pending[0].page < start.page
            or (
                pending[0].page == start.page
                and pending[0].top > start.baseline
            )
        ):
            ordered.append(pending.popleft().block)
        ordered.append(block)
    ordered.extend(item.block for item in pending)
    return ordered


def _blocks(  # ruff: ignore[too-many-arguments] - one input per earlier step
    drafts: Sequence[DraftBlock],
    texts: Sequence[str],
    *,
    levels: Mapping[int, tuple[int, HeadingSource]],
    kinds: Mapping[int, BlockKind],
    placed: Sequence[Placed],
    file: str,
    counter: Counter[str],
) -> list[Block]:
    """Make the normalized blocks of a document, with their section paths.

    Args:
        drafts: The draft blocks of the document, in reading order.
        texts: The text of each block, in the same order.
        levels: The level and the source of each heading, by position of
            its block.
        kinds: The kind of each contents or index page, by position of the
            page.
        placed: The image and table blocks, in the order of `place_order`.
        file: The name of the PDF file, which the blocks point to.
        counter: The counts of the normalizations.

    Returns:
        The blocks, each located on the page of its first line, and the image
        and table blocks among them, each before the first block that starts
        under its top, as `_with_placed` puts them. The entries of a list of
        figures or tables, as `figure_list` finds them, are contents.
    """
    listed = figure_list(texts)
    blocks = []
    for index, (draft, text) in enumerate(zip(drafts, texts, strict=True)):
        page = draft.lines[0].page
        kind, raw_text = _kind(
            draft,
            text,
            page_kind=kinds.get(page)
            or (BlockKind.TOC if index in listed else None),
            heading=index in levels,
            counter=counter,
        )
        heading = kind is BlockKind.HEADING
        blocks.append(
            normalized_block(
                kind,
                raw_text,
                SourceRef(file, page=page + 1),
                counter,
                level=levels[index][0] if heading else None,
                attrs={HEADING_SOURCE: levels[index][1]} if heading else None,
            ),
        )
    blocks = _with_placed(
        blocks,
        [draft.lines[0] for draft in drafts],
        placed,
    )
    blocks = strip_heading_numbers(blocks, counter)
    blocks = strip_caption_numbers(blocks, counter)
    assign_section_paths(blocks)
    return blocks


def _language_sample(blocks: Sequence[Block]) -> str:
    """Return the text of the first `LANG_SAMPLE_BLOCKS` blocks of text.

    Args:
        blocks: The blocks of the document, in reading order.

    Returns:
        Their texts, the image blocks left out, one blank apart.
    """
    return " ".join(
        islice(
            (b.text for b in blocks if b.kind is not BlockKind.IMAGE),
            LANG_SAMPLE_BLOCKS,
        ),
    )


def _refused_warnings(labels: Sequence[str]) -> list[str]:
    """Warn about the tables kept as text, as their cells do not match.

    Args:
        labels: The label of the page of each such table, in page order.

    Returns:
        One warning that counts them and lists the labels of the first
        `SCANNED_LABELS_MAX` of their pages, or none if there is none.
    """
    if not labels:
        return []
    pages = _page_list(list(dict.fromkeys(labels)))
    warning = (
        f"{len(labels)} table(s) kept as text, their cells not matching the"
        f" text of their page: {pages}"
    )
    return [warning]


def _page_list(labels: Sequence[str]) -> str:
    """List the labels of the pages that a warning is about.

    Args:
        labels: The labels of the pages, in page order.

    Returns:
        The first `SCANNED_LABELS_MAX` labels, comma separated, and an
        ellipsis after them if there are more.
    """
    listed = ", ".join(labels[:SCANNED_LABELS_MAX])
    if len(labels) > SCANNED_LABELS_MAX:
        listed += _MORE_LABELS
    return listed


def _is_scanned(page: _Page, segments: Sequence[Segment]) -> bool:
    """Tell whether a page is probably scanned.

    Args:
        page: The page.
        segments: Its segments, chrome removed.

    Returns:
        Whether the page keeps no segment once the chrome is removed, and
        images cover at least `SCANNED_MIN_COVERAGE` of it.
    """
    return not segments and page.image_coverage >= SCANNED_MIN_COVERAGE


def _scanned_warnings(
    pdf: _PdfFile,
    kept: Sequence[Sequence[Segment]],
) -> list[str]:
    """Warn about the pages that are probably scanned, as `_is_scanned` says.

    Args:
        pdf: What is read from the file.
        kept: The segments of each page, chrome removed.

    Returns:
        One warning that counts the scanned pages and lists the labels of the
        first `SCANNED_LABELS_MAX` of them, or none if no page is scanned.
    """
    labels = [
        page.label
        for page, segments in zip(pdf.pages, kept, strict=True)
        if _is_scanned(page, segments)
    ]
    if not labels:
        return []
    listed = _page_list(labels)
    return [f"{len(labels)} page(s) without text, probably scanned: {listed}"]


def _angled_warnings(pdf: _PdfFile) -> list[str]:
    """Warn about the pages whose text at an angle was left out.

    Args:
        pdf: What is read from the file.

    Returns:
        One warning that counts the pages with characters left out for their
        angle and lists the labels of the first `SCANNED_LABELS_MAX` of
        them, or none if no page has such characters.
    """
    labels = [page.label for page in pdf.pages if page.angled]
    if not labels:
        return []
    listed = _page_list(labels)
    return [f"{len(labels)} page(s) with text at an angle, left out: {listed}"]


def _figure_warnings(pdf: _PdfFile) -> list[str]:
    """Warn about the pages with more vector figures than are rendered.

    Args:
        pdf: What is read from the file.

    Returns:
        One warning for each page whose figures after the first
        `MAX_PAGE_FIGURES` were not rendered, in page order.
    """
    return [
        f"page {page.label}: {page.unrendered} vector figure(s) not rendered,"
        f" beyond the first {MAX_PAGE_FIGURES}"
        for page in pdf.pages
        if page.unrendered
    ]


def _link_warnings(pdf: _PdfFile) -> list[str]:
    """Warn about the links, and the texts of links, that were left unread.

    Args:
        pdf: What is read from the file.

    Returns:
        One warning that counts the pages whose links after the first
        `MAX_PAGE_LINKS` of the reader were not read, then one that counts
        the links whose text was not read, for `LINK_TEXT_WORK` of the
        reader, and their pages, each listing the labels of the first
        `SCANNED_LABELS_MAX` pages, if there are such pages; then one if
        links were left unread for `MAX_DOCUMENT_LINKS`. Each limit is the
        one that the reading used.
    """
    warnings: list[str] = []
    cut = [page.label for page in pdf.pages if page.more_links]
    if cut:
        warnings.append(
            f"{len(cut)} page(s) with more than {reader.MAX_PAGE_LINKS}"
            f" links, the others were not read: {_page_list(cut)}",
        )
    untexted = [page for page in pdf.pages if page.untexted_links]
    if untexted:
        total = sum(page.untexted_links for page in untexted)
        labels = _page_list([page.label for page in untexted])
        warnings.append(
            f"{len(untexted)} page(s) where the text of {total} links was"
            f" not read: {labels}",
        )
    if pdf.links_cut:
        warnings.append(
            f"more than {MAX_DOCUMENT_LINKS} links in the document,"
            " the others were not read",
        )
    return warnings


def _label_warnings(pdf: _PdfFile) -> list[str]:
    """Warn about the pages that were numbered rather than labelled.

    Args:
        pdf: What is read from the file.

    Returns:
        One warning if the labels asked reached `PAGE_LABEL_WORK` of the
        reader, as the reading used it, before the last page; or none.
    """
    if not pdf.labels_cut:
        return []
    return [
        (
            f"the page labels exceed {reader.PAGE_LABEL_WORK} bytes,"
            " the other pages are numbered"
        ),
    ]


def _outline_warnings(pdf: _PdfFile) -> list[str]:
    """Warn about the bookmarks that were left unread.

    Args:
        pdf: What is read from the file.

    Returns:
        One warning if bookmarks were not read, after the first
        `MAX_BOOKMARKS` of the reader or once their titles reached
        `OUTLINE_TITLE_WORK` bytes, each as the reading used it; or none.
    """
    match pdf.outline_cut:
        case "bookmarks":
            warning = (
                f"the outline has more than {reader.MAX_BOOKMARKS} bookmarks,"
                " the others were not read"
            )
        case "titles":
            warning = (
                f"the outline titles exceed {reader.OUTLINE_TITLE_WORK} bytes,"
                " the other bookmarks were not read"
            )
        case _:
            return []
    return [warning]


def _tag_warnings(tags: TagTree | None) -> list[str]:
    """Warn about a structure tree that was not read in full.

    Args:
        tags: The structure tree, if it was read.

    Returns:
        One warning that tells why the tree was not read in full, unless the
        document has no tree; or none.
    """
    if tags is None or tags.problem in {None, NO_STRUCTURE_TREE}:
        return []
    return [f"{tags.problem}; the checks of the tags may be incomplete"]


def _outline(bookmarks: Sequence[Bookmark]) -> tuple[OutlineEntry, ...]:
    """Make the outline of a document from its bookmarks.

    Args:
        bookmarks: The bookmarks, in the order of the document.

    Returns:
        One entry for each bookmark, with its printable title, cut at
        `METADATA_LIMIT` characters, and its page from 1, if it has one.
    """
    return tuple(
        OutlineEntry(
            bookmark.level,
            printable(bookmark.title, limit=METADATA_LIMIT),
            None if bookmark.page is None else bookmark.page + 1,
        )
        for bookmark in bookmarks
    )


def _page_facts(
    pdf: _PdfFile,
    kept: Sequence[Sequence[Segment]],
) -> tuple[PdfPageFacts, ...]:
    """Gather what the audit reads of each page.

    Args:
        pdf: What is read from the file.
        kept: The segments of each page, chrome removed.

    Returns:
        The facts of each page, in order: its number from 1, its label, its
        kept characters, those without Unicode mapping, whether it is
        scanned, as `_is_scanned` tells, and its links.
    """
    return tuple(
        PdfPageFacts(
            number,
            page.label,
            page.chars,
            page.unmapped,
            scanned=_is_scanned(page, segments),
            links=page.links,
        )
        for number, (page, segments) in enumerate(
            zip(pdf.pages, kept, strict=True),
            1,
        )
    )


def load_pdf(
    path: Path,
    cfg: PdfConfig,
    *,
    lang: str,
    images: bool = True,
    audit: bool = False,
) -> Document:
    """Load a PDF file.

    A file that cannot be read, that a user password protects, or that has a
    page that cannot be read raises an `InputError` that names the file.
    Settings that ask for the tables need the pdf-tables extra: without it,
    a `ConfigError` is raised before the file is read. The tables are read in
    a process started afresh, as `read_tables` tells, which imports the main
    script again: a script that calls this function with the tables must run
    its own code under ``if __name__ == "__main__":``.

    Args:
        path: The PDF file.
        cfg: PDF extraction settings.
        lang: Language requested by the user, or ``"auto"``.
        images: Whether the raster images give blocks and are read.
        audit: Whether to read the facts that only the audit uses: the
            characters, the scans, the links and the monospaced words of
            the pages, and the structure tree of a document that declares
            its tags.

    Returns:
        The document, its blocks in reading order. Its chrome lists the texts
        removed as running headers, footers and page numbers, and its
        warnings the pages that are probably scanned, then those whose text
        at an angle was left out, then those with more vector figures than
        are rendered, then those with more links than are read or links
        whose text was not read, then the pages numbered rather than
        labelled, then the bookmarks that were not read, then the images
        that could not be read, then those of the tables, as
        `read_kept_tables` of the table blocks gives them, and the tables
        kept as text, then why the structure tree was not read in full. Its
        title and its outline are always read, its page facts, its structure
        tree and its code words only for the audit. With the tables, as
        `read_kept_tables` reads them, each table kept is a block, joined to
        the tables that continue it, as `with_tables` of the table blocks
        makes them.
    """
    if cfg.tables:
        tables.require_pdfplumber()
    counter: Counter[str] = Counter()
    pdf = _read(path, counter, images=images, audit=audit)
    body = body_size(page.segments for page in pdf.pages)
    found = (
        read_kept_tables(path, pdf.pages, cfg) if cfg.tables else KeptTables()
    )
    kept, chrome = _strip_chrome(
        pdf,
        flow(pdf.pages, found),
        body=body,
        cfg=cfg,
        counter=counter,
    )
    # The segments of the tables take part in the cuts of their pages, so
    # that the text around a table is read in the order of the base reading.
    kinds, drafts = _drafts(
        text_segments(pdf.pages, kept, found),
        body=body,
        counter=counter,
        apart=table_breaks(found),
        hidden=found.taken,
    )
    texts = _texts(drafts, counter)
    levels = _levels(drafts, texts, pdf.bookmarks, body=body)
    images, chrome_images = _placed_images(
        pdf,
        file=path.name,
        share=cfg.repeat_share,
        counter=counter,
    )
    blocks = _blocks(
        drafts,
        texts,
        levels=levels,
        kinds=kinds,
        placed=with_tables(
            images,
            found,
            kept,
            file=path.name,
            counter=counter,
        ),
        file=path.name,
        counter=counter,
    )
    declared = declared_language(pdf.catalog.lang)
    tags = None
    if audit and pdf.catalog.tagged:
        from doc_checker.extract.pdf.tags import read_tags  # ruff: ignore[import-outside-top-level] - pypdf is only needed by the audit

        tags = read_tags(path)
    return Document(
        path=path,
        format="pdf",
        blocks=blocks,
        parts=[page.label for page in pdf.pages],
        lang=resolve_language(lang, declared, _language_sample(blocks)),
        declared_lang=declared,
        generator=pdf.generator,
        normalization=counter,
        chrome=chrome,
        chrome_keys=frozenset(),
        warnings=[
            *_scanned_warnings(pdf, text_segments(pdf.pages, kept, found)),
            *_angled_warnings(pdf),
            *_figure_warnings(pdf),
            *_link_warnings(pdf),
            *_label_warnings(pdf),
            *_outline_warnings(pdf),
            *image_warnings(blocks),
            *found.warnings,
            *_refused_warnings(found.refused),
            *_tag_warnings(tags),
        ],
        tagged=pdf.catalog.tagged,
        chrome_images=chrome_images,
        title=pdf.title,
        outline=_outline(pdf.bookmarks),
        pdf_pages=(
            _page_facts(pdf, text_segments(pdf.pages, kept, found))
            if audit
            else ()
        ),
        tags=tags,
        code_words=frozenset().union(*(page.code_words for page in pdf.pages)),
    )
