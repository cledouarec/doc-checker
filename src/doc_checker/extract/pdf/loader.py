# Copyright 2026 Christophe Le Douarec
"""Load a PDF file into a document.

The pages are read one at a time, and only the line segments of each page are
kept. The running headers, footers and page numbers are removed, the segments
of each page are ordered into lines, and the lines are grouped into blocks
across page and column breaks. Then the text of each block is joined, and the
block gets its kind: contents or index, code, heading, list item, caption or
paragraph.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.errors import InputError
from doc_checker.extract.common import (
    assign_section_paths,
    normalized_block,
    strip_caption_numbers,
    strip_heading_numbers,
)
from doc_checker.extract.pdf.blocks import (
    group_lines,
    merge_flows,
    strip_list_marker,
)
from doc_checker.extract.pdf.catalog import read_catalog
from doc_checker.extract.pdf.chrome import find_chrome, title_pages
from doc_checker.extract.pdf.headings import heading_levels
from doc_checker.extract.pdf.joining import code_text, prose_text, vocabulary
from doc_checker.extract.pdf.kinds import is_caption, toc_pages
from doc_checker.extract.pdf.layout import page_lines
from doc_checker.extract.pdf.reader import (
    open_pdf,
    read_bookmarks,
    read_metadata,
    read_page,
)
from doc_checker.extract.pdf.segments import body_size, build_segments
from doc_checker.model.document import (
    HEADING_SOURCE,
    BlockKind,
    Document,
    SourceRef,
)
from doc_checker.text.lang import declared_language, resolve_language
from doc_checker.text.normalize import ELLIPSIS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pypdfium2 as pdfium

    from doc_checker.config import PdfConfig
    from doc_checker.extract.pdf.blocks import DraftBlock
    from doc_checker.extract.pdf.catalog import PdfCatalog
    from doc_checker.extract.pdf.reader import Bookmark, PdfPageText
    from doc_checker.extract.pdf.segments import Segment
    from doc_checker.model.document import Block, HeadingSource

SCANNED_MIN_COVERAGE: Final = 0.5  # Smallest image share of a scanned page.
SCANNED_LABELS_MAX: Final = 10  # Most page labels that a warning lists.
LANG_SAMPLE_BLOCKS: Final = 200  # Blocks whose text tells the language.

_DIGITS: Final = re.compile(r"\d+")
_MORE_LABELS: Final = f", {ELLIPSIS}"  # Ends a list of labels that goes on.


@dataclass(frozen=True, slots=True)
class _Page:
    """A page, once its characters are grouped into segments.

    Attributes:
        label: Page label, such as ``iv``, or the page number from 1.
        height: Height of the crop box of the page turned upright, in points.
        image_coverage: Share of the page covered by images, from 0 to 1.
        segments: The segments of the page, in the order of their first
            character.
        angled: Number of characters left out because they do not follow the
            direction of the page.
    """

    label: str
    height: float
    image_coverage: float
    segments: tuple[Segment, ...]
    angled: int


@dataclass(frozen=True, slots=True)
class _PdfFile:
    """What is read from a PDF file.

    Attributes:
        catalog: The language and the tagging that the catalog declares.
        bookmarks: The outline, in the order of the document.
        title: The title of the metadata, if any.
        generator: The producer of the metadata, or else its creator, if any.
        pages: Every page of the document, in order.
    """

    catalog: PdfCatalog
    bookmarks: tuple[Bookmark, ...]
    title: str | None
    generator: str | None
    pages: tuple[_Page, ...]


def _count(counter: Counter[str], key: str, amount: int) -> None:
    """Add an amount to a count of normalizations, unless it is zero."""
    if amount:
        counter[key] += amount


def _read_page(
    pdf: pdfium.PdfDocument,
    index: int,
    path: Path,
) -> PdfPageText:
    """Read the characters of a page.

    Args:
        pdf: The document, opened by `open_pdf`.
        index: Position of the page, from 0.
        path: The PDF file, which the error names.

    Returns:
        The characters of the page, as `read_page` returns them.

    Raises:
        InputError: If the page cannot be read.
    """
    try:
        return read_page(pdf, index)
    except InputError as exc:
        msg = f"cannot read {path}: {exc}"
        raise InputError(msg) from None


def _read(path: Path, counter: Counter[str]) -> _PdfFile:
    """Read the catalog, the outline, the metadata and the pages of a PDF.

    The pages are read one at a time, and each one keeps its segments only,
    not its characters.

    Args:
        path: The PDF file.
        counter: The counts of the normalizations, in which `artifact` goes
            up by the characters dropped as artifacts, `unicode_map` by the
            characters without Unicode mapping, and `angled_text` by the
            characters left out for their angle.

    Returns:
        What is read from the file.
    """
    with open_pdf(path) as pdf:
        catalog = read_catalog(pdf)
        bookmarks = tuple(read_bookmarks(pdf))
        title, generator = read_metadata(pdf)
        pages = []
        for index in range(len(pdf)):
            page = _read_page(pdf, index, path)
            _count(counter, "artifact", page.artifacts)
            _count(counter, "unicode_map", page.unmapped)
            _count(counter, "angled_text", page.angled)
            segments = tuple(build_segments(page))
            pages.append(
                _Page(
                    page.label,
                    page.height,
                    page.image_coverage,
                    segments,
                    page.angled,
                ),
            )
    return _PdfFile(catalog, bookmarks, title, generator, tuple(pages))


def _strip_chrome(
    pdf: _PdfFile,
    *,
    body: float,
    cfg: PdfConfig,
    counter: Counter[str],
) -> tuple[list[list[Segment]], list[str]]:
    """Remove the running headers, footers and page numbers of every page.

    Args:
        pdf: What is read from the file.
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
    pages = [page.segments for page in pdf.pages]
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
) -> tuple[dict[int, BlockKind], list[DraftBlock]]:
    """Order the lines of each page and group them into draft blocks.

    Args:
        kept: The segments of each page, chrome removed, one item for each
            page of the document. The count of the pages bounds the numbers
            of the entries of a page that has no title of contents.
        body: The body size of the document, in points.
        counter: The counts of the normalizations, in which `page_break`
            goes up by one for each block joined across a break.

    Returns:
        The kind of each contents or index page, by position of the page, and
        the draft blocks of the document, in reading order.
    """
    lines = [
        line for segments in kept for line in page_lines(segments, body=body)
    ]
    kinds = toc_pages(lines, page_count=len(kept))
    return kinds, merge_flows(group_lines(lines), counter, page_kinds=kinds)


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


def _blocks(  # ruff: ignore[too-many-arguments] - one input per earlier step
    drafts: Sequence[DraftBlock],
    texts: Sequence[str],
    *,
    levels: Mapping[int, tuple[int, HeadingSource]],
    kinds: Mapping[int, BlockKind],
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
        file: The name of the PDF file, which the blocks point to.
        counter: The counts of the normalizations.

    Returns:
        The blocks, each located on the page of its first line.
    """
    blocks = []
    for index, (draft, text) in enumerate(zip(drafts, texts, strict=True)):
        page = draft.lines[0].page
        kind, raw_text = _kind(
            draft,
            text,
            page_kind=kinds.get(page),
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
    blocks = strip_heading_numbers(blocks, counter)
    blocks = strip_caption_numbers(blocks, counter)
    assign_section_paths(blocks)
    return blocks


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


def _scanned_warnings(
    pdf: _PdfFile,
    kept: Sequence[Sequence[Segment]],
) -> list[str]:
    """Warn about the pages that are probably scanned.

    A page is scanned if it keeps no segment once the chrome is removed, and
    images cover at least `SCANNED_MIN_COVERAGE` of it.

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
        if not segments and page.image_coverage >= SCANNED_MIN_COVERAGE
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


def load_pdf(path: Path, cfg: PdfConfig, *, lang: str) -> Document:
    """Load a PDF file.

    A file that cannot be read, that a user password protects, or that has a
    page that cannot be read raises an `InputError` that names the file.

    Args:
        path: The PDF file.
        cfg: PDF extraction settings.
        lang: Language requested by the user, or ``"auto"``.

    Returns:
        The document, its blocks in reading order. Its chrome lists the texts
        removed as running headers, footers and page numbers, and its
        warnings the pages that are probably scanned, then those whose text
        at an angle was left out.
    """
    counter: Counter[str] = Counter()
    pdf = _read(path, counter)
    body = body_size(page.segments for page in pdf.pages)
    kept, chrome = _strip_chrome(pdf, body=body, cfg=cfg, counter=counter)
    kinds, drafts = _drafts(kept, body=body, counter=counter)
    texts = _texts(drafts, counter)
    levels = _levels(drafts, texts, pdf.bookmarks, body=body)
    blocks = _blocks(
        drafts,
        texts,
        levels=levels,
        kinds=kinds,
        file=path.name,
        counter=counter,
    )
    sample = " ".join(block.text for block in blocks[:LANG_SAMPLE_BLOCKS])
    declared = declared_language(pdf.catalog.lang)
    return Document(
        path=path,
        format="pdf",
        blocks=blocks,
        parts=[page.label for page in pdf.pages],
        lang=resolve_language(lang, declared, sample),
        declared_lang=declared,
        generator=pdf.generator,
        normalization=counter,
        chrome=chrome,
        chrome_keys=frozenset(),
        warnings=[*_scanned_warnings(pdf, kept), *_angled_warnings(pdf)],
        tagged=pdf.catalog.tagged,
    )
