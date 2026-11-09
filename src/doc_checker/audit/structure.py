# Copyright 2026 Christophe Le Douarec
"""Check the structure of a document: headings, references, captions, pages.

A heading level is skipped, in an HTML page, by a heading more than one
level deeper than the heading before it in its page. In a PDF, whose
outline and styles cannot skip a level, only the headings of the structure
tree are compared, in its order; a heading without a level, a plain ``H``,
is ignored.

A section is empty when its heading is followed, in its page, by a heading
of its level or above, or ends the page, unless the page is made of that
heading alone, or content of the page that gives no block lies between
them: an embedded medium, a toctree, or a picture when the images are not
read. In a PDF, only the headings that a bookmark found head a section,
and a heading found by its style is content.

Two headings of one page are identical siblings when they have the same
parent heading, the same level and the same title, case ignored; in a PDF,
two bookmarks of the same parent with the same title, case ignored.

A reference is unresolved where a block that is neither code, formula,
table, contents, index nor image holds a lone ``??`` or ``[?]``, or, in
HTML only, a literal ``[text][label]``, that touches none of its inline
code, as `code_ranges` places it; an HTML page also tells the references
its DOM left without a target, each placed on the block of its source
line. Each block gives one draft per kind of reference, which marks the
first one.

A figure or a table without a caption is reported only in a document that
captions at least one figure, or table. A table, or a figure of an HTML
page, is captioned by a caption block just before or just after it, in its
page; a figure of a PDF, whose reading order may part it from its caption,
by a caption of its page a few blocks away. An HTML page, but the first
one, that no other page links to and that holds more than headings and
images is an orphan.

The texts of the document that a detail quotes are cut.

Every check walks the document once, with indexes built once.
"""

import re
from bisect import bisect_left, bisect_right
from collections import Counter
from typing import TYPE_CHECKING, Final

from doc_checker.audit.prose import code_ranges
from doc_checker.audit.sides import block_side, image_side, quote
from doc_checker.compare.drafts import ARROW, Draft, marked_side, marked_span
from doc_checker.extract.pdf.headings import MAX_LEVEL
from doc_checker.model.document import (
    FROM_OUTLINE,
    HEADING_SOURCE,
    BlockKind,
    SourceRef,
)
from doc_checker.spell.words import touches
from doc_checker.text.normalize import normalize_display, printable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.model.document import Block, Document
    from doc_checker.model.facts import HtmlPageFacts, Spot, TaggedElement
    from doc_checker.model.findings import Side
    from doc_checker.spell.words import Span

UNRESOLVED_MARK: Final = re.compile(r"(?<![\w?=])\?\?(?![\w?=])")
MISSING_CITATION: Final = re.compile(r"\[\?\]")
# One quantifier per bracket keeps the scan linear; the letter and the
# character class are checked in Python, on the match.
MARKDOWN_REFERENCE: Final = re.compile(
    r"(?<![\w\])\[])\[([^\[\]\n<>/]*)\]\[([^\[\]\n<>/]*)\](?![\[(])",
)
CHARACTER_CLASS: Final = re.compile(r"(?:[^\W_]-[^\W_])+")
_LETTER: Final = re.compile(r"[^\W\d_]")

_SKIPPED: Final = "structure.heading_skipped"
_EMPTY: Final = "structure.empty_section"
_DUPLICATE: Final = "structure.duplicate_heading"
_UNRESOLVED: Final = "structure.unresolved_reference"
_FIGURE: Final = "structure.figure_caption"
_TABLE: Final = "structure.table_caption"
_ORPHAN: Final = "structure.orphan_page"

_NEXT_HEADING: Final = "no content before the next heading of its level"
_PAGE_END: Final = "no content before the end of the page"
_DOCUMENT_END: Final = "no content before the end of the document"
_TOP: Final = "the top"
_XREF: Final = "xref"
_NOT_LINKED: Final = "no other page links to this page"
# The blocks whose text may hold an unresolved reference.
_NO_REFERENCES: Final = frozenset(
    {
        BlockKind.CODE,
        BlockKind.FORMULA,
        BlockKind.TABLE,
        BlockKind.TOC,
        BlockKind.INDEX,
        BlockKind.IMAGE,
    },
)
# The blocks that do not make a page worth linking to.
_NO_CONTENT: Final = frozenset({BlockKind.HEADING, BlockKind.IMAGE})
# What joins the texts of the blocks of a page, which no reference holds.
_JOIN: Final = "\n"
# The most characters searched for a dead reference among the blocks of its
# source line, after the reference before it, once the line has spent the
# budget of its searches.
TIE_SCAN_CHARS: Final = 10_000
# The most blocks between a figure of a PDF and its caption, on its page.
CAPTION_WINDOW: Final = 8
# The blocks that part a figure of a PDF from a caption beyond them.
_CAPTION_STOPS: Final = frozenset(
    {BlockKind.HEADING, BlockKind.TABLE, BlockKind.IMAGE},
)
# The pages a site needs before one of them can be an orphan.
_SITE_MIN_PAGES: Final = 2


def _title_key(text: str) -> str:
    """Return the key of a title: case and runs of blanks ignored."""
    return " ".join(text.casefold().split())


def _following(blocks: Sequence[Block]) -> list[int | None]:
    """Return where the next block of the file of each block is.

    Returns:
        For each block, the position of the next block of its file, or
        ``None`` for the last one.
    """
    following: list[int | None] = [None] * len(blocks)
    last: dict[str, int] = {}
    for index in range(len(blocks) - 1, -1, -1):
        file = blocks[index].source.file
        following[index] = last.get(file)
        last[file] = index
    return following


def _html_skips(blocks: Sequence[Block]) -> list[Draft]:
    """Return the headings deeper by more than one level than the one before.

    The heading before is that of the same page.
    """
    drafts: list[Draft] = []
    previous: dict[str, int] = {}
    for block in blocks:
        if block.kind is not BlockKind.HEADING or block.level is None:
            continue
        before = previous.get(block.source.file)
        previous[block.source.file] = block.level
        if before is not None and block.level > before + 1:
            drafts.append(
                Draft(
                    _SKIPPED,
                    None,
                    block_side(block),
                    None,
                    f"h{before} is followed by h{block.level}",
                    f"h{before}{ARROW}h{block.level}",
                ),
            )
    return drafts


def _outline_levels(headings: Sequence[TaggedElement]) -> list[int | None]:
    """Return the level of each tagged heading as an outline nests it.

    An outline cannot skip a level: a heading is one level below the last
    heading of a lower level before it, and `MAX_LEVEL` at most, as the
    levels of the bookmarks are. A heading without a level has none, and
    nests nothing.
    """
    above: list[int] = []
    levels: list[int | None] = []
    for element in headings:
        level = element.level
        if level is not None:
            while above and above[-1] >= level:
                above.pop()
            above.append(level)
        levels.append(None if level is None else min(len(above), MAX_LEVEL))
    return levels


class _TaggedPages:
    """The headings of each page of a PDF: tagged, and extracted.

    The extracted heading at the place of a tagged one, among the headings
    of its page, stands for it when the page has as many of each and every
    extracted heading of the page that a bookmark found has the level of
    its tagged heading, nested as `_outline_levels` nests it. The tags of a
    page may be out of order, as a single LaTeX run may leave them, though
    their number is right.
    """

    __slots__ = ("agrees", "extracted", "levels")

    def __init__(
        self,
        doc: Document,
        headings: Sequence[TaggedElement],
    ) -> None:
        """Gather the headings of the pages of a PDF.

        Args:
            doc: The PDF.
            headings: The headings of its structure tree, in its order.
        """
        self.levels: dict[int | None, list[int | None]] = {}
        for element, level in zip(
            headings,
            _outline_levels(headings),
            strict=True,
        ):
            self.levels.setdefault(element.page, []).append(level)
        self.extracted: dict[int, list[Block]] = {}
        for block in doc.blocks:
            page = block.source.page
            if block.kind is BlockKind.HEADING and page is not None:
                self.extracted.setdefault(page, []).append(block)
        self.agrees: dict[int, bool] = {}

    def heading(self, page: int | None, position: int) -> Block | None:
        """Return the extracted heading that stands for a tagged one, if any.

        Args:
            page: The page of the tagged heading.
            position: Its position among the tagged headings of its page.

        Returns:
            The extracted heading at that position, when it stands for the
            tagged one; else ``None``.
        """
        if page is None:
            return None
        blocks = self.extracted.get(page, [])
        levels = self.levels[page]
        if len(blocks) != len(levels):
            return None
        if page not in self.agrees:
            self.agrees[page] = all(
                level is None
                or block.attrs.get(HEADING_SOURCE) != FROM_OUTLINE
                or block.level == level
                for level, block in zip(levels, blocks, strict=True)
            )
        return blocks[position] if self.agrees[page] else None


def _tagged_side(
    doc: Document,
    element: TaggedElement,
    position: int,
    pages: _TaggedPages,
) -> Side:
    """Return the side of a tagged heading.

    Args:
        doc: The PDF.
        element: A heading of its structure tree.
        position: Position of the element among the tagged headings of its
            page.
        pages: The headings of each page.

    Returns:
        The side of the extracted heading that stands for the element, as
        `_TaggedPages` tells; else a side that shows the type of the
        element at its page.
    """
    block = pages.heading(element.page, position)
    if block is not None:
        return block_side(block)
    source = SourceRef(doc.path.name, page=element.page)
    return marked_side(source, element.type, kind=BlockKind.HEADING)


def _tagged_skips(doc: Document) -> list[Draft]:
    """Return the tagged headings deeper by more than one level than before.

    The headings of the structure tree are compared in its order; one
    without a level is neither compared nor the level before the next one.
    """
    if doc.tags is None:
        return []
    headings = [e for e in doc.tags.elements if e.role == "heading"]
    pages = _TaggedPages(doc, headings)
    seen: Counter[int | None] = Counter()
    drafts: list[Draft] = []
    before: int | None = None
    for element in headings:
        position = seen[element.page]
        seen[element.page] += 1
        level = element.level
        if level is None:
            continue
        if before is not None and level > before + 1:
            side = _tagged_side(doc, element, position, pages)
            drafts.append(
                Draft(
                    _SKIPPED,
                    None,
                    side,
                    None,
                    f"H{before} is followed by H{level}",
                    f"H{before}{ARROW}H{level}",
                ),
            )
        before = level
    return drafts


def _section_level(block: Block, *, pdf: bool) -> int | None:
    """Return the level of a heading that heads a section, if it does.

    In a PDF, only a heading that a bookmark found heads a section.
    """
    if block.kind is not BlockKind.HEADING:
        return None
    if pdf and block.attrs.get(HEADING_SOURCE) != FROM_OUTLINE:
        return None
    return block.level


def _has_blockless(
    lines: Sequence[int],
    block: Block,
    after: Block | None,
) -> bool:
    """Return whether content without block lies between a heading and after.

    Args:
        lines: Sorted source lines of the content of the page that gives no
            block.
        block: A heading of the page.
        after: The next block of the page, if any.

    Returns:
        Whether such content starts on the line of the heading or later,
        and on the line of the next block or before; ``False`` when the
        heading, or the next block, has no source line.
    """
    start = block.source.line
    end = None if after is None else after.source.line
    if start is None or (after is not None and end is None):
        return False
    first = bisect_left(lines, start)
    if first == len(lines):
        return False
    return end is None or lines[first] <= end


def _empty_sections(doc: Document) -> list[Draft]:
    """Return the headings of the sections without content.

    The content of an HTML page that gives no block, an embedded medium, a
    toctree or a picture whose image is not read, is content.
    """
    pdf = doc.format == "pdf"
    blocks = doc.blocks
    following = _following(blocks)
    sizes = Counter(block.source.file for block in blocks)
    blockless = {
        page.file: sorted((*page.media, *page.blockless))
        for page in doc.html_pages
    }
    drafts: list[Draft] = []
    for block, after in zip(blocks, following, strict=True):
        level = _section_level(block, pdf=pdf)
        if level is None:
            continue
        if after is None:
            if not pdf and sizes[block.source.file] == 1:
                continue
            detail = _DOCUMENT_END if pdf else _PAGE_END
        else:
            other = _section_level(blocks[after], pdf=pdf)
            if other is None or other > level:
                continue
            detail = _NEXT_HEADING
        lines = blockless.get(block.source.file)
        then = None if after is None else blocks[after]
        if lines and _has_blockless(lines, block, then):
            continue
        drafts.append(
            Draft(_EMPTY, None, block_side(block), None, detail, "empty"),
        )
    return drafts


def _duplicate(side: Side, parent: str | None) -> Draft:
    """Return the draft of a title that an earlier sibling has."""
    above = quote(parent) if parent else _TOP
    detail = f'same title as an earlier heading under "{above}"'
    return Draft(_DUPLICATE, None, side, None, detail, "duplicate")


def _html_duplicates(blocks: Sequence[Block]) -> list[Draft]:
    """Return the headings whose title an earlier sibling of its page has.

    Siblings have the same parent heading, or none, and the same level.
    """
    drafts: list[Draft] = []
    stacks: dict[str, list[tuple[int, int]]] = {}
    seen: set[tuple[str, int | None, int, str]] = set()
    for index, block in enumerate(blocks):
        level = block.level
        if block.kind is not BlockKind.HEADING or level is None:
            continue
        stack = stacks.setdefault(block.source.file, [])
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        stack.append((level, index))
        title = _title_key(block.text)
        if not title:
            continue
        key = (block.source.file, parent, level, title)
        if key in seen:
            above = None if parent is None else blocks[parent].text
            drafts.append(_duplicate(block_side(block), above))
        else:
            seen.add(key)
    return drafts


def _outline_duplicates(doc: Document) -> list[Draft]:
    """Return the bookmarks whose title an earlier one of its parent has."""
    drafts: list[Draft] = []
    stack: list[tuple[int, int]] = []
    seen: set[tuple[int | None, str]] = set()
    for index, entry in enumerate(doc.outline):
        while stack and stack[-1][0] >= entry.level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        stack.append((entry.level, index))
        title = _title_key(entry.title)
        if not title:
            continue
        if (parent, title) in seen:
            source = SourceRef(doc.path.name, page=entry.page)
            side = marked_side(source, entry.title, kind=BlockKind.HEADING)
            above = None if parent is None else doc.outline[parent].title
            drafts.append(_duplicate(side, above))
        else:
            seen.add((parent, title))
    return drafts


def _search_key(text: str) -> str:
    """Return a text as the blocks show it, its blanks collapsed."""
    return " ".join(normalize_display(text, Counter()).split())


class _Line:
    """The blocks of a page that start on one source line.

    A reference of that line is sought in their texts, blanks collapsed,
    after the reference before it: a page written on one line tells its
    blocks apart by their text only. The searches of a line read at most
    twice its text in all; then each reads `TIE_SCAN_CHARS` characters at
    most, so that the placing stays linear.
    """

    __slots__ = ("budget", "cursor", "indices", "starts", "text")

    def __init__(
        self,
        blocks: Sequence[Block],
        indices: Sequence[int],
    ) -> None:
        self.indices = indices
        self.starts: list[int] = []
        texts: list[str] = []
        offset = 0
        for index in indices:
            text = " ".join(blocks[index].text.split())
            self.starts.append(offset)
            texts.append(text)
            offset += len(text) + len(_JOIN)
        self.text = _JOIN.join(texts)
        self.cursor = 0
        self.budget = 2 * len(self.text)

    def place(self, spot: Spot) -> int:
        """Return the block of a reference of the line, by its text.

        Returns:
            The position of the block that holds the text of the reference,
            after the reference before it; else that of the block of the
            reference before it.
        """
        key = _search_key(spot.text)
        if not key:
            found = -1
        elif self.budget > 0:
            found = self.text.find(key, self.cursor)
            end = len(self.text) if found < 0 else found + len(key)
            self.budget -= end - self.cursor
        else:
            end = self.cursor + TIE_SCAN_CHARS
            found = self.text.find(key, self.cursor, end)
        if found >= 0:
            self.cursor = found + len(key)
        else:
            found = self.cursor
        return self.indices[bisect_right(self.starts, found) - 1]


def _place_spots(
    blocks: Sequence[Block],
    pages: Sequence[HtmlPageFacts],
) -> tuple[dict[int, list[Spot]], dict[str, list[Spot]]]:
    """Find the block of each reference that the DOM left without a target.

    A reference belongs to the last block of its page that starts on its
    source line or before it; among the blocks that start on that same
    line, `_Line` tells by the text.

    Returns:
        The references of each block, by its position, and those of each
        page that no block can hold: without a source line, or above the
        first block of the page.
    """
    files: dict[str, list[tuple[int, int]]] = {}
    for index, block in enumerate(blocks):
        line = block.source.line
        if line is not None:
            files.setdefault(block.source.file, []).append((line, index))
    placed: dict[int, list[Spot]] = {}
    lost: dict[str, list[Spot]] = {}
    for page in pages:
        if not page.unresolved:
            continue
        lined = sorted(files.get(page.file, []))
        lines = [line for line, _ in lined]
        ties: dict[int, _Line] = {}
        for spot in page.unresolved:
            end = 0 if spot.line is None else bisect_right(lines, spot.line)
            if not end:
                lost.setdefault(page.file, []).append(spot)
                continue
            first = bisect_left(lines, lines[end - 1])
            if end - first == 1:
                index = lined[first][1]
            else:
                if first not in ties:
                    ties[first] = _Line(
                        blocks,
                        [index for _, index in lined[first:end]],
                    )
                index = ties[first].place(spot)
            placed.setdefault(index, []).append(spot)
    return placed, lost


def _reference_side(block: Block, spot: Spot) -> Side:
    """Return the side of a block that marks a reference of its DOM.

    The reference is sought in the text of the block, in the form that
    blocks show, any run of blanks matching any run of blanks.
    """
    words = normalize_display(spot.text, Counter()).split()
    if len(words) == 1:
        start = block.text.find(words[0])
        span = None if start < 0 else (start, start + len(words[0]))
    else:
        pattern = re.compile(r"\s+".join(map(re.escape, words)))
        match = pattern.search(block.text) if words else None
        span = None if match is None else match.span()
    if span is None:
        return block_side(block)
    return marked_span(block.source, block.text, span, kind=block.kind)


def _xref(side: Side, spots: Sequence[Spot]) -> Draft:
    """Return the draft of the references of the DOM that a side shows."""
    if len(spots) == 1:
        text = normalize_display(spots[0].text, Counter())
        detail = f'reference "{quote(text)}" not resolved'
    else:
        detail = f"{len(spots)} unresolved reference(s) in this block"
    return Draft(_UNRESOLVED, None, side, None, detail, _XREF)


def _lost_xrefs(file: str, spots: Sequence[Spot]) -> list[Draft]:
    """Return the drafts of the references of the DOM that no block shows."""
    drafts: list[Draft] = []
    for spot in spots:
        text = normalize_display(spot.text, Counter())
        side = marked_side(SourceRef(file, line=spot.line), text, text)
        drafts.append(_xref(side, (spot,)))
    return drafts


def _text_references(
    block: Block,
    patterns: Sequence[tuple[str, re.Pattern[str]]],
) -> list[Draft]:
    """Return one draft per kind of unresolved reference that a text holds.

    A reference that touches the inline code of the block is code, such as
    the ``??`` operator.
    """
    if block.kind in _NO_REFERENCES:
        return []
    drafts: list[Draft] = []
    code: list[Span] | None = None  # Placed once, if a pattern matches.
    for name, pattern in patterns:
        matches = [
            m
            for m in pattern.finditer(block.text)
            if pattern is not MARKDOWN_REFERENCE or _is_reference(m[1])
        ]
        if matches and block.code_spans:
            code = code_ranges(block) if code is None else code
            matches = [m for m in matches if not touches(m.span(), code)]
        if matches:
            side = marked_span(
                block.source,
                block.text,
                matches[0].span(),
                kind=block.kind,
            )
            detail = f"{len(matches)} unresolved reference(s) in this block"
            drafts.append(Draft(_UNRESOLVED, None, side, None, detail, name))
    return drafts


def _is_reference(text: str) -> bool:
    """Return whether the text of a ``[text][label]`` can name something.

    It holds a letter and is not a character class such as ``0-9a-f``.
    """
    return bool(_LETTER.search(text)) and not CHARACTER_CLASS.fullmatch(text)


def _unresolved_references(doc: Document) -> list[Draft]:
    """Return the drafts of the references that lead nowhere.

    The drafts of a block come in the order of its patterns, then that of
    the references of its DOM; the references of a page that no block shows
    follow the last block of the page, or, without blocks, the other drafts.
    """
    patterns = [("??", UNRESOLVED_MARK), ("[?]", MISSING_CITATION)]
    if doc.format == "html":
        patterns.append(("[text][label]", MARKDOWN_REFERENCE))
    blocks = doc.blocks
    placed, lost = _place_spots(blocks, doc.html_pages)
    last = {block.source.file: index for index, block in enumerate(blocks)}
    drafts: list[Draft] = []
    for index, block in enumerate(blocks):
        drafts += _text_references(block, patterns)
        spots = placed.get(index)
        if spots:
            drafts.append(_xref(_reference_side(block, spots[0]), spots))
        file = block.source.file
        if last[file] == index and file in lost:
            drafts += _lost_xrefs(file, lost[file])
    for page in doc.html_pages:
        if page.file not in last and page.file in lost:
            drafts += _lost_xrefs(page.file, lost[page.file])
    return drafts


def _captioned(blocks: Sequence[Block], index: int) -> bool:
    """Return whether a caption block of its page is just around a block."""
    file = blocks[index].source.file
    return any(
        0 <= other < len(blocks)
        and blocks[other].kind is BlockKind.CAPTION
        and blocks[other].source.file == file
        for other in (index - 1, index + 1)
    )


def _same_figure(first: Block, second: Block) -> bool:
    """Return whether two blocks of a PDF are images of one figure."""
    return (
        first.kind is BlockKind.IMAGE
        and second.kind is BlockKind.IMAGE
        and first.source.page == second.source.page
    )


def _caption_near(
    blocks: Sequence[Block],
    page: int | None,
    steps: range,
) -> bool:
    """Return whether a caption of a page is reached along some blocks.

    A heading, a table, an image or a block of another page stops the walk.
    """
    for index in steps:
        block = blocks[index]
        if block.source.page != page or block.kind in _CAPTION_STOPS:
            return False
        if block.kind is BlockKind.CAPTION:
            return True
    return False


def _figure_bounds(blocks: Sequence[Block], index: int) -> tuple[int, int]:
    """Return the first and the last image of the figure of an image."""
    start = end = index
    while start > 0 and _same_figure(blocks[start - 1], blocks[start]):
        start -= 1
    while end + 1 < len(blocks) and _same_figure(blocks[end], blocks[end + 1]):
        end += 1
    return start, end


def _pdf_captioned(
    blocks: Sequence[Block],
    figures: Sequence[int],
) -> list[bool]:
    """Return whether each figure of a PDF has a caption near it.

    The images that follow one another on a page make one figure. It is
    captioned when a caption of its page is reached, `CAPTION_WINDOW`
    blocks at most before its first image or after its last, as
    `_caption_near` walks: the reading order of a PDF may set a few blocks,
    such as the rows of a table, between a figure and its caption.
    """
    captioned: list[bool] = []
    bounds, found = (-1, -1), False
    for index in figures:
        if not bounds[0] <= index <= bounds[1]:
            bounds = _figure_bounds(blocks, index)
            start, end = bounds
            page = blocks[index].source.page
            before = range(start - 1, max(start - 1 - CAPTION_WINDOW, -1), -1)
            after = range(end + 1, min(end + 1 + CAPTION_WINDOW, len(blocks)))
            found = _caption_near(blocks, page, before) or _caption_near(
                blocks,
                page,
                after,
            )
        captioned.append(found)
    return captioned


def _uncaptioned(
    indices: Sequence[int],
    captioned: Sequence[bool],
) -> list[int]:
    """Return the blocks without caption, if any of the blocks has one."""
    if not any(captioned):
        return []
    return [
        index for index, has in zip(indices, captioned, strict=True) if not has
    ]


def _caption_drafts(doc: Document) -> list[Draft]:
    """Return the figures, then the tables, without caption.

    The images removed as page chrome are no figures. A table, or a figure
    of an HTML page, is captioned by a caption just before or after it; a
    figure of a PDF, as `_pdf_captioned` tells.
    """
    blocks = doc.blocks
    figures = [
        index
        for index, block in enumerate(blocks)
        if block.kind is BlockKind.IMAGE
        and (
            block.image is None or block.image.digest not in doc.chrome_images
        )
    ]
    tables = [
        index
        for index, block in enumerate(blocks)
        if block.kind is BlockKind.TABLE
    ]
    if doc.format == "pdf":
        figured = _pdf_captioned(blocks, figures)
    else:
        figured = [_captioned(blocks, index) for index in figures]
    drafts = [
        Draft(
            _FIGURE,
            None,
            image_side(blocks[index]),
            None,
            pattern="figure",
        )
        for index in _uncaptioned(figures, figured)
    ]
    tabled = [_captioned(blocks, index) for index in tables]
    drafts += [
        Draft(_TABLE, None, block_side(blocks[index]), None, pattern="table")
        for index in _uncaptioned(tables, tabled)
    ]
    return drafts


def _orphan_pages(doc: Document) -> list[Draft]:
    """Return the pages of a site that no other page links to.

    The first page of the reading order, and a page of headings and images
    only, are never orphans.
    """
    pages = doc.html_pages
    if doc.format != "html" or len(pages) < _SITE_MIN_PAGES:
        return []
    linked = {
        target
        for page in pages
        for target in page.links_to
        if target != page.file
    }
    content = {
        block.source.file
        for block in doc.blocks
        if block.kind not in _NO_CONTENT
    }
    return [
        Draft(
            _ORPHAN,
            None,
            marked_side(SourceRef(page.file), printable(page.file)),
            None,
            _NOT_LINKED,
            "orphan",
        )
        for page in pages[1:]
        if page.file not in linked and page.file in content
    ]


def structure_drafts(doc: Document) -> list[Draft]:
    """Return the drafts of the structure of a document.

    Args:
        doc: The audited document.

    Returns:
        The drafts of the skipped heading levels, the empty sections, the
        identical sibling headings, the unresolved references, the figures
        then the tables without caption, and the orphan pages, each in the
        order of the document.
    """
    pdf = doc.format == "pdf"
    return [
        *(_tagged_skips(doc) if pdf else _html_skips(doc.blocks)),
        *_empty_sections(doc),
        *(_outline_duplicates(doc) if pdf else _html_duplicates(doc.blocks)),
        *_unresolved_references(doc),
        *_caption_drafts(doc),
        *_orphan_pages(doc),
    ]
