# Copyright 2026 Christophe Le Douarec
"""Make the blocks of the ruled tables of a PDF, with the pdf-tables extra.

The pages that the reader read as they lie, neither turned nor too crowded
for their drawings to be looked for nor too dense, are searched for ruled
tables by `read_tables` of `table_process`, in a process of its own. A table
in the top or the bottom band of its page is no table. Any other table takes
the place of the segments of the reader in its box only when its cells hold
their text exactly, read as the cells are, and no segment crosses its edge;
otherwise it stays text, and the loader warns about it. The segments of the
tables kept leave the flow of the text before the running headers are
looked for, and come back for the rule of the scanned pages and the facts
of the pages. A table continued on the next page, with no text or image
between, is joined to its start, and the header row that it repeats is
removed. The tables kept also break the flow of the text: the text before a
table never joins the text after it.
"""

import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Protocol

from doc_checker.extract.common import normalized_block
from doc_checker.extract.pdf import table_process, tables
from doc_checker.model.document import CELL_SEPARATOR, BlockKind, SourceRef
from doc_checker.text.normalize import normalize_display, strict_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from doc_checker.config import PdfConfig
    from doc_checker.extract.pdf.layout import Line
    from doc_checker.extract.pdf.segments import Segment
    from doc_checker.extract.pdf.tables import (
        Box,
        PdfTable,
        SkipReason,
        TablePage,
    )
    from doc_checker.model.document import Block

# Most points by which a segment may cross the edge of a table: one whose
# centre lies in the table may run out of its box by as much, across, and any
# other may reach into it by as much, across or up. A segment that crosses
# further holds text both in and out of the table.
TABLE_OVERHANG: Final = 2.0

# What tells why pages were not searched for tables, by reason.
_SKIPPED: Final[Mapping[SkipReason, str]] = {
    "dense": "dense",
    "rules": "of many rules",
    "rotated": "turned",
    "size": "of another size",
}


class TablePageFacts(Protocol):
    """What the tables need of a page, as the loader keeps it.

    Attributes:
        label: Page label, such as ``iv``, or the page number from 1.
        width: Width of the crop box of the page turned upright, in points.
        height: Height of the crop box of the page turned upright, in points.
        segments: The segments of the page, in the order of their first
            character.
        chars: Number of characters that the page keeps.
        figures: The boxes of its vector figures, from the lower left corner
            of its crop box.
        turned: Whether the reader turned the page upright.
        crowded: Whether the page has too many paths for its drawings to be
            looked for.
    """

    label: str
    width: float
    height: float
    segments: Sequence[Segment]
    chars: int
    figures: Sequence[Box]
    turned: bool
    crowded: bool


@dataclass(frozen=True, slots=True)
class Placed:
    """An image or a table block, and where its image or its table lies.

    Attributes:
        page: Position of the page of the image or of the start of the
            table, from 0.
        top: Top edge of the image or of the table on its page turned
            upright, in points.
        x0: Left edge of the image or of the table.
        block: The block.
        bottom: Bottom edge of the image or of the start of the table.
    """

    page: int
    top: float
    x0: float
    block: Block
    bottom: float


def place_order(placed: Placed) -> tuple[int, float, float]:
    """Return the order of a placed block: page, top down, left to right."""
    return placed.page, -placed.top, placed.x0


@dataclass(frozen=True, slots=True)
class KeptTables:
    """The tables read from a document, and the segments that they take.

    Attributes:
        found: The tables kept, by page, then from the top of the page down.
        taken: The segments of each page whose centre lies in a table kept.
        warnings: Why tables were not read, then the pages not searched for
            them.
        refused: The label of the page of each table kept as text, as its
            cells do not hold the text of the page, in page order.
    """

    found: tuple[PdfTable, ...] = ()
    taken: tuple[frozenset[Segment], ...] = ()
    warnings: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()


def _beside(box: Box, line: Line) -> bool:
    """Tell whether a box and a line share a stretch across the page.

    Args:
        box: The box.
        line: The line.

    Returns:
        Whether they overlap from left to right, edges apart.
    """
    return box[0] < line.x1 and line.x0 < box[2]


def _under(box: Box, line: Line) -> bool:
    """Tell whether a box lies under a line, beside it.

    Args:
        box: The box.
        line: The line.

    Returns:
        Whether the top of the box lies at or under the baseline of the
        line, and they share a stretch across the page.
    """
    return box[3] <= line.baseline and _beside(box, line)


def _over(box: Box, line: Line) -> bool:
    """Tell whether a box lies over a line, beside it.

    Args:
        box: The box.
        line: The line.

    Returns:
        Whether the bottom of the box lies at or over the baseline of the
        line, and they share a stretch across the page.
    """
    return box[1] >= line.baseline and _beside(box, line)


@dataclass(frozen=True, slots=True)
class TableBreaks:
    """The tables kept in a document, as breaks in the flow of its text.

    Attributes:
        boxes: The boxes of the tables kept, by position of their page.
    """

    boxes: Mapping[int, tuple[Box, ...]]

    @classmethod
    def of(cls, found: Sequence[PdfTable]) -> TableBreaks:
        """Gather the boxes of some tables by page.

        Args:
            found: The tables kept.

        Returns:
            The breaks of the tables.
        """
        boxes: dict[int, list[Box]] = {}
        for table in found:
            boxes.setdefault(table.page, []).append(table.box)
        return cls({page: tuple(each) for page, each in boxes.items()})

    def __call__(self, previous: Line, line: Line) -> bool:
        """Tell whether a table lies between two lines of the reading order.

        On one page, where the line after lies under the line before, a
        table lies between them if it lies under the line before and over
        the line after, beside one of them. Where the line after rises, at a
        break such as a column break, a table lies between them if it lies
        under the line before, beside it, but not under the line after,
        beside it; or over the line after, beside it, but not over the line
        before, beside it: a table over both or under both is read before
        both or after both. Across a page break, a table lies between them
        if it lies under the line before on its page, beside it, or over the
        line after on its page, beside it, or on a page between theirs.

        Args:
            previous: The line before.
            line: The line after.

        Returns:
            Whether a table lies between them.
        """
        if previous.page == line.page:
            boxes = self.boxes.get(line.page, ())
            if line.baseline <= previous.baseline:
                return any(
                    box[3] <= previous.baseline
                    and box[1] >= line.baseline
                    and (_beside(box, previous) or _beside(box, line))
                    for box in boxes
                )
            return any(
                (_under(box, previous) and not _under(box, line))
                or (_over(box, line) and not _over(box, previous))
                for box in boxes
            )
        return (
            any(
                _under(box, previous)
                for box in self.boxes.get(previous.page, ())
            )
            or any(_over(box, line) for box in self.boxes.get(line.page, ()))
            or any(
                page in self.boxes
                for page in range(previous.page + 1, line.page)
            )
        )


def _table_pages(
    pages: Sequence[TablePageFacts],
    skipped: Counter[SkipReason],
) -> list[TablePage]:
    """Choose the pages to search for tables, and count those left out.

    A page without characters is neither searched nor counted: half the
    cells of a table at least hold text. Three kinds of pages are left out
    and counted. A page that the reader turned upright is ``rotated``: the
    tables are found on the page as it lies. A page whose drawings the
    reader did not look for is of many ``rules``: no figure would keep its
    drawings from being tables. A page of more characters than
    `MAX_TABLE_PAGE_CHARS` of the tables, as the module holds it when this
    is called, is ``dense``.

    Args:
        pages: The pages of the document.
        skipped: The counts of the pages left out, by reason, which go up.

    Returns:
        The pages to search, in order, as the reader found them.
    """
    limit = tables.MAX_TABLE_PAGE_CHARS
    wanted: list[TablePage] = []
    for index, page in enumerate(pages):
        if not page.chars:
            continue
        if page.turned:
            skipped["rotated"] += 1
        elif page.crowded:
            skipped["rules"] += 1
        elif page.chars > limit:
            skipped["dense"] += 1
        else:
            wanted.append(
                tables.TablePage(
                    index,
                    page.width,
                    page.height,
                    page.figures,
                    page.chars,
                ),
            )
    return wanted


def _centre_in(box: Box, segment: Segment) -> bool:
    """Tell whether the centre of a segment lies in a box.

    Args:
        box: The box.
        segment: The segment.

    Returns:
        Whether the middle of the segment, across and up, lies in the box,
        edges included.
    """
    x = (segment.x0 + segment.x1) / 2
    y = (segment.bottom + segment.top) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _crosses(box: Box, segment: Segment) -> bool:
    """Tell whether a segment crosses the edge of a table.

    Args:
        box: The box of the table.
        segment: A segment of its page.

    Returns:
        Whether the segment, if its centre lies in the box, runs out of it,
        across, by more than `TABLE_OVERHANG`; or, if not, reaches into it
        by more than `TABLE_OVERHANG` both across and up.
    """
    if _centre_in(box, segment):
        return (
            segment.x0 < box[0] - TABLE_OVERHANG
            or segment.x1 > box[2] + TABLE_OVERHANG
        )
    across = min(segment.x1, box[2]) - max(segment.x0, box[0])
    up = min(segment.top, box[3]) - max(segment.bottom, box[1])
    return across > TABLE_OVERHANG and up > TABLE_OVERHANG


def _in_a_band(table: PdfTable, height: float, cfg: PdfConfig) -> bool:
    """Tell whether a table lies in the header band or the footer band.

    Args:
        table: The table.
        height: The height of its page, in points.
        cfg: The PDF settings, which give the bands.

    Returns:
        Whether its box lies wholly within the top `header_band` or the
        bottom `footer_band` of the height of its page, as the ruled header
        of a page template does.
    """
    _, bottom, _, top = table.box
    return (
        bottom >= height * (1 - cfg.header_band)
        or top <= height * cfg.footer_band
    )


def _table_key(texts: Iterable[str]) -> str:
    """Return the key of some texts for the check of a table.

    Args:
        texts: The texts, in order.

    Returns:
        Their strict key, as the comparison makes it, without any blank:
        pdfminer and the reader may put the blanks between words elsewhere.
    """
    scratch: Counter[str] = Counter()
    text = normalize_display(" ".join(texts), scratch)
    return "".join(strict_key(text, scratch).split())


def _cell_order(table: PdfTable, segment: Segment) -> tuple[int, int, float]:
    """Tell where a segment of a table comes when its cells are read.

    Args:
        table: The table.
        segment: A segment whose centre lies in its box.

    Returns:
        The row of its middle, from the top, and the column of its left
        edge, from the left, as the cells of the table are read, then how
        low its baseline lies.
    """
    up = [-edge for edge in table.row_edges]
    row = bisect_right(up, -_middle(segment)) - 1
    column = bisect_right(table.columns, segment.x0) - 1
    return (
        min(max(row, 0), len(table.rows) - 1),
        min(max(column, 0), len(table.columns) - 2),
        -segment.baseline,
    )


def _with_blanks(
    rows: Sequence[Sequence[str]],
    text: str,
) -> tuple[tuple[str, ...], ...] | None:
    """Give the cells of some rows the blanks of the text of the reader.

    The text of the cells and that of the reader have the same key: the
    characters that count in it are the same, in the same order. Both are
    composed first, as the key composes them, so that an accent that PDFium
    gives apart from its letter goes with it. The text of the reader is then
    cut where the cells are, and each cell takes its part.

    Args:
        rows: The text of each cell of each row.
        text: The text of the reader in the box of the table, its segments
            read as the cells are, whose key is that of `rows`.

    Returns:
        The rows, each cell the part of `text`, composed, from the first to
        the last character that gives its key, as `_table_key` makes it,
        each run of blanks a single space; None if a character of `text`
        gives the key of two cells, or the keys differ.
    """
    text = unicodedata.normalize("NFC", text)
    units: dict[str, str] = {}  # What each character gives a key, once.
    keys: list[str] = []
    for char in text:
        unit = units.get(char)
        if unit is None:
            unit = units[char] = _table_key((char,))
        keys.append(unit)
    at = 0
    read: list[tuple[str, ...]] = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            want = _table_key((unicodedata.normalize("NFC", cell),))
            got = ""
            start = end = at
            while len(got) < len(want) and at < len(keys):
                if keys[at]:
                    start = at if not got else start
                    got += keys[at]
                    end = at + 1
                at += 1
            if got != want:
                return None
            cells.append(" ".join(text[start:end].split()))
        read.append(tuple(cells))
    return tuple(read)


def _read_rows(
    table: PdfTable,
    inside: Sequence[Segment],
) -> tuple[tuple[str, ...], ...] | None:
    """Return the rows of a table whose text the reader finds in its box.

    The text of a table comes from pdfminer, a second reading of the file:
    the table takes the place of the text of the reader only where they are
    the same. The segments are read as the cells are, row after row, each
    row from left to right, each cell from the top down, as `_cell_order`
    tells, then left to right.

    Args:
        table: The table.
        inside: The segments of its page whose centre lies in its box.

    Returns:
        The rows of the table, if the key of their cells, row after row, is
        that of the segments, as `_table_key` makes it; else, if the key of
        the rows but the first is that of the segments, those rows: the
        reader leaves out a header row marked as an artifact, as LibreOffice
        marks the header rows that it repeats; else None. None, too, if the
        segments hold no text. The cells of the rows returned take the
        blanks of the reader, as `_with_blanks` gives them, so that a table
        never changes where the words of the reader begin and end; they
        keep those of pdfminer only where a character would fall across two
        cells.
    """
    ordered = sorted(
        inside,
        key=lambda segment: (*_cell_order(table, segment), segment.x0),
    )
    text = " ".join(segment.text for segment in ordered)
    reader = _table_key((text,))
    if not reader:
        return None
    for rows in (table.rows, table.rows[1:]):
        if _table_key(cell for row in rows for cell in row) == reader:
            return _with_blanks(rows, text) or rows
    return None


def _skip_warnings(skipped: Mapping[SkipReason, int]) -> list[str]:
    """Warn about the pages that were not searched for tables.

    Args:
        skipped: The number of the pages not searched, by reason.

    Returns:
        One warning that counts them, and each reason, in the order of
        `SKIP_REASONS` of the tables; or none if every page was searched.
    """
    total = sum(skipped.values())
    if not total:
        return []
    reasons = ", ".join(
        f"{skipped[reason]} {_SKIPPED[reason]}"
        for reason in tables.SKIP_REASONS
        if skipped.get(reason)
    )
    return [f"{total} page(s) not searched for tables: {reasons}"]


def read_kept_tables(
    path: Path,
    pages: Sequence[TablePageFacts],
    cfg: PdfConfig,
) -> KeptTables:
    """Find the ruled tables of a document, and the segments that they take.

    The pages chosen by `_table_pages` are searched by `read_tables` of
    `table_process`, in a process of its own, if there is one at least. A
    table in a band of its page, as `_in_a_band` tells, is no table: its text
    stays in the flow, where the running headers and footers are looked for.
    A table that a segment crosses, as `_crosses` tells, or whose cells do
    not hold the text that the reader finds in its box, as `_read_rows`
    tells, is kept as text. A table kept holds the rows that `_read_rows`
    returns.

    Args:
        path: The PDF file.
        pages: The pages of the document.
        cfg: The PDF settings, which give the bands.

    Returns:
        The tables kept and the segments that each page gives them, with
        the warnings of the search, then the one of `_skip_warnings`, and
        the label of the page of each table kept as text.
    """
    skipped: Counter[SkipReason] = Counter()
    wanted = _table_pages(pages, skipped)
    found, notes, counted = (
        table_process.read_tables(path, wanted, page_count=len(pages))
        if wanted
        else ({}, [], {})
    )
    skipped.update(counted)
    kept: list[PdfTable] = []
    taken: list[set[Segment]] = [set() for _ in pages]
    refused: list[str] = []
    for index, page_tables in sorted(found.items()):
        page = pages[index]
        for table in page_tables:
            if _in_a_band(table, page.height, cfg):
                continue
            inside = [
                segment
                for segment in page.segments
                if segment not in taken[index]
                and _centre_in(table.box, segment)
            ]
            # The exact check of `_read_rows` refuses on its own a table that
            # a segment crosses, with its middle in the table or out of it:
            # the text of the segments in the box then has more, or less,
            # than the cells. `_crosses` stays as a defence in depth.
            rows = (
                None
                if any(_crosses(table.box, s) for s in page.segments)
                else _read_rows(table, inside)
            )
            if rows is None:
                refused.append(page.label)
            else:
                # A table kept without its first row loses its first edge.
                edges = table.row_edges[len(table.rows) - len(rows) :]
                kept.append(replace(table, rows=rows, row_edges=edges))
                taken[index].update(inside)
    return KeptTables(
        tuple(kept),
        tuple(frozenset(segments) for segments in taken),
        (*notes, *_skip_warnings(skipped)),
        tuple(refused),
    )


def _middle(segment: Segment) -> float:
    """Return the height of the middle of a segment, between its edges."""
    return (segment.bottom + segment.top) / 2


def _continued(
    previous: PdfTable,
    table: PdfTable,
    kept: Sequence[Sequence[Segment]],
    images: Mapping[int, Sequence[Placed]],
) -> bool:
    """Tell whether a table continues the one before it, nothing between.

    Args:
        previous: The table before.
        table: The table that may continue it.
        kept: The segments of each page, chrome and tables removed.
        images: The image blocks kept, by position of their page.

    Returns:
        Whether `table` continues `previous`, as `continues` of the tables
        tells, and no kept segment or image lies under `previous` on its
        page, nor over `table` on its page.
    """
    return (
        tables.continues(previous, table)
        and not any(
            _middle(segment) < previous.box[1]
            for segment in kept[previous.page]
        )
        and not any(
            _middle(segment) > table.box[3] for segment in kept[table.page]
        )
        and not any(
            image.top <= previous.box[1]
            for image in images.get(previous.page, ())
        )
        and not any(
            image.bottom >= table.box[3]
            for image in images.get(table.page, ())
        )
    )


def _table_block(
    pieces: Sequence[PdfTable],
    *,
    file: str,
    counter: Counter[str],
) -> Placed:
    """Make the block of a table, from its pieces on successive pages.

    Args:
        pieces: The table, then the pieces that continue it.
        file: The name of the PDF file, which the block points to.
        counter: The counts of the normalizations, in which `table_header`
            goes up by one for each piece whose first row, the same as the
            first row of the table, is removed.

    Returns:
        The table block, at the page and the top of its first piece. Its text
        is each row, its cells joined by `CELL_SEPARATOR`, one row a line.
    """
    first = pieces[0]
    rows = list(first.rows)
    for piece in pieces[1:]:
        repeated = piece.rows[:1] == first.rows[:1]
        if repeated:
            counter["table_header"] += 1
        rows += piece.rows[1:] if repeated else piece.rows
    text = "\n".join(CELL_SEPARATOR.join(row) for row in rows)
    block = normalized_block(
        BlockKind.TABLE,
        text,
        SourceRef(file, page=first.page + 1),
        counter,
    )
    return Placed(first.page, first.box[3], first.box[0], block, first.box[1])


def _table_blocks(
    found: Sequence[PdfTable],
    kept: Sequence[Sequence[Segment]],
    images: Sequence[Placed],
    *,
    file: str,
    counter: Counter[str],
) -> list[Placed]:
    """Join the tables continued from page to page, and make their blocks.

    Args:
        found: The tables kept, by page, then from the top of the page down.
        kept: The segments of each page, chrome and tables removed.
        images: The image blocks kept, in the order of `place_order`.
        file: The name of the PDF file, which the blocks point to.
        counter: The counts of the normalizations.

    Returns:
        One block for each table and the tables that continue it, as
        `_continued` tells, in the order of `place_order`.
    """
    by_page: dict[int, list[Placed]] = {}
    for image in images:
        by_page.setdefault(image.page, []).append(image)
    joined: list[list[PdfTable]] = []
    for table in found:
        if joined and _continued(joined[-1][-1], table, kept, by_page):
            joined[-1].append(table)
        else:
            joined.append([table])
    return [
        _table_block(pieces, file=file, counter=counter) for pieces in joined
    ]


def flow(
    pages: Sequence[TablePageFacts],
    found: KeptTables,
) -> list[Sequence[Segment]]:
    """Return the segments of each page that are not in a table.

    Args:
        pages: The pages of the document.
        found: The tables of the document.

    Returns:
        The segments of each page, but those that the tables take.
    """
    if not found.found:
        return [page.segments for page in pages]
    return [
        [segment for segment in page.segments if segment not in taken]
        for page, taken in zip(pages, found.taken, strict=True)
    ]


def with_tables(
    images: Sequence[Placed],
    found: KeptTables,
    kept: Sequence[Sequence[Segment]],
    *,
    file: str,
    counter: Counter[str],
) -> list[Placed]:
    """Add the table blocks of a document to its image blocks.

    Args:
        images: The image blocks, in the order of `place_order`.
        found: The tables of the document.
        kept: The segments of each page, chrome and tables removed.
        file: The name of the PDF file, which the blocks point to.
        counter: The counts of the normalizations.

    Returns:
        The image and the table blocks, as `_table_blocks` makes the latter,
        in the order of `place_order`, the images first where two are level.
    """
    if not found.found:
        return list(images)
    blocks = _table_blocks(
        found.found,
        kept,
        images,
        file=file,
        counter=counter,
    )
    return sorted([*images, *blocks], key=place_order)


def text_segments(
    pages: Sequence[TablePageFacts],
    kept: Sequence[Sequence[Segment]],
    found: KeptTables,
) -> list[Sequence[Segment]]:
    """Return the segments of each page that are text, those of tables too.

    Args:
        pages: The pages of the document.
        kept: The segments of each page, chrome and tables removed.
        found: The tables of the document.

    Returns:
        The kept segments of each page and those that its tables take, in
        the order of the page: the segments that the page is cut in to be
        read, and those that the rule of the scanned pages and the facts of
        the pages read.
    """
    if not found.found:
        return list(kept)
    text: list[Sequence[Segment]] = []
    for page, segments, taken in zip(pages, kept, found.taken, strict=True):
        read = taken.union(segments)
        text.append([segment for segment in page.segments if segment in read])
    return text


def table_breaks(found: KeptTables) -> TableBreaks | None:
    """Return the tables kept in a document as breaks in its text.

    Args:
        found: The tables of the document.

    Returns:
        The breaks of the tables, as `TableBreaks` tells them; None if no
        table is kept.
    """
    return TableBreaks.of(found.found) if found.found else None
