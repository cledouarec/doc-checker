# Copyright 2026 Christophe Le Douarec
"""Find the ruled tables of the pages of a PDF, with pdfplumber.

pdfplumber, a second parser of the same untrusted file, comes with the
pdf-tables extra: it is imported only when the tables are asked for, by
`require_pdfplumber`. It finds the grids that the rules of a page draw. A grid
is kept as a table if it has 2 rows and 2 columns at least, every cell of each
row and column, half of its cells at least holding text, no cell whose text
pdfminer could not map to Unicode, and its text off the edges of its columns,
which the labels of a drawing over help lines touch; and if it lies within the
crop box of the page, over no vector figure. Its box and the edges of its
columns are given as the reader gives positions: from the lower left corner of
the crop box, y going up.

A page is not searched, only counted, if its ``/Rotate`` turns it, if its crop
box is not the size that the reader found, or if it is beyond a budget. Before
pdfminer reads a page, the work of its reading is counted from its content
streams and the forms that they draw, as `PageWork` of `table_work` tells,
against `MAX_TABLE_PAGE_WORK`. Once read, the page may hold no more than
`MAX_TABLE_PAGE_CHARS` characters and `MAX_TABLE_PAGE_EDGES` rules, which
pdfplumber may cross `MAX_TABLE_PAGE_CROSSINGS` times at most. pdfminer and
pdfplumber say nothing while they read. An error on a page is a warning for
that page; an error while the file is opened, or pages that are not those of
the reader, are one warning, and no table is read.

These budgets are a fast first filter, and `find_tables` runs the search under
them alone. `read_tables` of `table_process` bounds it: it runs `find_tables`
in a process of its own, whose CPU time, memory and duration are limited for
the whole document, and which is killed past them, so that no file can make
the search run longer.
"""

import logging
import re
import warnings
from bisect import bisect_left, bisect_right
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final, Literal

from doc_checker.errors import ConfigError
from doc_checker.extract.pdf.table_work import (
    MAX_TABLE_PAGE_WORK,
    OverWorkError,
    PageWork,
)
from doc_checker.text.normalize import ELLIPSIS, printable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import ModuleType

    from pdfplumber.page import Page
    from pdfplumber.table import Table, TableSettings

# A box on a page: its left edge, its bottom, its right edge and its top, in
# points, from the lower left corner of the crop box of the page, y going up.
type Box = tuple[float, float, float, float]
# Why a page was not searched: beyond `MAX_TABLE_PAGE_WORK` or
# `MAX_TABLE_PAGE_CHARS`, beyond `MAX_TABLE_PAGE_EDGES` or
# `MAX_TABLE_PAGE_CROSSINGS`, turned by its ``/Rotate``, or of another size
# than the reader found.
type SkipReason = Literal["dense", "rules", "rotated", "size"]

# The tables of a document by position of their page, the warnings, and the
# number of the pages not searched by reason, as `find_tables` gives them.
type TableReading = tuple[
    dict[int, list[PdfTable]],
    list[str],
    dict[SkipReason, int],
]

SKIP_REASONS: Final[tuple[SkipReason, ...]] = (
    "dense",
    "rules",
    "rotated",
    "size",
)
# Most characters of a page that is searched: pdfplumber reads each of them.
MAX_TABLE_PAGE_CHARS: Final = 20_000
# Most rules of a page that is searched, as pdfplumber counts them: its lines,
# the four sides of its rectangles and the segments of its curves. Pages of
# ruled tables made by pdflatex and reportlab have 19 to 337; the 1,323 pages
# of the PGF manual have 79 as a median, 408 at the 95th percentile and 1,156
# at the 99th, and three pages of drawings have more.
MAX_TABLE_PAGE_EDGES: Final = 2_000
# Most pairs of a vertical and a horizontal rule of a page that is searched,
# once pdfplumber has snapped and joined its rules: pdfplumber crosses each
# pair, then works on the crossings in a time that grows as their square, so
# that a page of 1,000 rules across 1,000 others would take hours. Pages of
# ruled tables have 196 pairs at most, and tables of 20,000 characters a page
# 1,539. At the limit, a full grid takes 57 ms at most without text, and
# 0.16 s with two digits in each cell, 5 times less than the time that
# `read_tables` allows the page; 0.13 s with one digit, 3.7 times less. A page
# that this time could not pay for is counted, not searched.
MAX_TABLE_PAGE_CROSSINGS: Final = 1_600
# Least gap between a character of a table, not blank, and the edge of a
# column, in points: the cells of a table pad their text, where a drawing over
# help lines puts its labels on them. The ruled tables of fpdf2 keep 2.8, those
# of pdflatex and reportlab 6, and the help lines of the PGF manual 0 or less.
RULE_GAP: Final = 1.0
# Largest gap between the edges of two columns that continue each other.
COLUMN_TOLERANCE: Final = 2.0
# Largest gap between the sizes of a page that pdfplumber and the reader find.
SIZE_TOLERANCE: Final = 1.0
REASON_LIMIT: Final = 80  # Most characters of an error of pdfplumber told.

_NO_EXTRA: Final = (
    "reading the tables of PDF documents needs the pdf-tables extra:"
    " pip install 'doc-checker[pdf-tables]'"
)
NOT_READ: Final = "tables not read: "
_NO_MATCH: Final = f"{NOT_READ}the pages do not match"
_LEAST_EDGES: Final = 3  # The edges of two columns, or of two rows.
# The text that pdfminer gives a glyph without Unicode mapping.
_UNMAPPED: Final = "(cid:"
# A memory address in an error, which differs from one run to the next.
_ADDRESS: Final = re.compile(r"0x[0-9a-fA-F]+")

# pdfminer logs what it finds amiss in a file, and pdfplumber too, which are
# no concern of the user of a command line: their messages are dropped.
_PDFMINER_LOGGER: Final = logging.getLogger("pdfminer")
_PDFMINER_LOGGER.addHandler(logging.NullHandler())
_PDFMINER_LOGGER.propagate = False
_PDFPLUMBER_LOGGER: Final = logging.getLogger("pdfplumber")
_PDFPLUMBER_LOGGER.addHandler(logging.NullHandler())
_PDFPLUMBER_LOGGER.propagate = False


@dataclass(frozen=True, slots=True)
class PdfTable:
    """A ruled table of a page.

    Attributes:
        page: Position of the page, from 0.
        box: The box of the table.
        columns: The edges of its columns, from left to right: one more than
            its columns, in the coordinates of `box`.
        rows: The text of each cell of each row, from the top down, each run
            of blanks a single space.
        row_edges: The edges of its rows, from the top down: one more than
            its rows, in the coordinates of `box`; none if not known.
    """

    page: int
    box: Box
    columns: tuple[float, ...]
    rows: tuple[tuple[str, ...], ...]
    row_edges: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class TablePage:
    """A page to search for tables, as the reader found it.

    Attributes:
        index: Position of the page, from 0.
        width: Width of its crop box, in points.
        height: Height of its crop box, in points.
        figures: The boxes of its vector figures.
        chars: The number of its characters, which the time that
            `read_tables` allows the page grows with, up to
            `MAX_TABLE_PAGE_CHARS`.
    """

    index: int
    width: float
    height: float
    figures: tuple[Box, ...]
    chars: int


def require_pdfplumber() -> ModuleType:
    """Import pdfplumber, which the pdf-tables extra installs.

    Returns:
        The pdfplumber package, its ``table`` module imported.

    Raises:
        ConfigError: If pdfplumber cannot be imported.
    """
    try:
        import pdfplumber.table  # ruff: ignore[import-outside-top-level] - optional extra
    except ImportError:
        raise ConfigError(_NO_EXTRA) from None
    return pdfplumber


def _reason(exc: Exception) -> str:
    """Tell an error of pdfminer or pdfplumber, the same from run to run.

    Args:
        exc: The error.

    Returns:
        Its type and its message, if it has one, each memory address written
        ``0x`` and an ellipsis, printable and cut at `REASON_LIMIT`
        characters.
    """
    message = _ADDRESS.sub(f"0x{ELLIPSIS}", str(exc))
    name = type(exc).__name__
    return printable(
        f"{name}: {message}" if message else name,
        limit=REASON_LIMIT,
    )


def rule_crossings(page: Page, settings: TableSettings) -> int:
    """Count the pairs of rules of a page that pdfplumber would cross.

    The rules are snapped, joined and filtered as `Page.find_tables` of
    pdfplumber does with the same settings.

    Args:
        page: The page, as pdfplumber reads it.
        settings: The settings of the search of tables.

    Returns:
        The number of its vertical rules times that of its horizontal ones.
    """
    plumber = require_pdfplumber()
    least = settings.edge_min_length_prefilter
    rules = plumber.table.merge_edges(
        [
            *plumber.utils.filter_edges(page.edges, "v", min_length=least),
            *plumber.utils.filter_edges(page.edges, "h", min_length=least),
        ],
        snap_x_tolerance=settings.snap_x_tolerance,
        snap_y_tolerance=settings.snap_y_tolerance,
        join_x_tolerance=settings.join_x_tolerance,
        join_y_tolerance=settings.join_y_tolerance,
    )
    rules = plumber.utils.filter_edges(
        rules,
        min_length=settings.edge_min_length,
    )
    vertical = sum(rule["orientation"] == "v" for rule in rules)
    return vertical * (len(rules) - vertical)


def _skip_reason(
    page: Page,
    wanted: TablePage,
    settings: TableSettings,
) -> SkipReason | None:
    """Tell why a page is not searched, if it is not.

    Args:
        page: The page, as pdfplumber reads it.
        wanted: The page, as the reader found it.
        settings: The settings of the search of tables.

    Returns:
        ``rotated`` if its ``/Rotate`` turns it, ``size`` if its crop box is
        not the size of `wanted`, to `SIZE_TOLERANCE`, ``dense`` beyond
        `MAX_TABLE_PAGE_WORK`, counted before pdfminer reads the page, or
        beyond `MAX_TABLE_PAGE_CHARS`, ``rules`` beyond
        `MAX_TABLE_PAGE_EDGES` or `MAX_TABLE_PAGE_CROSSINGS`, each limit as
        the module holds it when called; else None.
    """
    if page.rotation:
        return "rotated"
    x0, top, x1, bottom = page.cropbox
    if (
        abs(x1 - x0 - wanted.width) > SIZE_TOLERANCE
        or abs(bottom - top - wanted.height) > SIZE_TOLERANCE
    ):
        return "size"
    try:
        PageWork(MAX_TABLE_PAGE_WORK).read(page.page_obj)
    except OverWorkError:
        return "dense"
    if len(page.chars) > MAX_TABLE_PAGE_CHARS:
        return "dense"
    if (
        len(page.edges) > MAX_TABLE_PAGE_EDGES
        or rule_crossings(page, settings) > MAX_TABLE_PAGE_CROSSINGS
    ):
        return "rules"
    return None


def _grid(table: Table) -> tuple[list[float], list[float]] | None:
    """Return the column edges and the row edges of a full grid.

    Args:
        table: A table that pdfplumber found.

    Returns:
        The left and right edges of its columns, then the tops and bottoms
        of its rows, in the coordinates of pdfplumber, if each of its cells
        is the meeting of a column and a row, and each such meeting a cell,
        with 2 rows and 2 columns at least; else None.
    """
    cells = {tuple(cell) for cell in table.cells}
    xs = sorted({cell[0] for cell in cells} | {cell[2] for cell in cells})
    ys = sorted({cell[1] for cell in cells} | {cell[3] for cell in cells})
    if len(xs) < _LEAST_EDGES or len(ys) < _LEAST_EDGES:
        return None
    grid = {
        (left, top, right, bottom)
        for left, right in pairwise(xs)
        for top, bottom in pairwise(ys)
    }
    return (xs, ys) if cells == grid else None


def _rows(
    plumber: ModuleType,
    chars: Sequence[dict[str, Any]],
    xs: Sequence[float],
    ys: Sequence[float],
) -> tuple[tuple[str, ...], ...] | None:
    """Read the text of each cell of a grid.

    A character is in the cell where its middle lies, as pdfplumber tells;
    the characters of a cell are joined by pdfplumber. A character that is
    not blank, and comes closer than `RULE_GAP` to the edge of a column, or
    over it, is a label of a drawing: the cells of a table pad their text.

    Args:
        plumber: The pdfplumber package.
        chars: The characters of the page, as pdfplumber reads them.
        xs: The edges of the columns of the grid, from left to right.
        ys: The edges of its rows, from the top down.

    Returns:
        The text of each cell of each row, each run of blanks a single
        space, if no character between the top and the bottom of the grid
        comes too close to the edge of a column, half of the cells at least
        hold text, and none holds a glyph without Unicode mapping; else
        None.
    """
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for char in chars:
        down = (char["top"] + char["bottom"]) / 2
        if not ys[0] <= down < ys[-1]:
            continue
        left, right = char["x0"], char["x1"]
        edge = bisect_left(xs, left - RULE_GAP)
        if (
            edge < len(xs)
            and xs[edge] < right + RULE_GAP
            and char["text"].strip()
        ):
            return None
        across = (left + right) / 2
        if xs[0] <= across < xs[-1]:
            cell = (bisect_right(ys, down) - 1, bisect_right(xs, across) - 1)
            cells.setdefault(cell, []).append(char)
    rows = tuple(
        tuple(
            " ".join(plumber.utils.extract_text(cells[row, column]).split())
            if (row, column) in cells
            else ""
            for column in range(len(xs) - 1)
        )
        for row in range(len(ys) - 1)
    )
    texts = [text for row in rows for text in row]
    if any(_UNMAPPED in text for text in texts):
        return None
    if 2 * sum(map(bool, texts)) < len(texts):
        return None
    return rows


def _overlaps(box: Box, other: Box) -> bool:
    """Tell whether two boxes overlap.

    Args:
        box: A box.
        other: Another box.

    Returns:
        Whether they share an area, not only an edge.
    """
    return (
        box[0] < other[2]
        and other[0] < box[2]
        and box[1] < other[3]
        and other[1] < box[3]
    )


def _page_tables(
    plumber: ModuleType,
    page: Page,
    wanted: TablePage,
    settings: TableSettings,
) -> list[PdfTable]:
    """Find the tables of a page.

    Args:
        plumber: The pdfplumber package.
        page: The page, as pdfplumber reads it.
        wanted: The page, as the reader found it.
        settings: The settings of the search of tables.

    Returns:
        The grids of the page that are tables, as the module tells, from the
        top of the page down, then from left to right.
    """
    left, top, right, bottom = page.cropbox
    found = []
    for table in page.find_tables(table_settings=settings):
        grid = _grid(table)
        if grid is None:
            continue
        xs, ys = grid
        if (
            xs[0] < left - SIZE_TOLERANCE
            or xs[-1] > right + SIZE_TOLERANCE
            or ys[0] < top - SIZE_TOLERANCE
            or ys[-1] > bottom + SIZE_TOLERANCE
        ):
            continue
        box = (xs[0] - left, bottom - ys[-1], xs[-1] - left, bottom - ys[0])
        if any(_overlaps(box, figure) for figure in wanted.figures):
            continue
        rows = _rows(plumber, page.chars, xs, ys)
        if rows is not None:
            columns = tuple(x - left for x in xs)
            edges = tuple(bottom - y for y in ys)
            found.append(PdfTable(wanted.index, box, columns, rows, edges))
    return sorted(found, key=lambda table: (-table.box[3], table.box[0]))


def check_pages(pages: Sequence[TablePage], page_count: int) -> None:
    """Refuse the pages that are not pages of the file.

    Args:
        pages: The pages to search.
        page_count: The number of pages that the reader found in the file.

    Raises:
        ValueError: If a page is not one of the `page_count` pages.
    """
    for wanted in pages:
        if not 0 <= wanted.index < page_count:
            msg = f"page {wanted.index} is not one of {page_count} pages"
            raise ValueError(msg)


def find_tables(
    path: Path,
    pages: Sequence[TablePage],
    *,
    page_count: int,
) -> TableReading:
    """Find the ruled tables of some pages of a PDF, with pdfplumber.

    The search runs in this process, without the limits of `read_tables`,
    which runs it in a process of its own. pdfplumber must be installed, as
    `require_pdfplumber` tells. The pages are read one at a time, and each
    page drops what pdfplumber read of it once searched. pdfminer and
    pdfplumber say nothing while they read.

    Args:
        path: The PDF file.
        pages: The pages to search, as the reader found them.
        page_count: The number of pages that the reader found in the file.

    Returns:
        The tables of each page that has some, by position of the page, as
        `_page_tables` finds them; the warnings, one for each page where
        pdfplumber failed, ``page N: tables not read:`` and the error, in
        the order of `pages`; and the number of the pages not searched, by
        reason, as `_skip_reason` tells, in the order of `SKIP_REASONS`.
        If pdfplumber fails to open the file or to list its pages, or does
        not find `page_count` pages, the only warning tells it, and no table
        is read. A page that is not one of the pages of the file raises a
        `ValueError`, as `check_pages` tells.

    Raises:
        MemoryError: If memory runs out, which ends the search.
    """
    check_pages(pages, page_count)
    plumber = require_pdfplumber()
    settings = plumber.table.TableSettings()
    found: dict[int, list[PdfTable]] = {}
    notes: list[str] = []
    skipped: Counter[SkipReason] = Counter()
    if not pages:
        return found, notes, {}
    with ExitStack() as stack:
        stack.enter_context(warnings.catch_warnings(action="ignore"))
        try:
            pdf = stack.enter_context(plumber.open(path))
            plumber_pages = pdf.pages
        except MemoryError:
            raise
        except Exception as exc:  # ruff: ignore[blind-except] - pdfminer raises many kinds of errors on hostile files
            return {}, [f"{NOT_READ}{_reason(exc)}"], {}
        if len(plumber_pages) != page_count:
            return {}, [_NO_MATCH], {}
        for wanted in pages:
            page = plumber_pages[wanted.index]
            try:
                reason = _skip_reason(page, wanted, settings)
                tables = (
                    []
                    if reason
                    else _page_tables(plumber, page, wanted, settings)
                )
            except MemoryError:
                raise
            except Exception as exc:  # ruff: ignore[blind-except] - pdfminer raises many kinds of errors on hostile files
                notes.append(
                    f"page {wanted.index + 1}: {NOT_READ}{_reason(exc)}",
                )
                continue
            finally:
                page.close()
            if reason:
                skipped[reason] += 1
            elif tables:
                found[wanted.index] = tables
    counts = {reason: skipped[reason] for reason in SKIP_REASONS}
    return found, notes, {key: n for key, n in counts.items() if n}


def continues(previous: PdfTable, table: PdfTable) -> bool:
    """Tell whether a table continues the one before it, on the next page.

    Args:
        previous: The table before.
        table: The table that may continue it.

    Returns:
        Whether `table` is on the page after that of `previous`, with as
        many columns, each edge at `COLUMN_TOLERANCE` at most from the same
        edge of `previous`.
    """
    return (
        table.page == previous.page + 1
        and len(table.columns) == len(previous.columns)
        and all(
            abs(edge - before) <= COLUMN_TOLERANCE
            for edge, before in zip(
                table.columns,
                previous.columns,
                strict=True,
            )
        )
    )
