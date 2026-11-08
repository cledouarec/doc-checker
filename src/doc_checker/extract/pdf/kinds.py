# Copyright 2026 Christophe Le Douarec
"""Find the contents and index pages of a PDF, and the captions of its figures.

The contents and the index list the sections or the terms of a document, each
with the number of its page. Two builds of the same documentation do not break
their pages in the same places, so these numbers differ although the text does
not. The blocks of such pages get kinds of their own, `BlockKind.TOC` and
`BlockKind.INDEX`, so that the comparison can skip them.

A page of one of these kinds is made of entries, which are lines that end with
a page number. A title such as "Contents" tells which kind a page is, and the
pages after it keep that kind as long as they are made of entries too. The
numbers of a contents page never go back, whereas those of a table of settings
usually do, so a contents page goes on over a page only if its numbers are in
order. Without a title, a page is a contents page if its entries run to their
numbers along dot leaders.

A page that has no title, the pages that go on after a titled one included,
has one more condition: a line counts as an entry, or as a line with dot
leaders, only if the arabic number at its end is at most the count of the
pages of the document. A table of values with dot leaders, such as
"Max load . . . 500", is then no contents page in a document of fewer pages.
The order of the numbers is still read on every line that ends like an entry,
those above the count included, as a number that cannot be a page number
speaks against a contents. A page that has a title keeps its entries whatever
their numbers, as the pages of a volume may be numbered from 301. A roman
numeral always counts.

A list of figures or of tables, as LibreOffice writes it, is contents too,
whatever the rest of its page: a run of at least `TOC_MIN_LINES` blocks that
each start like a caption and end like an entry, with dot leaders or
without, right under the title of such a list. Their numbers are not bounded
by the count of the pages: LibreOffice may keep the page numbers that a list
had in the document it converts. Without that title, a run of captions is
no list, as captions stacked on a page may all end in a number or an
acronym; only a page of contents, as above, types such entries.
"""

import re
from itertools import pairwise
from typing import TYPE_CHECKING, Final

from doc_checker.model.document import BlockKind
from doc_checker.text.normalize import loose_key
from doc_checker.text.numbers import CAPTION_LABEL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.extract.pdf.layout import Line

# The titles of the pages, as `loose_key` writes them.
TOC_TITLES: Final = frozenset(
    {"contents", "table of contents", "table des matieres", "sommaire"},
)
INDEX_TITLES: Final = frozenset({"index"})
# The titles of the lists of figures and tables, as `loose_key` writes them.
FIGURE_LIST_TITLES: Final = frozenset(
    {
        "list of figures",
        "list of tables",
        "list of illustrations",
        "table of figures",
        "figure index",
        "illustration index",
        "table index",
        "liste des figures",
        "liste des tableaux",
        "liste des illustrations",
        "table des figures",
        "table des illustrations",
    },
)
TOC_MIN_LINES: Final = 3  # Fewest lines of a contents or an index page.
TOC_MIN_SHARE: Final = 0.5  # Smallest share of the lines that are entries.

# An entry ends with a page number, arabic or roman, that follows a blank or
# dot leaders. The dots may touch the number. Any line that ends with a word
# made of roman letters, such as "mix" or "did", looks like an entry too, and
# so does any line of a table that ends with a number, so this alone does not
# tell a page: it needs a title, or a page of that kind before it.
_TOC_ENTRY: Final = re.compile(
    r"(?:(?:\.\s*){3,}|\s)(?:\d{1,4}|[ivxlcdm]{1,6})$",
    re.IGNORECASE,
)
# An entry with dot leaders: at least three dots, with blanks or not between
# them, and then the number. Without a title, only these tell a page.
_LEADERS: Final = re.compile(
    r"(?:\.\s*){3,}(?:\d{1,4}|[ivxlcdm]{1,6})$",
    re.IGNORECASE,
)
# The arabic number at the end of an entry, which `_TOC_ENTRY` has matched
# already: a blank or a dot comes before the digits, and nothing after them.
_PAGE_NUMBER: Final = re.compile(r"\d+$")
# A caption starts with its label and the number of the figure or the table,
# which may have several parts, as in 2.1 or 3-2.
_CAPTION: Final = re.compile(
    rf"^{CAPTION_LABEL}\s+\d+(?:[.\-]\d+)*\b",
)


def _is_in_page_order(entries: Sequence[str]) -> bool:
    """Tell whether the page numbers of some entries never go back.

    Args:
        entries: The stripped text of the lines of a page that match
            `_TOC_ENTRY`, in reading order.

    Returns:
        Whether the arabic numbers at the end of the lines never decrease,
        a number that repeats being fine. A line that ends with a roman
        numeral has no such number and is left out.
    """
    numbers = [
        int(match[0])
        for entry in entries
        if (match := _PAGE_NUMBER.search(entry))
    ]
    return all(low <= high for low, high in pairwise(numbers))


def _is_within_pages(entry: str, page_count: int | None) -> bool:
    """Tell whether the number at the end of an entry can be a page number.

    Args:
        entry: The stripped text of an entry, which matches `_TOC_ENTRY`.
        page_count: The number of pages of the document, or None if there is
            no limit.

    Returns:
        Whether the arabic number at the end of the entry is at most
        `page_count`, which is always the case without one. An entry that
        ends with a roman numeral has no such number, so it is always within
        the pages.
    """
    if page_count is None:
        return True
    match = _PAGE_NUMBER.search(entry)
    return match is None or int(match[0]) <= page_count


def _page_kind(
    page: Sequence[Line],
    previous: BlockKind | None,
    *,
    page_count: int | None,
) -> BlockKind | None:
    """Return the kind of a page, from its lines.

    The title of the page is the loose key of its first line. The lines that
    match `_TOC_ENTRY` once stripped are its matches, and those that match
    `_LEADERS` its leaders. If the title is one of `TOC_TITLES` or of
    `INDEX_TITLES`, its entries are all of its matches. If not, its entries
    are the matches whose arabic number is within the pages, as
    `_is_within_pages` tells, and only the leaders that are within the pages
    count. Entries or leaders make a share if there are at least
    `TOC_MIN_SHARE` times as many of them as there are lines, all the lines
    of the page being counted. A page goes on with the contents before it if
    its matches are in page order, as `_is_in_page_order` tells. The order is
    read on every match, those that are no entries included: a number above
    the pages speaks against a contents.

    Args:
        page: The lines of the page, in reading order.
        previous: The kind of the page before it, or None if that page is
            neither a contents page nor an index page.
        page_count: The number of pages of the document, or None if there is
            no limit.

    Returns:
        None if the page has fewer than `TOC_MIN_LINES` lines. Otherwise:

        - `BlockKind.TOC` if the entries make a share, and the title is one
          of `TOC_TITLES` or the page before is a contents page and the
          matches are in page order.
        - Else `BlockKind.INDEX` if the title is one of `INDEX_TITLES` or the
          page before is an index page, and the entries make a share.
        - Else `BlockKind.TOC` if the leaders make a share.
        - Else None.
    """
    if len(page) < TOC_MIN_LINES:
        return None
    needed = TOC_MIN_SHARE * len(page)
    title = loose_key(page[0].text)
    # A title keeps the numbers of its page, whatever they are.
    is_titled = title in TOC_TITLES or title in INDEX_TITLES
    bound = None if is_titled else page_count
    texts = [line.text.strip() for line in page]
    # The order of the numbers is read on every match, those above the page
    # count included: such a number speaks against a contents.
    matches = [text for text in texts if _TOC_ENTRY.search(text)]
    entries = [text for text in matches if _is_within_pages(text, bound)]
    enough_entries = len(entries) >= needed
    goes_on = previous is BlockKind.TOC and _is_in_page_order(matches)
    if enough_entries and (title in TOC_TITLES or goes_on):
        return BlockKind.TOC
    if enough_entries and (
        title in INDEX_TITLES or previous is BlockKind.INDEX
    ):
        return BlockKind.INDEX
    leaders = sum(
        _LEADERS.search(text) is not None and _is_within_pages(text, bound)
        for text in texts
    )
    if leaders >= needed:
        return BlockKind.TOC
    return None


def toc_pages(
    lines: Sequence[Line],
    *,
    page_count: int | None = None,
) -> dict[int, BlockKind]:
    """Find the contents pages and the index pages of a document.

    The pages are examined in order, each one from its own lines and from the
    kind of the page before it. A page with fewer than `TOC_MIN_LINES` lines,
    and a page with none, are neither a contents page nor an index page, so
    that the page after them goes on with nothing.

    A page that follows a contents page, and that neither has the title of a
    contents nor is mostly made of entries with dot leaders, goes on with
    that contents only if the arabic numbers at the end of its entries never
    decrease, as the page numbers of a contents do. A table of values after a
    contents page, such as the settings of a service, usually breaks this
    order, so the page is not typed and the comparison reads it. The guard has
    a limit: a table whose last column rises, right after a contents page, is
    still taken for the rest of the contents, unless fewer than half of its
    lines end with a number within the count of the pages. The numbers of an
    index are in no order, so the pages of an index are not checked in this
    way.

    A page whose title is neither that of a contents nor that of an index has
    one more guard: a line counts as an entry, or as a line with dot leaders,
    only if the arabic number at its end is at most `page_count`. A table of
    values, such as "Max load . . . 500", is then no contents page in a
    document of fewer pages. The lines left out still count as lines of the
    page. The guard on the order of the numbers, above, reads all the lines
    that end like an entry, those left out included, as a number that cannot
    be a page number speaks against a contents. A roman numeral always
    counts. A page that has such a title keeps its entries whatever their
    numbers, as the pages of a volume may be numbered from 301.

    Args:
        lines: The lines of the document, as `page_lines` returns them. The
            `page` of a line is the position of its page, from 0, and the
            lines of one page are in reading order.
        page_count: The number of pages of the document, or None to set no
            limit to the numbers.

    Returns:
        The kind of each contents page, `BlockKind.TOC`, and of each index
        page, `BlockKind.INDEX`, by position of the page, in the order of the
        pages. The other pages are left out.
    """
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)
    kinds: dict[int, BlockKind] = {}
    for page in sorted(by_page):
        kind = _page_kind(
            by_page[page],
            kinds.get(page - 1),
            page_count=page_count,
        )
        if kind is not None:
            kinds[page] = kind
    return kinds


def figure_list(texts: Sequence[str]) -> frozenset[int]:
    """Find the entries of the lists of figures or tables of a document.

    Args:
        texts: The text of each block of the document, in reading order.

    Returns:
        The positions of the blocks in runs of at least `TOC_MIN_LINES`
        consecutive blocks whose stripped text is a caption, as `is_caption`
        tells, and an entry, as `_TOC_ENTRY` reads it, when the block right
        before the run is a title of `FIGURE_LIST_TITLES`.
    """
    found: set[int] = set()
    run: list[int] = []
    for index, text in enumerate([*texts, ""]):
        entry = text.strip()
        if is_caption(entry) and _TOC_ENTRY.search(entry):
            run.append(index)
            continue
        if (
            len(run) >= TOC_MIN_LINES
            and run[0] > 0
            and loose_key(texts[run[0] - 1]) in FIGURE_LIST_TITLES
        ):
            found.update(run)
        run = []
    return frozenset(found)


def is_caption(text: str) -> bool:
    """Tell whether a text is the caption of a figure, a table or a listing.

    Args:
        text: The text of a block.

    Returns:
        Whether it starts with ``Figure``, ``Fig.``, ``Table``, ``Tableau``,
        ``Listing``, ``Illustration``, ``Image`` or ``Dessin``, and then a
        number such as ``2``, ``2.1`` or ``3-2``.
    """
    return _CAPTION.match(text) is not None
