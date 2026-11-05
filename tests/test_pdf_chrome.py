# Copyright 2026 Christophe Le Douarec
"""Tests for finding the headers, footers and page numbers of a PDF."""

from itertools import product
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import PdfConfig
from doc_checker.extract.pdf.chrome import (
    CHROME_MAX_SIZE,
    HEADING_MIN_SIZE,
    band_of,
    find_chrome,
    neutral_key,
    page_number,
    title_key,
    title_pages,
)
from doc_checker.extract.pdf.reader import Bookmark
from tests.pdf_builders import make_segment

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from doc_checker.extract.pdf.segments import Segment

CFG = PdfConfig()
HEIGHT = 842.0
CODENAMES = ("alpha", "bravo", "charlie", "delta")  # Words without a number.


def _found(
    pages: list[list[Segment]],
    titles: dict[str, frozenset[int]] | None = None,
    body: float = 10.0,
    cfg: PdfConfig = CFG,
) -> list[tuple[int, str]]:
    """Return the page and the text of every chrome segment, in order."""
    chrome = find_chrome(
        pages,
        [HEIGHT] * len(pages),
        titles or {},
        labels=[str(page + 1) for page in range(len(pages))],
        body=body,
        cfg=cfg,
    )
    return sorted((segment.page, segment.text) for segment in chrome)


def _body(page: int) -> Segment:
    """Return a line of body text in the middle of a page."""
    return make_segment(
        f"Body text of page {page} goes here.",
        72.0,
        600.0,
        page=page,
    )


def _banners(count: int, on: Iterable[int]) -> list[list[Segment]]:
    """Make pages of body text, with a header on the pages listed in `on`."""
    pages = [[_body(page)] for page in range(count)]
    for page in on:
        pages[page].append(
            make_segment("Handbook", 72.0, 800.0, size=9.0, page=page),
        )
    return pages


def _footers(size: float) -> list[list[Segment]]:
    """Make four pages of body text, each with a footer of the given size."""
    return [
        [_body(page), make_segment("Draft", 72.0, 30.0, size=size, page=page)]
        for page in range(4)
    ]


def _footer_row(texts: Iterable[str]) -> list[Segment]:
    """Make a page of body text, with a footer segment for each text."""
    return [_body(0)] + [
        make_segment(text, 72.0 + 40.0 * index, 30.0, size=9.0)
        for index, text in enumerate(texts)
    ]


def test_band_lines_repeated_per_parity() -> None:
    """Lines of the bands that repeat on the pages of a parity are chrome."""
    pages = [
        [
            _body(page),
            make_segment("Draft", 72.0, 30.0, size=9.0, page=page),
            make_segment(
                "Left header" if page % 2 == 0 else "Right header",
                72.0,
                800.0,
                size=9.0,
                page=page,
            ),
        ]
        for page in range(6)
    ]
    assert _found(pages) == sorted(
        [(page, "Draft") for page in range(6)]
        + [
            (page, "Left header" if page % 2 == 0 else "Right header")
            for page in range(6)
        ],
    )


def test_single_or_body_lines_are_kept() -> None:
    """A line seen once, or outside the zones, is kept."""
    pages = [
        [
            _body(page),
            make_segment("Repeated body line.", 72.0, 500.0, page=page),
        ]
        for page in range(4)
    ]
    pages[0].append(make_segment("Unique banner", 72.0, 800.0, size=9.0))
    assert _found(pages) == []


def test_large_band_text_is_never_chrome() -> None:
    """Large text in a band, such as a title, is kept."""
    pages = [
        [
            _body(page),
            make_segment("Big title", 72.0, 800.0, size=20.0, page=page),
        ]
        for page in range(4)
    ]
    assert _found(pages) == []


def test_running_titles() -> None:
    """A running title is chrome, but the title of its own page is kept."""
    titles = {"limits": frozenset({1, 3}), "example": frozenset()}
    pages = [
        [_body(0), make_segment("Example", 72.0, 800.0, size=9.0, page=0)],
        [
            _body(1),
            make_segment("Limits", 72.0, 800.0, size=12.5, page=1),  # Heading.
        ],
        [_body(2), make_segment("Limits", 400.0, 800.0, size=9.0, page=2)],
        [
            _body(3),
            make_segment("Limits", 400.0, 800.0, size=9.0, page=3),
            make_segment("Limits", 72.0, 650.0, size=14.0, page=3),
        ],
    ]
    assert _found(pages, titles, body=12.0) == [
        (0, "Example"),
        (2, "Limits"),
        (3, "Limits"),
    ]
    chrome = find_chrome(
        pages,
        [HEIGHT] * 4,
        titles,
        labels=["1", "2", "3", "4"],
        body=12.0,
        cfg=CFG,
    )
    assert all(segment.size not in {12.5, 14.0} for segment in chrome)


def test_title_pages() -> None:
    """The titles come from the bookmarks, the document and large text."""
    pages = [
        [make_segment("Header", 72.0, 800.0, size=20.0, page=0)],
        [],
        [],
        [make_segment("User guide", 72.0, 700.0, size=20.0, page=3)],
    ]
    marks = [
        Bookmark(0, "1 Getting started", 0),
        Bookmark(1, "Installation", None),
    ]
    assert title_pages(
        pages,
        [HEIGHT] * 4,
        marks,
        "Example",
        body=10.0,
        cfg=CFG,
    ) == {
        "getting started": frozenset({0}),
        "installation": frozenset(),
        "example": frozenset(),
        "user guide": frozenset({3}),
    }


def test_keys_and_bands() -> None:
    """The keys neutralize numbers and the bands follow the page edges."""
    assert neutral_key("Page 12 of 30") == "page # of #"
    assert title_key("Chapter 2. Getting Started") == "getting started"
    assert title_key("1.2 Installation") == "installation"
    assert not title_key("2")
    assert (
        band_of(make_segment("x", 72.0, 800.0, size=9.0), HEIGHT, CFG) == "top"
    )
    assert (
        band_of(make_segment("x", 72.0, 760.0, size=9.0), HEIGHT, CFG) is None
    )
    assert (
        band_of(make_segment("x", 72.0, 30.0, size=9.0), HEIGHT, CFG)
        == "bottom"
    )
    assert band_of(make_segment("x", 72.0, 70.0), HEIGHT, CFG) is None


def test_repeat_threshold_follows_the_pages() -> None:
    """A text repeats on two pages at least, and on a share of the pages."""
    # Two pages of each parity: one page is not a repeat, two are.
    assert _found(_banners(4, [0])) == []
    assert _found(_banners(4, [0, 2])) == [(0, "Handbook"), (2, "Handbook")]
    # Ten even pages: 30 % of them is three pages.
    assert _found(_banners(20, [0, 2])) == []
    assert _found(_banners(20, [0, 2, 4])) == [
        (0, "Handbook"),
        (2, "Handbook"),
        (4, "Handbook"),
    ]
    half = PdfConfig(repeat_share=0.5)
    assert _found(_banners(20, range(0, 8, 2)), cfg=half) == []
    assert len(_found(_banners(20, range(0, 10, 2)), cfg=half)) == 5


def test_repeat_share_counts_the_pages_of_a_parity() -> None:
    """The share is taken among the pages of the parity of the page."""
    whole = PdfConfig(repeat_share=1.0)
    # Three even pages and two odd ones: pages 0 and 2 miss page 4.
    assert _found(_banners(5, [0, 1, 2, 3]), cfg=whole) == [
        (1, "Handbook"),
        (3, "Handbook"),
    ]
    assert len(_found(_banners(5, range(5)), cfg=whole)) == 5


def test_repeats_count_distinct_pages() -> None:
    """Two copies of a text on one page are one page of repeats."""
    pages = _banners(4, [0])
    pages[0].append(make_segment("Handbook", 300.0, 800.0, size=9.0))
    assert _found(pages) == []


def test_repeats_need_the_same_band_and_parity() -> None:
    """A text repeats in one band, on pages of one parity."""
    pages = [[_body(page)] for page in range(4)]
    pages[0].append(make_segment("Draft", 72.0, 800.0, size=9.0, page=0))
    pages[1].append(make_segment("Draft", 72.0, 800.0, size=9.0, page=1))
    pages[2].append(make_segment("Draft", 72.0, 30.0, size=9.0, page=2))
    assert _found(pages) == []


def test_repeats_need_the_same_size() -> None:
    """The size is part of the repeat, so a title is not a running header."""
    sizes = [9.0, 9.0, 9.0, 9.0, 11.0, 11.0]
    pages = [
        [
            _body(page),
            make_segment("Limits", 72.0, 800.0, size=size, page=page),
        ]
        for page, size in enumerate(sizes)
    ]
    assert _found(pages) == [(page, "Limits") for page in range(4)]


def test_repeats_ignore_digits() -> None:
    """Texts that differ by their numbers are one repeated text."""
    texts = [f"Rev {page} - Confidential" for page in range(4)]
    pages = [
        [_body(page), make_segment(text, 72.0, 30.0, size=9.0, page=page)]
        for page, text in enumerate(texts)
    ]
    assert _found(pages) == [(page, text) for page, text in enumerate(texts)]


def test_chrome_size_limit() -> None:
    """Text of a band is chrome up to 1.15 body sizes, and kept above."""
    assert len(_found(_footers(CHROME_MAX_SIZE * 10.0))) == 4
    assert _found(_footers(CHROME_MAX_SIZE * 10.0 + 0.5)) == []


def test_large_text_escapes_every_rule() -> None:
    """Large text of a band is kept, whichever rule would take it."""
    pages = [
        [
            _body(page),
            make_segment("Limits", 72.0, 800.0, size=20.0, page=page),
            make_segment("7", 400.0, 800.0, size=20.0, page=page),
        ]
        for page in range(4)
    ]
    assert _found(pages, {"limits": frozenset()}) == []


def test_rules_apply_only_in_the_zones() -> None:
    """Text outside the zones is kept, whichever rule would take it."""
    pages = [
        [
            _body(page),
            make_segment("Draft", 72.0, 500.0, page=page),
            make_segment("7", 300.0, 500.0, page=page),
            make_segment("Limits", 400.0, 500.0, page=page),
        ]
        for page in range(4)
    ]
    assert _found(pages, {"limits": frozenset()}) == []


def test_page_number_forms() -> None:
    """Page numbers come in several forms, unlike titles with a number."""
    numbers = {
        "12": "12",
        "100": "100",
        "(7)": "7",
        "- 7 -": "7",
        "iv": "iv",
        "XIV": "xiv",
        "Page 12": "12",
        "p. 4": "4",
        "3/10": "3",
        "4 / 30": "4",
        "Page 3/10": "3",
        "4 de 30": "4",
        "Page 4 of 30": "4",
        "Page 4 sur 30": "4",
    }
    others = ["Table 3", "Chapter 3", "3 apples", "Page", "Page two"]
    assert {text: page_number(text) for text in numbers} == numbers
    assert [page_number(text) for text in others] == [None] * len(others)


def test_words_and_figures_like_page_numbers_are_kept() -> None:
    """Words of roman letters, acronyms and decimals are not page numbers."""
    texts = [
        "mix",
        "civil",
        "did",
        "il",
        "xml",
        "CLI",
        "mm",
        "cm",
        "ml",
        "ID",
        "CI",
        "CD",
        "DVD",
        "Description",
        "2.5",
        "1,200",
    ]
    assert [page_number(text) for text in texts] == [None] * len(texts)
    assert _found([_footer_row(texts)]) == []


def test_roman_page_numbers_stop_at_99() -> None:
    """Only well formed roman numerals from 1 to 99 are page numbers."""
    numerals = ["i", "iv", "ix", "xiv", "xl", "xlix", "xcix"]
    others = ["c", "ci", "cv", "iiii", "vv", "il", "iix", "xxxx", "vx"]
    assert [page_number(text) for text in numerals] == numerals
    assert [page_number(text) for text in others] == [None] * len(others)


def test_two_numbers_are_a_page_number_around_a_slash() -> None:
    """Two numbers with no word between them need a slash to be a page."""
    with_slash = ["3/10", "4 / 30", "Page 3/10", "p. 3/10"]
    without = ["2.5", "1,200", "12-14", "Page 2.5", "3 10"]
    assert [page_number(text) for text in with_slash] == ["3", "4", "3", "3"]
    assert [page_number(text) for text in without] == [None] * len(without)


def test_page_numbers_follow_the_pages() -> None:
    """A page number is chrome when it follows the pages, a cell is not."""
    pages = [
        [
            _body(page),
            make_segment(str(page + 3), 300.0, 30.0, size=9.0, page=page),
        ]
        for page in range(3)
    ]
    pages[0].append(make_segment("17", 400.0, 30.0, size=9.0))
    pages[1].append(make_segment("42", 400.0, 30.0, size=9.0, page=1))
    assert _found(pages) == [(0, "3"), (1, "4"), (2, "5")]


def test_a_page_number_alone_is_its_label() -> None:
    """On a single page, a page number is chrome if it is the page label."""
    page = [
        _body(0),
        make_segment("iv", 300.0, 30.0, size=9.0),
        make_segment("7", 400.0, 30.0, size=9.0),
    ]
    chrome = find_chrome(
        [page],
        [HEIGHT],
        {},
        labels=["iv"],
        body=10.0,
        cfg=CFG,
    )
    assert [segment.text for segment in chrome] == ["iv"]


def test_page_number_offsets_need_a_share_of_the_pages() -> None:
    """An offset makes page numbers once shared by 30 % of the pages."""
    pages = [[_body(page)] for page in range(10)]
    # Two cells whose values minus their pages agree by chance.
    pages[0].append(make_segment("17", 300.0, 30.0, size=9.0))
    pages[1].append(make_segment("18", 300.0, 30.0, size=9.0, page=1))
    assert _found(pages) == []
    # Numbers that follow the pages from 5 on, on three of the ten pages.
    for page in range(2, 5):
        pages[page].append(
            make_segment(str(page + 3), 300.0, 30.0, size=9.0, page=page),
        )
    assert _found(pages) == [(2, "5"), (3, "6"), (4, "7")]


def test_outermost_rows_beyond_the_bands() -> None:
    """The first and last rows of a page count up to 20 % from the edge."""
    pages = [
        [
            make_segment(
                f"Body line {row} of page {page} goes on.",
                72.0,
                700.0 - 14 * row,
                page=page,
            )
            for row in range(5)
        ]
        + [
            make_segment("Example manual", 72.0, 740.0, size=9.0, page=page),
            make_segment(str(page + 1), 300.0, 92.0, size=9.0, page=page),
        ]
        for page in range(4)
    ]
    assert _found(pages) == sorted(
        [(page, "Example manual") for page in range(4)]
        + [(page, str(page + 1)) for page in range(4)],
    )


def test_inner_rows_beyond_the_bands_are_kept() -> None:
    """A row near the edge but inside the page text is never chrome."""
    ends = ["Alpha ends.", "Beta ends.", "Gamma ends.", "Delta ends."]
    pages = [
        [
            _body(page),
            make_segment(str(page + 1), 300.0, 120.0, size=9.0, page=page),
            make_segment(ends[page], 72.0, 100.0, page=page),
        ]
        for page in range(4)
    ]
    assert _found(pages) == []


def test_lone_numbers_repeat_as_they_are() -> None:
    """A lone number repeats only with the same value."""
    same = [
        [_body(p), make_segment("2026", 300.0, 30.0, size=9.0, page=p)]
        for p in range(4)
    ]
    cells = [
        [
            _body(p),
            make_segment(str(10 * p + 7), 300.0, 30.0, size=9.0, page=p),
        ]
        for p in range(4)
    ]
    assert _found(same) == [(page, "2026") for page in range(4)]
    assert _found(cells) == []


def _row_pages(
    baseline: float,
    rows: Sequence[Sequence[str]],
) -> list[list[Segment]]:
    """Make pages of body text, each with a row of cells at a baseline."""
    return [
        [_body(page)]
        + [
            make_segment(text, 72.0 + 150 * column, baseline, page=page)
            for column, text in enumerate(cells)
        ]
        for page, cells in enumerate(rows)
    ]


def test_texts_beyond_the_bands_repeat_with_their_numbers() -> None:
    """Beyond the bands, a text repeats only as it is, numbers included."""
    steps = [[f"Step {page + 2}"] for page in range(4)]
    checks = [["Yes"] for _ in range(4)]
    # Alone on the last line beyond the band, a step is no repeat, a check is.
    assert _found(_row_pages(100.0, steps)) == []
    assert _found(_row_pages(100.0, checks)) == [(p, "Yes") for p in range(4)]
    # In the band, the steps repeat once their numbers are left out.
    assert _found(_row_pages(30.0, steps)) == [
        (page, f"Step {page + 2}") for page in range(4)
    ]


def test_lines_beyond_the_bands_are_chrome_whole() -> None:
    """Beyond the bands, a line is chrome whole or not at all."""
    table = [
        [f"Step {page + 7}", f"{10 * page + 23} ms", "Yes"]
        for page in range(4)
    ]
    tasks = [[f"{name} done", "Yes"] for name in CODENAMES]
    # A row of a table keeps all its cells, the repeated check included.
    assert _found(_row_pages(100.0, table)) == []
    assert _found(_row_pages(100.0, tasks)) == []
    # A segment too large to be chrome keeps its line as well.
    totals = _row_pages(100.0, [["Yes"] for _ in range(4)])
    for page, segments in enumerate(totals):
        segments.append(
            make_segment("Total", 300.0, 100.0, size=16.0, page=page),
        )
    assert _found(totals) == []
    # In a band, each segment is chrome on its own.
    assert _found(_row_pages(30.0, tasks)) == [(p, "Yes") for p in range(4)]


def test_lines_of_chrome_beyond_the_bands_are_removed() -> None:
    """A header or a footer beyond the bands holds only chrome, and goes."""
    header = ["Limits", "{}"]  # A running title and the page number.
    footer = ["Example manual", "Page {}"]  # A repeat and the page number.
    pages = [
        [_body(page)]
        + [
            make_segment(
                text.format(page + 1),
                x,
                baseline,
                size=9.0,
                page=page,
            )
            for baseline, texts in ((740.0, header), (100.0, footer))
            for x, text in zip((72.0, 400.0), texts, strict=True)
        ]
        for page in range(4)
    ]
    assert _found(pages, {"limits": frozenset()}) == sorted(
        (page, text.format(page + 1))
        for page in range(4)
        for text in header + footer
    )


def test_cells_like_chrome_stay_with_their_row() -> None:
    """A cell equal to a title or to a page label stays in its table row."""
    # A cell equal to the title of a section, on the last line of a page.
    pages = [[_body(page)] for page in range(4)]
    pages[0] += [
        make_segment("Installation", 72.0, 80.0),
        make_segment("Added step 3", 300.0, 80.0),
    ]
    assert _found(pages, {"installation": frozenset({3})}) == []
    # A total equal to the label of the only page.
    total = [
        _body(0),
        make_segment("Total", 72.0, 80.0),
        make_segment("1", 300.0, 80.0),
    ]
    assert _found([total]) == []


def test_short_documents_count_repeats_on_all_pages() -> None:
    """In a document of fewer than four pages, repeats count on all pages."""
    assert _found(_footers(9.0)[:2]) == [(0, "Draft"), (1, "Draft")]
    assert _found(_footers(9.0)[:3]) == [
        (0, "Draft"),
        (1, "Draft"),
        (2, "Draft"),
    ]
    assert _found(_footers(9.0)[:1]) == []


def test_short_documents_decide_for_all_their_pages() -> None:
    """In a document of three pages, every page counts on all the pages."""
    assert _found(_banners(3, [1, 2])) == [(1, "Handbook"), (2, "Handbook")]


@pytest.mark.parametrize(
    ("gap", "chrome"),
    [(30.0, True), (20.5, True), (20.0, False), (12.0, False)],
)
def test_outermost_lines_beyond_the_bands_need_a_gap(
    gap: float,
    *,
    chrome: bool,
) -> None:
    """Beyond the bands, an outermost line is chrome past a gap of 2 sizes."""
    pages = [
        [
            make_segment("Example manual", 72.0, 740.0, page=page),
            make_segment(f"First line {page}.", 72.0, 740.0 - gap, page=page),
            make_segment(f"Middle line {page}.", 72.0, 420.0, page=page),
            make_segment(f"Last line {page}.", 72.0, 100.0 + gap, page=page),
            make_segment(str(page + 1), 300.0, 100.0, page=page),
        ]
        for page in range(4)
    ]
    expected = [(page, "Example manual") for page in range(4)]
    expected += [(page, str(page + 1)) for page in range(4)]
    assert _found(pages) == (sorted(expected) if chrome else [])


def test_a_line_alone_on_its_page_stands_apart() -> None:
    """With no other line on its page, an outermost line stands apart."""
    pages = [
        [make_segment("Example manual", 72.0, 740.0, size=9.0, page=page)]
        for page in range(4)
    ]
    assert _found(pages) == [(page, "Example manual") for page in range(4)]


def test_the_gap_skips_the_rest_of_the_line() -> None:
    """The gap runs to the next line, past the segments of the same line."""
    texts = {
        "Example manual": 740.0,
        "Draft": 738.5,  # Same first line, 1.5 pt lower.
        "Internal": 100.0,
        "Printed copy": 101.5,  # Same last line, 1.5 pt higher.
    }
    pages = [
        [_body(page)]
        + [
            make_segment(
                text,
                72.0 + 100 * index,
                baseline,
                size=9.0,
                page=page,
            )
            for index, (text, baseline) in enumerate(texts.items())
        ]
        for page in range(4)
    ]
    assert _found(pages) == sorted(product(range(4), texts))


def test_the_gap_counts_in_the_larger_size() -> None:
    """The gap is measured in the larger size of the line and the next one."""
    pages = [
        [
            make_segment("Example manual", 72.0, 740.0, size=9.0, page=page),
            # 30 pt are less than 2 sizes of 16 pt, though 3.3 sizes of 9 pt.
            make_segment(
                f"{page + 1} Methods",
                72.0,
                710.0,
                size=16.0,
                page=page,
            ),
            _body(page),
            make_segment(
                f"Note {page + 1}.",
                72.0,
                120.0,
                size=9.0,
                page=page,
            ),
            # 20 pt are less than 2 sizes of 11 pt, though 2.2 sizes of 9 pt.
            make_segment("Printed copy", 72.0, 100.0, size=11.0, page=page),
        ]
        for page in range(4)
    ]
    assert _found(pages) == []


def test_bands_need_no_gap() -> None:
    """A line in a band is chrome however close the next line is."""
    pages = [
        [
            make_segment("Example manual", 72.0, 770.0, size=9.0, page=page),
            make_segment(f"Line one of page {page}.", 72.0, 758.0, page=page),
            _body(page),
        ]
        for page in range(4)
    ]
    assert _found(pages) == [(page, "Example manual") for page in range(4)]


def _table_pages(count: int) -> list[list[Segment]]:
    """Make pages of a table whose rows are 18 pt apart, from 746 to 80."""
    pages: list[list[Segment]] = [[] for _ in range(count)]
    for page, row in product(range(count), range(38)):
        step, baseline = 38 * page + row, 746.0 - 18 * row
        pages[page] += [
            make_segment(f"Step {step + 1}", 72.0, baseline, page=page),
            make_segment(f"{10 * step + 7} ms", 200.0, baseline, page=page),
            make_segment("Yes", 300.0, baseline, page=page),
        ]
    return pages


def test_table_rows_at_the_edges_are_kept() -> None:
    """The first and last rows of a table, close to the next ones, are kept."""
    pages = _table_pages(6)
    assert _found(pages) == []
    # A changed value stays in the text, so that the comparison reports it.
    pages[2][-2] = make_segment("99 ms", 200.0, 80.0, page=2)
    assert _found(pages) == []


def test_a_chapter_heading_twins_its_running_title() -> None:
    """A heading too large to be chrome twins a running title on its page."""
    titles = {"methods": frozenset({0}), "results": frozenset({2})}
    pages = [
        [
            _body(page),
            make_segment(
                "Methods" if page < 2 else "Results",
                72.0,
                30.0,
                size=9.0,
                page=page,
            ),
        ]
        for page in range(6)
    ]
    pages[0].append(make_segment("1 Methods", 72.0, 750.0, size=16.0))
    assert _found(pages, titles) == [
        (0, "Methods"),
        (1, "Methods"),
        *((page, "Results") for page in range(2, 6)),
    ]


def test_page_numbers_have_nine_digits_at_most() -> None:
    """A run of more than nine digits is no page number."""
    assert page_number("123456789") == "123456789"
    assert page_number("1234567890") is None
    assert page_number("Page 1234567890 of 3") is None
    huge = make_segment("9" * 5000, 300.0, 30.0, size=9.0)
    assert _found([[_body(0), huge]]) == []


def test_repeated_chapter_page_numbers_are_still_repeats() -> None:
    """A footer such as 2-3 on every page is removed as a repeat."""
    texts = ["2-1", "2-2", "2-3", "2-4"]
    pages = [
        [_body(page), make_segment(text, 72.0, 30.0, size=9.0, page=page)]
        for page, text in enumerate(texts)
    ]
    assert _found(pages) == list(enumerate(texts))


def test_running_title_needs_a_title_key() -> None:
    """A text whose title key is empty is never a running title."""
    page = [_body(0), make_segment("Chapter 3", 72.0, 800.0, size=9.0)]
    assert _found([page], {"": frozenset()}) == []


def test_title_in_a_band_needs_a_twin_outside_the_bands() -> None:
    """On the page of a title, a copy that may be chrome is no twin."""
    titles = {"limits": frozenset({0})}
    page = [
        _body(0),
        make_segment("Limits", 72.0, 800.0, size=9.0),
        make_segment("Limits", 300.0, 800.0, size=9.0),
    ]
    assert _found([page], titles) == []


def test_running_titles_ignore_numbers_and_chapter_words() -> None:
    """A header is the title of its chapter whatever its numbering."""
    pages = [
        [
            _body(0),
            make_segment("1.2 Getting started", 72.0, 700.0, size=14.0),
        ],
        [
            _body(1),
            make_segment(
                "Chapter 1. Getting Started",
                72.0,
                800.0,
                size=9.0,
                page=1,
            ),
        ],
    ]
    titles = title_pages(pages, [HEIGHT] * 2, [], None, body=10.0, cfg=CFG)
    assert _found(pages, titles) == [(1, "Chapter 1. Getting Started")]


def test_title_pages_unite_the_sources() -> None:
    """The pages of a title are united, and empty keys and small text go."""
    edge = HEADING_MIN_SIZE * 10.0  # The smallest size of a title.
    pages: list[list[Segment]] = [[] for _ in range(7)]
    pages.append(
        [
            make_segment("Intro", 72.0, 700.0, size=14.0, page=7),
            make_segment("Small", 72.0, 600.0, size=11.0, page=7),
            make_segment("Edge", 72.0, 500.0, size=edge, page=7),
            make_segment("12", 72.0, 400.0, size=14.0, page=7),
        ],
    )
    marks = [
        Bookmark(0, "Intro", 0),
        Bookmark(1, "1. Intro", 5),
        Bookmark(0, "Chapter 3", 6),
    ]
    assert title_pages(
        pages,
        [HEIGHT] * 8,
        marks,
        "2026",
        body=10.0,
        cfg=CFG,
    ) == {"intro": frozenset({0, 5, 7}), "edge": frozenset({7})}


def test_title_key_drops_the_chapter_word_and_the_numbers() -> None:
    """The key of a title keeps neither its chapter word nor its numbers."""
    assert title_key("Chapitre 3 : Présentation") == "presentation"
    assert title_key("Section 4.2: Limits") == "limits"
    assert not title_key("Part 1")
    assert title_key("Version 2.0") == "version # #"
    assert neutral_key("Rév. 2026-11") == "rev # #"


def test_band_limits_are_inclusive() -> None:
    """A segment that reaches the edge of a band lies in the band."""
    cfg = PdfConfig(header_band=0.25, footer_band=0.25)
    # The line box of a 10 pt segment spans from 2.5 pt below its baseline to
    # 8.5 pt above it, and the bands of a 100 pt page start at 25 and 75.
    assert band_of(make_segment("x", 72.0, 66.5), 100.0, cfg) == "top"
    assert band_of(make_segment("x", 72.0, 66.0), 100.0, cfg) is None
    assert band_of(make_segment("x", 72.0, 27.5), 100.0, cfg) == "bottom"
    assert band_of(make_segment("x", 72.0, 28.0), 100.0, cfg) is None


def test_top_band_wins_when_the_bands_overlap() -> None:
    """A segment in both bands is in the top one."""
    cfg = PdfConfig(header_band=0.5, footer_band=0.5)
    assert band_of(make_segment("x", 72.0, 45.0), 100.0, cfg) == "top"


def test_roman_page_numbers_need_two_pages_only() -> None:
    """Roman numbers that follow two pages are page numbers."""
    pages = [[_body(page)] for page in range(10)]
    for page, number in enumerate(("i", "ii")):
        pages[page].append(
            make_segment(number, 300.0, 30.0, size=9.0, page=page),
        )
    # Arabic numbers on two pages of ten still are not enough.
    for page in (5, 6):
        pages[page].append(
            make_segment(str(page + 3), 300.0, 30.0, size=9.0, page=page),
        )
    assert _found(pages) == [(0, "i"), (1, "ii")]
