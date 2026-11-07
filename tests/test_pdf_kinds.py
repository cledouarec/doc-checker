# Copyright 2026 Christophe Le Douarec
"""Tests for the detection of contents pages, index pages and captions."""

from typing import TYPE_CHECKING

import pytest

from doc_checker.extract.pdf.kinds import is_caption, toc_pages
from doc_checker.model.document import BlockKind
from tests.pdf_builders import make_line

if TYPE_CHECKING:
    from doc_checker.extract.pdf.layout import Line

# Three lines that end with a number, and three that end like a sentence.
ENTRIES = ("Reference 8", "Command line 9", "Index 12")
SENTENCES = (
    "The service accepts 100 requests per second.",
    "Requests above the limit are rejected.",
    "Retries happen later.",
)
# The settings of a service: lines that end with numbers, which go back.
SETTINGS = ("timeout 30", "retries 3", "workers 4", "port 8080")


def _page(page: int, *texts: str) -> list[Line]:
    """Make the lines of a page, from the top down, one for each text."""
    return [
        make_line(text, 800.0 - 14.0 * i, page=page)
        for i, text in enumerate(texts)
    ]


def _contents(page: int) -> list[Line]:
    """Make a contents page: a title and three entries with dot leaders."""
    return _page(
        page,
        "Contents",
        "Getting started . . . . . 2",
        "Installation . . . 3",
        "Limits . . . 10",
    )


def test_toc_with_title_and_continuation() -> None:
    """A titled contents page goes on over the pages of entries after it."""
    lines = [*_contents(0), *_page(1, *ENTRIES), *_page(2, *SENTENCES)]
    assert toc_pages(lines) == {0: BlockKind.TOC, 1: BlockKind.TOC}


def test_index_page() -> None:
    """A page titled Index, whose lines end with numbers, is an index."""
    lines = _page(0, "Index", "cache, 3, 12", "configuration, 2", "service, 4")
    assert toc_pages(lines) == {0: BlockKind.INDEX}


def test_leaders_without_title_and_plain_numbers() -> None:
    """Dot leaders make a contents page without a title, plain numbers not."""
    plain = _page(0, "Version 2", "Release 3", "Build 4")
    leaders = _page(
        1,
        "Getting started . . . 2",
        "Installation . . . 3",
        "Limits . . . 10",
    )
    assert toc_pages([*plain, *leaders]) == {1: BlockKind.TOC}
    assert toc_pages(_page(0, "Contents", "Only two lines")) == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Figure 1. Overview", True),
        ("Table 2: Options", True),
        ("Tableau 3 \U00002013 Valeurs", True),
        ("Fig. 4 Overview", True),
        ("Listing 5", True),
        ("Table of contents", False),
        ("Figures show the flow.", False),
        ("The table 2 lists", False),
        ("Figure 12", True),
        ("Figure 2.1 Layout", True),
        ("Table 3-2: Options", True),
        ("Figure A. Overview", False),
        ("Fig 4 Overview", False),
    ],
)
def test_is_caption(text: str, expected: bool) -> None:  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
    """A caption starts with a label and a number, such as Figure 2.1."""
    assert is_caption(text) is expected


def test_no_lines_no_pages() -> None:
    """A document without lines has no contents page."""
    assert toc_pages([]) == {}


def test_pages_come_in_order() -> None:
    """The pages are told in page order, whatever the order of the lines."""
    lines = [*_page(1, *ENTRIES), *_contents(0)]
    assert list(toc_pages(lines)) == [0, 1]


@pytest.mark.parametrize(
    "title",
    [
        "CONTENTS",
        "Table of Contents",
        "Table des matières",
        "Table\xa0des\xa0matières",
        "SOMMAIRE :",
    ],
)
def test_toc_titles_are_compared_loosely(title: str) -> None:
    """Case, accents, punctuation and no-break spaces do not matter."""
    lines = _page(0, title, "Overview 2", "Install 3", "Limits 10")
    assert toc_pages(lines) == {0: BlockKind.TOC}


def test_a_title_is_the_whole_line() -> None:
    """A line that only starts with a title word is no title."""
    lines = _page(0, "Contents of the box", "Cable 1", "Manual 2", "Charger 3")
    assert toc_pages(lines) == {}


@pytest.mark.parametrize("title", ["Contents", "Index"])
def test_a_title_alone_is_not_enough(title: str) -> None:
    """Under the title of a contents or of an index, text is only text."""
    assert toc_pages(_page(0, title, *SENTENCES)) == {}


def test_min_lines() -> None:
    """Two lines never make a contents page, but three can."""
    assert toc_pages(_page(0, "Contents", "Overview 1")) == {}
    assert toc_pages(_page(0, "Contents", "Overview 1", "Limits 10")) == {
        0: BlockKind.TOC,
    }


def test_a_titled_page_needs_half_of_its_lines_as_entries() -> None:
    """After a title, half of the lines must be entries: fewer do not do."""
    half = _page(0, "Contents", "Overview 1", "Preface", "Limits 10")
    fewer = _page(0, "Contents", "Overview 1", "Preface", "Notes")
    assert toc_pages(half) == {0: BlockKind.TOC}
    assert toc_pages(fewer) == {}


def test_leaders_need_half_of_the_lines() -> None:
    """Without a title, half of the lines run to a number along dots."""
    half = _page(0, "Setup . . . 2", "Usage . . . 3", "Overview", "Notes")
    fewer = _page(0, "Setup . . . 2", "Usage 3", "Overview", "Notes")
    assert toc_pages(half) == {0: BlockKind.TOC}
    assert toc_pages(fewer) == {}


def test_numbers_beyond_the_page_count_are_no_entries() -> None:
    """Without a title, a table of values with dot leaders is no contents."""
    values = _page(
        0,
        "Max load . . . . . . 500",
        "Timeout . . . . . . 30",
        "Retries . . . . . . 3",
    )
    assert toc_pages(values, page_count=10) == {}
    assert toc_pages(values) == {0: BlockKind.TOC}
    titled = _page(
        0,
        "Contents",
        "Getting started . . . 2",
        "Limits . . . 300",
    )
    assert toc_pages(titled, page_count=10) == {0: BlockKind.TOC}


def test_a_number_equal_to_the_page_count_is_an_entry() -> None:
    """A number equal to the page count counts, and one above does not."""
    page = _page(0, "Setup . . . 2", "Usage . . . 10", "Overview")
    assert toc_pages(page, page_count=10) == {0: BlockKind.TOC}
    assert toc_pages(page, page_count=9) == {}


def test_a_page_after_the_contents_is_bounded_by_the_page_count() -> None:
    """Numbers that rise but exceed the page count do not go on a contents."""
    rising = _page(1, "Timeout 30", "Retries 300", "Port 800")
    lines = [*_contents(0), *rising]
    assert toc_pages(lines) == {0: BlockKind.TOC, 1: BlockKind.TOC}
    assert toc_pages(lines, page_count=10) == {0: BlockKind.TOC}


@pytest.mark.parametrize("page_count", [None, 10, 100])
def test_the_order_guard_reads_the_numbers_above_the_page_count(
    page_count: int | None,
) -> None:
    """The order guard reads the numbers above the page count too."""
    lines = [*_contents(0), *_page(1, *SETTINGS)]
    assert toc_pages(lines, page_count=page_count) == {0: BlockKind.TOC}


@pytest.mark.parametrize(
    ("texts", "kind"),
    [
        pytest.param(
            (
                "Preface . . . iv",
                "Foreword . . . ix",
                "Max load . . . 500",
                "Timeout . . . 600",
            ),
            BlockKind.TOC,
            id="roman numerals",
        ),
        pytest.param(
            ("Index", "cache, 3, 12", "tls, 7", "service, 400"),
            BlockKind.INDEX,
            id="index title",
        ),
    ],
)
def test_roman_numerals_and_titled_pages_are_not_bounded(
    texts: tuple[str, ...],
    kind: BlockKind,
) -> None:
    """A roman numeral always counts, and so do the numbers under a title."""
    assert toc_pages(_page(0, *texts), page_count=10) == {0: kind}


def test_page_numbers_may_be_roman_and_touch_their_dots() -> None:
    """An entry may end with a roman numeral, set against its dots."""
    # The blanks at the end of a line do not count.
    lines = _page(
        0,
        "Contents",
        "Preface.....iv ",
        "Foreword.....ix ",
        "Appendix.....XIV ",
    )
    assert toc_pages(lines) == {0: BlockKind.TOC}


def test_index_goes_on_over_pages() -> None:
    """The pages of entries after an index page are index pages as well."""
    lines = [
        *_page(0, "Index", "cache, 3, 12", "configuration, 2", "service, 4"),
        *_page(1, "timeout, 7", "tls, 9, 14", "upgrade, 5"),
        *_page(2, *SENTENCES),
        *_page(3, "retry, 6", "proxy, 8", "queue, 10"),
    ]
    assert toc_pages(lines) == {0: BlockKind.INDEX, 1: BlockKind.INDEX}


@pytest.mark.parametrize(
    "between",
    [
        pytest.param(("Overview 1", "Limits 10"), id="short"),
        pytest.param((), id="missing"),
        pytest.param(SENTENCES, id="sentences"),
        pytest.param(SETTINGS, id="settings"),
    ],
)
def test_the_run_ends_on_a_page_that_is_not_typed(
    between: tuple[str, ...],
) -> None:
    """Entries after a page that is no contents page are only entries."""
    lines = [*_contents(0), *_page(1, *between), *_page(2, *ENTRIES)]
    assert toc_pages(lines) == {0: BlockKind.TOC}


@pytest.mark.parametrize("count", [1, 4])
def test_settings_pages_do_not_go_on_with_the_contents(count: int) -> None:
    """Pages of settings after the contents are no part of it."""
    lines = _contents(0)
    for page in range(1, count + 1):
        lines += _page(page, *SETTINGS)
    assert toc_pages(lines) == {0: BlockKind.TOC}


@pytest.mark.parametrize(
    ("texts", "kind"),
    [
        pytest.param(ENTRIES, BlockKind.TOC, id="rising"),
        pytest.param(
            ("Setup 5", "Usage 5", "Limits 5"),
            BlockKind.TOC,
            id="equal",
        ),
        pytest.param(
            ("Preface vii", "Overview 1", "Foreword ix", "Install 3"),
            BlockKind.TOC,
            id="roman numerals left out",
        ),
        pytest.param(SETTINGS, None, id="settings"),
        pytest.param(
            ("Overview 3", "Install 4", "Limits 2"),
            None,
            id="one step back",
        ),
        pytest.param(
            ("Contents", "Overview 9", "Install 4", "Limits 12"),
            BlockKind.TOC,
            id="title",
        ),
        pytest.param(
            ("Setup . . . 9", "Usage . . . 4", "Limits . . . 12"),
            BlockKind.TOC,
            id="leaders",
        ),
        pytest.param(
            ("Index", "cache, 3, 12", "configuration, 2", "service, 4"),
            BlockKind.INDEX,
            id="index title",
        ),
    ],
)
def test_a_page_goes_on_with_the_contents_if_its_numbers_rise(
    texts: tuple[str, ...],
    kind: BlockKind | None,
) -> None:
    """Without a title or leaders, page numbers that go back end the run."""
    lines = [*_contents(0), *_page(1, *texts)]
    assert toc_pages(lines).get(1) is kind


@pytest.mark.parametrize(
    "text",
    ["Illustration 2: Logo", "Image 3 - Logo", "Dessin 4 : Plan"],
)
def test_more_caption_labels(text: str) -> None:
    """Office suites name their figures in other ways."""
    assert is_caption(text)
