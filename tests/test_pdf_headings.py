# Copyright 2026 Christophe Le Douarec
"""Tests for the detection of headings from bookmarks or from text styles."""

import pytest

from doc_checker.extract.pdf.blocks import DraftBlock
from doc_checker.extract.pdf.chrome import HEADING_MIN_SIZE
from doc_checker.extract.pdf.headings import (
    HEADING_MAX_LINES,
    HEADING_MAX_WORDS,
    bookmark_levels,
    bookmark_window,
    heading_levels,
    style_levels,
)
from doc_checker.extract.pdf.reader import Bookmark
from tests.pdf_builders import make_line

BODY = 11.0  # The body size of the test documents, in points.


def _block(  # ruff: ignore[too-many-arguments] - one keyword per attribute of a block
    text: str,
    *,
    page: int = 0,
    size: float = BODY,
    bold: bool = False,
    lines: int = 1,
    code: bool = False,
) -> DraftBlock:
    """Make a block of identical lines, 14 points apart."""
    rows = tuple(
        make_line(
            text,
            700.0 - 14 * i,
            page=page,
            size=size,
            bold=bold,
            mono=code,
        )
        for i in range(lines)
    )
    return DraftBlock(lines=rows, code=code, flow_start=False)


def _marked(blocks: list[DraftBlock], *marks: Bookmark) -> dict[int, int]:
    """Match bookmarks with blocks whose text is that of their first line."""
    texts = [block.lines[0].text for block in blocks]
    return bookmark_levels(blocks, texts, marks)


def _styled(*blocks: DraftBlock, body: float = BODY) -> dict[int, int]:
    """Rank blocks whose text is that of their first line."""
    texts = [block.lines[0].text for block in blocks]
    return style_levels(blocks, texts, body=body)


def test_bookmark_levels() -> None:
    """Bookmarks give a level to the blocks they match, in outline order."""
    blocks = [
        _block("Getting started", size=20.0, bold=True),
        _block("Intro text."),
        _block("1.2 Installation", page=1, size=14.0, bold=True),
        _block("Body.", page=1),
    ]
    texts = ["Getting started", "Intro text.", "1.2 Installation", "Body."]
    marks = [
        Bookmark(0, "Getting started", 0),
        Bookmark(1, "Installation", 1),
        Bookmark(1, "Missing", 2),
    ]
    assert bookmark_levels(blocks, texts, marks) == {0: 1, 2: 2}


def test_bookmark_pages_and_code() -> None:
    """A bookmark skips a block off its pages and a block of code."""
    texts = ["Limits"]
    blocks = [_block("Limits", page=5, size=14.0, bold=True)]
    off_page = [Bookmark(1, "Limits", 1)]
    no_page = [Bookmark(1, "Limits", None)]
    assert bookmark_levels(blocks, texts, off_page) == {}
    assert bookmark_levels(blocks, texts, no_page) == {0: 2}
    code = [_block("Limits", code=True)]
    assert bookmark_levels(code, texts, [Bookmark(0, "Limits", None)]) == {}


def test_style_levels() -> None:
    """Styles make ranks, numbers make levels, and the rest is text."""
    specs = [
        ("Getting started", 20.0, True),
        ("Installation", 14.0, True),
        ("Body text sentence.", 11.0, False),
        ("Note", 11.0, True),
        ("Ends with colon:", 14.0, True),
        ("2.1 Numbered", 14.0, True),
        ("3.1.4 Deep detail", 14.0, True),
        ("Run-in bold", 11.0, True),
    ]
    blocks = [_block(text, size=size, bold=bold) for text, size, bold in specs]
    texts = [text for text, _, _ in specs]
    assert style_levels(blocks, texts, body=11.0) == {
        0: 1,
        1: 2,
        5: 2,
        6: 3,
        7: 3,
    }
    tall = [_block("Big but long", size=20.0, bold=True, lines=4)]
    assert style_levels(tall, ["Big but long"], body=11.0) == {}


def test_nothing_to_match() -> None:
    """Without blocks, or without bookmarks, there is no heading."""
    assert bookmark_levels([], [], [Bookmark(0, "Title", 0)]) == {}
    assert bookmark_levels([_block("Title")], ["Title"], []) == {}
    assert style_levels([], [], body=BODY) == {}


@pytest.mark.parametrize(
    ("page", "expected"),
    [(1, {}), (2, {0: 1}), (3, {0: 1}), (4, {})],
)
def test_bookmark_page_window(page: int, expected: dict[int, int]) -> None:
    """A bookmark on page 2 matches a block on page 2 or on the next one."""
    blocks = [_block("Limits", page=page)]
    assert _marked(blocks, Bookmark(0, "Limits", 2)) == expected


def test_bookmark_window() -> None:
    """A bookmark is looked for on its page and the next, from the pointer."""
    pages = [0, 0, 1, 2, 2, 3, 5]
    assert bookmark_window(pages, 0, 2) == range(3, 6)
    assert bookmark_window(pages, 4, 2) == range(4, 6)
    assert bookmark_window(pages, 0, 4) == range(6, 7)
    assert not bookmark_window(pages, 6, 0)
    assert bookmark_window(pages, 2, None) == range(2, 7)


def test_bookmarks_match_distinct_blocks_in_order() -> None:
    """Two bookmarks with one title match two blocks, one after the other."""
    blocks = [_block("Overview"), _block("Overview")]
    marks = [Bookmark(0, "Overview", None), Bookmark(1, "Overview", None)]
    assert _marked(blocks, *marks) == {0: 1, 1: 2}


def test_a_bookmark_never_looks_back() -> None:
    """A bookmark out of order finds nothing before the previous match."""
    blocks = [_block("Alpha"), _block("Beta")]
    marks = [Bookmark(0, "Beta", None), Bookmark(0, "Alpha", None)]
    assert _marked(blocks, *marks) == {1: 1}


def test_a_missing_bookmark_does_not_move_on() -> None:
    """A bookmark with no block is ignored: the next ones are still found."""
    blocks = [_block("Alpha"), _block("Beta")]
    marks = [
        Bookmark(0, "Alpha", None),
        Bookmark(0, "Nowhere", None),
        Bookmark(0, "Beta", None),
    ]
    assert _marked(blocks, *marks) == {0: 1, 1: 1}


def test_a_bookmark_skips_code_for_a_later_block() -> None:
    """The block of code with the title is passed over, not the search."""
    blocks = [_block("Limits", code=True), _block("Limits")]
    assert _marked(blocks, Bookmark(0, "Limits", None)) == {1: 1}


def test_bookmark_levels_are_capped() -> None:
    """The level is the depth plus one, and six at the most."""
    titles = ["Alpha", "Beta", "Gamma", "Delta"]
    blocks = [_block(title) for title in titles]
    marks = [
        Bookmark(depth, title, None)
        for depth, title in zip([0, 4, 5, 9], titles, strict=True)
    ]
    assert _marked(blocks, *marks) == {0: 1, 1: 5, 2: 6, 3: 6}


def test_bookmarks_and_blocks_are_compared_by_title_key() -> None:
    """Numbers, a chapter word, case and accents do not tell titles apart."""
    blocks = [
        _block("Chapter 2. Getting Started"),
        _block("2.1 D\xe9marrage"),
    ]
    marks = [
        Bookmark(0, "1.2 getting started", 0),
        Bookmark(1, "Demarrage", 0),
    ]
    assert _marked(blocks, *marks) == {0: 1, 1: 2}


def test_bookmarks_without_a_key_are_ignored() -> None:
    """A title that is only a number or a chapter word matches nothing."""
    blocks = [_block("Chapter 3"), _block("3")]
    marks = [
        Bookmark(0, "Chapter 3", None),
        Bookmark(0, "3", None),
        Bookmark(0, "", None),
    ]
    assert _marked(blocks, *marks) == {}


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        pytest.param(
            _block("Title", size=HEADING_MIN_SIZE * 10.0),
            {0: 1},
            id="large-enough",
        ),
        pytest.param(_block("Title", size=11.0), {}, id="not-large-enough"),
        pytest.param(
            _block("Title", size=12.0, lines=3),
            {0: 1},
            id="large-lines",
        ),
        pytest.param(
            _block("Title", size=10.0, bold=True),
            {0: 1},
            id="bold-line",
        ),
        pytest.param(
            _block("Title", size=10.0, bold=True, lines=2),
            {},
            id="bold-lines",
        ),
        pytest.param(
            _block("Title", size=9.5, bold=True),
            {},
            id="bold-below-body",
        ),
        pytest.param(_block("Title", size=10.0), {}, id="body-text"),
    ],
)
def test_style_threshold(block: DraftBlock, expected: dict[int, int]) -> None:
    """A heading is large, or a bold line at least as big as the body."""
    assert _styled(block, body=10.0) == expected


def test_style_shape_limits() -> None:
    """A heading has at most three lines and fifteen words."""
    words = " ".join(["word"] * HEADING_MAX_WORDS)
    lines = HEADING_MAX_LINES
    assert _styled(_block(words, size=14.0, lines=lines)) == {0: 1}
    assert _styled(_block(f"{words} more", size=14.0)) == {}
    assert _styled(_block("Title", size=14.0, lines=lines + 1)) == {}


@pytest.mark.parametrize(
    "text",
    ["Title.", "Title,", "Title;", "Title:", "Title\xa0:"],
)
def test_a_sentence_end_is_no_heading(text: str) -> None:
    """A stop, comma, semicolon or colon at the end rules out a heading."""
    assert _styled(_block(text, size=14.0)) == {}


@pytest.mark.parametrize("text", ["How does it work?", "Options (advanced)"])
def test_a_question_or_a_bracket_can_end_a_heading(text: str) -> None:
    """Only a stop, a comma, a semicolon or a colon ends a sentence."""
    assert _styled(_block(text, size=14.0)) == {0: 1}


@pytest.mark.parametrize("text", ["Note", "WARNING", "See also", "Tip"])
def test_admonition_titles_are_no_headings(text: str) -> None:
    """The default title of an admonition stays a paragraph."""
    assert _styled(_block(text, size=20.0, bold=True)) == {}


def test_only_a_whole_admonition_title_is_left_out() -> None:
    """A title that merely starts with the word of an admonition is kept."""
    block = _block("Notes on the release", size=14.0, bold=True)
    assert _styled(block) == {0: 1}


def test_code_is_no_heading() -> None:
    """A block of code is no heading, however it is set."""
    assert _styled(_block("Title", size=20.0, bold=True, code=True)) == {}


def test_ranks_by_size_then_weight() -> None:
    """The larger size ranks first, and bold goes before regular."""
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    specs = [
        (14.0, False),
        (14.0, True),
        (16.0, False),
        (12.5, True),
        (14.0, True),
    ]
    blocks = [
        _block(name, size=size, bold=bold)
        for name, (size, bold) in zip(names, specs, strict=True)
    ]
    assert _styled(*blocks) == {0: 3, 1: 2, 2: 1, 3: 4, 4: 2}


def test_ranks_come_from_the_headings_only() -> None:
    """A style that no heading has takes no rank."""
    note = _block("Note", size=20.0, bold=True)
    tall = _block("Big but long", size=18.0, lines=4)
    title = _block("Installation", size=14.0, bold=True)
    assert _styled(note, tall, title) == {2: 1}


def test_ranks_are_capped_at_six_levels() -> None:
    """The seventh style and the next ones share the sixth level."""
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"]
    blocks = [_block(name, size=20.0 - i) for i, name in enumerate(names)]
    levels = _styled(*blocks, body=10.0)
    assert levels == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 6}


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("1 Overview", 1),
        ("1. Overview", 1),
        ("1.2.3 Details", 3),
        ("1.2.3. Details", 3),
        ("1.2.3.4.5.6.7 Details", 6),
        ("A.1.2 Details", 3),
        ("B.4 Tables", 2),
        ("1.2", 2),
        ("1.Overview", 2),
        ("2) Overview", 2),
        ("v1.2 Overview", 2),
    ],
)
def test_numbering_gives_the_level(text: str, level: int) -> None:
    """The parts of a number set the level, over the rank of the style."""
    blocks = [
        _block("Intro", size=20.0, bold=True),
        _block(text, size=14.0, bold=True),
    ]
    # Without a number, the second style ranks second.
    assert _styled(*blocks)[1] == level


def _levels(
    blocks: list[DraftBlock],
    *marks: Bookmark,
) -> dict[int, tuple[int, str]]:
    """Find the headings of blocks whose text is that of their first line."""
    texts = [block.lines[0].text for block in blocks]
    return heading_levels(blocks, texts, marks, body=BODY)


def test_a_partial_outline_is_completed_by_the_styles() -> None:
    """Blocks styled like marked headings, or smaller ones, are headings."""
    blocks = [
        _block("Getting started", size=20.0, bold=True),
        _block("Installation", page=1, size=14.0, bold=True),
        _block("Body text of the section.", page=1),
        _block("User guide", page=2, size=20.0, bold=True),
        _block("2.1 Projects", page=2, bold=True),
        _block("Option Default", page=2, bold=True),
        _block("Release title", page=3, size=24.0, bold=True),
    ]
    assert _levels(blocks, Bookmark(0, "Getting started", 0)) == {
        0: (1, "outline"),
        1: (2, "style"),
        3: (1, "style"),
        4: (2, "style"),
    }


def test_an_appendix_number_completes_the_outline() -> None:
    """A title numbered as a section of an appendix is a heading."""
    blocks = [
        _block("Appendices", size=20.0, bold=True),
        _block("A.1 Glossary", page=1, bold=True),
        _block("Body text of the appendix.", page=1),
    ]
    assert _levels(blocks, Bookmark(0, "Appendices", 0)) == {
        0: (1, "outline"),
        1: (2, "style"),
    }


def test_without_an_outline_the_styles_tell_the_headings() -> None:
    """With no bookmark, every heading comes from the styles."""
    blocks = [
        _block("Getting started", size=20.0, bold=True),
        _block("Body text of the section."),
    ]
    assert _levels(blocks) == {0: (1, "style")}
