# Copyright 2026 Christophe Le Douarec
"""Tests for the reading order and the lines of a PDF page."""

import threading
from typing import TYPE_CHECKING

import pytest

from doc_checker.extract.pdf.layout import Line, leaf_lines, page_lines, xy_cut
from tests.pdf_builders import make_segment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.extract.pdf.segments import Segment

BODY = 10.0
# How the page made by `_grid` reads, with a cut between columns and without.
BY_COLUMN = ["L0", "L1", "R0", "R1"]
BY_ROW = ["L0 R0", "L1 R1"]


def _order(segments: Sequence[Segment]) -> list[str]:
    """Return the text of the lines of a page, in reading order."""
    return [line.text for line in page_lines(segments, body=BODY)]


def _leaves(segments: Sequence[Segment]) -> list[list[str]]:
    """Return the texts of the segments of each leaf, in reading order."""
    return [[s.text for s in leaf] for leaf in xy_cut(segments, body=BODY)]


def _cells(*cells: tuple[str, float, float]) -> list[Segment]:
    """Make segments on one baseline from (text, x0, width) triples."""
    return [
        make_segment(text, x0, 700.0, width=width) for text, x0, width in cells
    ]


def _grid(gap: float, left: float, right: float) -> list[Segment]:
    """Make two rows of a left and a right cell, `gap` points apart."""
    return [
        make_segment(f"{side}{row}", x0, 700.0 - 12.0 * row, width=width)
        for row in range(2)
        for side, x0, width in (("L", 0.0, left), ("R", left + gap, right))
    ]


def _columns(rows: int) -> list[Segment]:
    """Make two columns of the given number of lines, below a title line."""
    return [
        make_segment(f"{side}{row}", x0, 788.0 - 12.0 * row, width=190.0)
        for row in range(rows)
        for side, x0 in (("L", 50.0), ("R", 260.0))
    ]


def _title(baseline: float) -> Segment:
    """Make a line of text as wide as the two columns."""
    return make_segment("Title", 50.0, baseline, width=400.0)


def test_title_above_two_columns() -> None:
    """A title that spans two columns comes first, then each column."""
    segments = [
        make_segment(
            "A wide title spanning both columns",
            50.0,
            800.0,
            size=14.0,
            width=400.0,
        ),
    ]
    for i in range(3):
        segments.extend(
            (
                make_segment(
                    f"left line {i} of the first column",
                    50.0,
                    760.0 - 12 * i,
                    width=190.0,
                ),
                make_segment(
                    f"right line {i} of the second column",
                    260.0,
                    760.0 - 12 * i,
                    width=190.0,
                ),
            ),
        )
    expected = [
        "A wide title spanning both columns",
        *(f"left line {i} of the first column" for i in range(3)),
        *(f"right line {i} of the second column" for i in range(3)),
    ]
    assert _order(segments) == expected
    assert _order(segments[::-1]) == expected


def test_far_list_labels_stay_on_their_line() -> None:
    """The labels of a list stay with their text, however far apart."""
    segments = []
    for i in range(3):
        segments.extend(
            (
                make_segment(f"{i + 1}.", 50.0, 700.0 - 12 * i, width=10.0),
                make_segment(
                    f"item text number {i} that is long enough",
                    90.0,
                    700.0 - 12 * i,
                    width=300.0,
                ),
            ),
        )
    assert _order(segments) == [
        f"{i + 1}. item text number {i} that is long enough" for i in range(3)
    ]


def test_table_is_read_row_by_row() -> None:
    """The cells of a table are read row by row, not column by column."""
    rows = [("Option", "Default"), ("timeout", "30"), ("retries", "3")]
    segments = [
        segment
        for i, (a, b) in enumerate(rows)
        for segment in (
            make_segment(a, 50.0, 600.0 - 18 * i, width=35.0),
            make_segment(b, 160.0, 600.0 - 18 * i, width=30.0),
        )
    ]
    assert _order(segments) == ["Option Default", "timeout 30", "retries 3"]


def test_short_second_column() -> None:
    """A second column of one line is read after the whole first column."""
    left = [
        make_segment(
            f"left column line {i} with plenty of words",
            50.0,
            760.0 - 12 * i,
            width=190.0,
        )
        for i in range(5)
    ]
    right = [make_segment("invite other members.", 260.0, 760.0, width=91.0)]
    assert _order(left + right)[-1] == "invite other members."
    assert _order(left + right)[:5] == [s.text for s in left]


def test_line_attributes() -> None:
    """A line takes its text, extent, size and style from its segments."""
    (line,) = page_lines(
        [
            make_segment("1", 50.0, 700.0, width=5.0),
            make_segment("pip install", 70.0, 700.0, mono=True),
        ],
        body=BODY,
    )
    assert (line.text, line.x0, line.x1, line.mono, line.bold, line.size) == (
        "1 pip install",
        50.0,
        125.0,
        True,
        False,
        10.0,
    )
    assert page_lines([], body=BODY) == []


def test_nothing_to_cut() -> None:
    """No segment gives no leaf, and one segment gives one leaf."""
    segment = make_segment("alone", 50.0, 700.0)
    assert xy_cut([], body=BODY) == []
    assert xy_cut([segment], body=BODY) == [[segment]]
    assert leaf_lines([]) == []


@pytest.mark.parametrize(
    ("gap", "left", "right", "expected"),
    [
        pytest.param(10.0, 80.0, 40.0, BY_COLUMN, id="limits-included"),
        pytest.param(10.0, 40.0, 80.0, BY_COLUMN, id="wide-on-right"),
        pytest.param(9.9, 80.0, 40.0, BY_ROW, id="gap-too-small"),
        pytest.param(10.0, 79.9, 40.0, BY_ROW, id="not-wide-enough"),
        pytest.param(10.0, 80.0, 39.9, BY_ROW, id="not-narrow-enough"),
        pytest.param(10.0, 39.9, 80.0, BY_ROW, id="narrow-on-left"),
    ],
)
def test_columns_need_a_gap_and_two_wide_enough_sides(
    gap: float,
    left: float,
    right: float,
    expected: list[str],
) -> None:
    """Columns need a gap of a body size, and a wide and a narrow side."""
    assert _order(_grid(gap, left, right)) == expected


def test_the_widest_column_gap_is_cut_first() -> None:
    """The widest gap between columns is cut before a narrower one."""
    cells = _cells(("A", 0.0, 45.0), ("B", 60.0, 45.0), ("C", 135.0, 90.0))
    # Cutting the narrower gap first would also split A from B.
    assert _leaves(cells) == [["A", "B"], ["C"]]


def test_equal_column_gaps_are_cut_from_the_left() -> None:
    """Of two equal gaps between columns, the left one is cut first."""
    cells = _cells(("A", 0.0, 90.0), ("B", 105.0, 45.0), ("C", 165.0, 45.0))
    # Cutting the right gap first would also split B from C.
    assert _leaves(cells) == [["A"], ["B", "C"]]


def test_a_wide_gap_that_splits_no_columns_hides_no_other() -> None:
    """A wide gap that is not between columns never hides a narrower one."""
    cells = _cells(("S", 0.0, 20.0), ("A", 60.0, 90.0), ("B", 165.0, 45.0))
    assert _leaves(cells) == [["S", "A"], ["B"]]


def test_the_largest_line_gap_is_cut_first() -> None:
    """A line below the columns is cut off before the gaps between lines."""
    footer = make_segment("Footer", 50.0, 730.0, width=400.0)
    segments = [_title(800.0), *_columns(3), footer]
    assert _order(segments) == [
        "Title",
        *(f"L{row}" for row in range(3)),
        *(f"R{row}" for row in range(3)),
        "Footer",
    ]


def test_equal_line_gaps_are_cut_from_the_top() -> None:
    """With equal gaps, the title is cut off before the columns are read."""
    segments = [_title(800.0), *_columns(3)]
    assert _order(segments) == [
        "Title",
        *(f"L{row}" for row in range(3)),
        *(f"R{row}" for row in range(3)),
    ]


def test_a_tall_segment_blocks_the_cuts_between_lines_it_spans() -> None:
    """The lines next to a drop cap are not cut apart from each other."""
    segments = [
        make_segment("T", 50.0, 700.0, size=40.0, width=30.0),
        make_segment("first line", 70.0, 722.0, width=200.0),
        make_segment("second line", 70.0, 704.0, width=200.0),
    ]
    assert len(xy_cut(segments, body=BODY)) == 1


def test_overlapping_and_identical_segments_are_one_leaf() -> None:
    """Segments that overlap, or sit on each other, are never cut apart."""
    twin = make_segment("same", 50.0, 700.0)
    assert _leaves([twin, twin, twin]) == [["same", "same", "same"]]
    assert _leaves(
        [
            make_segment("one", 50.0, 700.0, width=100.0),
            make_segment("two", 120.0, 695.0, width=100.0),
            make_segment("three", 90.0, 705.0, width=100.0),
        ],
    ) == [["one", "two", "three"]]


def test_a_cut_that_keeps_every_segment_is_no_cut() -> None:
    """A part that no cut can shrink is a leaf, so that the cutting ends."""
    normal = make_segment("Normal text above.", 72.0, 645.0)
    # A negative size puts the top of the line box below its bottom.
    inverted = make_segment("FLIPPED", 72.0, 630.0, size=-60.0)
    leaves: list[list[list[Segment]]] = []
    worker = threading.Thread(
        target=lambda: leaves.append(xy_cut([normal, inverted], body=BODY)),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=5.0)
    assert leaves == [[[normal, inverted]]]


def test_a_tall_page_is_cut_without_deep_recursion() -> None:
    """Over a thousand lines in a row are cut one by one, not recursively."""
    segments = [
        make_segment(f"line {i}", 50.0, 20000.0 - 12.0 * i)
        for i in range(1200)
    ]
    assert _leaves(segments) == [[s.text] for s in segments]


@pytest.mark.parametrize(
    ("shift", "text"),
    [
        pytest.param(5.0, ["A B"], id="on-the-limit"),
        pytest.param(5.1, ["A", "B"], id="beyond-the-limit"),
    ],
)
def test_baselines_join_within_half_the_size(
    shift: float,
    text: list[str],
) -> None:
    """Segments join a line while their baselines differ by half a size."""
    segments = [
        make_segment("A", 50.0, 700.0),
        make_segment("B", 100.0, 700.0 - shift),
    ]
    assert [line.text for line in leaf_lines(segments)] == text


def test_a_line_is_measured_from_its_first_segment() -> None:
    """The shift is measured from the first segment, not from the last."""
    segments = [
        make_segment("A", 50.0, 700.0),
        make_segment("B", 100.0, 697.0),
        make_segment("C", 150.0, 694.0),
    ]
    assert [line.text for line in leaf_lines(segments)] == ["A B", "C"]


def test_a_large_segment_joins_a_line_from_further_away() -> None:
    """The tolerance is half the size of the segment that joins the line."""
    segments = [
        make_segment("A", 50.0, 700.0),
        make_segment("B", 100.0, 690.0, size=40.0),
        make_segment("C", 300.0, 690.0),
    ]
    assert [line.text for line in leaf_lines(segments)] == ["A B", "C"]


def test_a_line_follows_its_leftmost_segment() -> None:
    """A line reads from the left, and starts where its leftmost one does."""
    segments = [
        make_segment("B", 100.0, 700.0, page=3),
        make_segment("A", 50.0, 699.0, page=3),
    ]
    assert leaf_lines(segments) == [
        Line(
            page=3,
            text="A B",
            x0=50.0,
            x1=105.0,
            baseline=699.0,
            size=10.0,
            mono=False,
            bold=False,
        ),
    ]


def test_a_line_covers_every_segment_it_joins() -> None:
    """The right edge of a line is the furthest one, not the last one."""
    segments = [
        make_segment("wide", 50.0, 700.0, width=200.0),
        make_segment("inside", 100.0, 700.0, width=20.0),
    ]
    (line,) = leaf_lines(segments)
    assert (line.x0, line.x1) == (50.0, 250.0)


def test_a_line_takes_the_size_of_its_longest_text() -> None:
    """The size is that of the longest text, and of the first on a tie."""
    longest = [
        make_segment("a", 50.0, 700.0, size=8.0),
        make_segment("longest", 100.0, 700.0, size=12.0),
        make_segment("mid", 200.0, 700.0, size=9.0),
    ]
    assert [line.size for line in leaf_lines(longest)] == [12.0]
    tie = [
        make_segment("abc", 50.0, 700.0, size=8.0),
        make_segment("xyz", 100.0, 700.0, size=12.0),
    ]
    assert [line.size for line in leaf_lines(tie)] == [8.0]


def test_a_line_number_set_apart_does_not_give_the_size_of_its_code() -> None:
    """A tiny number in a segment of its own leaves its code line its size."""
    number = make_segment("12", 40.0, 700.0, size=6.0)
    brace = make_segment("}", 72.0, 700.0, mono=True)
    (line,) = leaf_lines([number, brace])
    assert (line.text, line.mono, line.size) == ("12 }", True, 10.0)


def test_a_monospaced_number_is_code_and_gives_the_size_of_its_line() -> None:
    """A monospaced number is code, and gives the size of its line."""
    number = make_segment("12", 40.0, 700.0, size=6.0)
    code = make_segment("42", 72.0, 700.0, mono=True)
    (line,) = leaf_lines([number, code])
    assert (line.text, line.mono, line.size) == ("12 42", True, 10.0)
    # Shorter than the number, it is still the one segment that counts.
    short = make_segment("4", 72.0, 700.0, mono=True)
    (line,) = leaf_lines([number, short])
    assert (line.text, line.mono, line.size) == ("12 4", True, 10.0)


def _line_size(parts: list[tuple[str, float]]) -> float:
    """Return the size of the line made of (text, size) pairs."""
    segments = [
        make_segment(text, 50.0 + 100.0 * i, 700.0, size=size)
        for i, (text, size) in enumerate(parts)
    ]
    (line,) = leaf_lines(segments)
    return line.size


@pytest.mark.parametrize(
    ("parts", "size"),
    [
        pytest.param([("12", 6.0), ("}", 10.0)], 10.0, id="line-number"),
        pytest.param(
            [("1 000", 6.0), ("kg", 10.0)],
            10.0,
            id="number-with-a-blank",
        ),
        pytest.param(
            [("a", 8.0), ("12345", 12.0), ("bb", 9.0)],
            9.0,
            id="longest-of-the-others",
        ),
        pytest.param([("12", 6.0)], 6.0, id="number-alone"),
        pytest.param([("12", 6.0), ("34", 10.0)], 6.0, id="numbers-only"),
        pytest.param(
            [("1", 6.0), ("234", 10.0)],
            10.0,
            id="longest-of-the-numbers",
        ),
        pytest.param([("v2", 6.0), ("}", 10.0)], 6.0, id="not-a-number"),
        pytest.param(
            [("Total", 10.0), ("1234567", 10.0)],
            10.0,
            id="cell-set-like-the-text",
        ),
    ],
)
def test_numbers_beside_text_do_not_give_the_size_of_a_line(
    parts: list[tuple[str, float]],
    size: float,
) -> None:
    """Segments of digits only give the size of a line if nothing else does."""
    assert _line_size(parts) == size


def _mono_line(parts: list[tuple[str, bool]]) -> bool:
    """Return whether the line made of (text, mono) pairs is monospaced."""
    segments = [
        make_segment(text, 50.0 + 100.0 * i, 700.0, mono=is_mono)
        for i, (text, is_mono) in enumerate(parts)
    ]
    (line,) = leaf_lines(segments)
    return line.mono


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param([("pip", True), ("install", True)], id="all-mono"),
        pytest.param([("12", False), ("x = 1", True)], id="line-number"),
        pytest.param([("7", False), ("42", True)], id="digits-are-code"),
    ],
)
def test_a_line_is_mono_if_its_segments_are(
    parts: list[tuple[str, bool]],
) -> None:
    """A line is monospaced if all its segments are, or its code is."""
    assert _mono_line(parts)


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param([("pip", True), ("install", False)], id="one-is-not"),
        pytest.param([("x", False)], id="none-is"),
        pytest.param(
            [("12", False), ("x = 1", True), ("note", False)],
            id="line-number-and-text",
        ),
        pytest.param(
            [("12", False), ("34", False), ("x", True)],
            id="only-the-first-number",
        ),
        pytest.param([("x", True), ("12", False)], id="number-last"),
        pytest.param([("v2", False), ("x", True)], id="not-a-number"),
        pytest.param([("12", False)], id="number-alone"),
    ],
)
def test_a_line_is_not_mono_if_a_segment_is_not(
    parts: list[tuple[str, bool]],
) -> None:
    """Only a leading line number may be left out of the monospaced test."""
    assert not _mono_line(parts)


def test_a_line_is_bold_if_every_segment_is() -> None:
    """A line is bold only if all of its segments are."""
    bold = make_segment("a", 50.0, 700.0, bold=True)
    plain = make_segment("b", 100.0, 700.0)
    assert leaf_lines([bold, bold])[0].bold is True
    assert leaf_lines([bold, plain])[0].bold is False
