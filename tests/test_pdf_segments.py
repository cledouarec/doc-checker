# Copyright 2026 Christophe Le Douarec
"""Tests for grouping the characters of a PDF page into line segments."""

from functools import partial
from typing import TYPE_CHECKING

import pytest

from doc_checker.extract.pdf.segments import body_size, build_segments
from tests.pdf_builders import make_chars, make_page

if TYPE_CHECKING:
    from doc_checker.extract.pdf.reader import PdfChar


def _texts(*runs: list[PdfChar]) -> list[str]:
    """Return the texts of the segments of a page made of the runs."""
    return [segment.text for segment in build_segments(make_page(*runs))]


def _mono(*runs: list[PdfChar]) -> bool:
    """Tell whether the one segment of a page made of the runs is mono."""
    (segment,) = build_segments(make_page(*runs))
    return segment.mono


def test_one_line_is_one_segment() -> None:
    """A line without a wide gap is one segment, with its box and style."""
    page = make_page(make_chars("Install the package", 72.0, 700.0))
    (segment,) = build_segments(page)
    assert (
        segment.text,
        segment.x0,
        segment.x1,
        segment.baseline,
        segment.size,
    ) == ("Install the package", 72.0, 167.0, 700.0, 10.0)
    assert (segment.page, segment.mono, segment.bold) == (0, False, False)
    assert (segment.top, segment.bottom) == (708.5, 697.5)
    assert segment.width == pytest.approx(95.0)


def test_segments_carry_the_page_index() -> None:
    """The segments know their page, and an empty page has none."""
    page = make_page(make_chars("text", 72.0, 700.0), index=3)
    assert [segment.page for segment in build_segments(page)] == [3]
    assert build_segments(make_page()) == []


def test_new_baseline_or_backward_move_starts_a_segment() -> None:
    """A new baseline, or a move back to the left, starts a new segment."""
    assert _texts(
        make_chars("first line", 72.0, 700.0),
        make_chars("second line", 72.0, 686.0),
    ) == ["first line", "second line"]
    assert _texts(
        make_chars("right", 300.0, 700.0),
        make_chars("left", 72.0, 700.0),
    ) == ["right", "left"]


def test_gaps() -> None:
    """A wide gap splits a segment, and a narrow one becomes a space."""
    assert _texts(
        make_chars("Option ", 72.0, 600.0),
        make_chars("Default", 130.0, 600.0),
    ) == ["Option", "Default"]  # 23 pt > 2 em
    assert _texts(
        make_chars("Option ", 72.0, 600.0),
        make_chars("Default", 119.0, 600.0),
    ) == ["Option Default"]
    assert _texts(
        make_chars("Option", 72.0, 600.0),
        make_chars("Default", 110.0, 600.0),
    ) == ["Option Default"]  # space inserted: 8 pt > 0.3 em


def test_segment_limits_are_strict() -> None:
    """A shift, a move back or a gap splits a segment only beyond its limit."""
    ab = make_chars("ab", 72.0, 700.0)  # Its last character starts at 77.
    # At 10 pt, a baseline shift has to exceed 5 pt.
    assert _texts(ab, make_chars("c", 82.0, 705.0)) == ["abc"]
    assert _texts(ab, make_chars("c", 82.0, 705.5)) == ["ab", "c"]
    # A move back has to exceed 5 pt from the start of the previous character.
    assert _texts(ab, make_chars("c", 72.0, 700.0)) == ["abc"]
    assert _texts(ab, make_chars("c", 71.5, 700.0)) == ["ab", "c"]
    # A gap has to exceed 20 pt to split, and 3 pt to get a space.
    assert _texts(ab, make_chars("c", 102.0, 700.0)) == ["ab c"]
    assert _texts(ab, make_chars("c", 102.5, 700.0)) == ["ab", "c"]
    assert _texts(ab, make_chars("c", 85.0, 700.0)) == ["abc"]
    assert _texts(ab, make_chars("c", 85.5, 700.0)) == ["ab c"]


def test_space_uses_the_larger_size_and_skips_blanks() -> None:
    """A space follows the larger size, and never goes next to a blank."""
    # A gap of 2 pt is under 0.3 of the larger size, 10 pt, but not of 6 pt.
    assert _texts(
        make_chars("a", 72.0, 700.0, size=6.0),
        make_chars("b", 77.0, 700.0),
    ) == ["ab"]
    assert _texts(
        make_chars("a", 72.0, 700.0),
        make_chars("b", 79.0, 700.0, size=6.0),
    ) == ["ab"]
    # The gap before a blank gets no space, as the blank is one already.
    assert _texts(
        make_chars("ab", 72.0, 700.0),
        make_chars(" c", 86.0, 700.0),
    ) == ["ab c"]


def test_superscript_and_blanks() -> None:
    """A superscript joins its line, and blanks never start a segment."""
    assert _texts(
        make_chars("text", 72.0, 700.0),
        make_chars("1", 92.0, 703.5, size=6.0),
    ) == ["text1"]
    assert _texts(
        make_chars("1", 72.0, 703.5, size=6.0),
        make_chars("text", 75.0, 700.0),
    ) == ["1text"]
    assert _texts(
        make_chars("  padded  ", 72.0, 700.0),
        make_chars("   ", 72.0, 600.0),
    ) == ["padded"]


def test_styles_and_rounding() -> None:
    """A segment is mono or bold if all its characters are, size rounds."""
    page = make_page(
        make_chars("pip install", 72.0, 600.0, size=8.97, mono=True),
    )
    (code,) = build_segments(page)
    assert (code.mono, code.bold, code.size) == (True, False, 9.0)
    plain = make_chars("a ", 72.0, 500.0)
    bold = make_chars("b", 82.0, 500.0, bold=True)
    assert build_segments(make_page(plain, bold))[0].bold is False
    # The size rounds to the half point, not to the point.
    page = make_page(make_chars("text", 72.0, 400.0, size=10.4))
    assert build_segments(page)[0].size == pytest.approx(10.5)


def test_styles_ignore_blanks() -> None:
    """Blanks do not decide whether a segment is mono or bold."""
    page = make_page(
        make_chars("pip", 72.0, 600.0, mono=True, bold=True),
        make_chars(" ", 87.0, 600.0),
        make_chars("install", 92.0, 600.0, mono=True, bold=True),
    )
    (code,) = build_segments(page)
    assert (code.mono, code.bold) == (True, True)
    page = make_page(
        make_chars("run ", 72.0, 600.0),
        make_chars("pip", 92.0, 600.0, mono=True),
    )
    (mixed,) = build_segments(page)
    assert (mixed.mono, mixed.bold) == (False, False)


def test_a_leading_line_number_in_another_font_keeps_code_monospaced() -> None:
    """Proportional digits before code are its line number."""
    run = make_chars("1", 59.0, 700.0, size=6.0) + make_chars(
        "import os",
        73.0,
        700.0,
        mono=True,
    )
    (segment,) = build_segments(make_page(run))
    assert (segment.text, segment.mono, segment.size) == (
        "1 import os",
        True,
        10.0,
    )
    (number,) = build_segments(
        make_page(make_chars("42", 59.0, 650.0, size=6.0)),
    )
    (prose,) = build_segments(make_page(make_chars("3 apples", 72.0, 600.0)))
    assert (number.mono, prose.mono) == (False, False)


def test_a_line_number_does_not_give_the_size_of_its_segment() -> None:
    """The size of a segment is that of its code, not of the number before."""
    tiny = partial(make_chars, baseline=700.0, size=6.0)
    brace = make_chars("}", 73.0, 700.0, mono=True)
    # The number wins a count that it ties or outnumbers: 9, 10 and 100 have
    # one digit or more for the one character of code.
    for digits in ("9", "10", "100"):
        number = tiny(digits, 62.0 - 3.0 * len(digits))
        (segment,) = build_segments(make_page(number, brace))
        assert (segment.text, segment.mono, segment.size) == (
            f"{digits} }}",
            True,
            10.0,
        )


def test_a_number_gives_the_size_of_its_segment_only_if_it_is_alone() -> None:
    """Digits that open a segment give its size only if nothing follows."""
    (alone,) = build_segments(
        make_page(make_chars("42", 59.0, 700.0, size=6.0)),
    )
    # Before prose too, the 14 pt digits tie with the 9 pt letters, and lose.
    (prose,) = build_segments(
        make_page(
            make_chars("12", 50.0, 700.0, size=14.0),
            make_chars("of", 70.0, 700.0, size=9.0),
        ),
    )
    assert (alone.size, prose.size) == (6.0, 9.0)


def test_only_digits_that_open_a_segment_are_a_line_number() -> None:
    """Digits count as a line number only if they open the segment, whole."""
    code = make_chars("import os", 73.0, 700.0, mono=True)
    tiny = partial(make_chars, baseline=700.0, size=6.0)
    # The digits may be as long as the number, and mono digits are code.
    assert _mono(tiny("12", 55.0), code)
    assert _mono(tiny("1", 59.0, mono=True), code)
    # Any other character in the proportional font is no line number.
    assert not _mono(tiny("a1", 55.0), code)
    assert not _mono(tiny("1:", 55.0), code)
    # Numbers set apart are not one number, and digits after the code are not
    # a line number either.
    assert not _mono(tiny("1 2", 52.0), code)
    assert not _mono(code, tiny("1", 130.0))


def test_box_and_baseline() -> None:
    """The box spans the characters that are not blank, from the first line."""
    page = make_page(
        make_chars("text", 72.0, 700.0),
        make_chars("1", 92.0, 703.5, size=6.0),
    )
    (segment,) = build_segments(page)
    assert segment.baseline == pytest.approx(700.0)
    padded = make_page(make_chars("  padded  ", 72.0, 700.0))
    (segment,) = build_segments(padded)
    assert (segment.x0, segment.x1) == (82.0, 112.0)
    # The second glyph overlaps the first one, and starts to its left.
    overlap = make_page(
        make_chars("a", 72.0, 700.0),
        make_chars("b", 70.0, 700.0),
    )
    (segment,) = build_segments(overlap)
    assert (segment.text, segment.x0, segment.x1) == ("ab", 70.0, 77.0)


def test_size_is_the_commonest_one_and_ties_go_to_the_first() -> None:
    """A segment has the commonest size of its characters, else the first."""
    page = make_page(
        make_chars("a", 72.0, 700.0),
        make_chars("bc", 77.0, 700.0, size=12.0),
    )
    (segment,) = build_segments(page)
    assert segment.size == pytest.approx(12.0)
    for first, second in ((10.0, 12.0), (12.0, 10.0)):
        page = make_page(
            make_chars("ab", 72.0, 700.0, size=first),
            make_chars("cd", 72.0 + first, 700.0, size=second),
        )
        (tied,) = build_segments(page)
        assert tied.size == pytest.approx(first)


def test_body_size() -> None:
    """The body size is the commonest size of the text that is not mono."""
    body = build_segments(
        make_page(make_chars("x" * 100, 72.0, 700.0, size=11.0)),
    )
    title = build_segments(
        make_page(make_chars("Title", 72.0, 750.0, size=14.0)),
    )
    code = build_segments(
        make_page(make_chars("y" * 200, 72.0, 600.0, size=9.0, mono=True)),
    )
    assert body_size([body, title, code]) == pytest.approx(11.0)
    assert body_size([]) == pytest.approx(10.0)
    assert body_size([code]) == pytest.approx(10.0)  # No text but mono.


def test_body_size_counts_characters_that_are_not_blank() -> None:
    """Blanks have no weight in the body size, and a tie takes the smaller."""
    solid = build_segments(
        make_page(make_chars("x" * 10, 72.0, 700.0, size=12.0)),
    )
    airy = build_segments(
        make_page(make_chars("y y y y y y", 72.0, 600.0, size=9.0)),
    )
    # The airy line has 11 characters but 6 that are not blank.
    assert body_size([airy, solid]) == pytest.approx(12.0)
    tie = build_segments(
        make_page(make_chars("y" * 10, 72.0, 600.0, size=9.0)),
    )
    assert body_size([solid, tie]) == pytest.approx(9.0)
