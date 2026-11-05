# Copyright 2026 Christophe Le Douarec
"""Tests for the grouping of PDF lines into blocks, and their merge."""

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from doc_checker.extract.pdf.blocks import (
    group_lines,
    merge_flows,
    paragraph_pitches,
    strip_list_marker,
)
from doc_checker.model.document import BlockKind
from tests.pdf_builders import make_line

if TYPE_CHECKING:
    from doc_checker.extract.pdf.layout import Line


def _groups(lines: list[Line]) -> list[list[str]]:
    """Return the texts of the lines of each block."""
    return [
        [line.text for line in block.lines] for block in group_lines(lines)
    ]


def _flows(lines: list[Line]) -> list[bool]:
    """Return, for each block, whether it opens a page or a column."""
    return [block.flow_start for block in group_lines(lines)]


def _joined(first: Line, second: Line) -> bool:
    """Return whether two lines, one page apart, end up in one block."""
    return len(merge_flows(group_lines([first, second]), Counter())) == 1


def test_spacing_splits_paragraphs() -> None:
    """A gap wider than one and a half sizes starts a new paragraph."""
    lines = [
        make_line("First line of a paragraph", 700.0),
        make_line("second line.", 688.0),
        make_line("Next paragraph.", 668.0),
    ]
    assert _groups(lines) == [
        ["First line of a paragraph", "second line."],
        ["Next paragraph."],
    ]


def test_spacing_limits_are_strict() -> None:
    """A gap splits a block only beyond 1.5 sizes, or 2.5 for code."""
    # At 10 pt, the distance between the baselines has to exceed 15 pt for
    # text, and 25 pt for code. Such a gap is not a break of the flow.
    for mono, limit in ((False, 15.0), (True, 25.0)):
        stay = [
            make_line("a", 700.0, mono=mono),
            make_line("b", 700.0 - limit, mono=mono),
        ]
        split = [
            make_line("a", 700.0, mono=mono),
            make_line("b", 699.5 - limit, mono=mono),
        ]
        assert _groups(stay) == [["a", "b"]]
        assert _groups(split) == [["a"], ["b"]]
        assert _flows(split) == [False, False]


def test_spacing_uses_the_larger_size() -> None:
    """The limit of a gap is measured with the larger of the two sizes."""
    # 15.5 pt is beyond 1.5 x 10 pt, but not beyond 1.5 x 10.5 pt.
    assert _groups(
        [make_line("a", 700.0), make_line("b", 684.5, size=10.5)],
    ) == [["a", "b"]]
    assert _groups(
        [make_line("a", 700.0, size=10.5), make_line("b", 684.5)],
    ) == [["a", "b"]]


def test_paragraph_pitches() -> None:
    """The pitch of the wrapped lines of paragraphs is measured per size."""
    lines = [
        make_line("one two three four five six", 700.0),
        make_line("seven eight nine ten eleven", 682.5),
        make_line("twelve thirteen fourteen fifteen", 665.0),
        make_line("end.", 647.5),
        make_line("A short line", 600.0),
        make_line("after it", 590.0),
    ]
    assert paragraph_pitches(lines) == {10.0: 17.5}
    code = [
        make_line("x = 1 + 2 + 3 + 4", 700.0, mono=True),
        make_line("y = 2", 690.0, mono=True),
    ]
    assert paragraph_pitches(code) == {}


# Four lines whose three pairs are never measured, each kind for one reason
# only. In all kinds but the last, the lines are 12 pt apart and their pairs
# do not wrap a paragraph. The lines on the same baseline wrap one, but their
# pitch is not positive.
_WORDS = "one two three four five"
_NOT_WRAPS = {
    "short line": [
        make_line("one two three four", 600.0 - 12 * i) for i in range(4)
    ],
    "sentence end": [
        make_line(f"{_WORDS}.", 600.0 - 12 * i) for i in range(4)
    ],
    "list item": [
        make_line(f"\U00002022 {_WORDS}", 600.0 - 12 * i) for i in range(4)
    ],
    "code": [make_line(_WORDS, 600.0 - 12 * i, mono=True) for i in range(4)],
    "new page": [make_line(_WORDS, 600.0 - 12 * i, page=i) for i in range(4)],
    "bold": [
        make_line(_WORDS, 600.0 - 12 * i, bold=i % 2 == 1) for i in range(4)
    ],
    "same baseline": [make_line(_WORDS, 600.0) for _ in range(4)],
}


@pytest.mark.parametrize("kind", list(_NOT_WRAPS))
def test_paragraph_pitches_skip_the_other_pairs(kind: str) -> None:
    """Pairs that do not wrap a paragraph are not measured, however many."""
    wrapped = [
        make_line("one two three four five six", 700.0),
        make_line("seven eight nine ten eleven", 686.0),
        make_line("twelve thirteen fourteen end.", 672.0),
    ]
    assert paragraph_pitches(wrapped + _NOT_WRAPS[kind]) == {10.0: 14.0}


def test_paragraph_pitches_prefer_the_smaller_on_a_tie() -> None:
    """Of two pitches as frequent as each other, the smaller one is kept."""
    lines = [
        make_line("one two three four five six", 700.0),
        make_line("seven eight nine ten eleven", 686.0),
        make_line("twelve thirteen fourteen end.", 672.0),
        make_line("one two three four five six", 640.0),
        make_line("seven eight nine ten eleven", 628.0),
        make_line("twelve thirteen fourteen end.", 616.0),
    ]
    assert paragraph_pitches(lines) == {10.0: 12.0}


def test_paragraph_pitches_round_the_larger_size() -> None:
    """A pair has the larger size of its lines, rounded to half a point."""
    mixed = [
        make_line("one two three four five six", 700.0),
        make_line("seven eight nine ten eleven", 686.0, size=10.5),
        make_line("twelve thirteen fourteen fifteen", 672.0),
    ]
    assert paragraph_pitches(mixed) == {10.5: 14.0}
    # A quarter of a point goes to the even half point, as Python rounds: 20.5
    # half points make 20, and 21.5 make 22.
    quarters = [
        make_line("one two three four five six", 700.0, size=10.25),
        make_line("seven eight nine ten end.", 686.25, size=10.25),
        make_line("one two three four five six", 600.0, size=10.75),
        make_line("seven eight nine ten end.", 585.75, size=10.75),
    ]
    assert paragraph_pitches(quarters) == {10.0: 14.0, 11.0: 14.0}


def test_wide_line_spacing_keeps_paragraphs_whole() -> None:
    """Lines 1.75 sizes apart stay together, a wider gap still splits."""
    words = "word " * 8
    first = [make_line(f"{words}{i}", 700.0 - 17.5 * i) for i in range(6)]
    first.append(make_line("end of it.", 595.0))
    second = [make_line(f"{words}{i}", 571.5 - 17.5 * i) for i in range(3)]
    blocks = group_lines(first + second)
    assert [len(block.lines) for block in blocks] == [7, 3]


def test_small_paragraph_space_splits_wide_lines() -> None:
    """Lines 1.75 sizes apart split at a gap beyond 1.1 pitches."""
    words = "word " * 8
    first = [make_line(f"{words}{i}", 700.0 - 17.5 * i) for i in range(6)]
    first.append(make_line("end of it.", 595.0))
    # The pitch is 17.5 pt, and 1.1 pitches make 19.25 pt. A space of 3 pt
    # after the paragraph, a gap of 20.5 pt, splits it, as does a gap of
    # 19.5 pt; a gap of 19 pt does not.
    for gap, count in ((20.5, 2), (19.5, 2), (19.0, 1)):
        second = [
            make_line(f"{words}{i}", 595.0 - gap - 17.5 * i) for i in range(3)
        ]
        assert len(group_lines(first + second)) == count


def test_first_line_indent_starts_a_paragraph() -> None:
    """A line indented after the end of a sentence starts a paragraph."""
    lines = [
        make_line("The first paragraph runs over a line", 700.0),
        make_line("and ends here.", 686.0),
        make_line("The second one is indented and runs", 672.0, x0=87.0),
        make_line("on the next line.", 658.0),
        make_line("Then a sentence ends at the margin.", 644.0),
        make_line("A line at the margin goes on.", 630.0),
    ]
    assert _groups(lines) == [
        ["The first paragraph runs over a line", "and ends here."],
        [
            "The second one is indented and runs",
            "on the next line.",
            "Then a sentence ends at the margin.",
            "A line at the margin goes on.",
        ],
    ]


def test_indents_inside_items_and_entries_continue_them() -> None:
    """An indent inside a list item, or after a comma, starts nothing."""
    item = [
        make_line("\U00002022 Install the package.", 700.0),
        make_line("Then restart it.", 688.0, x0=84.0),
        make_line("\U00002022 Next item", 676.0),
    ]
    assert _groups(item) == [
        ["\U00002022 Install the package.", "Then restart it."],
        ["\U00002022 Next item"],
    ]
    entry = [
        make_line("Knuth, D. The Art of Computer Programming,", 700.0),
        make_line("volume 1, third edition.", 688.0, x0=90.0),
    ]
    assert len(group_lines(entry)) == 1


def test_hanging_indents_continue_their_entries() -> None:
    """Lines indented after a sentence end, in a hanging indent, start none."""
    lamport = [
        make_line("Lamport, L. LaTeX: A Document Preparation System.", 700.0),
        make_line("Addison-Wesley, 1994.", 688.0, x0=90.0),
    ]
    # No wrapped line shows whether the first lines are indented.
    assert len(group_lines(lamport)) == 1
    # A wrap to the hanging indent shows that they are not.
    knuth = [
        make_line("Knuth, D. E. The Art of Computer Programming,", 676.0),
        make_line("volume 1. Addison-Wesley, 1997.", 664.0, x0=90.0),
    ]
    assert len(group_lines(lamport + knuth)) == 1
    # A wrap back to the margin, after an entry that ends without a full
    # stop, only ties with it.
    tufte = [
        make_line("Tufte, E. R. The Visual Display of Information", 676.0),
        make_line("Graphics Press, Cheshire, CT, USA", 664.0, x0=90.0),
        make_line("Strunk, W. The Elements of Style.", 652.0),
        make_line("Longman, 1999.", 640.0, x0=90.0),
    ]
    assert len(group_lines(lamport + tufte)) == 1


def test_indented_one_liners_each_start_a_paragraph() -> None:
    """Indented paragraphs of a single line each start a block."""
    lines = [
        make_line("The last line of a long paragraph ends here.", 700.0),
        make_line("A short paragraph.", 688.0, x0=87.0),
        make_line("Another short one.", 676.0, x0=87.0),
        make_line("A long one opens with its indent and", 664.0, x0=87.0),
        make_line("goes on at the margin.", 652.0),
    ]
    assert _groups(lines) == [
        ["The last line of a long paragraph ends here."],
        ["A short paragraph."],
        ["Another short one."],
        ["A long one opens with its indent and", "goes on at the margin."],
    ]


def test_indent_without_wraps_starts_a_paragraph() -> None:
    """Without a wrap, most lines at the margin make an indent count."""
    lines = [
        make_line("The first paragraph opens on this line.", 700.0),
        make_line("It goes on at the margin and ends here.", 688.0),
        make_line("The second one is indented, and ends.", 676.0, x0=87.0),
        make_line("It goes on at the margin too and ends.", 664.0),
    ]
    assert _groups(lines) == [
        [
            "The first paragraph opens on this line.",
            "It goes on at the margin and ends here.",
        ],
        [
            "The second one is indented, and ends.",
            "It goes on at the margin too and ends.",
        ],
    ]


def test_hanging_wraps_outweigh_the_lines_at_the_margin() -> None:
    """A wrap to a hanging indent wins over most lines at the margin."""
    lines = [
        make_line("Knuth, D. The Art of Computer Programming,", 700.0),
        make_line("volume 1. Addison-Wesley, 1997.", 688.0, x0=90.0),
        make_line("Lamport, L. LaTeX: A Document Preparation System.", 676.0),
        make_line("Addison-Wesley, 1994.", 664.0, x0=90.0),
        make_line("Tufte, E. Envisioning Information.", 652.0),
        make_line("Wirth, N. Algorithms and Data Structures.", 640.0),
    ]
    assert [len(block.lines) for block in group_lines(lines)] == [6]


def test_lines_of_an_indented_quote_stay_together() -> None:
    """A sentence end in a quote, from its second line on, starts nothing."""
    lines = [
        make_line("A paragraph opens with its indent and", 700.0, x0=87.0),
        make_line("introduces a quote:", 688.0),
        make_line("The quote is set apart from the text and", 676.0, x0=100.0),
        make_line("runs over a few lines. Here it ends.", 664.0, x0=100.0),
        make_line("And then it goes on for a while.", 652.0, x0=100.0),
    ]
    assert _groups(lines) == [
        ["A paragraph opens with its indent and", "introduces a quote:"],
        [
            "The quote is set apart from the text and",
            "runs over a few lines. Here it ends.",
            "And then it goes on for a while.",
        ],
    ]


def test_items_keep_their_indented_lines() -> None:
    """An item keeps its indented lines where first lines are indented."""
    lines = [
        make_line("\U00002022 Install it with the options", 700.0, x0=87.0),
        make_line("and restart the service once it is installed.", 688.0),
        make_line("Then check that it runs.", 676.0, x0=87.0),
    ]
    assert len(group_lines(lines)) == 1


def test_indented_code_after_a_colon_stays_whole() -> None:
    """A line of code indented after a colon goes on with its sample."""
    lines = [
        make_line("for item in items:", 700.0, mono=True),
        make_line("total += item", 689.0, x0=96.0, mono=True),
        make_line("print(total)", 678.0, mono=True),
    ]
    (block,) = group_lines(lines)
    assert (len(block.lines), block.code) == (3, True)


def test_list_items_and_continuations() -> None:
    """Each bullet starts an item, and an indented line continues it."""
    lines = [
        make_line("\U00002022 First item", 640.0),
        make_line("continued here", 628.0, x0=84.0),
        make_line("\U00002022 Second item", 616.0),
    ]
    assert _groups(lines) == [
        ["\U00002022 First item", "continued here"],
        ["\U00002022 Second item"],
    ]


def test_numbered_items_start_blocks() -> None:
    """A number starts an item, but a decimal or an abbreviation does not."""
    lines = [
        make_line("1. Install", 700.0),
        make_line("2) Configure", 688.0),
        make_line("3.14 is required", 676.0),
        make_line("e.g. this", 664.0),
    ]
    assert _groups(lines) == [
        ["1. Install"],
        ["2) Configure", "3.14 is required", "e.g. this"],
    ]


def test_code_is_not_split_on_what_looks_like_a_marker() -> None:
    """A line of code that starts like a bullet or a number stays in code."""
    lines = [
        make_line("x = 1", 700.0, mono=True),
        make_line("- y", 689.0, mono=True),
        make_line("1. z", 678.0, mono=True),
    ]
    (block,) = group_lines(lines)
    assert (len(block.lines), block.code) == (3, True)


def test_blank_numbered_lines_stay_in_their_code() -> None:
    """A lone number that goes on with the numbers of code is a blank line."""
    lines = [
        make_line("1 import os", 700.0, mono=True),
        make_line("2", 688.0, size=6.0),
        make_line("3 def main():", 676.0, mono=True),
        make_line("4 main()", 664.0, mono=True),
    ]
    assert [(len(b.lines), b.code) for b in group_lines(lines)] == [(4, True)]
    other = [
        make_line("1 import os", 700.0, mono=True),
        make_line("7", 688.0, size=6.0),
    ]
    assert [b.code for b in group_lines(other)] == [True, False]


def test_blank_numbered_lines_follow_each_other() -> None:
    """Blank numbered lines in a row take the size of the code before them."""
    lines = [
        make_line("8 import os", 700.0, mono=True),
        make_line("9", 688.0, size=6.0),
        make_line("10", 676.0, size=6.0),
        make_line("11 main()", 664.0, mono=True),
    ]
    (block,) = group_lines(lines)
    assert {(line.mono, line.size) for line in block.lines} == {(True, 10.0)}


@pytest.mark.parametrize(
    ("first", "number", "mono"),
    [
        ("1 import os", "3", True),
        ("1 import os", "1", True),
        ("1 import os", "2 apples", True),
        ("999999999 x", "1000000000", True),
        ("1 import os", "2", False),
    ],
    ids=["skipped", "repeated", "more", "ten-digits", "prose"],
)
def test_other_lone_numbers_stay_text(
    first: str,
    number: str,
    *,
    mono: bool,
) -> None:
    """A number is a blank line only if it goes on with the code before it."""
    lines = [
        make_line(first, 700.0, mono=mono),
        make_line(number, 688.0, size=6.0),
    ]
    last = group_lines(lines)[-1].lines[-1]
    assert (last.mono, last.size) == (False, 6.0)


@pytest.mark.parametrize(
    ("page", "x0"),
    [(1, 72.0), (0, 300.0)],
    ids=["page", "column"],
)
def test_blank_numbered_lines_open_pages_and_columns(
    page: int,
    x0: float,
) -> None:
    """A blank numbered line that opens a page or a column stays in code."""
    lines = [
        make_line("1 import os", 80.0, mono=True),
        make_line("2", 780.0, size=6.0, page=page, x0=x0),
        make_line("3 main()", 768.0, page=page, x0=x0, mono=True),
    ]
    blocks = group_lines(lines)
    assert [(b.code, b.flow_start) for b in blocks] == [
        (True, False),
        (True, True),
    ]
    assert {(line.mono, line.size) for line in blocks[1].lines} == {
        (True, 10.0),
    }
    assert len(merge_flows(blocks, Counter())) == 1


def test_code_and_style_changes() -> None:
    """The end of a code sample, or a change of boldness, opens a block."""
    lines = [
        make_line("x = 1", 600.0, mono=True),
        make_line("y = 2", 589.0, mono=True),
        make_line("z = 3", 567.0, mono=True),
        make_line("After code.", 540.0),
        make_line("Note", 520.0, bold=True),
        make_line("Body text.", 506.0),
    ]
    blocks = group_lines(lines)
    assert [len(b.lines) for b in blocks] == [3, 1, 1, 1]
    assert [b.code for b in blocks] == [True, False, False, False]


def test_size_limit_is_strict() -> None:
    """A size gap starts a new block only beyond half a point."""
    assert _groups(
        [make_line("a", 700.0), make_line("b", 688.0, size=10.5)],
    ) == [["a", "b"]]
    assert _groups(
        [make_line("a", 700.0), make_line("b", 688.0, size=10.6)],
    ) == [["a"], ["b"]]


def test_flow_breaks() -> None:
    """A new page or column, or a line that goes up, opens a flow."""
    blocks = group_lines(
        [
            make_line("bottom of page", 80.0),
            make_line("top of next page", 780.0, page=1),
        ],
    )
    assert [b.flow_start for b in blocks] == [False, True]
    blocks = group_lines(
        [
            make_line("bottom of left column", 80.0),
            make_line("top of right column", 780.0, x0=300.0),
        ],
    )
    assert [b.flow_start for b in blocks] == [False, True]


def test_flow_breaks_each_on_its_own() -> None:
    """A page, a rise or a line aside breaks the flow without the others."""
    # At the same place, but on the next page.
    assert _flows([make_line("a", 700.0), make_line("b", 688.0, page=1)]) == [
        False,
        True,
    ]
    # Below the previous line, but wholly to its right.
    assert _flows(
        [make_line("a", 700.0, width=100.0), make_line("b", 688.0, x0=200.0)],
    ) == [False, True]
    # Above the previous line, at the same place: at 10 pt, the baseline has
    # to rise by more than 5 pt.
    assert _flows([make_line("a", 700.0), make_line("b", 705.0)]) == [False]
    assert _flows([make_line("a", 700.0), make_line("b", 705.5)]) == [
        False,
        True,
    ]


def test_flow_break_wins_over_the_other_rules() -> None:
    """A line that breaks the flow and changes the style opens a flow."""
    lines = [
        make_line("Text at the bottom of a page", 80.0),
        make_line("Heading", 780.0, page=1, size=14.0, bold=True),
    ]
    assert _flows(lines) == [False, True]


def test_merge_flows() -> None:
    """A sentence cut by a break is joined, unless the break is natural."""
    counter: Counter[str] = Counter()
    split = group_lines(
        [
            make_line("the scheduler processes 1,000 jobs per", 80.0),
            make_line("minute on a single core.", 780.0, page=1),
        ],
    )
    (merged,) = merge_flows(split, counter)
    assert [line.text for line in merged.lines] == [
        "the scheduler processes 1,000 jobs per",
        "minute on a single core.",
    ]
    assert counter["page_break"] == 1
    for first, second in (
        (
            make_line("A complete sentence.", 80.0),
            make_line("next page text", 780.0, page=1),
        ),
        (
            make_line("an open sentence", 80.0),
            make_line("\U00002022 item", 780.0, page=1),
        ),
        (
            make_line("Bold heading", 80.0, bold=True),
            make_line("text", 780.0, page=1),
        ),
        (
            make_line("small text", 80.0, size=8.0),
            make_line("text", 780.0, page=1),
        ),
    ):
        assert len(merge_flows(group_lines([first, second]), Counter())) == 2
    code = group_lines(
        [
            make_line("x = 1", 80.0, mono=True),
            make_line("y = 2", 780.0, page=1, mono=True),
        ],
    )
    assert len(merge_flows(code, Counter())) == 1


@pytest.mark.parametrize(
    ("ending", "merged"),
    [
        ("1,000 jobs per", True),
        ("first, second,", True),
        ("a word cut by a hyphen-", True),
        ("requires Python 3.14", True),
        ("It ends here.", False),
        ("Is it?", False),
        ("Stop!", False),
        ("as follows:", False),
        ("first; second;", False),
        ('He said "go."', False),
        ("(see above.)", False),
        ("[1, 2, 3.]", False),
        ("\U0000201cDone.\U0000201d", False),
        ("it\U00002019s done.\U00002019", False),
        ("\xabDone.\xbb", False),
    ],
)
def test_merge_follows_the_end_of_the_sentence(
    ending: str,
    *,
    merged: bool,
) -> None:
    """Only a line that does not end like a sentence joins the next page."""
    first = make_line(ending, 80.0)
    assert _joined(first, make_line("go on", 780.0, page=1)) is merged


def test_merge_bold_and_size_limits() -> None:
    """Bold text, or a size gap beyond half a point, stops the merge."""
    open_line = make_line("an open sentence", 80.0)
    assert _joined(open_line, make_line("go on", 780.0, page=1))
    assert not _joined(open_line, make_line("Go on", 780.0, page=1, bold=True))
    # The sizes may differ by half a point, but not more.
    assert _joined(open_line, make_line("go on", 780.0, page=1, size=10.5))
    assert not _joined(open_line, make_line("go on", 780.0, page=1, size=10.6))
    assert _joined(
        make_line("an open sentence", 80.0, size=10.5),
        make_line("go on", 780.0, page=1),
    )


def test_merge_chains_over_several_pages() -> None:
    """A block that spans pages is one block, and each join is counted."""
    counter: Counter[str] = Counter()
    lines = [
        make_line("A first paragraph.", 700.0),
        make_line("The scheduler", 80.0, page=1),
        make_line("runs the jobs", 780.0, page=2),
        make_line("in parallel.", 780.0, page=3),
        make_line("A new page.", 780.0, page=4),
    ]
    blocks = group_lines(lines)
    first, spanning, last = merge_flows(blocks, counter)
    assert [line.text for line in first.lines] == ["A first paragraph."]
    assert [line.page for line in spanning.lines] == [1, 2, 3]
    # The merged block opens where its first block did.
    assert (spanning.flow_start, spanning.code) == (True, False)
    assert [line.text for line in last.lines] == ["A new page."]
    assert counter["page_break"] == 2
    assert len(blocks) == 5  # The blocks that were given are left alone.


def test_merge_needs_a_flow_start() -> None:
    """Blocks of the same flow are never joined, even after an open line."""
    counter: Counter[str] = Counter()
    split = group_lines(
        [make_line("an open sentence", 700.0), make_line("next", 660.0)],
    )
    assert len(merge_flows(split, counter)) == 2
    assert counter["page_break"] == 0


def test_merge_keeps_code_and_text_apart() -> None:
    """Code is not joined to text across a break, nor text to code."""
    assert not _joined(
        make_line("x = 1", 80.0, mono=True),
        make_line("then the text", 780.0, page=1),
    )
    assert not _joined(
        make_line("an open sentence", 80.0),
        make_line("y = 2", 780.0, page=1, mono=True),
    )


def test_merge_counts_the_joins_of_code() -> None:
    """The join of two code blocks is counted as well."""
    counter: Counter[str] = Counter()
    code = group_lines(
        [
            make_line("x = 1", 80.0, mono=True),
            make_line("y = 2", 780.0, page=1, mono=True),
        ],
    )
    (merged,) = merge_flows(code, counter)
    assert (merged.code, merged.flow_start) == (True, False)
    assert counter["page_break"] == 1


def test_merge_stops_at_a_change_of_page_kind() -> None:
    """A block is never joined across a change of page kind."""
    split = group_lines(
        [
            make_line("Limits . . . . . 3", 80.0),
            make_line("This guide explains the service.", 780.0, page=1),
        ],
    )
    toc = {0: BlockKind.TOC}
    assert len(merge_flows(split, Counter(), page_kinds=toc)) == 2
    assert len(merge_flows(split, Counter())) == 1


@pytest.mark.parametrize("number", ["1", "iv", "Page 3"])
def test_merge_never_continues_a_page_number(number: str) -> None:
    """A continuation is never joined to a block that is a page number."""
    split = group_lines(
        [
            make_line(number, 80.0),
            make_line("cache rule quota archive", 780.0, page=1),
        ],
    )
    assert len(merge_flows(split, Counter())) == 2


def test_empty_input() -> None:
    """There is nothing to group or to merge without a line."""
    assert group_lines([]) == []
    assert merge_flows([], Counter()) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\U00002022 First item", ("First item", "bullet")),
        ("\U00002013 Python 3.14", ("Python 3.14", "bullet")),
        ("\U00002014 Dash", ("Dash", "bullet")),
        ("\xb7 Dot", ("Dot", "bullet")),
        ("\U0000f0b7 Word bullet", ("Word bullet", "bullet")),
        ("\U0000f0a7 Square", ("Square", "bullet")),
        ("\U0000f0d8 Arrow", ("Arrow", "bullet")),
        ("\U0000f076 Diamond", ("Diamond", "bullet")),
        ("\U0000f0fc Check", ("Check", "bullet")),
        ("\U0000f0b7Tight", ("Tight", "bullet")),
        ("\U0000f0b7", ("\U0000f0b7", None)),
        ("\U00002022Tight", ("\U00002022Tight", None)),
        ("- x", ("x", "bullet")),
        ("1. Install", ("Install", "list_number")),
        ("12) Twelve", ("Twelve", "list_number")),
        ("a) Option", ("Option", "list_number")),
        ("(iv) Last", ("Last", "list_number")),
        ("3.14 is required", ("3.14 is required", None)),
        ("e.g. this", ("e.g. this", None)),
        ("1234. A year", ("1234. A year", None)),
        ("-5 degrees", ("-5 degrees", None)),
        ("\U00002022", ("\U00002022", None)),
        ("Plain text", ("Plain text", None)),
        ("", ("", None)),
    ],
)
def test_strip_list_marker(
    text: str,
    expected: tuple[str, str | None],
) -> None:
    """The bullet or the number of a list item is removed and named."""
    assert strip_list_marker(text) == expected
