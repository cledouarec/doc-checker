# Copyright 2026 Christophe Le Douarec
"""Tests for the text of draft blocks: hyphenation, indentation and numbers."""

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.extract.pdf.blocks import DraftBlock
from doc_checker.extract.pdf.joining import (
    code_text,
    is_hyphen_break,
    prose_text,
    vocabulary,
)
from tests.pdf_builders import make_line

if TYPE_CHECKING:
    from doc_checker.extract.pdf.layout import Line

CODE_CHAR = 5.4  # Width of a character of the code font, in points.


def _prose(
    *texts: str,
    vocabulary: frozenset[str] = frozenset(),
) -> tuple[str, Counter[str]]:
    """Join lines that follow one another, and return the text and counts."""
    counter: Counter[str] = Counter()
    lines = [make_line(text, 700.0 - 12 * i) for i, text in enumerate(texts)]
    return prose_text(lines, vocabulary, counter), counter


def _block(*texts: str, code: bool = False) -> DraftBlock:
    """Make a block of lines that follow one another."""
    lines = tuple(
        make_line(text, 700.0 - 12 * i, mono=code)
        for i, text in enumerate(texts)
    )
    return DraftBlock(lines=lines, code=code, flow_start=False)


def _code_line(
    text: str,
    baseline: float,
    *,
    column: float = 0.0,
    page: int = 0,
) -> Line:
    """Make a line of code that starts `column` characters after the margin."""
    return make_line(
        text,
        baseline,
        x0=72.0 + column * CODE_CHAR,
        width=len(text) * CODE_CHAR,
        size=9.0,
        page=page,
        mono=True,
    )


def test_lines_are_joined_with_spaces() -> None:
    """Lines are joined with a space, and each join is counted."""
    text, counter = _prose("Install the", "package now.")
    assert (text, counter["line_break"]) == ("Install the package now.", 1)


def test_hyphenation() -> None:
    """A hyphen goes if the document has the whole word, and stays if not."""
    assert _prose(
        "the configu-",
        "ration file.",
        vocabulary=frozenset({"configuration"}),
    )[0] == ("the configuration file.")
    text, counter = _prose("internet connec-", "tion is needed.")
    assert (text, counter["hyphenation"], counter["line_break"]) == (
        "internet connec-tion is needed.",
        1,
        0,
    )
    assert _prose("the non-", "blocking mode")[0] == "the non-blocking mode"


def test_a_hyphen_after_a_blank_is_no_break() -> None:
    """A hyphen after a blank does not join lines."""
    assert _prose("value -", "1 here.")[0] == "value - 1 here."


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("built for x86-", "64 only.", "built for x86-64 only."),
        ("signed with SHA-", "256 digests.", "signed with SHA-256 digests."),
        ("over Wi-", "Fi links.", "over Wi-Fi links."),
        ("an IPv4-", "only gateway.", "an IPv4-only gateway."),
        ("see Python-", "Next steps.", "see Python-Next steps."),
    ],
)
def test_compound_words_keep_their_hyphen(
    first: str,
    second: str,
    expected: str,
) -> None:
    """A hyphen glued to a letter or a digit joins lines, and stays."""
    text, counter = _prose(first, second, vocabulary=frozenset({"ipv4only"}))
    assert (text, counter["hyphenation"], counter["line_break"]) == (
        expected,
        1,
        0,
    )


def test_hyphenation_ignores_case() -> None:
    """The word that a hyphen splits is looked up in lower case."""
    text, _ = _prose(
        "the Configu-",
        "ration file.",
        vocabulary=frozenset({"configuration"}),
    )
    assert text == "the Configuration file."


@pytest.mark.parametrize("hyphen", ["-", "\U00002010", "\xad"])
def test_hyphenation_characters(hyphen: str) -> None:
    """Each kind of hyphen goes if the word is known, and stays if not."""
    known = frozenset({"configuration"})
    text, _ = _prose(f"the configu{hyphen}", "ration file.", vocabulary=known)
    assert text == "the configuration file."
    text, _ = _prose(f"the configu{hyphen}", "ration file.")
    assert text == f"the configu{hyphen}ration file."


def test_hyphenation_of_a_compound_word() -> None:
    """The word before the hyphen is a whole token, hyphens inside included."""
    text, _ = _prose(
        "the non-block-",
        "ing mode",
        vocabulary=frozenset({"non-blocking"}),
    )
    assert text == "the non-blocking mode"


def test_hyphenation_before_a_symbol() -> None:
    """A lowercase symbol that is no word, such as a circled a, is no error."""
    text, counter = _prose("the configu-", "\U000024d0")
    assert (text, counter["hyphenation"]) == ("the configu-\U000024d0", 1)


def test_breaks_are_counted_one_by_one() -> None:
    """A paragraph counts its hyphen breaks and its other line breaks."""
    text, counter = _prose(
        "A short configu-",
        "ration of the con-",
        "nection layer is",
        "needed.",
        vocabulary=frozenset({"configuration"}),
    )
    assert text == "A short configuration of the con-nection layer is needed."
    assert (counter["hyphenation"], counter["line_break"]) == (2, 1)


def test_one_line_is_unchanged() -> None:
    """A lone line gives its text, and nothing is counted."""
    text, counter = _prose("Install the package.")
    assert text == "Install the package."
    assert not counter


@pytest.mark.parametrize(
    ("text", "following"),
    [
        ("the configu-", "ration"),
        ("the configu\U00002010", "ration"),
        ("the configu\xad", "ration"),
        ("un café-", "établi"),
    ],
)
def test_hyphen_break(text: str, following: str) -> None:
    """A letter, a hyphen and a lowercase letter on the next line break."""
    first, second = make_line(text, 700.0), make_line(following, 688.0)
    assert is_hyphen_break(first, second)


@pytest.mark.parametrize(
    ("text", "following"),
    [
        ("the configu-", "Ration"),
        ("the configu-", "1st"),
        ("the configu-", "(ration"),
        ("value -", "ration"),
        ("section 3-", "ration"),
        ("the configu\U00002013", "ration"),
        ("the configu", "ration"),
        ("-", "ration"),
    ],
)
def test_no_hyphen_break(text: str, following: str) -> None:
    """Any other end of a line, or start of the next one, is no break."""
    first, second = make_line(text, 700.0), make_line(following, 688.0)
    assert not is_hyphen_break(first, second)


def test_vocabulary_skips_fragments() -> None:
    """The pieces of a word that a hyphen splits are not in the vocabulary."""
    block = DraftBlock(
        lines=(
            make_line("the configuration file.", 700.0),
            make_line("internet connec-", 688.0),
            make_line("tion is needed.", 676.0),
        ),
        code=False,
        flow_start=False,
    )
    assert vocabulary([block]) == frozenset(
        {"the", "configuration", "file", "internet", "is", "needed"},
    )


def test_vocabulary_words() -> None:
    """Words, lower case, from every block, code included, numbers left out."""
    blocks = [
        _block("The Cache holds 100 entries,", "non-blocking mode."),
        _block("Config = load()", code=True),
    ]
    prose = {"the", "cache", "holds", "entries", "non-blocking", "mode"}
    code = {"config", "load"}
    assert vocabulary(blocks) == prose | code


def test_vocabulary_breaks_stay_inside_a_block() -> None:
    """Only consecutive lines of one block make a hyphen break."""
    blocks = [_block("the connec-"), _block("tion is here.")]
    assert vocabulary(blocks) == frozenset(
        {"the", "connec", "tion", "is", "here"},
    )


def test_vocabulary_word_cut_twice() -> None:
    """A line between two hyphen breaks loses its first and its last word."""
    block = _block("configu-", "ra-", "tion is here.")
    assert vocabulary([block]) == frozenset({"is", "here"})


def test_code_indentation_and_runs() -> None:
    """The indentation comes from the positions, run by run."""
    lines = [
        make_line("def main():", 600.0, width=11 * 5.4, size=9.0, mono=True),
        make_line(
            "return 0",
            589.0,
            x0=72.0 + 4 * 5.4,
            width=8 * 5.4,
            size=9.0,
            mono=True,
        ),
        make_line(
            "main()",
            780.0,
            x0=300.0,
            width=6 * 5.4,
            size=9.0,
            page=1,
            mono=True,
        ),
    ]
    assert code_text(lines, Counter()) == "def main():\n    return 0\nmain()"


def test_code_width_comes_from_the_longest_line() -> None:
    """A short line, with a width that is not typical, sets no indentation."""
    lines = [
        make_line("x", 600.0, width=4.0, size=9.0, mono=True),
        _code_line("return 0", 589.0, column=2),
    ]
    assert code_text(lines, Counter()) == "x\n  return 0"


def test_code_a_line_that_goes_up_starts_a_run() -> None:
    """The lines of a second column have a margin of their own."""
    lines = [
        _code_line("def f():", 600.0),
        _code_line("return 1", 589.0, column=4),
        _code_line("def g():", 700.0, column=30),
        _code_line("return 2", 689.0, column=34),
    ]
    assert code_text(lines, Counter()) == (
        "def f():\n    return 1\ndef g():\n    return 2"
    )


def test_code_a_small_rise_keeps_the_run() -> None:
    """The baseline has to rise by more than half a size to start a run."""
    # At 9 pt, the limit is 4.5 pt.
    for rise, expected in ((4.5, "a\n    b"), (5.0, "a\nb")):
        lines = [
            _code_line("a", 600.0),
            _code_line("b", 600.0 + rise, column=4),
        ]
        assert code_text(lines, Counter()) == expected


def test_code_far_indentation_keeps_the_run() -> None:
    """A line indented beyond the end of the previous one is in its run."""
    lines = [_code_line("x = f(", 600.0), _code_line("1)", 589.0, column=30)]
    assert code_text(lines, Counter()) == "x = f(\n" + " " * 30 + "1)"


def test_code_without_width() -> None:
    """Lines with no width cannot be measured: no indentation, no error."""
    first = replace(make_line("a", 600.0, mono=True), x1=72.0)
    second = replace(make_line("b", 589.0, x0=100.0, mono=True), x1=100.0)
    assert code_text([first, second], Counter()) == "a\nb"


def test_code_line_numbers() -> None:
    """Consecutive line numbers go, with the blanks that all lines share."""
    counter: Counter[str] = Counter()
    numbered = [
        make_line("1  def main():", 600.0, mono=True),
        make_line("2      return 0", 589.0, mono=True),
    ]
    assert code_text(numbered, counter) == "def main():\n    return 0"
    assert counter["code_line_number"] == 2
    gaps = [
        make_line("1 a", 600.0, mono=True),
        make_line("3 b", 589.0, mono=True),
    ]
    assert code_text(gaps, Counter()) == "1 a\n3 b"
    assert (
        code_text([make_line("42 is the answer", 600.0, mono=True)], Counter())
        == "42 is the answer"
    )


def test_code_line_numbers_across_a_break() -> None:
    """Numbers go on in the next page, and may be aligned on the right."""
    counter: Counter[str] = Counter()
    lines = [
        _code_line("9  a = 1", 600.0, column=1),
        _code_line("10  b = 2", 589.0),
        _code_line("11  c = 3", 780.0, page=1),
        _code_line("12      d = 4", 769.0, page=1),
    ]
    assert code_text(lines, counter) == "a = 1\nb = 2\nc = 3\n    d = 4"
    assert counter["code_line_number"] == 4


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        pytest.param(
            ["1  import os", "2", "3  def main():", "4      return 0"],
            "import os\ndef main():\n    return 0",
            id="in the middle",
        ),
        pytest.param(
            ["1", "2  a = 1", "3  b = 2"],
            "a = 1\nb = 2",
            id="first",
        ),
        pytest.param(
            ["1  a = 1", "2  b = 2", "3"],
            "a = 1\nb = 2",
            id="last",
        ),
        pytest.param(
            ["1  a = 1", "2", "3", "4  b = 2"],
            "a = 1\nb = 2",
            id="two in a row",
        ),
        pytest.param(
            ["9  a = 1", "10", "11  b = 2"],
            "a = 1\nb = 2",
            id="numbers of two digits",
        ),
    ],
)
def test_code_line_numbers_with_empty_lines(
    texts: list[str],
    expected: str,
) -> None:
    """An empty line has only its number: it goes, and it is counted."""
    counter: Counter[str] = Counter()
    lines = [_code_line(t, 600.0 - 11 * i) for i, t in enumerate(texts)]
    assert code_text(lines, counter) == expected
    assert counter["code_line_number"] == len(texts)


def test_code_empty_line_between_right_aligned_numbers() -> None:
    """The number of an empty line may be indented like the others."""
    counter: Counter[str] = Counter()
    lines = [
        _code_line("8  a = 1", 600.0, column=1),
        _code_line("9", 589.0, column=1),
        _code_line("10  b = 2", 578.0),
    ]
    assert code_text(lines, counter) == "a = 1\nb = 2"
    assert counter["code_line_number"] == 3


def test_code_line_numbers_have_at_most_nine_digits() -> None:
    """Nine digits make a line number, ten or more are code."""
    nine = [
        _code_line("123456789  a", 600.0),
        _code_line("123456790  b", 589.0),
    ]
    assert code_text(nine, Counter()) == "a\nb"
    ten = [
        _code_line("1234567890  a", 600.0),
        _code_line("1234567891  b", 589.0),
    ]
    assert code_text(ten, Counter()) == "1234567890  a\n1234567891  b"


def test_code_huge_numbers_are_not_parsed() -> None:
    """Python refuses to parse 4300 digits or more: that must not raise."""
    digits = "9" * 5000
    lines = [
        _code_line(f"{digits} a", 600.0),
        _code_line(f"{digits} b", 589.0),
    ]
    assert code_text(lines, Counter()) == f"{digits} a\n{digits} b"


@pytest.mark.parametrize(
    "texts",
    [
        ["1 a", "b"],
        ["2 a", "1 b"],
        ["1 a", "1 b"],
        ["1 a", "2b"],
        ["10", "11", "12"],
        ["1", "2"],
    ],
)
def test_code_numbers_are_kept(texts: list[str]) -> None:
    """Digits stay if the lines do not count up one by one or hold no code."""
    counter: Counter[str] = Counter()
    lines = [_code_line(t, 600.0 - 11 * i) for i, t in enumerate(texts)]
    assert code_text(lines, counter) == "\n".join(texts)
    assert not counter
