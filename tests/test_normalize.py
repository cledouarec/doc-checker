# Copyright 2026 Christophe Le Douarec
"""Tests for text normalization."""

import time
from collections import Counter

import pytest

from doc_checker.text.normalize import (
    count_marks,
    loose_key,
    normalize_display,
    printable,
    strict_key,
    strip_accents,
)


def test_display_expands_ligatures() -> None:
    """Ligature code points become plain letters and are counted."""
    counter: Counter[str] = Counter()
    text = "con\U0000fb01guration"
    assert normalize_display(text, counter) == "configuration"
    assert counter["ligature"] == 1


def test_display_removes_invisibles_keeps_replacement_char() -> None:
    """Invisible characters are removed, U+FFFD is kept."""
    counter: Counter[str] = Counter()
    text = "a\xadb\U0000200bc\U0000fffd"
    assert normalize_display(text, counter) == "abc\U0000fffd"
    assert counter["invisible"] == 2


def test_display_applies_nfc() -> None:
    """Decomposed characters are composed."""
    assert normalize_display("e\U00000301t\xe9", Counter()) == "\xe9t\xe9"


def test_strict_key_unifies_spaces_quotes_dashes() -> None:
    """Special spaces, quotes and dashes are unified."""
    counter: Counter[str] = Counter()
    text = "l\U00002019option\xa0\U00002014 \xab\xa0x\xa0\xbb"
    assert strict_key(text, counter) == 'l\'option - "x"'
    assert counter["quote"] == 3
    assert counter["dash"] == 1


def test_strict_key_removes_french_spacing() -> None:
    """Spaces before French high punctuation are removed."""
    counter: Counter[str] = Counter()
    text = "Attention\U0000202f: oui\xa0?"
    assert strict_key(text, counter) == "Attention: oui?"
    assert counter["punct_spacing"] == 2


def test_strict_key_reads_a_run_of_spaces_once() -> None:
    """A long run of spaces costs one reading, whatever follows it."""
    counter: Counter[str] = Counter()
    start = time.perf_counter()
    assert strict_key("a" + " " * 200_000 + ",a", counter) == "a ,a"
    assert strict_key("a" + "\xa0" * 40_000 + ",a", counter) == "a ,a"
    assert counter["punct_spacing"] == 0
    assert strict_key("a" + " " * 200_000 + ";a", counter) == "a;a"
    assert strict_key("a" + "\xa0" * 40_000 + ";a", counter) == "a;a"
    assert counter["punct_spacing"] == 2
    assert time.perf_counter() - start < 1


def test_strict_key_collapses_whitespace_and_anchor_glyph() -> None:
    """Whitespace runs collapse and anchor glyphs disappear."""
    counter: Counter[str] = Counter()
    assert strict_key("  Install \n now\xb6 ", counter) == "Install now"
    assert counter["anchor_glyph"] == 1


def test_strict_key_ellipsis() -> None:
    """The ellipsis character becomes three dots."""
    assert strict_key("wait\U00002026", Counter()) == "wait..."


def test_strict_key_code_keeps_lines() -> None:
    """Code keeps its lines, indentation and quotes."""
    code = "\nx = 1   \n  y = \U0000201cz\U0000201d\n\n"
    expected = "x = 1\n  y = \U0000201cz\U0000201d"
    assert strict_key(code, Counter(), code=True) == expected


def test_loose_key_folds_case_accents_punctuation() -> None:
    """The loose key ignores case, accents and punctuation."""
    assert loose_key("L'Été, déjà !") == "l ete deja"


def test_strip_accents_and_count_marks() -> None:
    """Accents are stripped and counted."""
    assert strip_accents("été") == "ete"
    assert count_marks("été") == 2
    assert count_marks("ete") == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Page\x1b[2J 3", "Page [2J 3"),
        ("a\U0000202eb\U00002066c", "a b c"),
        ("  x\n\ty  ", "x y"),
        ("\x85x\x9b", "x"),
        ("caf\xe9 \U0001f600", "caf\xe9 \U0001f600"),
    ],
)
def test_printable(raw: str, expected: str) -> None:
    """Controls and bidi overrides become blanks, and blanks collapse."""
    assert printable(raw) == expected


def test_printable_cuts_long_texts() -> None:
    """A text longer than the limit is cut and ends with an ellipsis."""
    assert printable("abcdef", limit=4) == "abcd\U00002026"
    assert printable("abcd", limit=4) == "abcd"


@pytest.mark.parametrize(
    "char",
    [
        "\x00",
        "\x08",
        "\x0b",
        "\x1f",
        "\x7f",
        "\x9f",
        "\U0000202a",
        "\U0000202e",
        "\U00002066",
        "\U00002069",
    ],
)
def test_printable_blanks_out_controls_and_bidi_characters(char: str) -> None:
    """Each control and bidi character becomes a blank, ends of ranges too."""
    assert printable(f"a{char}b") == "a b"


@pytest.mark.parametrize("char", ["~", "\xa1", "\U00002065", "\U0000206a"])
def test_printable_keeps_what_lies_next_to_the_ranges(char: str) -> None:
    """The characters just outside the ranges are kept."""
    assert printable(f"a{char}b") == f"a{char}b"
