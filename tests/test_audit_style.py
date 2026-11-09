# Copyright 2026 Christophe Le Douarec
"""Tests for the style checks: repeats, long sentences, foreign blocks."""

import time

from doc_checker.audit.prose import Prose, excluded_spans
from doc_checker.audit.style import (
    foreign_block,
    long_sentences,
    repeated_words,
)
from doc_checker.model.document import BlockKind
from doc_checker.model.findings import Severity
from tests.builders import make_block, sentence

KNOWN = {"le", "the", "vous", "that", "is", "windows", "as"}.__contains__
HOSTILE_RUNS = (
    "a" * 200_000,
    " " * 200_000,
    "a" + " " * 200_000 + "a",
    "a " * 100_000,
    "a a" * 70_000,
    "a-" * 100_000,
    "a'" * 100_000,
    "\xe9" * 200_000 + " " + "\xe9" * 200_000,
    "." * 200_000 + "a",
    "a." * 100_000,
    "ab " * 100_000,
    "ab" + " " * 200_000 + "ab ab",
)


def _repeats(text: str, lang: str = "fr") -> list[str]:
    """Return the patterns of the repeated words of a text."""
    block = make_block(text)
    marks = repeated_words(block.text, lang, excluded_spans(block), KNOWN)
    return [mark.pattern for mark in marks]


def test_repeated_words() -> None:
    """A known word written twice is marked, with the exceptions."""
    block = make_block("On lit le le fichier")
    (mark,) = repeated_words(block.text, "fr", (), KNOWN)
    assert block.text[slice(*mark.span)] == "le le"
    assert mark.rule == "text.repeated_word"
    assert mark.pattern == "le"
    assert mark.detail == '"le" is written twice'
    assert _repeats("The the agent", "en") == ["the"]
    assert _repeats("windows Windows") == []
    assert _repeats("vous vous connectez") == []
    assert _repeats("that that", "en") == []
    assert _repeats("dy dy") == []
    assert _repeats("C C") == []
    assert _repeats("as-is is", "en") == []
    assert _repeats("l'le le") == []
    assert _repeats("see le\nle") == []
    assert _repeats("Le le") == ["le"]
    assert _repeats("The The", "en") == ["the"]
    assert _repeats("the The", "en") == []
    assert _repeats("had had", "en") == []


def test_repeated_words_stop_at_a_line_break() -> None:
    """A table value on the next row is no repeated word."""
    table = make_block("No\nNo", BlockKind.TABLE)
    known = {"no", "the"}.__contains__
    assert repeated_words(table.text, "en", (), known) == []
    row = make_block("the the | x", BlockKind.TABLE)
    assert len(repeated_words(row.text, "en", (), known)) == 1


def test_a_run_of_one_word_is_no_repeat() -> None:
    """Three of one word or more in a row are meant: emphasis or labels."""
    known = {"really", "thick", "the", "click", "here"}.__contains__
    for text in (
        "It is really really really fast.",
        "thick thick thick thick thick",
        "Click Click Click Click",
        "here here here\nhere here here",
        "the the The",
    ):
        block = make_block(text)
        assert repeated_words(block.text, "en", (), known) == [], text
    block = make_block("so the the end, then the the the")
    marks = repeated_words(block.text, "en", (), known)
    assert [block.text[slice(*mark.span)] for mark in marks] == ["the the"]
    assert marks[0].span == (3, 10)
    line = make_block("the\nthe the")
    assert len(repeated_words(line.text, "en", (), known)) == 1


def test_repeated_words_skips_excluded_parts() -> None:
    """A repetition that touches an excluded part is left out."""
    text = "le le"
    assert repeated_words(text, "fr", [(3, 5)], KNOWN) == []
    assert repeated_words(text, "fr", [(5, 8)], KNOWN) == []


def test_long_sentences() -> None:
    """A sentence over a threshold is marked in full."""
    block = make_block(sentence(41))
    (mark,) = long_sentences(block.text, (), minor=40, major=60)
    assert mark.severity is Severity.MINOR
    assert mark.pattern == "over 40 words"
    assert mark.detail == "41 words in one sentence"
    assert mark.span == (0, len(block.text))
    assert (
        long_sentences(make_block(sentence(40)).text, (), minor=40, major=60)
        == []
    )
    (major,) = long_sentences(
        make_block(sentence(61)).text,
        (),
        minor=40,
        major=60,
    )
    assert major.severity is Severity.MAJOR
    assert major.pattern == "over 60 words"
    (custom,) = long_sentences(
        make_block(sentence(35)).text,
        (),
        minor=30,
        major=50,
    )
    assert custom.severity is Severity.MINOR
    table = " ".join(["word"] * 80)
    assert long_sentences(table, (), minor=40, major=60) == []


def test_long_sentences_skip_excluded_parts() -> None:
    """The words of an excluded part are not counted."""
    text = sentence(39)[:-1] + " one two three."
    assert len(long_sentences(text, (), minor=40, major=60)) == 1
    start = len(text) - len(" one two three.") + 1
    excluded = [(start, len(text) - 1)]
    assert long_sentences(text, excluded, minor=40, major=60) == []


def test_sentence_builder() -> None:
    """The builder returns sentences of the exact length."""
    for count in range(1, 70):
        assert len(sentence(count).split()) == count


def test_foreign_block() -> None:
    """A block of another language is marked as a whole."""
    block = make_block("This is a block in English for you.")
    prose = Prose(block, lang="en", own="en", excluded=())
    mark = foreign_block(prose, "fr")
    assert mark is not None
    assert mark.rule == "text.block_language"
    assert mark.span == (0, len(block.text))
    assert mark.pattern == "en"
    assert mark.detail == (
        "this block reads as English, the document as French"
    )
    assert foreign_block(prose, "en") is None
    assert foreign_block(prose, None) is None
    assert foreign_block(prose, "de") is None
    assert foreign_block(Prose(block, "en", None, ()), "fr") is None
    assert foreign_block(Prose(block, "en", "de", ()), "fr") is None


def test_style_stays_linear_on_hostile_runs() -> None:
    """Each pattern reads a long run of one shape in linear time."""
    blocks = [make_block(run) for run in HOSTILE_RUNS]
    start = time.perf_counter()
    for block in blocks:
        excluded = excluded_spans(block)
        repeated_words(block.text, "fr", excluded, lambda _: True)
        long_sentences(block.text, excluded, minor=40, major=60)
    # Under 2 s; a pattern that restarted at every place of a run would take
    # minutes.
    assert time.perf_counter() - start < 15
