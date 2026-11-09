# Copyright 2026 Christophe Le Douarec
"""Tests for the prose of the audit: language, excluded parts, sentences."""

import time
from dataclasses import replace

import pytest

from doc_checker.audit.prose import (
    ABBREVIATIONS,
    CODE_MARK,
    CODE_SCAN_FACTOR,
    SENTENCE_END,
    SENTENCE_KINDS,
    SPELLED_KINDS,
    Mark,
    code_like,
    code_ranges,
    count_words,
    document_language,
    excluded_spans,
    prose_blocks,
    sentences,
)
from doc_checker.model.document import BlockKind
from doc_checker.text.lang import (
    BLOCK_EVIDENCE_MIN,
    LANGUAGE_NAMES,
    block_language,
    strong_language,
)
from tests.builders import make_audit_document, make_block

# An English sentence of six stopwords: the, the, and, it, to, the.
ENGLISH = "The agent reads the file and writes it to the disk."
# A French sentence of six stopwords: le, la, et, la, dans, le.
FRENCH = "Le service lit la configuration et la garde dans le cache."
CODE = "\\node (a) at (1,0) {A}; \\draw (a) -- (b);"
# Long runs of one shape, on which a pattern that restarts at every place of
# the run would be quadratic.
HOSTILE_RUNS = (
    "a" * 200_000,
    "." * 200_000 + "a",
    "?" * 200_000 + "a",
    "a" + " " * 200_000 + ",a",
    "1" * 200_000 + "km",
    "X" * 200_000 + "\xe8me",
    "a@" * 100_000,
    ".)" * 100_000 + "a",
    "a. " * 70_000,
    "Ab. " * 50_000,
    "-" * 200_000 + ">",
    "a" * 200_000 + ". B",
    '."' * 100_000 + "a",
    ". \xbb" * 70_000 + "a",
    "a.\xa0\xbb" * 50_000,
)


def _span(text: str, part: str) -> tuple[int, int]:
    """Return the place of the first occurrence of a part of a text."""
    start = text.index(part)
    return start, start + len(part)


def test_code_spans_are_placed_in_order() -> None:
    """Each code is sought after the one before it, never inside a word."""
    block = replace(make_block("Use a and b, then a."), code_spans=("a", "a"))
    assert code_ranges(block) == [(4, 5), (18, 19)]
    missing = replace(block, code_spans=("b", "zz", "", " ", "a"))
    assert code_ranges(missing) == [(10, 11), (18, 19)]


def test_code_spans_are_placed_as_whole_words() -> None:
    """Only an edge of a code that is a word character may not touch one."""
    call = replace(make_block("Call foo()s or foo()."), code_spans=("foo()",))
    suffix = replace(
        make_block("Name it orbit.toml, not .toml."),
        code_spans=(".toml",),
    )
    assert code_ranges(call) == [(5, 10)]
    assert code_ranges(suffix) == [(13, 18)]


def test_code_spans_are_located_within_a_budget() -> None:
    """Codes that cannot be found cost a few readings of the text at most."""
    block = replace(make_block("a " * 100_000), code_spans=("b",) * 50_000)
    glued = replace(make_block("a" * 200_000), code_spans=("a",) * 50_000)
    dashes = replace(
        make_block("x-a" * 70_000 + "x"),
        code_spans=("-a",) * 50_000,
    )
    start = time.perf_counter()
    assert code_ranges(block) == []
    assert code_ranges(glued) == []
    assert code_ranges(dashes) == []
    assert time.perf_counter() - start < 1


def test_codes_after_the_budget_are_not_placed() -> None:
    """Once the text was read `CODE_SCAN_FACTOR` times, the search stops."""
    block = make_block("zz " + "a " * 1_000)
    spent = replace(block, code_spans=("q",) * CODE_SCAN_FACTOR + ("zz",))
    left = replace(block, code_spans=("q",) * (CODE_SCAN_FACTOR - 1) + ("zz",))
    assert code_ranges(spent) == []
    assert code_ranges(left) == [(0, 2)]


def test_block_language() -> None:
    """A block reads as a language from six of its stopwords, three to one."""
    assert BLOCK_EVIDENCE_MIN == 6
    assert block_language(ENGLISH) == "en"
    assert block_language(FRENCH) == "fr"
    five = "The agent reads the file and writes it to disk."
    assert block_language(five) is None
    assert block_language("the and is of to in le la les des du de") is None
    assert strong_language(ENGLISH) is None
    assert strong_language(ENGLISH, minimum=BLOCK_EVIDENCE_MIN) == "en"
    assert dict(LANGUAGE_NAMES) == {"fr": "French", "en": "English"}


def test_excluded_spans() -> None:
    """URLs, e-mails and code-like chunks are no prose, sorted and merged."""
    block = make_block(
        "See https: //tug.org/x, mail a@b.org, run --force on src/x.py.",
    )
    text = block.text
    parts = ("https: //tug.org/x", "a@b.org", "--force", "src/x.py")
    assert excluded_spans(block) == tuple(_span(text, p) for p in parts)
    assert excluded_spans(make_block("plain words only")) == ()


def test_dotted_names_are_no_prose() -> None:
    """A file extension or a dotfile is no prose; the sentence still ends."""
    block = make_block("Edit (.gitignore) and .txt. Then go.")
    text = block.text
    assert excluded_spans(block) == (
        _span(text, ".gitignore"),
        _span(text, ".txt"),
    )
    found = sentences(text, excluded_spans(block))
    assert [text[a:b] for a, b in found] == [
        "Edit (.gitignore) and .txt.",
        "Then go.",
    ]


def test_inline_code_is_no_prose() -> None:
    """The inline code of a block joins its other excluded parts."""
    block = replace(
        make_block("Run orbit start, see (www.x.org/a) or x.y."),
        code_spans=("orbit start", "x.y"),
    )
    text = block.text
    parts = ("orbit start", "www.x.org/a", "x.y")
    assert excluded_spans(block) == tuple(_span(text, p) for p in parts)


def test_code_like_blocks() -> None:
    """Four chunks or more, a quarter of them with a mark of code, is code."""
    assert code_like(CODE)
    assert not code_like("Option | Default | timeout | 30")
    assert not code_like("a=1 b=2 c=3")
    assert code_like("a=1 b=2 c=3 d")
    assert code_like("one two three a->b")
    assert code_like("one two three a::b")
    assert not code_like("one two three four a::b")
    assert not code_like(ENGLISH)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two words here.", ["One.", "Two words here."]),
        (
            "See e.g. the guide. Then stop.",
            ["See e.g. the guide.", "Then stop."],
        ),
        ("Ask J. R. Smith. Done!", ["Ask J. R. Smith.", "Done!"]),
        ("Version 3.12 is out. Yes.", ["Version 3.12 is out.", "Yes."]),
        ("It ends here. and goes on.", ["It ends here. and goes on."]),
        (
            "Cf. la page. Puis\U00002026 fin.",
            ["Cf. la page.", "Puis\U00002026 fin."],
        ),
        ("No end here", []),
    ],
)
def test_sentences(text: str, expected: list[str]) -> None:
    """Sentences end at . ! ? or an ellipsis, not after an abbreviation."""
    assert [text[a:b] for a, b in sentences(text, ())] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "  He said \xabStop.\xbb Then (see p. 4.) It ran?! Yes\U00002026",
            [
                "He said \xabStop.\xbb",
                "Then (see p. 4.)",
                "It ran?!",
                "Yes\U00002026",
            ],
        ),
        ("Voir (cf. la note). Fin.", ["Voir (cf. la note).", "Fin."]),
        ("Use etc. Then go.", ["Use etc. Then go."]),
        ("Step 1. Do it.", ["Step 1.", "Do it."]),
        ("Wait... What?", ["Wait...", "What?"]),
        ("Read it.\nThen go.", ["Read it.", "Then go."]),
        ("It costs 3.5.Then", []),
    ],
)
def test_sentence_edges(text: str, expected: list[str]) -> None:
    """Closing quotes and brackets end with their sentence, blanks do not."""
    assert [text[a:b] for a, b in sentences(text, ())] == expected


@pytest.mark.parametrize(
    ("text", "codes", "expected"),
    [
        (
            (
                "Read https://x.org/a. Edit src/x.py! Type stop. Now in the"
                " box. Done."
            ),
            ("stop. Now",),
            [
                "Read https://x.org/a.",
                "Edit src/x.py!",
                "Type stop. Now in the box.",
                "Done.",
            ],
        ),
        (
            "Read \U00002018https://x.org/a.\U00002019 Then go.",
            (),
            ["Read \U00002018https://x.org/a.\U00002019", "Then go."],
        ),
        (
            "See https://x.org/b\U00002026 Next one.",
            (),
            ["See https://x.org/b\U00002026", "Next one."],
        ),
        (
            "It is done. make builds it.",
            ("make",),
            ["It is done.", "make builds it."],
        ),
        ("It is done. make builds it.", (), ["It is done. make builds it."]),
        (
            "Run it. https://x.org is next. src/x.py holds it.",
            (),
            ["Run it.", "https://x.org is next.", "src/x.py holds it."],
        ),
    ],
)
def test_a_sentence_may_end_next_to_an_excluded_part(
    text: str,
    codes: tuple[str, ...],
    expected: list[str],
) -> None:
    """A sentence ends before or after a URL or code, never inside one.

    The punctuation after a URL or a code chunk still ends a sentence, and
    so does one before an excluded part that starts with a lower-case
    letter; a punctuation inside an excluded part ends nothing.
    """
    block = replace(make_block(text), code_spans=codes)
    assert block.text == text
    found = sentences(text, excluded_spans(block))
    assert [text[a:b] for a, b in found] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Il dit \xab Stop. \xbb Puis il part.",
            ["Il dit \xab Stop. \xbb", "Puis il part."],
        ),
        (
            "Il dit \xab\xa0Stop\xa0! \xbb Puis\U0000202f? \xbb Fin.",
            ["Il dit \xab\xa0Stop\xa0! \xbb", "Puis\U0000202f? \xbb", "Fin."],
        ),
        (
            "Il dit \xab\U0000202fStop.\U0000202f\xbb Puis il part.",
            ["Il dit \xab\U0000202fStop.\U0000202f\xbb", "Puis il part."],
        ),
        (
            "Il cite \xab etc. \xbb Puis il part.",
            ["Il cite \xab etc. \xbb Puis il part."],
        ),
        (
            "He said \U0000201cStop. \U0000201d Then left.",
            ["He said \U0000201cStop. \U0000201d", "Then left."],
        ),
    ],
)
def test_a_spaced_closing_quote_stays_with_its_sentence(
    text: str,
    expected: list[str],
) -> None:
    """A closing guillemet after a space ends with its sentence."""
    assert [text[a:b] for a, b in sentences(text, ())] == expected


def test_abbreviations() -> None:
    """The abbreviations are lower case, without their final dot."""
    assert {"e.g", "c.-\xe0-d", "j.-c", "r\xe9f", "cie"} <= ABBREVIATIONS
    assert all(a == a.casefold() for a in ABBREVIATIONS)
    assert not any(a.endswith(".") for a in ABBREVIATIONS)
    assert not {"ref", "id", "no"} & ABBREVIATIONS


@pytest.mark.parametrize(
    "text",
    [
        "See the ref. It works.",
        "Set the id. It works.",
        "Default: no. It works.",
    ],
)
def test_technical_nouns_end_a_sentence(text: str) -> None:
    """A ref, an id or a no before a dot is a word, not an abbreviation."""
    assert len(sentences(text, ())) == 2


def test_words_of_a_sentence_leave_code_out() -> None:
    """Words and numbers count, but not those that touch an excluded part."""
    block = replace(
        make_block("Run orbit start now."),
        code_spans=("orbit start",),
    )
    text = block.text
    whole = (0, len(text))
    assert count_words(text, whole, excluded_spans(block)) == 2
    assert count_words(text, whole, ()) == 4
    version = "Version 3.12 is out."
    assert count_words(version, (0, len(version)), ()) == 4
    assert count_words(version, (8, len(version)), ()) == 3


def test_document_language() -> None:
    """The text tells the language, unless the user names one."""
    doc = make_audit_document(*(make_block(FRENCH) for _ in range(20)))
    assert (doc.lang, doc.declared_lang) == ("en", "en")
    assert document_language(doc, "auto") == "fr"
    assert document_language(doc, "en") == "en"
    weak = make_audit_document(make_block(FRENCH))
    assert document_language(weak, "auto") == "en"
    code = make_audit_document(make_block(CODE, BlockKind.CODE), lang="de")
    assert document_language(code, "auto") == "de"
    assert document_language(replace(code, lang=None), "auto") is None


def test_prose_blocks() -> None:
    """Prose blocks keep their order, checked in their own language."""
    heading = make_block(ENGLISH, BlockKind.HEADING, level=1)
    english = make_block(ENGLISH)
    french = make_block(FRENCH, BlockKind.QUOTE)
    short = make_block("Voir la page.", BlockKind.LIST_ITEM)
    table = make_block("Option | Default | timeout | 30", BlockKind.TABLE)
    note = make_block(ENGLISH, BlockKind.FOOTNOTE)
    doc = make_audit_document(
        heading,
        make_block("x = 1", BlockKind.CODE),
        english,
        make_block("E = mc^2", BlockKind.FORMULA),
        french,
        make_block("Figure", BlockKind.IMAGE),
        short,
        make_block("Contents", BlockKind.TOC),
        table,
        make_block("orbit, 4", BlockKind.INDEX),
        make_block(CODE),
        note,
        lang="fr",
        declared_lang="fr",
    )
    found = prose_blocks(doc, "fr")
    assert [(p.block, p.own, p.lang) for p in found] == [
        (heading, None, "fr"),
        (english, "en", "en"),
        (french, "fr", "fr"),
        (short, None, "fr"),
        (table, None, "fr"),
        (note, "en", "en"),
    ]
    assert all(p.excluded == excluded_spans(p.block) for p in found)
    unknown = [p.lang for p in prose_blocks(doc, None)]
    assert unknown == [None, "en", "fr", None, None, "en"]


def test_kinds_of_prose() -> None:
    """Sentences are judged in the spelled kinds but headings and tables."""
    assert {
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST_ITEM,
        BlockKind.TABLE,
        BlockKind.CAPTION,
        BlockKind.QUOTE,
        BlockKind.ADMONITION,
        BlockKind.FOOTNOTE,
    } == SPELLED_KINDS
    assert SENTENCE_KINDS < SPELLED_KINDS
    assert {
        BlockKind.HEADING,
        BlockKind.TABLE,
    } == SPELLED_KINDS - SENTENCE_KINDS


def test_a_mark_has_the_default_severity() -> None:
    """A mark without a severity takes that of its rule."""
    mark = Mark("text.ordinal", (0, 3), "\xe8me", "detail")
    assert mark.severity is None


def test_prose_stays_linear_on_hostile_runs() -> None:
    """Each pattern and step reads a long run of one shape in linear time."""
    blocks = [make_block(run) for run in HOSTILE_RUNS]
    start = time.perf_counter()
    for block in blocks:
        text = block.text
        for pattern in (SENTENCE_END, CODE_MARK):
            sum(1 for _ in pattern.finditer(text))
        excluded = excluded_spans(block)
        for span in sentences(text, excluded):
            count_words(text, span, excluded)
        count_words(text, (0, len(text)), excluded)
        code_like(text)
        block_language(text)
    prose_blocks(make_audit_document(*blocks), "en")
    # About 5 s on a desktop, 13 s on a slower one; a step that restarted
    # at every place of a run would take many minutes.
    assert time.perf_counter() - start < 60
