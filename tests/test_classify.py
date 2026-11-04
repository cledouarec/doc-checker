# Copyright 2026 Christophe Le Douarec
"""Tests for word level differences and their classification."""

import pytest

from doc_checker.compare.classify import (
    HunkContext,
    classify_hunk,
    has_encoding_damage,
)
from doc_checker.compare.textdiff import diff_tokens
from doc_checker.model.findings import Severity
from doc_checker.text.tokenize import TokenKind, tokenize

C = Severity.CRITICAL
MA = Severity.MAJOR
MI = Severity.MINOR
IN = Severity.INFO


def _classify(
    ref: str,
    new: str,
    lang: str | None = "en",
) -> list[tuple[str, Severity]]:
    ref_tokens, new_tokens = tokenize(ref), tokenize(new)
    vocabulary = frozenset(
        t.text.casefold() for t in ref_tokens if t.kind is TokenKind.WORD
    )
    ctx = HunkContext(
        ref,
        new,
        ref_tokens,
        new_tokens,
        vocabulary,
        lang,
        lang,
    )
    return [
        (c.rule, c.severity)
        for c in (
            classify_hunk(h, ctx) for h in diff_tokens(ref_tokens, new_tokens)
        )
    ]


@pytest.mark.parametrize(
    ("ref", "new", "lang", "expected"),
    [
        (
            "Read the état page.",
            "Read the \xc3\xa9tat page.",
            "fr",
            ("text.encoding_issue", C),
        ),
        (
            "Accepts 100 requests.",
            "Accepts 10 requests.",
            "en",
            ("text.number_changed", C),
        ),
        (
            "Uses port 8080 by default.",
            "Uses port by default.",
            "en",
            ("text.number_changed", C),
        ),
        (
            "Processes 1,000 jobs.",
            "Processes 1000 jobs.",
            "en",
            ("text.number_format", IN),
        ),
        ("Prend 1,5 Go.", "Prend 1.5 Go.", "fr", ("text.number_format", IN)),
        (
            "Do not delete the cache.",
            "Do delete the cache.",
            "en",
            ("text.negation_changed", C),
        ),
        (
            "Ne supprimez pas le cache.",
            "Supprimez le cache.",
            "fr",
            ("text.negation_changed", C),
        ),
        (
            "The efficient scheduler.",
            "The ecient scheduler.",
            "en",
            ("text.ligature_lost", MA),
        ),
        (
            "Edit the configuration file.",
            "Edit the con guration file.",
            "en",
            ("text.ligature_lost", MA),
        ),
        ("Pendant l'été.", "Pendant l'ete.", "fr", ("text.accent_lost", MA)),
        (
            "Pendant l'ete.",
            "Pendant l'été.",
            "fr",
            ("text.accent_changed", MI),
        ),
        (
            "Install Python now.",
            "Install python now.",
            "en",
            ("text.case_changed", MI),
        ),
        (
            "Install it, then run.",
            "Install it then run.",
            "en",
            ("text.punctuation_changed", IN),
        ),
        (
            "You must restart.",
            "You may restart.",
            "en",
            ("text.modality_changed", MA),
        ),
        (
            "The installer downloads files.",
            "The installer donwloads files.",
            "en",
            ("text.typo", MA),
        ),
        (
            "Copy from the form.",
            "Copy form the form.",
            "en",
            ("text.word_replaced", MA),
        ),
        (
            "Install the package with the default settings.",
            "Install the package.",
            "en",
            ("text.words_missing", MA),
        ),
        (
            "Install the package now.",
            "Install the package.",
            "en",
            ("text.words_missing", MI),
        ),
        (
            "Install the package.",
            "Install the package quickly.",
            "en",
            ("text.words_added", MI),
        ),
        (
            "The service restarts automatically after updates.",
            "A different sentence replaces everything here completely.",
            "en",
            ("text.modified", MA),
        ),
    ],
)
def test_classify_hunk(
    ref: str,
    new: str,
    lang: str,
    expected: tuple[str, Severity],
) -> None:
    """Each kind of difference gets its rule and severity."""
    assert _classify(ref, new, lang) == [expected]


def test_identical_text_has_no_hunk() -> None:
    """Identical texts have no difference."""
    tokens = tokenize("Same text here.")
    assert diff_tokens(tokens, tokens) == []


@pytest.mark.parametrize(
    ("text", "damaged"),
    [
        ("état", False),
        ("Ça va", False),
        ("\xc3\xa9tat", True),
        ("a\U0000fffdb", True),
        ("l\xe2\U000020ac\U00002122app", True),
        ("x\U0000e000y", True),
    ],
)
def test_has_encoding_damage(text: str, damaged: bool) -> None:  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
    """Replacement, private use and mojibake characters are detected."""
    assert has_encoding_damage(text) is damaged
