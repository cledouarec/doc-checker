# Copyright 2026 Christophe Le Douarec
"""Tests for tokenization and number canonicalization."""

import pytest

from doc_checker.text.numbers import canonical_number
from doc_checker.text.tokenize import TokenKind, tokenize


def _texts(text: str) -> list[str]:
    return [token.text for token in tokenize(text)]


def test_tokenize_kinds_and_offsets() -> None:
    """Tokens carry their kind and their position in the text."""
    tokens = tokenize("Port 8080, ok.")
    assert [(t.text, t.kind) for t in tokens] == [
        ("Port", TokenKind.WORD),
        ("8080", TokenKind.NUMBER),
        (",", TokenKind.PUNCT),
        ("ok", TokenKind.WORD),
        (".", TokenKind.PUNCT),
    ]
    assert (tokens[1].start, tokens[1].end) == (5, 9)


def test_tokenize_french_elisions() -> None:
    """French elided articles and pronouns are separate tokens."""
    assert _texts("l'installation jusqu'à s'il") == [
        "l'",
        "installation",
        "jusqu'",
        "à",
        "s'",
        "il",
    ]


def test_tokenize_keeps_contractions_and_compounds() -> None:
    """English contractions and hyphenated words stay whole."""
    assert _texts("don't use non-blocking I/O") == [
        "don't",
        "use",
        "non-blocking",
        "I",
        "/",
        "O",
    ]


def test_tokenize_numbers() -> None:
    """Grouped numbers, decimals and versions are single tokens."""
    assert _texts("1 000,50 et 1,000.5 version 1.2.3 3D") == [
        "1 000,50",
        "et",
        "1,000.5",
        "version",
        "1.2.3",
        "3D",
    ]


@pytest.mark.parametrize(
    ("raw", "lang", "expected"),
    [
        ("1 000,50", "fr", "1000.50"),
        ("1,000.50", "en", "1000.50"),
        ("1.000,50", "fr", "1000.50"),
        ("1,000", "en", "1000"),
        ("1,5", "fr", "1.5"),
        ("3,14", None, "3.14"),
        ("1,000", None, "1000"),
        ("3.10", "en", "3.10"),
        ("1.2.3", "fr", "1.2.3"),
        ("2,000,000", "en", "2000000"),
        ("100", "fr", "100"),
    ],
)
def test_canonical_number(raw: str, lang: str | None, expected: str) -> None:
    """Numbers are reduced to a language independent form."""
    assert canonical_number(raw, lang) == expected
