# Copyright 2026 Christophe Le Douarec
"""Split texts into word, number and punctuation tokens."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class TokenKind(StrEnum):
    """Kind of a token."""

    WORD = "word"
    NUMBER = "number"
    PUNCT = "punct"


@dataclass(frozen=True, slots=True)
class Token:
    """A token and its position in the source text.

    Attributes:
        text: Token text.
        kind: Token kind.
        start: Offset of the first character in the source text.
        end: Offset just after the last character in the source text.
    """

    text: str
    kind: TokenKind
    start: int
    end: int


# Alternatives are tried in order: numbers grouped by spaces, plain numbers,
# French elisions (l', qu', jusqu'...), words, then any other symbol.
TOKEN_RE: Final = re.compile(
    r"""
    (?P<grouped>\d{1,3}(?:[ \xa0\U0000202f]\d{3})+(?:[.,]\d+)?(?!\w))
  | (?P<number>\d+(?:[.,]\d+)*(?!\w))
  | (?P<elision>\b(?i:qu|jusqu|lorsqu|puisqu|quoiqu|[cdjlmnst])
        ['\U00002019](?=\w))
  | (?P<word>\w+(?:['\U00002019\-]\w+)*)
  | (?P<punct>[^\w\s])
    """,
    re.VERBOSE,
)

_KINDS: Final = {
    "grouped": TokenKind.NUMBER,
    "number": TokenKind.NUMBER,
    "elision": TokenKind.WORD,
    "word": TokenKind.WORD,
    "punct": TokenKind.PUNCT,
}


def tokenize(text: str) -> list[Token]:
    """Split a text into tokens.

    Args:
        text: Text to split, usually a strict comparison key.

    Returns:
        The tokens in text order. Whitespace is not a token.
    """
    return [
        Token(match.group(), _KINDS[str(match.lastgroup)], *match.span())
        for match in TOKEN_RE.finditer(text)
    ]
