# Copyright 2026 Christophe Le Douarec
"""Word level differences between two texts."""

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from doc_checker.text.tokenize import Token, TokenKind

if TYPE_CHECKING:
    from collections.abc import Sequence

type HunkOp = Literal["replace", "delete", "insert"]

# A hyphen between two letters, where `[^\W\d_]` is a letter. A hyphen next to
# a digit is not one: it belongs to a code such as ``x86-64``.
_LETTER_HYPHEN: Final = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")


@dataclass(frozen=True, slots=True)
class Hunk:
    """A run of tokens that differs between two texts.

    Attributes:
        op: Whether tokens were replaced, deleted or inserted.
        ref_start: First reference token index.
        ref_end: Reference token index just after the run.
        new_start: First new token index.
        new_end: New token index just after the run.
    """

    op: HunkOp
    ref_start: int
    ref_end: int
    new_start: int
    new_end: int


def _key(token: Token) -> str:
    """Return the text a token is compared by.

    Args:
        token: Token to compare.

    Returns:
        The text of a word without the hyphens between two letters, else the
        text of the token.
    """
    if token.kind is TokenKind.WORD:
        return _LETTER_HYPHEN.sub("", token.text)
    return token.text


def diff_tokens(
    ref_tokens: Sequence[Token],
    new_tokens: Sequence[Token],
) -> list[Hunk]:
    """Return the runs of tokens that differ, in order.

    Words are compared without the hyphens between two letters: a line break
    hyphenation, or ``e-mail`` written ``email``, is not a difference. A hyphen
    next to a digit belongs to a code, such as ``x86-64`` or ``XJ-120-4``, and
    is compared. Other tokens are compared by their text.

    Args:
        ref_tokens: Tokens of the reference text.
        new_tokens: Tokens of the new text.

    Returns:
        One hunk per difference, never merged with its neighbors. Its indices
        are positions in the given tokens.
    """
    matcher = difflib.SequenceMatcher(
        None,
        [_key(token) for token in ref_tokens],
        [_key(token) for token in new_tokens],
        autojunk=False,
    )
    return [
        Hunk(cast("HunkOp", tag), i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def span_text(text: str, tokens: Sequence[Token], start: int, end: int) -> str:
    """Return the text covered by a run of tokens.

    Args:
        text: Text the tokens come from.
        tokens: Tokens of the text.
        start: First token index.
        end: Token index just after the run.

    Returns:
        The covered text with its original spacing, empty for an empty run.
    """
    if start >= end:
        return ""
    return text[tokens[start].start : tokens[end - 1].end]
