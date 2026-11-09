# Copyright 2026 Christophe Le Douarec
"""Show the document in the findings of the audit.

A text of the document that a detail or a pattern quotes is cut, by
`quote`. A side shows a whole block, by `block_side`, or a figure as
`describe` tells it, by `image_side`; the side of a link of a block marks
the text of that link, as `LinkMarks` places it.
"""

import re
from collections import Counter
from typing import TYPE_CHECKING, Final

from doc_checker.compare.aggregate import PATTERN_MAX_CHARS
from doc_checker.compare.drafts import marked_side, marked_span
from doc_checker.images.refs import describe
from doc_checker.text.normalize import normalize_display, truncate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.model.document import Block
    from doc_checker.model.findings import Side

# How many times the text of a block each pass of the placing of its links
# reads at most.
LINK_SCAN_FACTOR: Final = 2

_BLANKS: Final = re.compile(r"\s+")
_WORD_RUN: Final = re.compile(r"\w*")

type Span = tuple[int, int]


def quote(text: str) -> str:
    """Return a text of the document as a detail or a pattern quotes it."""
    return truncate(text, PATTERN_MAX_CHARS)


def block_side(block: Block) -> Side:
    """Return the side that shows a whole block."""
    return marked_side(block.source, block.text, kind=block.kind)


def image_side(block: Block) -> Side:
    """Return the side that shows a figure, as `describe` tells it."""
    text = block.text if block.image is None else describe(block.image)
    return marked_side(block.source, text, kind=block.kind, image=block.image)


def _is_word(char: str) -> bool:
    """Return whether a character is a letter, a digit or an underscore.

    That is what a word character of a regular expression is.
    """
    return char.isalnum() or char == "_"


class LinkMarks:
    """Place the links of a block in its text, to mark each one.

    The text of a link matches as whole words, any run of blanks matching
    any run of blanks. The links are placed in their order: each at the
    first match of its text after the place of the link before it; then,
    from the last link back to the first, each moves to the last match of
    its text before the place of the link after it. The text of a link
    that the block also holds before the link, outside any link, is thus
    not taken for the link; one that it holds after the link, outside any
    link, is: the text alone cannot tell them apart. A link whose text is
    not found after the link before it marks the place of the last link of
    the same text before it, or shows its block unmarked if there is none.

    Each pass reads at most about `LINK_SCAN_FACTOR` times the text of the
    block: every character a search reads counts, those of the candidates
    that fail as those of the match. Past that, the first pass places no
    more link and the second keeps the places of the first, so that the
    placing stays linear whatever the texts of the links.
    """

    __slots__ = ("block", "budget", "spans", "words")

    def __init__(self, block: Block, texts: Sequence[str]) -> None:
        """Place the links of a block.

        Args:
            block: The block.
            texts: The text of each link of the block, in order.
        """
        self.block = block
        self.words: dict[str, tuple[str, ...]] = {}
        self.budget = 0
        self.spans: list[Span | None] = self._forward(texts)
        self._backward(texts)
        last: dict[str, Span] = {}
        for index, text in enumerate(texts):
            span = self.spans[index]
            if span is None:
                self.spans[index] = last.get(text)
            else:
                last[text] = span

    def _words(self, text: str) -> tuple[str, ...]:
        """Return the words of the text of a link, as the blocks show them."""
        if text not in self.words:
            words = normalize_display(text, Counter()).split()
            self.words[text] = tuple(words)
        return self.words[text]

    def _match(self, words: Sequence[str], at: int, end: int) -> Span | None:
        """Return the match of some words at a place of the text, if any.

        The characters read after the first word are charged to the budget.

        Args:
            words: The words, at least one; the text holds the first one at
                that place, after no word character if it starts with one.
            at: The place.
            end: Where the match has to end, at the latest.

        Returns:
            The span of the words, whole words, if they end by ``end``.
        """
        text = self.block.text
        stop = at + len(words[0])
        for word in words[1:]:
            blanks = _BLANKS.match(text, stop, end)
            if blanks is None:
                return None
            self.budget -= blanks.end() - stop + len(word)
            if not text.startswith(word, blanks.end()):
                return None
            stop = blanks.end() + len(word)
        if stop > end:
            return None
        if (
            _is_word(words[-1][-1])
            and stop < len(text)
            and _is_word(text[stop])
        ):
            return None
        return at, stop

    def _find(self, words: Sequence[str], start: int, end: int) -> Span | None:
        """Return the first match of some words between two places.

        The characters read are charged to the budget; ``None`` once it is
        spent. A first word that starts with a word character and follows
        one cannot match anywhere in that run of word characters, which is
        skipped.
        """
        text = self.block.text
        first = words[0]
        head = _is_word(first[0])
        while self.budget > 0:
            at = text.find(first, start, end)
            if at < 0:
                self.budget -= end - start
                return None
            self.budget -= at - start + len(first)
            if head and at and _is_word(text[at - 1]):
                start = _WORD_RUN.match(text, at).end()
                self.budget -= start - at
                continue
            span = self._match(words, at, end)
            if span is not None:
                return span
            start = at + 1
        return None

    def _find_last(
        self,
        words: Sequence[str],
        start: int,
        end: int,
    ) -> Span | None:
        """Return the last match of some words between two places.

        The characters read are charged to the budget; ``None`` once it is
        spent.
        """
        text = self.block.text
        first = words[0]
        head = _is_word(first[0])
        stop = end
        while self.budget > 0:
            at = text.rfind(first, start, stop)
            if at < 0:
                self.budget -= stop - start
                return None
            self.budget -= stop - at
            if not (head and at and _is_word(text[at - 1])):
                span = self._match(words, at, end)
                if span is not None:
                    return span
            stop = at + len(first) - 1
        return None

    def _forward(self, texts: Sequence[str]) -> list[Span | None]:
        """Return the first match of each link after the link before it."""
        size = len(self.block.text)
        self.budget = LINK_SCAN_FACTOR * size
        cursor = 0
        spans: list[Span | None] = []
        for text in texts:
            span = None
            if self.budget > 0 and (words := self._words(text)):
                span = self._find(words, cursor, size)
                if span is not None:
                    cursor = span[1]
            spans.append(span)
        return spans

    def _backward(self, texts: Sequence[str]) -> None:
        """Move each placed link to its last match before the next link."""
        self.budget = LINK_SCAN_FACTOR * len(self.block.text)
        limit = len(self.block.text)
        for index in range(len(texts) - 1, -1, -1):
            span = self.spans[index]
            if span is None:
                continue
            if self.budget > 0:
                words = self._words(texts[index])
                span = self._find_last(words, span[0], limit) or span
                self.spans[index] = span
            limit = span[0]

    def side(self, index: int) -> Side:
        """Return the side of a link of the block, its text marked.

        Args:
            index: Position of the link among the texts the marks were
                given.

        Returns:
            The side of the block, the place of the link marked; the block
            unmarked if the link has no place.
        """
        span = self.spans[index]
        block = self.block
        if span is None:
            return block_side(block)
        return marked_span(block.source, block.text, span, kind=block.kind)
