# Copyright 2026 Christophe Le Douarec
"""Read the prose of a document as the checks of its text do.

The checks of the text read the blocks of `SPELLED_KINDS`, and those of
sentences the blocks of `SENTENCE_KINDS`, but not a block that looks like
code, as `code_like` tells. A block is checked in the language that it
plainly reads as, if it is a block of sentences, else in the language of the
text of the document, as `document_language` tells.

The excluded parts of a block are no prose: its URLs, its e-mail addresses,
its inline code, which `code_ranges` places from the code that an HTML page
marks, and its chunks that look like code. A sentence ends at a
`SENTENCE_END`, but not after an abbreviation or an initial, not before a
lower-case letter, and not inside an excluded part.
"""

import re
from bisect import bisect_left
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Final

from doc_checker.audit.accessibility import prose_sample
from doc_checker.model.document import BlockKind
from doc_checker.spell.words import (
    BLANKS,
    CHUNK,
    CODE_CHUNK,
    DOTTED,
    EMAIL,
    PROSE_EDGE,
    URL,
    is_word_char,
    merged_spans,
    touches,
)
from doc_checker.text.lang import AUTO, block_language, strong_language
from doc_checker.text.tokenize import TokenKind, tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.model.document import Block, Document
    from doc_checker.model.findings import Severity
    from doc_checker.spell.words import Span

# How many times the text of a block the search of its inline code reads at
# most.
CODE_SCAN_FACTOR: Final = 2
# The blocks whose words and typography are checked.
SPELLED_KINDS: Final = frozenset(
    {
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST_ITEM,
        BlockKind.TABLE,
        BlockKind.CAPTION,
        BlockKind.QUOTE,
        BlockKind.ADMONITION,
        BlockKind.FOOTNOTE,
    },
)
# The blocks of sentences: their sentences and their language are judged.
SENTENCE_KINDS: Final = SPELLED_KINDS - {BlockKind.HEADING, BlockKind.TABLE}
# What makes a chunk of a block a chunk of code, for `code_like`.
CODE_MARK: Final = re.compile(r"[\\{}=<>|$^~]|::|->")
# A block of at least `CODE_LIKE_MIN_CHUNKS` chunks, at least
# `CODE_LIKE_SHARE` of them with a `CODE_MARK`, looks like code.
CODE_LIKE_SHARE: Final = 0.25
CODE_LIKE_MIN_CHUNKS: Final = 4
# The punctuation that ends a sentence, then a closing guillemet or quote
# after one space, as French sets it, then its closing quotes and brackets,
# before a blank or the end. A match starts only at the start of a run of
# that punctuation, so a long run is read once.
SENTENCE_END: Final = re.compile(
    r"(?<![.!?\U00002026])[.!?\U00002026]+"
    r"(?:[ \xa0\U0000202f][\xbb\U0000201d])?"
    r"[\"'\U00002019\U0000201d\xbb)\]]*(?=\s|$)",
)
# The abbreviations, without case nor final dot, after which a dot does not
# end a sentence.
ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "cf",
        "vs",
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "fig",
        "figs",
        "eq",
        "vol",
        "p",
        "pp",
        "ch",
        "chap",
        "sec",
        "approx",
        "env",
        "ex",
        "m",
        "mme",
        "mlle",
        "art",
        "al",
        "resp",
        "c.-à-d",
        "p.ex",
        "n.b",
        "nb",
        "op",
        "cit",
        "ibid",
        "inc",
        "ltd",
        "jr",
        "sr",
        "min",
        "max",
        "réf",
        "éd",
        "ed",
        "eds",
        "trad",
        "coll",
        "dir",
        "av",
        "apr",
        "j.-c",
        "cie",
    },
)

# What the end of a URL loses: the punctuation of the sentence around it.
_URL_TAIL: Final = ".,;:!?)\xbb\U0000201d\U00002019\U00002026\"'"
# The punctuation that a match of `SENTENCE_END` starts with.
_FINAL: Final = ".!?\U00002026"
_DOT: Final = "."
# A chunk alone between blanks: a separator of the cells of a table.
_CELL: Final = "|"
_WORD_RUN: Final = re.compile(r"\w*")
_COUNTED: Final = frozenset({TokenKind.WORD, TokenKind.NUMBER})


@dataclass(frozen=True, slots=True)
class Mark:
    """A part of the text of a block that a check of the text points at.

    Attributes:
        rule: Identifier of the rule.
        span: Where the part lies in the text of the block.
        pattern: What the finding is about: the findings of a rule with the
            same pattern make the same mistake.
        detail: The explanation of the finding.
        severity: Severity of the finding, or ``None`` for the default
            severity of the rule.
    """

    rule: str
    span: Span
    pattern: str
    detail: str
    severity: Severity | None = None


@dataclass(frozen=True, slots=True)
class Prose:
    """A block of prose, as the checks of the text read it.

    Attributes:
        block: The block.
        lang: The language in which the block is checked: its own, else
            that of the text of the document, if known.
        own: The language that a block of `SENTENCE_KINDS` plainly reads
            as, as `block_language` tells, if it does.
        excluded: The parts of its text that are no prose, as
            `excluded_spans` returns them.
    """

    block: Block
    lang: str | None
    own: str | None
    excluded: tuple[Span, ...]


def _place(
    text: str,
    code: str,
    start: int,
    budget: int,
) -> tuple[Span | None, int]:
    """Return the first place of a code in a text, from a place, as a whole.

    A code that starts with a word character does not match after one, and
    a code that ends with one does not match before one: a code never
    stands inside a longer word.

    Args:
        text: The text.
        code: The code, not empty.
        start: Where the search starts.
        budget: How many more characters of the text may be read: every
            character a search reads counts, those of the candidates that
            fail as those of the match.

    Returns:
        The place, if the code was found before the budget was spent, and
        the budget left.
    """
    head = is_word_char(code[0])
    tail = is_word_char(code[-1])
    while budget > 0:
        stop = min(len(text), start + budget)
        at = text.find(code, start, stop)
        if at < 0:
            return None, budget - (stop - start)
        end = at + len(code)
        budget -= end - start
        glued = (head and at > 0 and is_word_char(text[at - 1])) or (
            tail and end < len(text) and is_word_char(text[end])
        )
        if not glued:
            return (at, end), budget
        if head:
            # Every other place in this run of word characters follows one;
            # the run is read up to the end of the search, within budget.
            start = _WORD_RUN.match(text, at, stop).end()
            budget -= start - at
        else:
            start = at + 1
    return None, budget


def code_ranges(block: Block) -> list[Span]:
    """Return the places of the inline code of a block in its text.

    Each of its `Block.code_spans` but the blank ones is sought after the
    place of the one before it, as a whole, as `_place` tells; one that is
    not found is left out, and the next one is sought after the place of
    the one before it. The search reads at most `CODE_SCAN_FACTOR` times
    the text: the codes after that are not placed.

    The words of a PDF set in a monospaced font, `Document.code_words`, are
    no inline code here: they are only spared by the spelling.

    Args:
        block: The block.

    Returns:
        The places, in order, none overlapping another.
    """
    text = block.text
    budget = CODE_SCAN_FACTOR * len(text)
    ranges: list[Span] = []
    cursor = 0
    for code in block.code_spans:
        if budget <= 0:
            break
        if not code.strip():
            continue
        span, budget = _place(text, code, cursor, budget)
        if span is not None:
            ranges.append(span)
            cursor = span[1]
    return ranges


def excluded_spans(block: Block) -> tuple[Span, ...]:
    """Return the parts of the text of a block that are no prose.

    They are its URLs, without the punctuation of the sentence at their
    end, its e-mail addresses, its inline code, as `code_ranges` places it,
    and each chunk, a run of non-blank characters, that starts with a
    `DOTTED` name, from its dot, or that `CODE_CHUNK` finds once the
    `PROSE_EDGE` of its edges are stripped, without those edges: the
    punctuation after a URL or a file name still ends a sentence.

    Args:
        block: The block.

    Returns:
        The parts, sorted, with those that overlap or touch joined.
    """
    text = block.text
    spans: list[Span] = [
        (match.start(), match.start() + len(match.group().rstrip(_URL_TAIL)))
        for match in URL.finditer(text)
    ]
    spans.extend(match.span() for match in EMAIL.finditer(text))
    spans.extend(code_ranges(block))
    for match in CHUNK.finditer(text):
        chunk = match.group()
        core = chunk.strip(PROSE_EDGE)
        if not core:
            continue
        first = match.start() + len(chunk) - len(chunk.lstrip(PROSE_EDGE))
        dotted = DOTTED.match(chunk)
        if dotted is not None:
            # The name starts at its dot, the one before its first letter.
            spans.append((match.start() + dotted.end() - 2, first + len(core)))
        elif CODE_CHUNK.search(core):
            spans.append((first, first + len(core)))
    return tuple(merged_spans(spans))


def code_like(text: str) -> bool:
    """Return whether the text of a block looks like code.

    Args:
        text: The text.

    Returns:
        Whether it has at least `CODE_LIKE_MIN_CHUNKS` chunks, runs of
        non-blank characters, and at least `CODE_LIKE_SHARE` of them have a
        `CODE_MARK`. A ``|`` alone, which separates the cells of a table,
        is no chunk.
    """
    chunks = marked = 0
    for match in CHUNK.finditer(text):
        chunk = match.group()
        if chunk == _CELL:
            continue
        chunks += 1
        if CODE_MARK.search(chunk):
            marked += 1
    return (
        chunks >= CODE_LIKE_MIN_CHUNKS and marked >= CODE_LIKE_SHARE * chunks
    )


def _overlaps(span: Span, spans: Sequence[Span]) -> bool:
    """Return whether a place shares a character with one of merged places.

    Args:
        span: The place.
        spans: Places sorted by start, none overlapping another.

    Returns:
        Whether one of ``spans`` has a character of ``span``.
    """
    start, end = span
    index = bisect_left(spans, end, key=itemgetter(0))
    return index > 0 and spans[index - 1][1] > start


def _word_before(text: str, end: int) -> str:
    """Return the word that ends at a place, without case.

    The word is the run of non-blank characters before the place, without
    the `PROSE_EDGE` of its start.
    """
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:end].lstrip(PROSE_EDGE).casefold()


def _ends(text: str, end: re.Match[str], excluded: Sequence[Span]) -> bool:
    """Return whether a match of `SENTENCE_END` ends a sentence.

    Args:
        text: The text.
        end: The match.
        excluded: The excluded parts of the text, sorted and merged.

    Returns:
        False if its punctuation lies in an excluded part, if a lower-case
        letter that starts no excluded part follows it after the blanks, or
        if it is a dot alone after a word of `ABBREVIATIONS` or a single
        letter; else True.
    """
    start = end.start()
    group = end.group()
    marks = group[: len(group) - len(group.lstrip(_FINAL))]
    if _overlaps((start, start + len(marks)), excluded):
        return False
    after = BLANKS.match(text, end.end()).end()
    if (
        after < len(text)
        and text[after].islower()
        and not _overlaps((after, after + 1), excluded)
    ):
        return False
    if marks != _DOT:
        return True
    word = _word_before(text, start)
    return word not in ABBREVIATIONS and not (
        len(word) == 1 and word.isalpha()
    )


def sentences(text: str, excluded: Sequence[Span]) -> list[Span]:
    """Return the sentences of a text that end.

    A sentence ends at a match of `SENTENCE_END` that `_ends` accepts; the
    text after the last end is no sentence.

    Each match is followed by a blank or the end of the text, so the words
    that `_word_before` reads before two matches never overlap: the text
    is read a few times at most, whatever its shape.

    Args:
        text: The text.
        excluded: Its excluded parts, sorted and merged.

    Returns:
        The places of the sentences, in order, without the blanks of their
        edges.
    """
    found: list[Span] = []
    start = 0
    for end in SENTENCE_END.finditer(text):
        if _ends(text, end, excluded):
            found.append((BLANKS.match(text, start).end(), end.end()))
            start = end.end()
    return found


def count_words(text: str, span: Span, excluded: Sequence[Span]) -> int:
    """Return how many words a part of a text has.

    Args:
        text: The text.
        span: The part, such as a sentence.
        excluded: The excluded parts of the text, sorted and merged.

    Returns:
        The words and numbers of the part, as `tokenize` reads them, that
        touch no excluded part.
    """
    start, end = span
    return sum(
        1
        for token in tokenize(text[start:end])
        if token.kind in _COUNTED
        and not touches((start + token.start, start + token.end), excluded)
    )


def document_language(doc: Document, requested: str) -> str | None:
    """Return the language of the text of a document.

    Args:
        doc: The document.
        requested: The language that the user names, or ``"auto"``.

    Returns:
        The language that the user names, unless ``"auto"``; else the
        language that its `prose_sample` plainly reads as, as
        `strong_language` tells; else the language of the document.
    """
    if requested != AUTO:
        return requested
    return strong_language(prose_sample(doc)) or doc.lang


def prose_blocks(doc: Document, lang: str | None) -> list[Prose]:
    """Return the prose of a document, as the checks of its text read it.

    Args:
        doc: The document.
        lang: The language of its text, if known.

    Returns:
        Its blocks of `SPELLED_KINDS` that do not look like code, as
        `code_like` tells, in order; a block of `SENTENCE_KINDS` is checked
        in the language that it plainly reads as, if it does.
    """
    found: list[Prose] = []
    for block in doc.blocks:
        if block.kind not in SPELLED_KINDS or code_like(block.text):
            continue
        own = (
            block_language(block.text)
            if block.kind in SENTENCE_KINDS
            else None
        )
        found.append(Prose(block, own or lang, own, excluded_spans(block)))
    return found
