# Copyright 2026 Christophe Le Douarec
"""Check the style of the prose of a block: repetition, length, language.

A word written twice in a row is a slip, unless the language wants it
(`REPEAT_ALLOWED`), the second one is a name, it is no word of the
dictionary, or the same word comes right before or after the pair: a run of
three or more is meant, as emphasis ("really really really") or as the
labels of a figure that stayed text. A sentence of more words than a
threshold is long. A block that plainly reads as another language than that
of the document is a block in another language.

The pattern of the repeated words starts only at the start of a run of
letters, so a long run is read once; the blanks before a pair are read
once, by the pair that they come before.
"""

import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from doc_checker.audit.prose import Mark, Prose, count_words, sentences
from doc_checker.audit.sides import quote
from doc_checker.model.findings import Severity
from doc_checker.spell.words import touches
from doc_checker.text.lang import LANGUAGE_NAMES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from doc_checker.spell.words import Span

# The words that a language repeats on purpose, without case.
REPEAT_ALLOWED: Final = MappingProxyType(
    {"fr": frozenset({"vous", "nous"}), "en": frozenset({"that", "had"})},
)
# A word of letters, not after a letter, a digit, an apostrophe or a
# hyphen, then the same word after blanks of one line (rows of a table are
# split by a line break), not before one of those, and not before the same
# word again after blanks of one line.
_REPEAT: Final = re.compile(
    r"(?<![\w'\U00002019-])([^\W\d_]+)[^\S\n]+(\1)(?![\w'\U00002019-])"
    r"(?![^\S\n]+\1(?![\w'\U00002019-]))",
    re.IGNORECASE,
)
# What a word may not touch, besides a word character: an apostrophe or a
# hyphen.
_WORD_EDGE: Final = frozenset("'\U00002019-")
_MIN_WORD: Final = 2
_REPEATED: Final = "text.repeated_word"
_LONG: Final = "text.long_sentence"
_FOREIGN: Final = "text.block_language"


def _glued(char: str) -> bool:
    """Return whether a character would make a word longer.

    It is a word character of a regular expression, an apostrophe or a
    hyphen, as `_REPEAT` reads them.
    """
    return char.isalnum() or char == "_" or char in _WORD_EDGE


def _follows_itself(text: str, start: int, word: str) -> bool:
    """Return whether the same word comes right before a place.

    Args:
        text: The text.
        start: The place, where a word starts.
        word: That word, as written.

    Returns:
        Whether blanks of one line, then the same word without case, as a
        whole, come right before the place.
    """
    end = start
    while end > 0 and text[end - 1] != "\n" and text[end - 1].isspace():
        end -= 1
    begin = end - len(word)
    return (
        end < start
        and begin >= 0
        and text[begin:end].casefold() == word.casefold()
        and not (begin > 0 and _glued(text[begin - 1]))
    )


def repeated_words(
    text: str,
    lang: str,
    excluded: Sequence[Span],
    known: Callable[[str], bool],
) -> list[Mark]:
    """Return the words that are written twice in a row.

    A word of at least two letters is repeated when the second one has the
    case of the first or is in lower case (``The the``, not ``windows
    Windows``), the word is known, and it is not one of
    `REPEAT_ALLOWED` for the language. A repetition that touches an
    excluded part is left out, and so is one that the same word follows or
    comes after, across blanks of one line: a run of three or more.

    Args:
        text: The text of the block.
        lang: The language of the block.
        excluded: Its excluded parts, sorted and merged.
        known: Tells whether a word, without case, is a word of the
            dictionary of the language. It must answer False for a word
            that the speller does not vouch for: the speller answers True
            for a word once its work budget is spent, so the caller passes
            a function that answers False then, else a run of non-words
            would become findings.

    Returns:
        A mark for each repetition, over both words.
    """
    allowed = REPEAT_ALLOWED.get(lang, frozenset())
    marks: list[Mark] = []
    for match in _REPEAT.finditer(text):
        first, second = match.group(1), match.group(2)
        word = first.casefold()
        if (
            len(word) < _MIN_WORD
            or word in allowed
            or not (second == first or second.islower())
            or touches(match.span(), excluded)
        ):
            continue
        if _follows_itself(text, match.start(), first) or not known(word):
            continue
        marks.append(
            Mark(
                _REPEATED,
                match.span(),
                quote(word),
                f'"{quote(word)}" is written twice',
            ),
        )
    return marks


def long_sentences(
    text: str,
    excluded: Sequence[Span],
    *,
    minor: int,
    major: int,
) -> list[Mark]:
    """Return the sentences that have too many words.

    Args:
        text: The text of the block.
        excluded: Its excluded parts, sorted and merged.
        minor: A sentence of more words is a minor finding.
        major: A sentence of more words is a major finding.

    Returns:
        A mark over each long sentence, as `sentences` ends them.
    """
    marks: list[Mark] = []
    for span in sentences(text, excluded):
        count = count_words(text, span, excluded)
        if count > major:
            severity, limit = Severity.MAJOR, major
        elif count > minor:
            severity, limit = Severity.MINOR, minor
        else:
            continue
        marks.append(
            Mark(
                _LONG,
                span,
                f"over {limit} words",
                f"{count} words in one sentence",
                severity,
            ),
        )
    return marks


def foreign_block(prose: Prose, lang: str | None) -> Mark | None:
    """Return the mark of a block that reads as another language.

    Args:
        prose: The block.
        lang: The language of the document.

    Returns:
        A mark over the whole text, if the block plainly reads as a
        language of `LANGUAGE_NAMES` that is not that of the document,
        another language of `LANGUAGE_NAMES`; else None.
    """
    own = prose.own
    if own is None or lang is None or own == lang:
        return None
    if own not in LANGUAGE_NAMES or lang not in LANGUAGE_NAMES:
        return None
    return Mark(
        _FOREIGN,
        (0, len(prose.block.text)),
        quote(own),
        f"this block reads as {LANGUAGE_NAMES[own]}, "
        f"the document as {LANGUAGE_NAMES[lang]}",
    )
