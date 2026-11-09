# Copyright 2026 Christophe Le Douarec
"""The words of a text that the speller checks.

A word is a run of non-blank characters, with the punctuation of its edges
stripped, cut at the en and em dashes. A run that touches an excluded
place, a URL or an e-mail address, or that looks like code or starts with
a dotted file name, gives no word.
`skip_reason` tells why a word is not spelled: too short, too long, too
many parts, a digit, a character that is not a Latin letter, an acronym or
a capital inside a part.

The checks of the text share the pieces of words defined here: the runs
of characters (`CHUNK`, `BLANKS`), the apostrophes and hyphens, the French
elided words (`ELISIONS`), the English possessive (`without_possessive`)
and the characters of a word (`is_word_char`).
"""

import re
from bisect import bisect_right
from dataclasses import dataclass
from itertools import islice
from operator import itemgetter
from typing import TYPE_CHECKING, Final

from doc_checker.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

type Span = tuple[int, int]

# The longest word that is spelled, in characters, and the most parts
# between hyphens it may have: the cost of a Hunspell lookup grows fast with
# the parts of a word.
MAX_WORD_CHARS: Final = 40
MAX_WORD_PARTS: Final = 5

# A URL, from its scheme or from "www." to the next blank; PDF extraction
# may leave a blank around the colon of the scheme. A scheme name followed
# by a colon is a mention of the scheme.
URL: Final = re.compile(
    r"(?:\b(?:https?|s?ftp|file|git|ssh|svn)\s?:\s?//|\bwww\.|\bmailto:)\S*"
    r"|\b(?:https?|s?ftp|file|mailto)(?=:)",
    re.IGNORECASE,
)
# An e-mail address. A match starts only at the start of a run of the
# characters of its name, so a long run without "@" is read once.
EMAIL: Final = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_QUOTES: Final = "\"'\U00002018\U00002019\U0000201c\U0000201d\xab\xbb"
# The punctuation of the edges of a word of prose.
PROSE_EDGE: Final = _QUOTES + "(),;:!?.\U00002026"
# What the edges of a word lose: the punctuation of prose and the marks of
# code or of a lightweight markup.
EDGE: Final = PROSE_EDGE + "[]{}<>*_`|~"
# What makes a run of non-blank characters look like code: a mark of code,
# a dot between a letter and a word character (a dot between two digits is
# a decimal number), or a leading hyphen (an option).
CODE_CHUNK: Final = re.compile(
    r"[/\\_=<>{}\[\]|`$%#@~^*+&]|::|[^\W\d_]\.\w|\w\.[^\W\d_]|^-",
)
# A run of non-blank characters that starts with a dot and a letter, after
# opening quotes or brackets, is code too: a file extension or a dotfile
# (".txt", ".gitignore"). An ellipsis ("...and") or a decimal (".5") is not.
# A match does not start inside a run of those quotes and brackets.
DOTTED: Final = re.compile(
    r"(?<![\"'(\[\U00002018\U0000201c\xab])"
    r"[\"'(\[\U00002018\U0000201c\xab]*\.[^\W\d_]",
)

# A run of non-blank characters, and a run of blanks, maybe empty.
CHUNK: Final = re.compile(r"\S+")
BLANKS: Final = re.compile(r"\s*")
# The apostrophe, the typographic one and the modifier letter apostrophe.
APOSTROPHES: Final = "'\U00002019\U000002bc"
APOSTROPHE: Final = re.compile(f"[{APOSTROPHES}]")
# The hyphen, and the hyphen and non-breaking hyphen of typography.
HYPHENS: Final = "-\U00002010\U00002011"
HYPHEN: Final = re.compile(f"[{HYPHENS}]")
# The French elided words that may start a word, before an apostrophe, without
# case: one letter ("l'agent"), or "qu" alone or ending a word ("qu'il").
ELISIONS: Final = frozenset(
    {"c", "d", "j", "l", "m", "n", "s", "t"}
    | {"qu", "jusqu", "lorsqu", "puisqu", "quoiqu"},
)

_LETTER: Final = (
    r"[A-Za-z\xc0-\xd6\xd8-\xf6\xf8-\U0000024f\U00001e00-\U00001eff]"
)
_APOSTROPHE: Final = f"[{APOSTROPHES}]"
_PART: Final = rf"{_LETTER}+(?:{_APOSTROPHE}{_LETTER}+)*"
# A word of Latin letters: each part between hyphens begins and ends with a
# letter and has an apostrophe only between two letters; the whole word may
# end with one apostrophe (a plural possessive).
LATIN_WORD: Final = re.compile(
    rf"{_PART}(?:[{HYPHENS}]{_PART})*{_APOSTROPHE}?",
)

_BETWEEN_DASHES: Final = re.compile(r"[^\U00002013\U00002014]+")
_PART_BREAK: Final = re.compile(f"[{HYPHENS}{APOSTROPHES}]")
_STRAIGHT: Final = str.maketrans(
    {"\U00002019": "'", "\U000002bc": "'"},
)
# The endings of an English possessive, ``'s`` before the apostrophe alone.
_POSSESSIVES: Final = tuple(
    f"{apostrophe}{ending}"
    for ending in ("s", "")
    for apostrophe in APOSTROPHES
)
_MIN_ALPHANUMERICS: Final = 2
# The most distinct runs whose words `words` remembers in one call: enough
# for a block, while a huge paragraph of distinct runs does not double its
# memory.
MEMO_MAX: Final = 4_096
_COMMENT: Final = "#"


@dataclass(frozen=True, slots=True)
class Word:
    """A word of a text, and its place in that text.

    Attributes:
        text: The word.
        start: Where the word starts in the text.
        end: Where the word ends in the text.
    """

    text: str
    start: int
    end: int


def touches(span: Span, spans: Sequence[Span]) -> bool:
    """Return whether a place overlaps or touches one of sorted places.

    Two places touch when one ends where the other starts.

    Args:
        span: The place.
        spans: Places sorted by start, none inside another, as merged
            places are.

    Returns:
        Whether one of ``spans`` overlaps or touches ``span``.
    """
    start, end = span
    index = bisect_right(spans, end, key=itemgetter(0))
    return index > 0 and spans[index - 1][1] >= start


def words(text: str, excluded: Sequence[Span] = ()) -> list[Word]:
    """Return the words of a text that may be spelled.

    A word comes from a run of non-blank characters that touches no
    excluded place and no match of `URL` or `EMAIL`, that does not start
    with a `DOTTED` name, and that has no match of `CODE_CHUNK` once the
    `PROSE_EDGE` of its edges are stripped. The
    run is cut at the en and em dashes, and each piece loses the `EDGE` of
    its edges; empty pieces give no word. The outcome of each distinct run
    is computed once per call, for the first `MEMO_MAX` distinct runs.

    Args:
        text: The text.
        excluded: Places of the text that give no word.

    Returns:
        The words, in the order of the text.
    """
    blocked = merged_spans(
        [
            *excluded,
            *(match.span() for match in URL.finditer(text)),
            *(match.span() for match in EMAIL.finditer(text)),
        ],
    )
    pieces_of: dict[str, tuple[Span, ...]] = {}
    found: list[Word] = []
    index = 0
    for match in CHUNK.finditer(text):
        start, end = match.span()
        while index < len(blocked) and blocked[index][1] < start:
            index += 1
        if index < len(blocked) and blocked[index][0] <= end:
            continue
        chunk = match.group()
        pieces = pieces_of.get(chunk)
        if pieces is None:
            pieces = _pieces(chunk)
            if len(pieces_of) < MEMO_MAX:
                pieces_of[chunk] = pieces
        found.extend(
            Word(chunk[first:last], start + first, start + last)
            for first, last in pieces
        )
    return found


def merged_spans(spans: Iterable[Span]) -> list[Span]:
    """Return places sorted, with those that overlap or touch joined.

    Args:
        spans: The places, in any order.

    Returns:
        The places sorted by start, none overlapping or touching another,
        as `touches` reads them.
    """
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _pieces(chunk: str) -> tuple[Span, ...]:
    """Return the places of the words of a run of non-blank characters.

    Args:
        chunk: The run.

    Returns:
        The places in the run of its words, none if it looks like code.
    """
    if DOTTED.match(chunk) or CODE_CHUNK.search(chunk.strip(PROSE_EDGE)):
        return ()
    pieces: list[Span] = []
    for part in _BETWEEN_DASHES.finditer(chunk):
        piece = part.group()
        core = piece.strip(EDGE)
        if core:
            first = part.start() + len(piece) - len(piece.lstrip(EDGE))
            pieces.append((first, first + len(core)))
    return tuple(pieces)


def skip_reason(word: str) -> str | None:
    """Return why a word is not spelled, if it is not.

    The reasons, in the order they are tested: ``"short"`` (fewer than two
    letters or digits), ``"long"`` (more than `MAX_WORD_CHARS`
    characters), ``"parts"`` (more than `MAX_WORD_PARTS` parts between
    hyphens, U+2010 and U+2011 included), ``"digit"``, ``"symbol"`` (not a
    `LATIN_WORD`),
    ``"acronym"`` (all its letters are capitals, once a possessive is
    removed) and ``"mixed"`` (a capital after the first letter of a part
    between hyphens and apostrophes).

    Args:
        word: The word.

    Returns:
        The reason, or None if the word is spelled.
    """
    reason: str | None = None
    if _is_short(word):
        reason = "short"
    elif len(word) > MAX_WORD_CHARS:
        reason = "long"
    elif sum(map(word.count, HYPHENS)) >= MAX_WORD_PARTS:
        reason = "parts"
    elif any(char.isdigit() for char in word):
        reason = "digit"
    elif LATIN_WORD.fullmatch(word) is None:
        reason = "symbol"
    elif _is_acronym(word):
        reason = "acronym"
    elif _is_mixed(word):
        reason = "mixed"
    return reason


def _is_short(word: str) -> bool:
    """Return whether a word has fewer than two letters or digits.

    The count stops at two, so a long run costs no more than its start.
    """
    alphanumerics = islice(filter(str.isalnum, word), _MIN_ALPHANUMERICS)
    return sum(1 for _ in alphanumerics) < _MIN_ALPHANUMERICS


def _is_acronym(word: str) -> bool:
    """Return whether all the letters of a word are capitals.

    A possessive ending is not part of the acronym: ``API's``.
    """
    return all(
        char.isupper() for char in without_possessive(word) if char.isalpha()
    )


def _is_mixed(word: str) -> bool:
    """Return whether a capital follows the first letter of a part.

    The parts of a word are between its hyphens and apostrophes.
    """
    return any(
        any(char.isupper() for char in part[1:])
        for part in _PART_BREAK.split(word)
    )


def fold(word: str) -> str:
    """Return the form under which words match whatever their case.

    Args:
        word: The word.

    Returns:
        The word case folded, its typographic apostrophes read ``'``.
    """
    return word.casefold().translate(_STRAIGHT)


def without_possessive(word: str) -> str:
    """Return a word without its English possessive, if it has one.

    Args:
        word: The word.

    Returns:
        The word without a final ``'s``, else without a final apostrophe,
        any of `APOSTROPHES`, when more comes before it; else the word. The
        ``s`` is in lower case: ``API'S`` keeps its ending.
    """
    for ending in _POSSESSIVES:
        if word.endswith(ending) and len(word) > len(ending):
            return word[: -len(ending)]
    return word


def is_word_char(char: str) -> bool:
    """Return whether a character is a letter, a digit or an underscore.

    That is what a word character of a regular expression is.
    """
    return char.isalnum() or char == "_"


def read_word_file(path: Path) -> tuple[str, ...]:
    """Read a file of personal words.

    The file is UTF-8, with or without a byte order mark, and has one word
    per line; blank lines and lines that start with ``#`` are ignored, and
    the blanks around a word are dropped.

    Args:
        path: The file.

    Returns:
        The words, in the order of the file.

    Raises:
        ConfigError: If the file cannot be read, is not UTF-8, or has a
            line with more than one word.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        msg = f"{path}: cannot read file: {exc.strerror or exc}"
        raise ConfigError(msg) from None
    except UnicodeDecodeError:
        msg = f"{path}: cannot read file: not UTF-8 text"
        raise ConfigError(msg) from None
    found: list[str] = []
    for number, line in enumerate(text.split("\n"), start=1):
        word = line.strip()
        if not word or word.startswith(_COMMENT):
            continue
        if any(char.isspace() for char in word):
            msg = f"{path}:{number}: expected one word per line"
            raise ConfigError(msg)
        found.append(word)
    return tuple(found)
