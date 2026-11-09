# Copyright 2026 Christophe Le Douarec
"""Text normalization: display text, comparison keys and printable labels.

Every replacement or removal of the display text and of the keys increments a
counter so that reports can show how much was normalized away instead of
hiding it.
"""

import re
import unicodedata
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections import Counter

# Ranges of a regular expression character class, which tell the characters of
# a text of a document that must not reach a reader or a terminal. The layout
# characters are the tab and the line feed, which a terminal keeps.
LAYOUT_CLASS: Final = r"\t\n"
# The other C0 controls, the delete and the C1 controls.
CONTROL_CLASS: Final = r"\x00-\x08\x0b-\x1f\x7f-\x9f"
# The bidirectional embeddings, overrides and isolates.
BIDI_CLASS: Final = r"\U0000202a-\U0000202e\U00002066-\U00002069"

_LIGATURES: Final = {
    "\U0000fb00": "ff",
    "\U0000fb01": "fi",
    "\U0000fb02": "fl",
    "\U0000fb03": "ffi",
    "\U0000fb04": "ffl",
    "\U0000fb05": "st",
    "\U0000fb06": "st",
}
_LIGATURE_RE: Final = re.compile(r"[\U0000fb00-\U0000fb06]")
_INVISIBLE_RE: Final = re.compile(
    r"[\xad\U0000200b-\U0000200d\U00002060\U0000feff]",
)
_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b-\x1f]")

_SPECIAL_SPACE_RE: Final = re.compile(
    r"[\xa0\U0000202f\U00002007\U00002009\U0000200a]",
)
_ANCHOR_GLYPH_RE: Final = re.compile(r"[\xb6\U0001f517]")
# A space before a high punctuation mark. A match starts only at the start
# of a run of spaces, so a long run that no such mark follows is read once.
_SPACE_BEFORE_RE: Final = re.compile(r"(?<![ ])[ ]+([;:!?\xbb])")
_SPACE_AFTER_RE: Final = re.compile(r"(\xab)[ ]+")
_SINGLE_QUOTE_RE: Final = re.compile(
    r"[\U00002018-\U0000201b\U00002039\U0000203a]",
)
_DOUBLE_QUOTE_RE: Final = re.compile(r"[\U0000201c-\U0000201f\xab\xbb]")
_DASH_RE: Final = re.compile(r"[\U00002010-\U00002015\U00002212]")
# What ends a text that was cut.
ELLIPSIS: Final = "\U00002026"
_ELLIPSIS_RE: Final = re.compile(ELLIPSIS)
_WHITESPACE_RE: Final = re.compile(r"[ \t\r\n\f\v]+")
_NON_WORD_RE: Final = re.compile(r"[^\w\s]")
_UNPRINTABLE_RE: Final = re.compile(
    f"[{LAYOUT_CLASS}{CONTROL_CLASS}{BIDI_CLASS}]",
)
_COMBINING_MARK: Final = "Mn"


def _count(
    pattern: re.Pattern[str],
    replacement: str,
    text: str,
    counter: Counter[str],
    key: str,
) -> str:
    """Replace every match of a pattern and count the replacements.

    Args:
        pattern: Pattern to replace.
        replacement: Replacement, which may use group references.
        text: Text to transform.
        counter: Counter to update.
        key: Counter key to increment.

    Returns:
        The transformed text.
    """
    result, count = pattern.subn(replacement, text)
    if count:
        counter[key] += count
    return result


def normalize_display(text: str, counter: Counter[str]) -> str:
    """Return the display form of an extracted text.

    Applies NFC, expands ligatures and removes invisible and control
    characters. U+FFFD is kept so that encoding damage stays detectable.

    Args:
        text: Text as extracted.
        counter: Normalization counter to update.

    Returns:
        The display text.
    """
    text = unicodedata.normalize("NFC", text)
    text, count = _LIGATURE_RE.subn(lambda m: _LIGATURES[m.group()], text)
    if count:
        counter["ligature"] += count
    text = _count(_INVISIBLE_RE, "", text, counter, "invisible")
    return _CONTROL_RE.sub("", text)


def strict_key(text: str, counter: Counter[str], *, code: bool = False) -> str:
    """Return the key used to diff two texts.

    Args:
        text: Display text.
        counter: Normalization counter to update.
        code: Whether the text is source code, which keeps its lines.

    Returns:
        The strict comparison key.
    """
    if code:
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip("\n")
    text = _count(_SPECIAL_SPACE_RE, " ", text, counter, "space")
    text = _count(_ANCHOR_GLYPH_RE, "", text, counter, "anchor_glyph")
    text = _count(_SPACE_BEFORE_RE, r"\1", text, counter, "punct_spacing")
    text = _count(_SPACE_AFTER_RE, r"\1", text, counter, "punct_spacing")
    text = _count(_SINGLE_QUOTE_RE, "'", text, counter, "quote")
    text = _count(_DOUBLE_QUOTE_RE, '"', text, counter, "quote")
    text = _count(_DASH_RE, "-", text, counter, "dash")
    text = _count(_ELLIPSIS_RE, "...", text, counter, "ellipsis")
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    """Return the text without combining marks.

    Args:
        text: Any text.

    Returns:
        The text with its accents removed, in NFC.
    """
    decomposed = unicodedata.normalize("NFD", text)
    kept = (
        char
        for char in decomposed
        if unicodedata.category(char) != _COMBINING_MARK
    )
    return unicodedata.normalize("NFC", "".join(kept))


def count_marks(text: str) -> int:
    """Return the number of combining marks of a text, after NFD.

    Args:
        text: Any text.

    Returns:
        The number of combining marks.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return sum(
        unicodedata.category(char) == _COMBINING_MARK for char in decomposed
    )


def loose_key(strict: str) -> str:
    """Return the key used to align blocks.

    Args:
        strict: Strict comparison key.

    Returns:
        The key without case, accents or punctuation.
    """
    text = strip_accents(strict.casefold())
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    """Return a text cut to a number of characters.

    Args:
        text: Any text.
        limit: Most characters kept.

    Returns:
        The text, or its start followed by `ELLIPSIS` if it is longer.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + ELLIPSIS


def printable(text: str, *, limit: int | None = None) -> str:
    """Return a text of a document that is safe to show on one line.

    Each control character and each bidirectional override or isolate becomes
    a blank, so that the text can neither move a cursor nor reorder the text
    around it. The runs of blanks, whatever their kind, become one space.

    Args:
        text: Any text, such as a page label, a metadata value or an
            identifier.
        limit: The most characters to keep, if any.

    Returns:
        The text on one line, without blanks at either end. If it is longer
        than `limit`, it is cut there and ends with an ellipsis.
    """
    text = " ".join(_UNPRINTABLE_RE.sub(" ", text).split())
    return text if limit is None else truncate(text, limit)
