# Copyright 2026 Christophe Le Douarec
"""Check the typography of the prose of a block, in French and in English.

French sets a space before ``; : ! ?`` and a closing guillemet, and after
an opening one, a space that does not break the line, one of
`NON_BREAKING`; English sets none before ``, ; : ! ?``. French quotes with
guillemets, English with quotation marks. An ordinal is written with the
letters its language wants, and a number is set apart from its unit, one
of `UNITS`, or in French one of `FRENCH_UNITS`. Two marks of punctuation
after a word, where one is meant, are a slip in any language.

A PDF does not mark its inline code, but it tells the words that it sets in
a monospaced font, its code words: a number glued to its unit, and a blank
or no space before a mark after a word, are code when that token is one of
their `code_tokens`, as ``1cm`` of ``xshift=1cm`` or ``tar`` before
``tar:``. Only the token of the mark is matched: a code word never hides a
part of the text.

The patterns read a text once: a match that could start inside a run of
the characters it starts with is refused there by a lookbehind, so a long
run is not read again from each of its places; a quotation is at most
`_QUOTED_MAX` characters long.
"""

import re
from bisect import bisect_left
from typing import TYPE_CHECKING, Final

from doc_checker.audit.prose import Mark
from doc_checker.audit.sides import quote
from doc_checker.compare.drafts import ARROW
from doc_checker.spell.words import BLANKS, touches
from doc_checker.text.lang import LANGUAGE_NAMES
from doc_checker.text.normalize import ELLIPSIS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from doc_checker.spell.words import Span

# The spaces that do not break the line: no-break, narrow no-break and
# figure spaces.
NON_BREAKING: Final = "\xa0\U0000202f\U00002007"
# The units that a number is set apart from, in any language. CSS and
# LaTeX lengths (px, pt, em) and the one-letter units (s, m, h, g) are left
# out: `16px` is a value, `4G` and `3D` are names.
UNITS: Final = (
    "km",
    "cm",
    "mm",
    "\xb5m",
    "nm",
    "kg",
    "mg",
    "ms",
    "\xb5s",
    "ns",
    "min",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "kB",
    "KB",
    "MB",
    "GB",
    "TB",
    "PB",
    "KiB",
    "MiB",
    "GiB",
    "TiB",
    "kW",
    "kWh",
    "mAh",
    "dB",
)
# The French storage units, units in French text only: in English, `To`
# after a footnote number starts a sentence ("10To be more precise").
FRENCH_UNITS: Final = ("Ko", "Mo", "Go", "To", "Kio", "Mio", "Gio", "Tio")

_FRENCH: Final = "fr"
_ENGLISH: Final = "en"
_SPACE_MISSING: Final = "text.space_missing"
_SPACE_BREAKABLE: Final = "text.space_breakable"
_SPACE_BEFORE: Final = "text.space_before_punctuation"
_QUOTES: Final = "text.quotes"
_ORDINAL: Final = "text.ordinal"
_UNIT_SPACE: Final = "text.unit_space"
_DOUBLE: Final = "text.double_punctuation"

# The spaces that may break the line: the space, the thin and hair spaces.
_BREAKABLE: Final = " \U00002009\U0000200a"
_OPENING: Final = "\xab"
_CLOSING: Final = "\xbb"
# A French high mark glued to a letter, or to a closing bracket,
# guillemet or quote.
_GLUED: Final = re.compile(
    r"(?<=[^\W\d_]|[)\xbb\U0000201d\"\U00002019])[;:!?]",
)
# What may follow a glued mark when it is not French punctuation, as in
# `a:b`, `C:\`, `?id=1&x` or `&amp;`, besides a letter, and a digit after
# `:`, `?` or `!` (`10:30`, `a?1`). The pairs `?!`, `!?` and `??` are left
# to the doubled punctuation, which does not judge them either.
_GLUED_ON_PURPOSE: Final = {
    ":": frozenset("/\\="),
    ";": frozenset("&"),
    "?": frozenset("=&!?"),
    "!": frozenset("=&?"),
}
# The marks before which LaTeX (babel-french) sets a thin space, which the
# extraction of a PDF often drops: a PDF does not tell whether they are
# glued. Before ``:`` and inside guillemets the space is a full one.
_THIN_SPACED: Final = frozenset(";!?")
# A guillemet glued to the text it opens or closes. An opening guillemet
# that a mark of punctuation follows, or a closing one that follows an
# opening guillemet or bracket, is the character itself, named in the text
# ("pour" then an opening guillemet and a comma).
_GLUED_OPENING: Final = re.compile(rf"{_OPENING}(?=[^\s,.;:!?){_CLOSING}])")
_GLUED_CLOSING: Final = re.compile(rf"(?<=[^\s({_OPENING}]){_CLOSING}")
# A breakable space between a word and a French high mark or a closing
# guillemet, or after an opening guillemet. The space after an opening
# guillemet is the guillemet's, even before a mark that the guillemets
# name.
_BREAKABLE_BEFORE: Final = re.compile(
    rf"(?<=[^\s{_OPENING}])[{_BREAKABLE}][;:!?{_CLOSING}]",
)
_BREAKABLE_AFTER: Final = re.compile(rf"{_OPENING}[{_BREAKABLE}](?=\S)")
# A word of letters and digits, then blanks, then an English mark that a
# blank or the end follows. A match starts only at the start of a word.
_SPACED: Final = re.compile(
    rf"(?<![^\W_])(?P<word>[^\W_]+)"
    rf"(?P<blanks>[{_BREAKABLE}{NON_BREAKING}]+)[,;:!?](?=\s|$)",
)
_ENGLISH_MARKS: Final = frozenset(",;:!?")
# The marks that end a sentence: one that a lower-case word follows is a
# character named in the text (`Use ? to mean`).
_ENDING_MARKS: Final = frozenset("!?")
# The words that never end a clause, articles and conjunctions: a mark
# after one of them is a character named in the text (`marked with a ?`,
# `* or ?`).
_NAMING_WORDS: Final = frozenset({"a", "an", "the", "and", "or", "either"})
# A French quotation between quotation marks, straight or curly, of 1 to
# `_QUOTED_MAX` characters; an English one between guillemets.
_QUOTED_MAX: Final = 80
_FRENCH_QUOTED: Final = re.compile(
    rf"(?<!\w)(?:\"[^\"\U0000201c\U0000201d]{{1,{_QUOTED_MAX}}}\""
    rf"|\U0000201c[^\"\U0000201c\U0000201d]{{1,{_QUOTED_MAX}}}\U0000201d)"
    r"(?!\w)",
)
_ENGLISH_QUOTED: Final = re.compile(
    rf"{_OPENING}[^{_OPENING}{_CLOSING}]{{1,{_QUOTED_MAX}}}{_CLOSING}",
)
# The quotation mark that French sets around a quotation inside guillemets.
_INNER_OPENING: Final = "\U0000201c"
_GUILLEMETS: Final = re.compile(rf"[{_OPENING}{_CLOSING}]")
# What a number may not follow, so that a match starts only at the start of
# a number, never inside a word, a version or a decimal number. A number is
# written in ASCII digits.
_NUMBER_START: Final = r"(?<![\w.,])"
_NUMBER: Final = r"[0-9]+"
# A Roman numeral from I to XXXIX, in capitals, as centuries and chapters
# are numbered: a name such as `Leme` or `Mème` is no ordinal. A numeral
# starts with one of its letters, so it is never empty.
_ROMAN: Final = r"(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})"
# The right ending of a French ordinal for each wrong one, before an
# optional plural ``s``. The endings that give ``re`` are only for one of
# `_FIRST`. The first is written with one of `_FIRST_ENDINGS`, whose gender
# the text does not tell: the detail names both, and the pattern of an
# ending that gives ``e`` names the first one.
_FRENCH_ENDINGS: Final = {
    "ième": "e",
    "ieme": "e",
    "ème": "e",
    "eme": "e",
    "éme": "e",
    "ière": "re",
    "ère": "re",
    "ere": "re",
}
_FIRST: Final = frozenset({"1", "I"})
_FIRST_ENDINGS: Final = ("er", "re")
_FRENCH_ORDINAL: Final = re.compile(
    rf"{_NUMBER_START}({_NUMBER}|{_ROMAN})"
    rf"({'|'.join(_FRENCH_ENDINGS)})(s?)\b",
)
_ENGLISH_ORDINAL: Final = re.compile(
    rf"{_NUMBER_START}({_NUMBER})(st|nd|rd|th)\b",
)
_ENGLISH_ENDINGS: Final = {"1": "st", "2": "nd", "3": "rd"}
# The last two digits of the ordinals that end in th whatever their last one.
_TEENS: Final = range(11, 14)
# The micro sign of `UNITS`, which the Greek mu of a formula stands for.
_MICRO: Final = "\xb5"
_MICROS: Final = "[\xb5\U000003bc]"


def _unit_pattern(units: Sequence[str]) -> re.Pattern[str]:
    """Return the pattern of a number glued to one of some units.

    The units are its alternatives, the longest first; the micro sign
    matches the Greek mu too.
    """
    names = "|".join(
        unit.replace(_MICRO, _MICROS)
        for unit in sorted(units, key=len, reverse=True)
    )
    return re.compile(
        rf"{_NUMBER_START}({_NUMBER}(?:[.,]{_NUMBER})?)({names})\b",
    )


_UNIT: Final = _unit_pattern(UNITS)
_FRENCH_UNIT: Final = _unit_pattern(UNITS + FRENCH_UNITS)
# A token of a code word: a run of word characters and dots.
_CODE_TOKEN: Final = re.compile(r"[\w.]+")
# Two marks where one is meant, after a word of letters (not after a hash,
# `4b9458b..`) and before a blank, a closing quote or bracket, or the end;
# never `...`, `?!`, `??`, `::` or `a..b`. A match starts only at the start
# of a word.
_DOUBLED: Final = re.compile(
    r"(?<!\w)[^\W\d_]+(?P<marks>,,|;;|!!|\.\.|[,;:]\.)"
    r"(?=[\s\"\xbb\U0000201d\U00002019)]|$)",
)


def code_tokens(code_words: Iterable[str]) -> frozenset[str]:
    """Return the tokens of the code words of a document, without case.

    Args:
        code_words: The words that the document sets as code, such as
            `Document.code_words`.

    Returns:
        The runs of word characters and dots of each word, without the
        dots of their edges, case folded: ``xshift=0.5cm`` gives
        ``xshift`` and ``0.5cm``.
    """
    return frozenset(
        token
        for word in code_words
        for match in _CODE_TOKEN.finditer(word)
        if (token := match.group().strip(".").casefold())
    )


def _word_before(text: str, end: int) -> str:
    """Return the run of letters and digits that ends at a place."""
    start = end
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    return text[start:end]


def _glued_on_purpose(mark: str, after: str) -> bool:
    """Return whether a mark glued to a word is no French punctuation.

    Args:
        mark: The mark, one of ``; : ! ?``.
        after: The character after it, or ``""`` at the end of the text.

    Returns:
        Whether the character after it is a letter, a digit after ``:``,
        ``?`` or ``!``, or one of `_GLUED_ON_PURPOSE`.
    """
    if not after:
        return False
    if after.isalpha() or (mark != ";" and after.isdigit()):
        return True
    return after in _GLUED_ON_PURPOSE[mark]


def _french_spaces(
    text: str,
    *,
    breakable: bool,
    code: frozenset[str],
) -> Iterator[Mark]:
    """Yield the French spaces missing or breakable around punctuation.

    Args:
        text: The text.
        breakable: Whether the text tells a breakable space from a
            non-breaking one, and keeps a thin space, as `typography_marks`
            reads it.
        code: The tokens of the code words of the document.

    Yields:
        The marks of the spaces: none before ``; ! ?`` when not
        ``breakable``, as in a PDF, which drops a thin space, and none
        before a mark glued to a word of ``code``.
    """
    for match in _GLUED.finditer(text):
        if not breakable and match.group() in _THIN_SPACED:
            continue
        if _glued_on_purpose(
            match.group(),
            text[match.end() : match.end() + 1],
        ):
            continue
        if code and _word_before(text, match.start()).casefold() in code:
            continue
        mark = quote(match.group())
        yield Mark(
            _SPACE_MISSING,
            match.span(),
            mark,
            f'no space before "{mark}"',
        )
    for pattern, place in (
        (_GLUED_OPENING, "after"),
        (_GLUED_CLOSING, "before"),
    ):
        for match in pattern.finditer(text):
            mark = quote(match.group())
            yield Mark(
                _SPACE_MISSING,
                match.span(),
                mark,
                f'no space {place} "{mark}"',
            )
    if not breakable:
        return
    for pattern, place, at in (
        (_BREAKABLE_BEFORE, "before", -1),
        (_BREAKABLE_AFTER, "after", 0),
    ):
        for match in pattern.finditer(text):
            mark = quote(match.group()[at])
            yield Mark(
                _SPACE_BREAKABLE,
                match.span(),
                mark,
                f'the space {place} "{mark}" can break the line',
            )


def _english_spaces(text: str, code: frozenset[str]) -> Iterator[Mark]:
    """Yield the blanks before English punctuation.

    A mark is a character named in the text, not punctuation, after one of
    `_NAMING_WORDS` (``marked with a ?``), when another of `_ENGLISH_MARKS`
    follows it after blanks (``the characters ; ! ? and :``), or when it is
    one of `_ENDING_MARKS` and a lower-case letter follows it after blanks
    (``Use ? to mean``). The mark after another mark follows a mark, not a
    word, and does not match. A mark after a word of ``code``, the tokens
    of the code words of the document, is code.
    """
    for match in _SPACED.finditer(text):
        word = match.group("word").casefold()
        if word in _NAMING_WORDS or word in code:
            continue
        after = BLANKS.match(text, match.end()).end()
        following = text[after : after + 1]
        mark = match.group()[-1]
        if following in _ENGLISH_MARKS or (
            mark in _ENDING_MARKS and following.islower()
        ):
            continue
        mark = quote(mark)
        yield Mark(
            _SPACE_BEFORE,
            (match.start("blanks"), match.end()),
            mark,
            f'space before "{mark}"',
        )


def _quotations(text: str, lang: str) -> Iterator[Mark]:
    """Yield the quotations set with the quotation marks of another language.

    A quotation starts with a letter or a digit, after blanks: quotation
    marks around a mark of punctuation name characters, such as an opening
    guillemet between brackets. In French, a quotation between curly
    quotation marks inside guillemets is the quotation in a quotation that
    French sets so, as `_inside_guillemets` tells.

    Args:
        text: The text.
        lang: Its language, French or English.

    Yields:
        A mark of each quotation, the quotation marks included.
    """
    pattern = _FRENCH_QUOTED if lang == _FRENCH else _ENGLISH_QUOTED
    guillemets: list[int] | None = None
    for match in pattern.finditer(text):
        quoted = match.group()
        if not quoted[1:-1].lstrip()[:1].isalnum():
            continue
        if lang == _FRENCH and quoted[0] == _INNER_OPENING:
            if guillemets is None:
                guillemets = [g.start() for g in _GUILLEMETS.finditer(text)]
            if _inside_guillemets(text, match.span(), guillemets):
                continue
        opening = quote(quoted[0])
        yield Mark(
            _QUOTES,
            match.span(),
            opening,
            f"quoted with {opening}{ELLIPSIS}{quote(quoted[-1])}"
            f" in {LANGUAGE_NAMES[lang]} text",
        )


def _inside_guillemets(
    text: str,
    span: Span,
    guillemets: Sequence[int],
) -> bool:
    """Return whether a part of a text lies inside guillemets.

    Args:
        text: The text.
        span: The part.
        guillemets: The places of the guillemets of the text, sorted.

    Returns:
        Whether the last guillemet before the part opens and the first
        one after it closes.
    """
    before = bisect_left(guillemets, span[0])
    after = bisect_left(guillemets, span[1])
    return (
        before > 0
        and text[guillemets[before - 1]] == _OPENING
        and after < len(guillemets)
        and text[guillemets[after]] == _CLOSING
    )


def _ordinal(
    span: Span,
    number: str,
    found: str,
    right: str,
    rights: Sequence[str],
) -> Mark:
    """Return the mark of an ordinal whose ending is wrong.

    Args:
        span: Where the ordinal lies.
        number: Its number.
        found: Its ending.
        right: The ending that the pattern names.
        rights: The endings that it may have, which the detail names.

    Returns:
        The mark, whose pattern changes the ending.
    """
    written = " or ".join(f'"{quote(number + form)}"' for form in rights)
    return Mark(
        _ORDINAL,
        span,
        f"{quote(found)}{ARROW}{quote(right)}",
        f'"{quote(number + found)}" is written {written}',
    )


def _french_ordinals(text: str) -> Iterator[Mark]:
    """Yield the French ordinals written with too many letters."""
    for match in _FRENCH_ORDINAL.finditer(text):
        number, ending, plural = match.groups()
        right = _FRENCH_ENDINGS[ending]
        if number in _FIRST:
            right = _FIRST_ENDINGS[0] if right == "e" else right
            rights = _FIRST_ENDINGS
        elif right == "re":
            continue
        else:
            rights = (right,)
        yield _ordinal(
            match.span(),
            number,
            ending + plural,
            right + plural,
            [form + plural for form in rights],
        )


def _english_ordinals(text: str) -> Iterator[Mark]:
    """Yield the English ordinals whose ending the number does not want."""
    for match in _ENGLISH_ORDINAL.finditer(text):
        number, ending = match.groups()
        if int(number[-2:]) in _TEENS:
            right = "th"
        else:
            right = _ENGLISH_ENDINGS.get(number[-1], "th")
        if ending != right:
            yield _ordinal(match.span(), number, ending, right, (right,))


def _units(
    text: str,
    lang: str | None,
    code: frozenset[str],
) -> Iterator[Mark]:
    """Yield the numbers glued to a unit, but those of ``code``.

    Args:
        text: The text.
        lang: Its language: the units are those of `UNITS`, and in French
            those of `FRENCH_UNITS` too.
        code: The tokens of the code words of the document: a number and
            its unit that are one of them are code.

    Yields:
        The marks of the numbers and their units.
    """
    pattern = _FRENCH_UNIT if lang == _FRENCH else _UNIT
    for match in pattern.finditer(text):
        if code and match.group().casefold() in code:
            continue
        number, unit = match.groups()
        yield Mark(
            _UNIT_SPACE,
            match.span(),
            quote(unit),
            f"no space between {quote(number)} and {quote(unit)}",
        )


def _doubled(text: str) -> Iterator[Mark]:
    """Yield the doubled punctuation after a word."""
    for match in _DOUBLED.finditer(text):
        marks = quote(match.group("marks"))
        yield Mark(
            _DOUBLE,
            match.span("marks"),
            marks,
            f'"{marks}" doubles the punctuation',
        )


def _marks(
    text: str,
    lang: str | None,
    *,
    breakable: bool,
    code: frozenset[str],
) -> Iterator[Mark]:
    """Yield the marks of every rule of a language, in no order."""
    if lang == _FRENCH:
        yield from _french_spaces(text, breakable=breakable, code=code)
        yield from _quotations(text, lang)
        yield from _french_ordinals(text)
    elif lang == _ENGLISH:
        yield from _english_spaces(text, code)
        yield from _quotations(text, lang)
        yield from _english_ordinals(text)
    yield from _units(text, lang, code)
    yield from _doubled(text)


def typography_marks(
    text: str,
    lang: str | None,
    excluded: Sequence[Span],
    *,
    breakable: bool,
    code: frozenset[str] = frozenset(),
) -> list[Mark]:
    """Return the marks of the typography of the prose of a block.

    In French: `text.space_missing`, `text.space_breakable` if
    ``breakable``, `text.quotes` and `text.ordinal`. In English:
    `text.space_before_punctuation`, `text.quotes` and `text.ordinal`. In
    any language, or none: `text.unit_space`, with the units of
    `FRENCH_UNITS` in French only, and `text.double_punctuation`.

    Args:
        text: The text of the block.
        lang: The language in which the block is checked, if known.
        excluded: The parts of the text that are no prose, sorted and
            merged, as `excluded_spans` returns them.
        breakable: Whether a breakable space is told from a non-breaking
            one, and a thin space is kept, as in HTML; a PDF may set a
            non-breaking space as a plain one, and drop a thin one, so
            that neither a breakable space nor a missing one before
            ``; ! ?`` is judged.
        code: The tokens of the code words of the document, as
            `code_tokens` returns them: a number glued to a unit, and a
            blank or a missing space before a mark after a word, are not
            judged when that number and unit, or that word, is one of them.

    Returns:
        The marks that touch no excluded part, sorted by place, then by
        rule.
    """
    return sorted(
        (
            mark
            for mark in _marks(text, lang, breakable=breakable, code=code)
            if not touches(mark.span, excluded)
        ),
        key=lambda mark: (mark.span, mark.rule),
    )
