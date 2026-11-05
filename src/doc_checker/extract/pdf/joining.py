# Copyright 2026 Christophe Le Douarec
"""Join the lines of a draft block into its text.

The lines of a paragraph are joined with spaces. A word that a line break
splits with a hyphen is put back together only if the document confirms it, by
using the whole word somewhere else: otherwise the hyphen stays, so that a
compound word such as ``non-blocking`` is not spoiled. Any other hyphen that
ends a line after a letter or a digit, and is followed by a letter or a digit,
is part of a compound word, such as ``x86-64`` or ``Wi-Fi``: the lines are
joined without a space and the hyphen stays. A code sample keeps its lines
apart: its indentation is rebuilt from the positions of the lines, and the
numbers of its lines are removed, with the empty lines that have only a number.
"""

import re
from itertools import pairwise, starmap
from typing import TYPE_CHECKING, Final

from doc_checker.extract.pdf.blocks import RISE_TOLERANCE, DraftBlock
from doc_checker.text.tokenize import TokenKind, tokenize

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Iterator, Sequence

    from doc_checker.extract.pdf.layout import Line

# The fewest lines for which a code sample can have line numbers.
CODE_NUMBERS_MIN_LINES: Final = 2
# The characters that end a line when a word is hyphenated or a compound word
# is cut at its hyphen: the hyphen-minus, the hyphen and the soft hyphen.
HYPHENS: Final = ("-", "\U00002010", "\xad")
# The line number that starts a line of code, then the blanks that follow it,
# unless the line is only the number. It has at most nine digits: a longer run
# is no line number, and Python refuses to parse thousands of digits.
_LINE_NUMBER: Final = re.compile(r"^\s*(\d{1,9})(?:( +)|$)")


def _words(text: str) -> list[str]:
    """Return the words of a text, as the tokenizer finds them."""
    return [
        token.text for token in tokenize(text) if token.kind is TokenKind.WORD
    ]


def is_hyphen_break(line: Line, following: Line) -> bool:
    """Return whether a word is hyphenated across two consecutive lines.

    Args:
        line: A line.
        following: The line after it, in reading order.

    Returns:
        Whether the text of `line` ends with one of the `HYPHENS` that follows
        a letter, and the text of `following` starts with a lowercase letter.
    """
    text = line.text
    return (
        text.endswith(HYPHENS)
        and len(text) > 1
        and text[-2].isalpha()
        and following.text[:1].islower()
    )


def _is_compound_break(line: Line, following: Line) -> bool:
    """Return whether a compound word is cut after its hyphen.

    Unlike `is_hyphen_break`, this accepts a digit before the hyphen, and any
    letter or digit after it, whatever its case: a hyphen between ``x86`` and
    ``64``, or between ``Wi`` and ``Fi``, is part of the word, and the line
    break does not remove it.

    Args:
        line: A line.
        following: The line after it, in reading order.

    Returns:
        Whether the text of `line` ends with one of the `HYPHENS` that follows
        a letter or a digit, and the text of `following` starts with a letter
        or a digit.
    """
    text = line.text
    return (
        text.endswith(HYPHENS)
        and len(text) > 1
        and text[-2].isalnum()
        and following.text[:1].isalnum()
    )


def _line_words(lines: Sequence[Line]) -> Iterator[list[str]]:
    """Yield the words of each line, but the pieces of a split word.

    Args:
        lines: The lines of a block, in reading order.

    Yields:
        The words of each line. Where a word is hyphenated at the end of a
        line, the last word of that line and the first word of the next one
        are the two pieces of it, and are left out.
    """
    breaks = list(starmap(is_hyphen_break, pairwise(lines)))
    # For each line, whether a break comes before it and whether one follows.
    padded = zip(lines, [False, *breaks], [*breaks, False], strict=True)
    for line, after_break, before_break in padded:
        words = _words(line.text)
        if after_break:
            words = words[1:]
        if before_break:
            words = words[:-1]
        yield words


def vocabulary(blocks: Sequence[DraftBlock]) -> frozenset[str]:
    """Collect the words that a document uses.

    The vocabulary tells which words the document uses, so that a word that a
    hyphen splits is put back together only if the whole word is known. The
    words of the code samples count too.

    Args:
        blocks: The blocks of the document, as `merge_flows` returns them.

    Returns:
        The words of all the lines, made with `casefold`, without the pieces
        of the words that a hyphen splits between two consecutive lines of a
        block.
    """
    return frozenset(
        word.casefold()
        for block in blocks
        for words in _line_words(block.lines)
        for word in words
    )


def _is_confirmed(
    line: Line,
    following: Line,
    known: frozenset[str],
) -> bool:
    """Return whether the document has the word that a hyphen break splits.

    Args:
        line: The line that ends with the hyphen.
        following: The line after it.
        known: The words of the document, as `vocabulary` returns them.

    Returns:
        Whether the last word of `line` glued to the first word of `following`
        is one of the known words, in lower case.
    """
    # Slices, not indexes: a line may start with a lowercase symbol that is no
    # word, such as a circled letter.
    pieces = _words(line.text)[-1:] + _words(following.text)[:1]
    return "".join(pieces).casefold() in known


def prose_text(
    lines: Sequence[Line],
    vocabulary: frozenset[str],
    counter: Counter[str],
) -> str:
    """Join the lines of a paragraph, or of any block that is not code.

    The lines are joined with a space, and `counter["line_break"]` goes up by
    one for each. Two lines are joined without a space instead, and
    `counter["hyphenation"]` goes up by one, if a word is hyphenated across
    them (see `is_hyphen_break`), or if a compound word is cut after its
    hyphen: the first line ends with a hyphen that follows a letter or a digit
    and the second one starts with a letter or a digit, as in ``x86-`` and
    ``64``. The hyphen is removed only from a hyphenated word, and only if the
    last word of the first line, glued to the first word of the next line and
    set in lower case, is in the vocabulary. Otherwise it stays, as the word
    may be a compound one: ``x86-`` and ``64`` give ``x86-64``.

    Args:
        lines: The lines of the block, at least one.
        vocabulary: The words of the document, as `vocabulary` returns them.
        counter: The counts of the normalizations.

    Returns:
        The text.
    """
    parts = [lines[0].text]
    for previous, line in pairwise(lines):
        hyphenated = is_hyphen_break(previous, line)
        if hyphenated or _is_compound_break(previous, line):
            counter["hyphenation"] += 1
            if hyphenated and _is_confirmed(previous, line, vocabulary):
                parts[-1] = previous.text[:-1]
        else:
            counter["line_break"] += 1
            parts.append(" ")
        parts.append(line.text)
    return "".join(parts)


def _starts_run(previous: Line, line: Line) -> bool:
    """Return whether a line of code starts a run.

    A run is a stretch of lines that share a left margin: the lines of a code
    sample in one column of one page. Unlike `group_lines`, this does not take
    a line that lies wholly to the right of the previous one for a break: it
    may only be indented far.

    Args:
        previous: The line before it, in reading order.
        line: The line.

    Returns:
        Whether the line is on another page, or its baseline rises above that
        of the previous one by more than `RISE_TOLERANCE` sizes, the size being
        the larger of the two.
    """
    size = max(previous.size, line.size)
    return (
        line.page != previous.page
        or line.baseline > previous.baseline + RISE_TOLERANCE * size
    )


def _indented(lines: Sequence[Line]) -> list[str]:
    """Rebuild the indentation of the lines of a code sample.

    The width of a character is measured on the longest line of the block,
    where the rounding errors weigh least. A line is indented by its distance
    to the left edge of its run, in characters, rounded.

    Args:
        lines: The lines of the block, at least one.

    Returns:
        The text of each line, with its leading spaces. No indentation if the
        longest line has no width.
    """
    longest = max(lines, key=lambda line: len(line.text))
    width = (longest.x1 - longest.x0) / len(longest.text)
    runs = [[lines[0]]]
    for previous, line in pairwise(lines):
        if _starts_run(previous, line):
            runs.append([line])
        else:
            runs[-1].append(line)
    texts = []
    for run in runs:
        margin = min(line.x0 for line in run)
        for line in run:
            indent = round((line.x0 - margin) / width) if width > 0 else 0
            texts.append(" " * indent + line.text)
    return texts


def _without_numbers(texts: Sequence[str]) -> list[str] | None:
    """Remove the line numbers from the lines of a code sample.

    The lines are numbered if there are at least `CODE_NUMBERS_MIN_LINES` of
    them, each starts with a number that is alone or followed by blanks, the
    numbers go up by one from line to line, and at least one line has code
    after its number. A line that is only a number is an empty line of the
    sample.

    Args:
        texts: The text of each line, with its indentation.

    Returns:
        The lines that have code, each without its number and without the
        shortest of the blanks that follow the numbers. The empty lines are
        left out, as blank lines are not rebuilt. None if the lines are not
        numbered.
    """
    matches = [m for text in texts if (m := _LINE_NUMBER.match(text))]
    if len(texts) < CODE_NUMBERS_MIN_LINES or len(matches) < len(texts):
        return None
    numbers = [int(match[1]) for match in matches]
    if any(after - before != 1 for before, after in pairwise(numbers)):
        return None
    coded = [
        (text, match)
        for text, match in zip(texts, matches, strict=True)
        if match.end() < len(text)
    ]
    if not coded:
        return None
    # The shortest blanks are the gap between the numbers and the code; the
    # others are the indentation of the code.
    gap = min(match.end() - match.end(1) for _, match in coded)
    return [text[match.end(1) + gap :] for text, match in coded]


def code_text(lines: Sequence[Line], counter: Counter[str]) -> str:
    """Join the lines of a code sample, one per line of text.

    The indentation is rebuilt from the positions of the lines. The lines form
    runs, which are stretches of lines that share a left margin: a new run
    starts on another page, or where the baseline rises, as in another column.
    A character is as wide as in the longest line of the block, and a line gets
    one leading space for each character that it lies to the right of the
    left edge of its run.

    If the block has at least `CODE_NUMBERS_MIN_LINES` lines, each starts with
    a number, alone or followed by blanks, the numbers go up by one from line
    to line, and at least one line has code after its number, these are the
    line numbers of the sample. Each number is removed, with the shortest of
    the blanks that follow the numbers on the lines that have code. A line
    that is only a number is an empty line of the sample: it is left out, as
    blank lines are not rebuilt. `counter["code_line_number"]` goes up by the
    number of lines, the empty ones included.

    Args:
        lines: The lines of the block, at least one.
        counter: The counts of the normalizations.

    Returns:
        The text, with its lines joined by a line feed.
    """
    texts = _indented(lines)
    unnumbered = _without_numbers(texts)
    if unnumbered is not None:
        counter["code_line_number"] += len(texts)
        texts = unnumbered
    return "\n".join(texts)
