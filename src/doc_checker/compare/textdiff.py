# Copyright 2026 Christophe Le Douarec
"""Word level differences between two texts.

Two long texts of lines, such as long tables or code, are compared line by
line first, as `diff_tokens` tells, so that the work stays linear in their
size.
"""

import bisect
import difflib
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final, Literal, cast

from doc_checker.text.tokenize import Token, TokenKind

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

type HunkOp = Literal["replace", "delete", "insert"]
type _Opcode = tuple[str, int, int, int, int]

# Least characters of a table or a code block, by its key, that is compared
# line by line: in the alignment, with the lines of its counterpart, and in
# the diff of a pair that holds it. The largest tables and code blocks of the
# PGF manual and of the git manual hold 4,592 and 6,500 characters of strict
# key, 2,989 and 3,075 of loose key; below this size, the comparison of two
# such blocks costs a time that grows as the square of their size.
LINES_MIN_CHARS: Final = 10_000
# Most products of the token counts of the two sides of a run of replaced
# lines, its equal ends left out, that is diffed token by token: a longer run
# is one modification. difflib takes about 30 ns a product on the rows of a
# table, 0.03 s a run at this bound, but its time grows faster than the
# product on crafted tokens: 3.0 s a run at this bound was measured, for
# alternating digits against pairs of them. A lower bound would make whole
# modifications of the runs of changed rows of real tables of digits.
RUN_MAX_CELLS: Final = 1_000_000
# Most products of the line counts of the two sides of a gap between lines
# found once on each side that is diffed line by line, at the same cost: a
# longer gap is one run of replaced lines.
GAP_MAX_CELLS: Final = 1_000_000

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
        whole: Whether the run is a run of lines too long to be diffed token
            by token, which is one modification.
    """

    op: HunkOp
    ref_start: int
    ref_end: int
    new_start: int
    new_end: int
    whole: bool = False


def without_letter_hyphens(text: str) -> str:
    """Return a text without the hyphens placed between two letters.

    Args:
        text: Any text.

    Returns:
        The text; a hyphen next to a digit stays.
    """
    return _LETTER_HYPHEN.sub("", text)


def _key(token: Token) -> str:
    """Return the text a token is compared by.

    Args:
        token: Token to compare.

    Returns:
        The text of a word without the hyphens between two letters, else the
        text of the token.
    """
    if token.kind is TokenKind.WORD:
        return without_letter_hyphens(token.text)
    return token.text


def _opcodes(
    ref: Sequence[Hashable],
    new: Sequence[Hashable],
) -> list[_Opcode]:
    """Return the opcodes of `difflib.SequenceMatcher`, without its junk."""
    matcher = difflib.SequenceMatcher(None, ref, new, autojunk=False)
    return matcher.get_opcodes()


def _shifted(opcodes: Sequence[_Opcode], ref: int, new: int) -> list[_Opcode]:
    """Return opcodes moved by offsets on each side."""
    return [
        (tag, i1 + ref, i2 + ref, j1 + new, j2 + new)
        for tag, i1, i2, j1, j2 in opcodes
    ]


def _unique_pairs(
    ref: Sequence[Hashable],
    new: Sequence[Hashable],
) -> list[tuple[int, int]]:
    """Return the longest chain of lines found once on each side, in order.

    Args:
        ref: The lines of the reference.
        new: The lines of the new text.

    Returns:
        The positions of such lines on both sides, both increasing: the
        longest increasing subsequence of them, by patience sorting.
    """
    ref_counts, new_counts = Counter(ref), Counter(new)
    new_positions = {
        line: j for j, line in enumerate(new) if new_counts[line] == 1
    }
    pairs = [
        (i, new_positions[line])
        for i, line in enumerate(ref)
        if ref_counts[line] == 1 and line in new_positions
    ]
    tails: list[int] = []
    tail_values: list[int] = []
    previous = [-1] * len(pairs)
    for index, (_, value) in enumerate(pairs):
        position = bisect.bisect_left(tail_values, value)
        if position:
            previous[index] = tails[position - 1]
        if position == len(tails):
            tails.append(index)
            tail_values.append(value)
        else:
            tails[position] = index
            tail_values[position] = value
    chain = []
    index = tails[-1] if tails else -1
    while index >= 0:
        chain.append(pairs[index])
        index = previous[index]
    return chain[::-1]


def _gap_opcodes(
    ref: Sequence[Hashable],
    new: Sequence[Hashable],
    bounds: tuple[int, int, int, int],
) -> list[_Opcode]:
    """Return the opcodes of the lines of a gap between two common lines.

    The lines equal at both ends of the gap are equal; those between are
    diffed by `difflib.SequenceMatcher` if the product of their counts is
    `GAP_MAX_CELLS` at most, else one run of replaced lines.
    """
    i1, i2, j1, j2 = bounds
    head = 0
    while (
        i1 + head < i2 and j1 + head < j2 and ref[i1 + head] == new[j1 + head]
    ):
        head += 1
    tail = 0
    while (
        i1 + head < i2 - tail
        and j1 + head < j2 - tail
        and ref[i2 - tail - 1] == new[j2 - tail - 1]
    ):
        tail += 1
    opcodes: list[_Opcode] = []
    if head:
        opcodes.append(("equal", i1, i1 + head, j1, j1 + head))
    start, stop = i1 + head, i2 - tail
    first, last = j1 + head, j2 - tail
    if (stop - start) * (last - first) <= GAP_MAX_CELLS:
        opcodes += _shifted(
            _opcodes(ref[start:stop], new[first:last]),
            start,
            first,
        )
    else:
        opcodes.append(("replace", start, stop, first, last))
    if tail:
        opcodes.append(("equal", i2 - tail, i2, j2 - tail, j2))
    return opcodes


def _line_opcodes(
    ref: Sequence[Hashable],
    new: Sequence[Hashable],
) -> list[_Opcode]:
    """Return the opcodes of two texts of lines, in a linear time.

    The lines found once on each side, in the same order, are equal, as in a
    patience diff; each gap between two of them is diffed alone.
    """
    opcodes: list[_Opcode] = []
    i = j = 0
    for line_i, line_j in [*_unique_pairs(ref, new), (len(ref), len(new))]:
        opcodes += _gap_opcodes(ref, new, (i, line_i, j, line_j))
        if line_i < len(ref):
            opcodes.append(("equal", line_i, line_i + 1, line_j, line_j + 1))
        i, j = line_i + 1, line_j + 1
    return [
        opcode
        for opcode in opcodes
        if opcode[1] < opcode[2] or opcode[3] < opcode[4]
    ]


def _line_keys(
    keys: Sequence[str],
    starts: Sequence[int],
) -> list[tuple[str, ...]]:
    """Return the keys of the tokens of each line, from its first token."""
    bounds = [*starts, len(keys)]
    return [tuple(keys[a:b]) for a, b in pairwise(bounds)]


def _by_lines(
    ref_keys: Sequence[str],
    new_keys: Sequence[str],
    lines: tuple[Sequence[int], Sequence[int]],
) -> list[Hunk]:
    """Return the hunks of two token keys of lines, diffed line by line.

    Args:
        ref_keys: The keys of the reference tokens.
        new_keys: The keys of the new tokens.
        lines: The position of the first token of each line, on each side.

    Returns:
        The hunks: a run of deleted or inserted lines is one hunk; the
        tokens that a run of replaced lines starts and ends with on both
        sides are equal, and those between are one hunk if a side has none,
        else are diffed if the product of their counts is `RUN_MAX_CELLS` at
        most, else are one whole hunk.
    """
    ref_bounds = [*lines[0], len(ref_keys)]
    new_bounds = [*lines[1], len(new_keys)]
    hunks: list[Hunk] = []
    for tag, i1, i2, j1, j2 in _line_opcodes(
        _line_keys(ref_keys, lines[0]),
        _line_keys(new_keys, lines[1]),
    ):
        if tag == "equal":
            continue
        start, stop = ref_bounds[i1], ref_bounds[i2]
        first, last = new_bounds[j1], new_bounds[j2]
        if tag == "replace":
            start, stop, first, last = _trimmed(
                ref_keys,
                new_keys,
                (start, stop, first, last),
            )
        if start == stop and first == last:
            continue
        if start == stop or first == last:
            op: HunkOp = "insert" if start == stop else "delete"
            hunks.append(Hunk(op, start, stop, first, last))
        elif (stop - start) * (last - first) <= RUN_MAX_CELLS:
            hunks += [
                Hunk(cast("HunkOp", op), a1, a2, b1, b2)
                for op, a1, a2, b1, b2 in _shifted(
                    _opcodes(ref_keys[start:stop], new_keys[first:last]),
                    start,
                    first,
                )
                if op != "equal"
            ]
        else:
            hunks.append(Hunk("replace", start, stop, first, last, whole=True))
    return hunks


def _trimmed(
    ref: Sequence[Hashable],
    new: Sequence[Hashable],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the bounds of two runs without the items equal at their ends.

    Args:
        ref: The items of the reference.
        new: The items of the new text.
        bounds: The start and the end of the run of each side.

    Returns:
        The bounds of what is left between the items equal at the start of
        both runs and those equal at their end, the start first.
    """
    start, stop, first, last = bounds
    while start < stop and first < last and ref[start] == new[first]:
        start, first = start + 1, first + 1
    while start < stop and first < last and ref[stop - 1] == new[last - 1]:
        stop, last = stop - 1, last - 1
    return start, stop, first, last


def diff_tokens(
    ref_tokens: Sequence[Token],
    new_tokens: Sequence[Token],
    *,
    lines: tuple[Sequence[int], Sequence[int]] | None = None,
) -> list[Hunk]:
    """Return the runs of tokens that differ, in order.

    Words are compared without the hyphens between two letters: a line break
    hyphenation, or ``e-mail`` written ``email``, is not a difference. A hyphen
    next to a digit belongs to a code, such as ``x86-64`` or ``XJ-120-4``, and
    is compared. Other tokens are compared by their text.

    Two texts of lines are compared line by line first: the lines found once
    on each side, in the same order, anchor the comparison, as in a patience
    diff, and the lines between two anchors are diffed by
    `difflib.SequenceMatcher`, or replaced as a whole past `GAP_MAX_CELLS`.
    The tokens of each run of replaced lines are then diffed, or the run is
    one whole hunk past `RUN_MAX_CELLS`.

    Args:
        ref_tokens: Tokens of the reference text.
        new_tokens: Tokens of the new text.
        lines: For texts compared line by line, the position of the first
            token of each line of each side, from 0 and increasing.

    Returns:
        One hunk per difference, never merged with its neighbors. Its indices
        are positions in the given tokens.
    """
    ref_keys = [_key(token) for token in ref_tokens]
    new_keys = [_key(token) for token in new_tokens]
    if lines is not None:
        return _by_lines(ref_keys, new_keys, lines)
    return [
        Hunk(cast("HunkOp", tag), i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in _opcodes(ref_keys, new_keys)
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
