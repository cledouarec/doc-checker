# Copyright 2026 Christophe Le Douarec
"""Global, deterministic alignment of the blocks of two documents.

1. Blocks whose loose key is unique on both sides become anchors.
2. The longest increasing subsequence of anchors gives the common order;
   the other anchors are moved blocks.
3. The gaps between consecutive anchors are aligned by dynamic
   programming, after looking for split and merged blocks.
4. Blocks still unmatched are paired as moved blocks when similar enough.

Two blocks are similar as `fuzz.ratio` finds their loose keys, but two long
blocks of lines, as `_by_lines` tells, as their lines are, in a time linear
in their size: `_line_similarity`.
"""

import bisect
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from rapidfuzz import fuzz, process

from doc_checker.compare.textdiff import LINES_MIN_CHARS
from doc_checker.model.document import Block, BlockKind
from doc_checker.text.normalize import loose_key, strict_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

MIN_ANCHOR_WORDS: Final = 4
MAX_DP_CELLS: Final = 40_000
GREEDY_WINDOW: Final = 50
SPLIT_MERGE_RATIO: Final = 0.9
MAX_SPLIT_PARTS: Final = 6
MAX_STRUCTURE_PARTS: Final = 100
# The work an alignment may spend, in units of `_Budget.take`: a base plus
# an amount per character of the loose keys of both documents.
ALIGN_WORK_BASE: Final = 10_000_000
ALIGN_WORK_PER_CHAR: Final = 1_000

_PERCENT: Final = 100
# The characters `fuzz.ratio` compares in one unit of work, per query block.
_CHARS_PER_WORD: Final = 64
# Units of work, at about 2 ns each, of a call of `fuzz.ratio` on two empty
# texts, and of one more choice of `process.extractOne`.
_CALL_COST: Final = 200
_CHOICE_COST: Final = 50
# Units of work per character, at about 2 ns each, of reading the lines of a
# block, once, and of comparing the lines of two blocks: about 180 ns and 12 ns
# a character were measured.
_LINES_READ_COST: Final = 100
_LINES_MATCH_COST: Final = 8
# The score of `fuzz.ratio` that a split or a merge must reach.
_SPLIT_SCORE: Final = SPLIT_MERGE_RATIO * _PERCENT
# The kinds whose blocks may split into many parts: a table into one block
# per cell or row, code into one block per line.
_STRUCTURES: Final = frozenset({BlockKind.TABLE, BlockKind.CODE})
_GROUPS: Final = {
    BlockKind.HEADING: "h",
    BlockKind.CODE: "c",
    BlockKind.TABLE: "t",
}
_OTHER_GROUP: Final = "x"


class PairKind(StrEnum):
    """How blocks of the two documents correspond."""

    EQUAL = "equal"
    MODIFIED = "modified"
    SPLIT = "split"
    MERGE = "merge"
    DELETED = "deleted"
    INSERTED = "inserted"


@dataclass(frozen=True, slots=True)
class Pair:
    """Blocks of the reference matched with blocks of the new document.

    Attributes:
        kind: Kind of correspondence.
        ref: Indices of the reference blocks.
        new: Indices of the new blocks.
        moved: Whether the blocks are out of the common order.
    """

    kind: PairKind
    ref: tuple[int, ...]
    new: tuple[int, ...]
    moved: bool = False


@dataclass(slots=True)
class _Budget:
    """The work left to an alignment, shared by all its fuzzy comparisons.

    A comparison of a query with some choices costs about
    ``(len(query) // 64 + 1)`` units per character of the choices, plus one
    per choice. Once a comparison does not fit, the budget is exhausted for
    good and every later block is only paired with an identical one.

    Attributes:
        limit: Units the alignment may spend.
        spent: Units spent so far.
        exhausted: Whether a comparison did not fit.
        skipped: The blocks, as (side, index), that were not fully searched.
        lines: The lines of the blocks read so far, as `_lines_of` gives
            them, by the identity of the block.
    """

    limit: int
    spent: int = 0
    exhausted: bool = False
    skipped: set[tuple[str, int]] = field(default_factory=set)
    lines: dict[int, Counter[str]] = field(default_factory=dict)

    def take(self, cost: int) -> bool:
        """Charge a cost and return whether it fits.

        Args:
            cost: Units of work of a comparison.

        Returns:
            Whether the comparison may be made.
        """
        if self.exhausted or self.spent + cost > self.limit:
            self.exhausted = True
            return False
        self.spent += cost
        return True

    def search(self, query: int, chars: int, count: int) -> bool:
        """Charge a query compared with choices by ``process.extractOne``.

        The pattern of the query is built once, then scanned against every
        choice.

        Args:
            query: Length of the query text.
            chars: Total length of the choices.
            count: Number of choices.

        Returns:
            Whether the comparison may be made.
        """
        return self.take(_search_cost(query, chars, count))


def _search_cost(query: int, chars: int, count: int) -> int:
    """Return the work of ``process.extractOne``, as `_Budget.search` tells."""
    cost = (query // _CHARS_PER_WORD + 1) * chars + count * _CHOICE_COST
    return cost + query + _CALL_COST


def _call_cost(first: int, second: int) -> int:
    """Return the work of ``fuzz.ratio`` on two texts of given lengths.

    A call builds the pattern of the longer text, then scans the shorter.
    """
    longest, shortest = max(first, second), min(first, second)
    return (longest // _CHARS_PER_WORD + 1) * shortest + longest + _CALL_COST


@dataclass(frozen=True, slots=True)
class Alignment:
    """The alignment of two block lists.

    Attributes:
        pairs: The pairs, in reference order.
        unsearched: Blocks that were compared by identical text only, the
            work limit being reached.
    """

    pairs: list[Pair]
    unsearched: int = 0


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A split or merge found inside a gap, in gap positions."""

    score: float
    ref_start: int
    ref_end: int
    new_start: int
    new_end: int
    pair: Pair


class _Exact:
    """The columns of a gap by loose key, to pair identical blocks in order.

    Attributes:
        positions: For each loose key, the positions of the columns that
            have it, in order.
    """

    def __init__(self, new: Sequence[Block], columns: Sequence[int]) -> None:
        self.positions: dict[str, list[int]] = {}
        for position, j in enumerate(columns):
            self.positions.setdefault(new[j].key_loose, []).append(position)

    def find(self, key: str, start: int, stop: int) -> int | None:
        """Return the first position in ``[start, stop)`` of a key."""
        positions = self.positions.get(key, [])
        index = bisect.bisect_left(positions, start)
        if index < len(positions) and positions[index] < stop:
            return positions[index]
        return None


def lines_compared(sizes: tuple[int, int], *, structure: bool) -> bool:
    """Return whether two texts are compared by their lines.

    The alignment and the diff of a pair ask the same question.

    Args:
        sizes: The characters of the loose key of each text, its blocks
            joined by blanks.
        structure: Whether one of them holds a table or a code block.

    Returns:
        Whether both have `LINES_MIN_CHARS` characters at least, and one
        holds a table or a code block.
    """
    return structure and min(sizes) >= LINES_MIN_CHARS


def pair_by_lines(ref: Sequence[Block], new: Sequence[Block]) -> bool:
    """Return whether the blocks of two sides are compared by their lines.

    Args:
        ref: The blocks of one side, in order.
        new: The blocks of the other side, in order.

    Returns:
        What `lines_compared` tells of their loose keys, joined by blanks.
    """
    return lines_compared(
        (_joined_size(ref), _joined_size(new)),
        structure=any(block.kind in _STRUCTURES for block in (*ref, *new)),
    )


def _joined_size(blocks: Sequence[Block]) -> int:
    """Return the length of the loose keys of blocks joined by blanks."""
    return sum(len(block.key_loose) for block in blocks) + max(
        len(blocks) - 1,
        0,
    )


def _by_lines(first: Block, second: Block) -> bool:
    """Return whether two blocks are compared by their lines.

    They are when both have `LINES_MIN_CHARS` characters of loose key at
    least, and one is a table or a code block, as `pair_by_lines` tells.
    """
    return pair_by_lines((first,), (second,))


def _lines_of(block: Block) -> Counter[str]:
    """Return the loose keys of the lines of a block, as a multiset.

    Args:
        block: Any block.

    Returns:
        The non-empty loose key of each line of a code block, or of each row
        of a table, when their keys joined by blanks are the loose key of the
        block; else its loose key alone, as for another block.
    """
    whole = Counter([block.key_loose] if block.key_loose else [])
    if block.kind is BlockKind.CODE:
        lined = block.key_strict
    elif block.kind is BlockKind.TABLE:
        lined = strict_key(block.text, Counter(), lines=True)
    else:
        return whole
    lines = [key for line in lined.split("\n") if (key := loose_key(line))]
    return Counter(lines) if " ".join(lines) == block.key_loose else whole


def _weight(lines: Mapping[str, int]) -> int:
    """Return the characters of lines, each with the blank that ends it."""
    return sum((len(line) + 1) * count for line, count in lines.items())


def _line_similarity(first: Counter[str], second: Counter[str]) -> float:
    """Return the similarity of two multisets of lines, in [0, 1].

    The characters of the lines found on both sides, each with a blank,
    count as found; the words of the other lines are found as a multiset in
    the same way, so that a line changed by a word still counts for its
    other words. The similarity is twice the characters found over all the
    characters of both sides, as `fuzz.ratio` measures it.

    Args:
        first: The lines of a text, as `_lines_of` gives them.
        second: The lines of another text.

    Returns:
        The similarity: 1 for two empty texts.
    """
    size = _weight(first) + _weight(second)
    if not size:
        return 1.0
    found = 0
    first_words: Counter[str] = Counter()
    second_words: Counter[str] = Counter()
    for line, count in first.items():
        other = second.get(line, 0)
        found += min(count, other) * (len(line) + 1)
        for word in line.split() if count > other else ():
            first_words[word] += count - other
    for line, count in second.items():
        other = first.get(line, 0)
        for word in line.split() if count > other else ():
            second_words[word] += count - other
    found += _weight(first_words & second_words)
    return 2 * found / size


def _read_cost(budget: _Budget, blocks: Sequence[Block]) -> int:
    """Return the work of reading the lines of the blocks not read yet."""
    unread = {
        id(block): block for block in blocks if id(block) not in budget.lines
    }
    return _LINES_READ_COST * sum(
        len(block.key_loose) for block in unread.values()
    )


def _match_cost(first: int, second: int) -> int:
    """Return the work of comparing the lines of two texts of given lengths."""
    return _LINES_MATCH_COST * (first + second) + _CALL_COST


def _block_lines(budget: _Budget, block: Block) -> Counter[str]:
    """Return the lines of a block, as `_lines_of` gives them, read once."""
    if id(block) not in budget.lines:
        budget.lines[id(block)] = _lines_of(block)
    return budget.lines[id(block)]


def _run_lines(budget: _Budget, blocks: Sequence[Block]) -> Counter[str]:
    """Return the lines of a run of blocks together."""
    found: Counter[str] = Counter()
    for block in blocks:
        found.update(_block_lines(budget, block))
    return found


def _pair_cost(first: Block, second: Block) -> int:
    """Return the work of `_similarity` on two blocks, their lines read."""
    if _by_lines(first, second):
        return _match_cost(len(first.key_loose), len(second.key_loose))
    return _call_cost(len(first.key_loose), len(second.key_loose))


def _similarity(first: Block, second: Block, budget: _Budget) -> float:
    """Return the similarity of two blocks, in [0, 1].

    It is that of their lines if `_by_lines` tells so, as
    `_line_similarity` measures it, else that of their loose keys, by
    `fuzz.ratio`. Its work, `_pair_cost`, and the reading of the lines,
    `_read_cost`, are charged beforehand.
    """
    if _by_lines(first, second):
        return _line_similarity(
            _block_lines(budget, first),
            _block_lines(budget, second),
        )
    return fuzz.ratio(first.key_loose, second.key_loose) / _PERCENT


def _searched_by_lines(query: Block, choices: Iterable[Block]) -> bool:
    """Return whether a query is compared by lines with some choice."""
    return len(query.key_loose) >= LINES_MIN_CHARS and any(
        _by_lines(query, choice) for choice in choices
    )


def _choices_cost(
    query: Block,
    choices: Mapping[int, Block],
    budget: _Budget,
) -> int:
    """Return the work of `_best_choice`, the reading of lines included."""
    lined = [block for block in choices.values() if _by_lines(query, block)]
    others = [
        len(block.key_loose)
        for block in choices.values()
        if not _by_lines(query, block)
    ]
    size = len(query.key_loose)
    cost = sum(_match_cost(size, len(block.key_loose)) for block in lined)
    if lined:
        cost += _read_cost(budget, [query, *lined])
    if others:
        cost += _search_cost(size, sum(others), len(others))
    return cost


def _best_choice(
    query: Block,
    choices: Mapping[int, Block],
    match: float,
    budget: _Budget,
) -> int | None:
    """Return the choice most similar to a query, as `_similarity` finds.

    The choices compared by lines are scored one by one, the others by
    ``process.extractOne``. The work, `_choices_cost`, is charged
    beforehand.

    Args:
        query: The block searched for.
        choices: The blocks searched, by their position, in order.
        match: Least similarity of the choice.
        budget: The work left, which holds the lines read.

    Returns:
        The position of the first choice of the highest similarity, if it is
        `match` at least.
    """
    order = {position: rank for rank, position in enumerate(choices)}
    others = {
        position: block.key_loose
        for position, block in choices.items()
        if not _by_lines(query, block)
    }
    best: tuple[float, int] | None = None
    found = (
        process.extractOne(
            query.key_loose,
            others,
            scorer=fuzz.ratio,
            score_cutoff=match * _PERCENT,
        )
        if others
        else None
    )
    if found is not None:
        best = (found[1] / _PERCENT, found[2])
    for position, block in choices.items():
        if position in others:
            continue
        score = _similarity(query, block, budget)
        if score >= match and (
            best is None
            or (score, -order[position]) > (best[0], -order[best[1]])
        ):
            best = (score, position)
    return None if best is None else best[1]


def _matched(
    ref: Sequence[Block],
    new: Sequence[Block],
    pair: tuple[int, int],
    *,
    moved: bool = False,
) -> Pair:
    """Return the pair of two matched blocks, equal or modified."""
    i, j = pair
    same = ref[i].key_strict == new[j].key_strict
    kind = PairKind.EQUAL if same else PairKind.MODIFIED
    return Pair(kind, (i,), (j,), moved=moved)


def _anchor_positions(blocks: Sequence[Block]) -> dict[str, int]:
    """Return the position of each anchor key found once among blocks."""
    keys = [
        f"{_GROUPS.get(block.kind, _OTHER_GROUP)}:{block.key_loose}"
        if block.kind is BlockKind.HEADING
        or len(block.key_loose.split()) >= MIN_ANCHOR_WORDS
        else None
        for block in blocks
    ]
    counts = Counter(key for key in keys if key is not None)
    return {
        key: index
        for index, key in enumerate(keys)
        if key is not None and counts[key] == 1
    }


def _longest_increasing(
    pairs: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the longest subsequence of pairs whose second item increases.

    Args:
        pairs: Pairs sorted by their first item.

    Returns:
        The longest increasing subsequence, found by patience sorting.
    """
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


def _anchors(
    ref: Sequence[Block],
    new: Sequence[Block],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the anchors in common order and the moved anchors."""
    ref_positions = _anchor_positions(ref)
    new_positions = _anchor_positions(new)
    anchors = sorted(
        (index, new_positions[key])
        for key, index in ref_positions.items()
        if key in new_positions
    )
    chain = _longest_increasing(anchors)
    in_chain = set(chain)
    return chain, [anchor for anchor in anchors if anchor not in in_chain]


def _unmatched(rows: Sequence[int], columns: Sequence[int]) -> list[Pair]:
    """Return the pairs of blocks found on one side only."""
    return [Pair(PairKind.DELETED, (i,), ()) for i in rows] + [
        Pair(PairKind.INSERTED, (), (j,)) for j in columns
    ]


def _greedy(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - the two documents, a gap of them and the shared budget
    ref: Sequence[Block],
    new: Sequence[Block],
    rows: Sequence[int],
    columns: Sequence[int],
    match: float,
    budget: _Budget,
) -> list[Pair]:
    """Return the pairs of a large gap, aligned greedily in a window.

    Once the budget is exhausted, a block is only paired with an identical
    one of the window.
    """
    pairs = []
    pointer = 0
    used: set[int] = set()
    exact = _Exact(new, columns)
    for i in rows:
        stop = min(pointer + GREEDY_WINDOW, len(columns))
        chars = sum(
            len(new[columns[k]].key_loose) for k in range(pointer, stop)
        )
        blocks = {k: new[columns[k]] for k in range(pointer, stop)}
        lined = _searched_by_lines(ref[i], blocks.values())
        if lined and budget.take(_choices_cost(ref[i], blocks, budget)):
            position = _best_choice(ref[i], blocks, match, budget)
        elif not lined and budget.search(
            len(ref[i].key_loose),
            chars,
            stop - pointer,
        ):
            window = {
                position: block.key_loose for position, block in blocks.items()
            }
            found = process.extractOne(
                ref[i].key_loose,
                window,
                scorer=fuzz.ratio,
                score_cutoff=match * _PERCENT,
            )
            position = None if found is None else found[2]
        else:
            if stop > pointer:
                budget.skipped.add(("ref", i))
            position = exact.find(ref[i].key_loose, pointer, stop)
        if position is None:
            pairs.append(Pair(PairKind.DELETED, (i,), ()))
            continue
        pairs.append(_matched(ref, new, (i, columns[position])))
        used.add(columns[position])
        pointer = position + 1
    pairs += [
        Pair(PairKind.INSERTED, (), (j,)) for j in columns if j not in used
    ]
    return pairs


def _dynamic(
    rows: Sequence[int],
    columns: Sequence[int],
    score: Callable[[int, int], float],
    match: float,
    pair_of: Callable[[int, int], Pair],
) -> list[Pair]:
    """Align a gap by dynamic programming, keeping the order.

    The alignment maximizes the sum of the similarities of the matched
    blocks, a match requiring a similarity of at least ``match``.

    Args:
        rows: Reference block indices of the gap.
        columns: New block indices of the gap.
        score: Similarity of the blocks at two positions of the gap.
        match: Minimum similarity of a match.
        pair_of: Builds the pair of two matched block indices.

    Returns:
        The pairs in order.
    """
    height, width = len(rows), len(columns)
    best = [[0.0] * (width + 1) for _ in range(height + 1)]
    for a, b in itertools.product(range(1, height + 1), range(1, width + 1)):
        value = max(best[a - 1][b], best[a][b - 1])
        similarity = score(a - 1, b - 1)
        if similarity >= match:
            value = max(value, best[a - 1][b - 1] + similarity)
        best[a][b] = value
    pairs: list[Pair] = []
    a, b = height, width
    while a and b:
        similarity = score(a - 1, b - 1)
        if (
            similarity >= match
            and best[a][b] == best[a - 1][b - 1] + similarity
        ):
            pairs.append(pair_of(rows[a - 1], columns[b - 1]))
            a, b = a - 1, b - 1
        elif best[a][b] == best[a - 1][b]:
            pairs.append(Pair(PairKind.DELETED, (rows[a - 1],), ()))
            a -= 1
        else:
            pairs.append(Pair(PairKind.INSERTED, (), (columns[b - 1],)))
            b -= 1
    pairs += [Pair(PairKind.DELETED, (rows[k],), ()) for k in range(a)][::-1]
    pairs += [Pair(PairKind.INSERTED, (), (columns[k],)) for k in range(b)][
        ::-1
    ]
    return pairs[::-1]


def _opens(first: Block, single: Block) -> bool:
    """Return whether a block reads like the start of another block.

    Args:
        first: The block that may open a run.
        single: The block that may have been split or merged.

    Returns:
        Whether the loose key of ``first`` scores `SPLIT_MERGE_RATIO`
        against as many first characters of the loose key of ``single``.
    """
    key = first.key_loose
    start = single.key_loose[: len(key)]
    score = fuzz.ratio(key, start, score_cutoff=_SPLIT_SCORE) / _PERCENT
    return score >= SPLIT_MERGE_RATIO


def _best_run(
    single: Block,
    runs: Sequence[Block],
    budget: _Budget,
    who: tuple[str, int],
) -> tuple[float, int, int] | None:
    """Return the best contiguous run of blocks equal to one block.

    Every run of 2 to `MAX_SPLIT_PARTS` blocks is tried. A table or a code
    block is also looked for in longer runs, of up to `MAX_STRUCTURE_PARTS`
    blocks, but only from a block that opens like it (`_opens`): a table
    split into a block per cell or row, or code split into a block per line,
    starts with the start of the table or of the code.

    A run is only scored if the length of its joined keys lets
    ``fuzz.ratio`` reach `SPLIT_MERGE_RATIO`: two texts of lengths ``L1``
    and ``L2`` score at most ``2 * min(L1, L2) / (L1 + L2)``. A run stops
    growing once it is too long.

    Args:
        single: The block that may have been split or merged.
        runs: Candidate blocks of the other side, in order.
        budget: The work left; the search ends when it is exhausted.
        who: The side and index of ``single``, counted as skipped then.

    Returns:
        The score, start and end of the best run, or ``None``.
    """
    structure = single.kind in _STRUCTURES
    size = len(single.key_loose)
    shortest = size * SPLIT_MERGE_RATIO / (2 - SPLIT_MERGE_RATIO)
    longest = size * (2 - SPLIT_MERGE_RATIO) / SPLIT_MERGE_RATIO
    best: tuple[float, int, int] | None = None
    for start in range(len(runs)):
        parts = MAX_SPLIT_PARTS
        prefix = min(len(runs[start].key_loose), size)
        # A part as long as `LINES_MIN_CHARS` opens no long run: it is
        # compared by its lines.
        if structure and prefix < LINES_MIN_CHARS:
            if not budget.take(_call_cost(prefix, prefix)):
                budget.skipped.add(who)
                return best
            if _opens(runs[start], single):
                parts = MAX_STRUCTURE_PARTS
        # The length of the keys of the run joined by blanks.
        length = len(runs[start].key_loose)
        lined = structure or runs[start].kind in _STRUCTURES
        for end in range(start + 2, min(start + parts, len(runs)) + 1):
            length += 1 + len(runs[end - 1].key_loose)
            lined = lined or runs[end - 1].kind in _STRUCTURES
            if length > longest:
                break
            if length < shortest:
                continue
            ratio = _run_similarity(
                single,
                runs[start:end],
                (length, lined),
                budget,
            )
            if ratio is None:
                budget.skipped.add(who)
                return best
            if ratio >= SPLIT_MERGE_RATIO and (
                best is None or ratio > best[0]
            ):
                best = (ratio, start, end)
    return best


def _run_similarity(
    single: Block,
    run: Sequence[Block],
    shape: tuple[int, bool],
    budget: _Budget,
) -> float | None:
    """Return the similarity of a block and a run of blocks, its work charged.

    Args:
        single: The block that may have been split or merged.
        run: The blocks of the other side that may be its parts.
        shape: The length of the keys of the run joined by blanks, and
            whether the block or the run holds a table or a code block.
        budget: The work left.

    Returns:
        The similarity of their lines, as `_line_similarity` measures it, if
        both are `LINES_MIN_CHARS` long and one holds a table or code; else
        that of the loose keys, by ``fuzz.ratio``, 0 below
        `SPLIT_MERGE_RATIO`; ``None`` if its work does not fit the budget.
    """
    length, lined = shape
    size = len(single.key_loose)
    if lines_compared((size, length), structure=lined):
        cost = _match_cost(size, length) + _read_cost(budget, [single, *run])
        if not budget.take(cost):
            return None
        return _line_similarity(
            _block_lines(budget, single),
            _run_lines(budget, run),
        )
    if not budget.take(_call_cost(size, length)):
        return None
    joined = " ".join(block.key_loose for block in run)
    return (
        fuzz.ratio(single.key_loose, joined, score_cutoff=_SPLIT_SCORE)
        / _PERCENT
    )


def _candidates(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - the two documents, a gap of them and the shared budget
    ref: Sequence[Block],
    new: Sequence[Block],
    rows: Sequence[int],
    columns: Sequence[int],
    scores: Sequence[Sequence[float]],
    budget: _Budget,
) -> list[_Candidate]:
    """Return the split and merge candidates of a gap."""
    found = []
    new_blocks = [new[j] for j in columns]
    for a, i in enumerate(rows):
        if max(scores[a]) >= SPLIT_MERGE_RATIO:
            continue
        best = _best_run(ref[i], new_blocks, budget, ("ref", i))
        if best is not None:
            ratio, start, end = best
            pair = Pair(PairKind.SPLIT, (i,), tuple(columns[start:end]))
            found.append(_Candidate(ratio, a, a + 1, start, end, pair))
    ref_blocks = [ref[i] for i in rows]
    for b, j in enumerate(columns):
        if max(row[b] for row in scores) >= SPLIT_MERGE_RATIO:
            continue
        best = _best_run(new[j], ref_blocks, budget, ("new", j))
        if best is not None:
            ratio, start, end = best
            pair = Pair(PairKind.MERGE, tuple(rows[start:end]), (j,))
            found.append(_Candidate(ratio, start, end, b, b + 1, pair))
    return found


def _compatible(first: _Candidate, second: _Candidate) -> bool:
    """Return whether two candidates neither overlap nor cross."""
    before = (
        first.ref_end <= second.ref_start and first.new_end <= second.new_start
    )
    after = (
        second.ref_end <= first.ref_start and second.new_end <= first.new_start
    )
    return before or after


def _select(candidates: list[_Candidate]) -> list[_Candidate]:
    """Return the best compatible candidates, in reference order."""
    ranked = sorted(
        candidates,
        key=lambda c: (-c.score, c.ref_start, c.new_start),
    )
    kept: list[_Candidate] = []
    for candidate in ranked:
        if all(_compatible(candidate, other) for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda c: c.ref_start)


def _align_gap(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - the two documents, a gap of them and the shared budget
    ref: Sequence[Block],
    new: Sequence[Block],
    rows: Sequence[int],
    columns: Sequence[int],
    match: float,
    budget: _Budget,
) -> list[Pair]:
    """Return the pairs of the blocks of a gap between two anchors."""
    if not rows or not columns:
        return _unmatched(rows, columns)
    if len(rows) * len(columns) > MAX_DP_CELLS:
        return _greedy(ref, new, rows, columns, match, budget)
    lined = {
        id(block): block
        for i in rows
        for j in columns
        if _by_lines(ref[i], new[j])
        for block in (ref[i], new[j])
    }
    cost = sum(
        _pair_cost(ref[i], new[j]) for i in rows for j in columns
    ) + _read_cost(budget, list(lined.values()))
    if not budget.take(cost):
        # No similarity matrix: align the identical blocks, in order.
        budget.skipped.update(("ref", i) for i in rows)
        return _dynamic(
            rows,
            columns,
            lambda a, b: float(
                ref[rows[a]].key_loose == new[columns[b]].key_loose,
            ),
            1.0,
            lambda i, j: _matched(ref, new, (i, j)),
        )
    scores = [
        [_similarity(ref[i], new[j], budget) for j in columns] for i in rows
    ]

    def pair_of(i: int, j: int) -> Pair:
        return _matched(ref, new, (i, j))

    pairs: list[Pair] = []
    row_start = column_start = 0
    for candidate in _select(
        _candidates(ref, new, rows, columns, scores, budget),
    ):
        pairs += _dynamic(
            rows[row_start : candidate.ref_start],
            columns[column_start : candidate.new_start],
            lambda a, b, r=row_start, c=column_start: scores[a + r][b + c],
            match,
            pair_of,
        )
        pairs.append(candidate.pair)
        row_start, column_start = candidate.ref_end, candidate.new_end
    pairs += _dynamic(
        rows[row_start:],
        columns[column_start:],
        lambda a, b: scores[a + row_start][b + column_start],
        match,
        pair_of,
    )
    return pairs


def _match_leftovers(
    ref: Sequence[Block],
    new: Sequence[Block],
    pairs: list[Pair],
    moved_match: float,
    budget: _Budget,
) -> list[Pair]:
    """Return the pairs with unmatched blocks paired as moved blocks.

    Each search is charged to the budget. Once it does not fit, the missing
    blocks left are only paired with the first added block left that has the
    same loose key.

    Args:
        ref: Reference blocks.
        new: New blocks.
        pairs: The pairs found so far.
        moved_match: Minimum similarity to pair a moved block.
        budget: The work left.

    Returns:
        The pairs.
    """
    inserted = {
        pair.new[0]: new[pair.new[0]].key_loose
        for pair in pairs
        if pair.kind is PairKind.INSERTED
    }
    # The added blocks that a long block may be compared with by lines.
    long_inserted = {
        j for j, key in inserted.items() if len(key) >= LINES_MIN_CHARS
    }
    chars = sum(len(key) for key in inserted.values())
    deleted = sorted(
        pair.ref[0] for pair in pairs if pair.kind is PairKind.DELETED
    )
    result = [
        pair
        for pair in pairs
        if pair.kind not in {PairKind.DELETED, PairKind.INSERTED}
    ]
    by_key: dict[str, deque[int]] | None = None
    for i in deleted:
        j = None
        if inserted and by_key is None:
            lined = _searched_by_lines(
                ref[i],
                (new[k] for k in long_inserted),
            )
            blocks = {k: new[k] for k in inserted} if lined else {}
            if lined and budget.take(_choices_cost(ref[i], blocks, budget)):
                j = _best_choice(ref[i], blocks, moved_match, budget)
            elif not lined and budget.search(
                len(ref[i].key_loose),
                chars,
                len(inserted),
            ):
                found = process.extractOne(
                    ref[i].key_loose,
                    inserted,
                    scorer=fuzz.ratio,
                    score_cutoff=moved_match * _PERCENT,
                )
                j = None if found is None else found[2]
            else:
                by_key = {}
                for position, key in inserted.items():
                    by_key.setdefault(key, deque()).append(position)
        if inserted and by_key is not None:
            budget.skipped.add(("ref", i))
            positions = by_key.get(ref[i].key_loose, deque())
            while positions and positions[0] not in inserted:
                positions.popleft()
            j = positions[0] if positions else None
        if j is None:
            result.append(Pair(PairKind.DELETED, (i,), ()))
            continue
        chars -= len(inserted.pop(j))
        long_inserted.discard(j)
        result.append(_matched(ref, new, (i, j), moved=True))
    result += [Pair(PairKind.INSERTED, (), (j,)) for j in sorted(inserted)]
    return result


def _in_reference_order(pairs: list[Pair]) -> list[Pair]:
    """Return pairs in reference order, insertions after predecessors."""
    matched_new = sorted(j for pair in pairs if pair.ref for j in pair.new)
    reference_of = {
        j: pair.ref[0] for pair in pairs if pair.ref for j in pair.new
    }

    def key(pair: Pair) -> tuple[int, int, int]:
        if pair.ref:
            return (pair.ref[0], 0, pair.new[0] if pair.new else -1)
        position = bisect.bisect_left(matched_new, pair.new[0])
        before = reference_of[matched_new[position - 1]] if position else -1
        return (before, 1, pair.new[0])

    return sorted(pairs, key=key)


def align_blocks(
    ref: Sequence[Block],
    new: Sequence[Block],
    *,
    match: float,
    moved_match: float,
    work: int | None = None,
) -> Alignment:
    """Align the blocks of a reference document with its new version.

    Args:
        ref: Reference blocks, in reading order.
        new: New blocks, in reading order.
        match: Minimum similarity to pair blocks in place.
        moved_match: Minimum similarity to pair a moved block.
        work: Units of work the alignment may spend; by default
            `ALIGN_WORK_BASE` plus `ALIGN_WORK_PER_CHAR` per character of the
            loose keys.

    Returns:
        The pairs, in reference order, and the number of blocks compared by
        identical text only because of the work limit.
    """
    if work is None:
        size = sum(len(block.key_loose) for block in (*ref, *new))
        work = ALIGN_WORK_BASE + ALIGN_WORK_PER_CHAR * size
    budget = _Budget(work)
    chain, moved = _anchors(ref, new)
    pairs = [_matched(ref, new, anchor, moved=True) for anchor in moved]
    used_ref = {i for i, _ in moved}
    used_new = {j for _, j in moved}
    bounds = [(-1, -1), *chain, (len(ref), len(new))]
    for (i0, j0), (i1, j1) in itertools.pairwise(bounds):
        rows = [i for i in range(i0 + 1, i1) if i not in used_ref]
        columns = [j for j in range(j0 + 1, j1) if j not in used_new]
        pairs += _align_gap(ref, new, rows, columns, match, budget)
        if i1 < len(ref):
            pairs.append(_matched(ref, new, (i1, j1)))
    pairs = _match_leftovers(ref, new, pairs, moved_match, budget)
    return Alignment(_in_reference_order(pairs), len(budget.skipped))


def align(
    ref: Sequence[Block],
    new: Sequence[Block],
    *,
    match: float,
    moved_match: float,
) -> list[Pair]:
    """Return the pairs of `align_blocks`, in reference order."""
    return align_blocks(
        ref,
        new,
        match=match,
        moved_match=moved_match,
    ).pairs
