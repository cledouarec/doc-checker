# Copyright 2026 Christophe Le Douarec
"""Global, deterministic alignment of the blocks of two documents.

1. Blocks whose loose key is unique on both sides become anchors.
2. The longest increasing subsequence of anchors gives the common order;
   the other anchors are moved blocks.
3. The gaps between consecutive anchors are aligned by dynamic
   programming, after looking for split and merged blocks.
4. Blocks still unmatched are paired as moved blocks when similar enough.
"""

import bisect
import itertools
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from rapidfuzz import fuzz, process

from doc_checker.model.document import Block, BlockKind

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MIN_ANCHOR_WORDS: Final = 4
MAX_DP_CELLS: Final = 40_000
GREEDY_WINDOW: Final = 50
SPLIT_MERGE_RATIO: Final = 0.9
MAX_SPLIT_PARTS: Final = 6

_PERCENT: Final = 100
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


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A split or merge found inside a gap, in gap positions."""

    score: float
    ref_start: int
    ref_end: int
    new_start: int
    new_end: int
    pair: Pair


def _similarity(first: Block, second: Block) -> float:
    """Return the similarity of the loose keys of two blocks, in [0, 1]."""
    return fuzz.ratio(first.key_loose, second.key_loose) / _PERCENT


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


def _greedy(
    ref: Sequence[Block],
    new: Sequence[Block],
    rows: Sequence[int],
    columns: Sequence[int],
    match: float,
) -> list[Pair]:
    """Return the pairs of a large gap, aligned greedily in a window."""
    pairs = []
    pointer = 0
    used: set[int] = set()
    for i in rows:
        window = {
            position: new[columns[position]].key_loose
            for position in range(
                pointer,
                min(pointer + GREEDY_WINDOW, len(columns)),
            )
        }
        found = process.extractOne(
            ref[i].key_loose,
            window,
            scorer=fuzz.ratio,
            score_cutoff=match * _PERCENT,
        )
        if found is None:
            pairs.append(Pair(PairKind.DELETED, (i,), ()))
            continue
        position = found[2]
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


def _best_run(
    single: Block,
    runs: Sequence[Block],
) -> tuple[float, int, int] | None:
    """Return the best contiguous run of blocks equal to one block.

    Args:
        single: The block that may have been split or merged.
        runs: Candidate blocks of the other side, in order.

    Returns:
        The score, start and end of the best run, or ``None``.
    """
    best: tuple[float, int, int] | None = None
    for start in range(len(runs)):
        for length in range(2, MAX_SPLIT_PARTS + 1):
            end = start + length
            if end > len(runs):
                break
            joined = " ".join(block.key_loose for block in runs[start:end])
            ratio = fuzz.ratio(single.key_loose, joined) / _PERCENT
            if ratio >= SPLIT_MERGE_RATIO and (
                best is None or ratio > best[0]
            ):
                best = (ratio, start, end)
    return best


def _candidates(
    ref: Sequence[Block],
    new: Sequence[Block],
    rows: Sequence[int],
    columns: Sequence[int],
    scores: Sequence[Sequence[float]],
) -> list[_Candidate]:
    """Return the split and merge candidates of a gap."""
    found = []
    new_blocks = [new[j] for j in columns]
    for a, i in enumerate(rows):
        if max(scores[a]) >= SPLIT_MERGE_RATIO:
            continue
        best = _best_run(ref[i], new_blocks)
        if best is not None:
            ratio, start, end = best
            pair = Pair(PairKind.SPLIT, (i,), tuple(columns[start:end]))
            found.append(_Candidate(ratio, a, a + 1, start, end, pair))
    ref_blocks = [ref[i] for i in rows]
    for b, j in enumerate(columns):
        if max(row[b] for row in scores) >= SPLIT_MERGE_RATIO:
            continue
        best = _best_run(new[j], ref_blocks)
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


def _align_gap(
    ref: Sequence[Block],
    new: Sequence[Block],
    rows: Sequence[int],
    columns: Sequence[int],
    match: float,
) -> list[Pair]:
    """Return the pairs of the blocks of a gap between two anchors."""
    if not rows or not columns:
        return _unmatched(rows, columns)
    if len(rows) * len(columns) > MAX_DP_CELLS:
        return _greedy(ref, new, rows, columns, match)
    scores = [[_similarity(ref[i], new[j]) for j in columns] for i in rows]

    def pair_of(i: int, j: int) -> Pair:
        return _matched(ref, new, (i, j))

    pairs: list[Pair] = []
    row_start = column_start = 0
    for candidate in _select(_candidates(ref, new, rows, columns, scores)):
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
) -> list[Pair]:
    """Return the pairs with unmatched blocks paired as moved blocks."""
    inserted = {
        pair.new[0]: new[pair.new[0]].key_loose
        for pair in pairs
        if pair.kind is PairKind.INSERTED
    }
    deleted = sorted(
        pair.ref[0] for pair in pairs if pair.kind is PairKind.DELETED
    )
    result = [
        pair
        for pair in pairs
        if pair.kind not in {PairKind.DELETED, PairKind.INSERTED}
    ]
    for i in deleted:
        found = (
            process.extractOne(
                ref[i].key_loose,
                inserted,
                scorer=fuzz.ratio,
                score_cutoff=moved_match * _PERCENT,
            )
            if inserted
            else None
        )
        if found is None:
            result.append(Pair(PairKind.DELETED, (i,), ()))
            continue
        j = found[2]
        del inserted[j]
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


def align(
    ref: Sequence[Block],
    new: Sequence[Block],
    *,
    match: float,
    moved_match: float,
) -> list[Pair]:
    """Align the blocks of a reference document with its new version.

    Args:
        ref: Reference blocks, in reading order.
        new: New blocks, in reading order.
        match: Minimum similarity to pair blocks in place.
        moved_match: Minimum similarity to pair a moved block.

    Returns:
        The pairs, in reference order.
    """
    chain, moved = _anchors(ref, new)
    pairs = [_matched(ref, new, anchor, moved=True) for anchor in moved]
    used_ref = {i for i, _ in moved}
    used_new = {j for _, j in moved}
    bounds = [(-1, -1), *chain, (len(ref), len(new))]
    for (i0, j0), (i1, j1) in itertools.pairwise(bounds):
        rows = [i for i in range(i0 + 1, i1) if i not in used_ref]
        columns = [j for j in range(j0 + 1, j1) if j not in used_new]
        pairs += _align_gap(ref, new, rows, columns, match)
        if i1 < len(ref):
            pairs.append(_matched(ref, new, (i1, j1)))
    pairs = _match_leftovers(ref, new, pairs, moved_match)
    return _in_reference_order(pairs)
