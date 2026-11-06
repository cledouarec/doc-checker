# Copyright 2026 Christophe Le Douarec
"""Changes of kind between the matched blocks of two documents.

The kinds of the reference blocks of a matched pair are compared with the
kinds of its new blocks. A heading, a table or a code block matched with
blocks among which none has its kind lost it: ``structure.heading_lost``,
``structure.table_lost`` and ``structure.code_lost``. The items of a list
matched with blocks among which none is a list item lost their list: one
``structure.list_lost`` for each run of consecutive items of the reference.
Any other change of kind, such as a paragraph that became a heading, is
``structure.kind_changed``, for information.

Between two PDF documents, the kinds come from fonts, bullets and the
layout, which the comparison ignores: their findings are information, with
a detail that says so, except the loss of a heading of the outline when the
new outline has headings of the same level, which keeps the severity of its
rule.

The levels of the headings are compared with a map learned from the pairs
of headings one to one: for each level of the reference, the level that
most of its headings took, or else that level moved by the most frequent
shift. A heading whose new level differs from the map is
``structure.heading_level``, so that a generator that shifts every heading
changes nothing.
"""

import itertools
from collections import Counter
from typing import TYPE_CHECKING, Final

from doc_checker.compare.drafts import ARROW, MATCHED, Draft, whole
from doc_checker.model.document import HEADING_SOURCE, BlockKind
from doc_checker.model.findings import Severity

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from doc_checker.compare.align import Pair
    from doc_checker.model.document import Block

_HEADING_LOST: Final = "structure.heading_lost"
_LIST_LOST: Final = "structure.list_lost"
_KIND_CHANGED: Final = "structure.kind_changed"
_HEADING_LEVEL: Final = "structure.heading_level"
# Each kind whose loss has a rule of its own, with that rule.
_LOST: Final = (
    (BlockKind.HEADING, _HEADING_LOST),
    (BlockKind.TABLE, "structure.table_lost"),
    (BlockKind.CODE, "structure.code_lost"),
)
_OUTLINE: Final = "outline"
_PDF_NOTE: Final = "; PDF kinds are inferred from the layout"


def _from_outline(block: Block) -> bool:
    """Return whether a block is a heading that a PDF outline gives."""
    return (
        block.kind is BlockKind.HEADING
        and block.attrs.get(HEADING_SOURCE) == _OUTLINE
    )


def _names(kinds: Iterable[BlockKind]) -> list[str]:
    """Return the sorted names of kinds, each one once."""
    return sorted({kind.value for kind in kinds})


def _outline_kept(
    ref_blocks: Sequence[Block],
    outline_levels: Collection[int | None],
) -> bool:
    """Return whether a lost heading has a level of the new outline.

    Args:
        ref_blocks: Reference blocks of the pair that lost a heading.
        outline_levels: Levels of the outline headings of the new document.

    Returns:
        Whether one of the blocks is a heading of the outline whose level
        the new outline has.
    """
    return any(
        _from_outline(block) and block.level in outline_levels
        for block in ref_blocks
    )


def _grade(detail: str, *, inferred: bool) -> tuple[Severity | None, str]:
    """Return the severity and the detail of a draft.

    Args:
        detail: Detail of the draft.
        inferred: Whether the draft rests on kinds that the layout of a PDF
            document tells.

    Returns:
        ``None``, the severity of the rule, and the detail; or, for kinds
        inferred from the layout, information and the detail followed by a
        note that says so.
    """
    if inferred:
        return Severity.INFO, detail + _PDF_NOTE
    return None, detail


def _pair_drafts(
    ref_blocks: Sequence[Block],
    new_blocks: Sequence[Block],
    *,
    pdf: bool,
    outline_levels: Collection[int | None],
) -> list[Draft]:
    """Return the drafts of the changes of kind of one matched pair.

    The items of a list are left to `_list_drafts`.
    """
    before = {block.kind for block in ref_blocks}
    after = {block.kind for block in new_blocks}
    if before == after:
        return []
    rules = [rule for kind, rule in _LOST if kind in before - after]
    if not rules and BlockKind.LIST_ITEM not in before:
        rules = [_KIND_CHANGED]
    old_names, new_names = _names(before), _names(after)
    detail = f"{' and '.join(old_names)} became {' and '.join(new_names)}"
    pattern = f"{'+'.join(old_names)}{ARROW}{'+'.join(new_names)}"
    ref_side, new_side = whole(ref_blocks, "equal"), whole(new_blocks, "equal")
    drafts = []
    for rule in rules:
        inferred = pdf and not (
            rule == _HEADING_LOST and _outline_kept(ref_blocks, outline_levels)
        )
        severity, text = _grade(detail, inferred=inferred)
        drafts.append(Draft(rule, severity, ref_side, new_side, text, pattern))
    return drafts


def _list_runs(blocks: Sequence[Block]) -> list[list[int]]:
    """Return the runs of consecutive list items, as block positions."""
    runs = []
    for is_item, run in itertools.groupby(
        range(len(blocks)),
        key=lambda index: blocks[index].kind is BlockKind.LIST_ITEM,
    ):
        if is_item:
            runs.append(list(run))
    return runs


def _list_drafts(
    pairs: Sequence[Pair],
    ref: Sequence[Block],
    new: Sequence[Block],
    *,
    pdf: bool,
) -> list[tuple[int, Draft]]:
    """Return one draft for each list of the reference that lost items.

    Args:
        pairs: The alignment of the two documents.
        ref: Reference blocks.
        new: New blocks.
        pdf: Whether the reference is a PDF document.

    Returns:
        Each draft, with the position of the pair of the first lost item of
        its list, in reference order.
    """
    position_of = {
        index: position
        for position, pair in enumerate(pairs)
        if pair.kind in MATCHED
        for index in pair.ref
    }
    found = []
    for run in _list_runs(ref):
        lost = [
            index
            for index in run
            if index in position_of
            and all(
                new[j].kind is not BlockKind.LIST_ITEM
                for j in pairs[position_of[index]].new
            )
        ]
        if not lost:
            continue
        matches = dict.fromkeys(
            j for index in lost for j in pairs[position_of[index]].new
        )
        new_blocks = [new[j] for j in matches]
        names = _names(block.kind for block in new_blocks)
        detail = (
            f"{len(lost)} of {len(run)} list items became "
            f"{' and '.join(names)}"
        )
        severity, text = _grade(detail, inferred=pdf)
        draft = Draft(
            _LIST_LOST,
            severity,
            whole([ref[index] for index in lost], "equal", separator="\n"),
            whole(new_blocks, "equal", separator="\n"),
            text,
            f"{BlockKind.LIST_ITEM.value}{ARROW}{'+'.join(names)}",
        )
        found.append((position_of[lost[0]], draft))
    return found


def kind_drafts(
    pairs: Sequence[Pair],
    ref: Sequence[Block],
    new: Sequence[Block],
    *,
    pdf: bool,
) -> dict[int, list[Draft]]:
    """Return the drafts of the changes of kind between matched blocks.

    Args:
        pairs: The alignment of the two documents.
        ref: Reference blocks, which the pairs index.
        new: New blocks, which the pairs index.
        pdf: Whether the reference is a PDF document, whose kinds are
            inferred from the layout.

    Returns:
        The drafts of each pair that has some, by position of the pair: the
        changes of kind of the pair, then the lists whose first lost item
        it holds.
    """
    outline_levels = frozenset(
        block.level for block in new if _from_outline(block)
    )
    found: dict[int, list[Draft]] = {}
    for position, pair in enumerate(pairs):
        if pair.kind not in MATCHED:
            continue
        drafts = _pair_drafts(
            [ref[i] for i in pair.ref],
            [new[j] for j in pair.new],
            pdf=pdf,
            outline_levels=outline_levels,
        )
        if drafts:
            found[position] = drafts
    for position, draft in _list_drafts(pairs, ref, new, pdf=pdf):
        found.setdefault(position, []).append(draft)
    return found


def _heading_pairs(
    pairs: Sequence[Pair],
    ref: Sequence[Block],
    new: Sequence[Block],
) -> list[tuple[int, Block, Block]]:
    """Return the matched one to one pairs of headings that have levels.

    Returns:
        The position, the reference heading and the new heading of each.
    """
    found = []
    for position, pair in enumerate(pairs):
        if (
            pair.kind not in MATCHED
            or len(pair.ref) != 1
            or len(pair.new) != 1
        ):
            continue
        before, after = ref[pair.ref[0]], new[pair.new[0]]
        if (
            before.kind is BlockKind.HEADING
            and after.kind is BlockKind.HEADING
            and before.level is not None
            and after.level is not None
        ):
            found.append((position, before, after))
    return found


def _most_common(counts: Counter[int]) -> tuple[int, int]:
    """Return the most frequent value and its count, the smallest on a tie."""
    top = min(counts, key=lambda value: (-counts[value], value))
    return top, counts[top]


def _levels_of(found: Sequence[tuple[int, Block, Block]]) -> dict[int, int]:
    """Return the expected new level of each reference level of headings."""
    shifts: Counter[int] = Counter()
    took: dict[int, Counter[int]] = {}
    for _, before, after in found:
        a, b = before.level, after.level
        shifts[b - a] += 1
        took.setdefault(a, Counter())[b] += 1
    if not shifts:
        return {}
    dominant = min(shifts, key=lambda d: (-shifts[d], abs(d), d))
    expected: dict[int, int] = {}
    previous: int | None = None
    for a in sorted(took):
        top, n = _most_common(took[a])
        e = top if 2 * n > took[a].total() else a + dominant
        if previous is not None and e <= expected[previous]:
            e = expected[previous] + (a - previous)
        expected[a] = e
        previous = a
    return expected


def expected_levels(
    pairs: Sequence[Pair],
    ref: Sequence[Block],
    new: Sequence[Block],
) -> dict[int, int]:
    """Return the level that each reference heading level maps to.

    Args:
        pairs: The alignment of the two documents.
        ref: Reference blocks, which the pairs index.
        new: New blocks, which the pairs index.

    Returns:
        For each reference level of the headings matched one to one, the
        new level that most of them took if that is a strict majority,
        otherwise the level moved by the most frequent shift (on a tie, the
        smallest in absolute value, then the smallest), kept strictly
        above the expected level of the level before it.
    """
    return _levels_of(_heading_pairs(pairs, ref, new))


def level_drafts(
    pairs: Sequence[Pair],
    ref: Sequence[Block],
    new: Sequence[Block],
    *,
    pdf: bool,
) -> dict[int, list[Draft]]:
    """Return the drafts of the headings whose level left the level map.

    Args:
        pairs: The alignment of the two documents.
        ref: Reference blocks, which the pairs index.
        new: New blocks, which the pairs index.
        pdf: Whether the reference is a PDF document; the levels of its
            headings are information unless both come from the outline.

    Returns:
        The draft of each pair of headings by position of the pair.
    """
    found = _heading_pairs(pairs, ref, new)
    expected = _levels_of(found)
    drafts: dict[int, list[Draft]] = {}
    for position, before, after in found:
        a, b = before.level, after.level
        e = expected[a]
        if b == e:
            continue
        detail = (
            f"level {a} became level {b}, "
            f"level {e} was expected from the other headings"
        )
        inferred = pdf and not (_from_outline(before) and _from_outline(after))
        severity, text = _grade(detail, inferred=inferred)
        draft = Draft(
            _HEADING_LEVEL,
            severity,
            whole([before], "equal"),
            whole([after], "equal"),
            text,
            f"{a}{ARROW}{b}",
        )
        drafts[position] = [draft]
    return drafts
