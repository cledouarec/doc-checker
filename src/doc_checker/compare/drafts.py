# Copyright 2026 Christophe Le Douarec
"""Draft findings, before configuration overrides and filters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.compare.align import PairKind
from doc_checker.model.findings import DiffSegment, Side

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.model.document import Block
    from doc_checker.model.findings import Severity

# The arrow of the patterns and of the changes of kind.
ARROW: Final = "\U00002192"
EXCERPT_MAX_CHARS: Final = 300
# The kinds of the pairs whose blocks the two documents share.
MATCHED: Final = frozenset(
    {PairKind.EQUAL, PairKind.MODIFIED, PairKind.SPLIT, PairKind.MERGE},
)

_ELLIPSIS: Final = "\U00002026"

type SegmentOp = Literal["equal", "delete", "insert"]


@dataclass(frozen=True, slots=True)
class Draft:
    """A finding before configuration overrides and filters.

    Attributes:
        rule: Identifier of the rule.
        severity: Severity of the finding, or ``None`` for the default
            severity of the rule.
        ref: Reference side, if any.
        new: New side, if any.
        detail: Optional extra explanation.
        pattern: What the finding changes, such as the kinds before and
            after a change of kind, joined by `ARROW`. Findings of one rule
            with the same pattern make the same change.
    """

    rule: str
    severity: Severity | None
    ref: Side | None
    new: Side | None
    detail: str | None = None
    pattern: str | None = None


def truncate(text: str, limit: int = EXCERPT_MAX_CHARS) -> str:
    """Return the text cut to a number of characters.

    Args:
        text: Any text.
        limit: Most characters kept.

    Returns:
        The text, or its start followed by an ellipsis if it is longer.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + _ELLIPSIS


def whole(
    blocks: Sequence[Block],
    op: SegmentOp,
    *,
    separator: str = " ",
) -> Side:
    """Return a side that shows the whole text of blocks.

    Args:
        blocks: The blocks of the side, at least one.
        op: How the text is marked.
        separator: What joins the texts of the blocks.

    Returns:
        A side at the first block, of its kind, whose one segment is the
        joined text, truncated.
    """
    text = separator.join(block.text for block in blocks)
    segments = (DiffSegment(op, truncate(text)),)
    return Side(blocks[0].source, segments, blocks[0].kind)
