# Copyright 2026 Christophe Le Douarec
"""Draft findings, before configuration overrides and filters.

The sides of a draft show the text of their blocks: whole, by `whole`, or
around the part that a finding points at, marked, by `marked_side`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.compare.align import PairKind
from doc_checker.model.findings import DiffSegment, Side
from doc_checker.text.normalize import ELLIPSIS, truncate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.model.document import (
        Block,
        BlockKind,
        ImageRef,
        SourceRef,
    )
    from doc_checker.model.findings import Severity

# The arrow of the patterns and of the changes of kind.
ARROW: Final = "\U00002192"
EXCERPT_MAX_CHARS: Final = 300
# The kinds of the pairs whose blocks the two documents share.
MATCHED: Final = frozenset(
    {PairKind.EQUAL, PairKind.MODIFIED, PairKind.SPLIT, PairKind.MERGE},
)

type SegmentOp = Literal["equal", "delete", "insert", "mark"]


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
        joined text, cut to `EXCERPT_MAX_CHARS` characters.
    """
    text = separator.join(block.text for block in blocks)
    segments = (DiffSegment(op, truncate(text, EXCERPT_MAX_CHARS)),)
    return Side(blocks[0].source, segments, blocks[0].kind)


def marked_span(
    source: SourceRef,
    text: str,
    span: tuple[int, int],
    *,
    kind: BlockKind | None = None,
    image: ImageRef | None = None,
) -> Side:
    """Return a side that marks a span of a text, with some text around it.

    Args:
        source: Where the text comes from.
        text: The text, such as the text of a block.
        span: Start and end of the marked part in the text.
        kind: Kind of the block of the text, if any.
        image: Image of the block of the text, if any.

    Returns:
        A side whose segments are the text before the span, its last
        ``EXCERPT_MAX_CHARS // 2`` characters after an `ELLIPSIS` if it is
        longer; the span, marked, cut to `EXCERPT_MAX_CHARS` characters;
        then the text after it, cut to ``EXCERPT_MAX_CHARS // 2``
        characters. Empty segments are left out. Only the parts kept are
        read, so that marking a long text costs no more than a short one.
    """
    start, end = span
    half = EXCERPT_MAX_CHARS // 2
    if start > half:
        before = ELLIPSIS + text[start - half : start]
    else:
        before = text[:start]
    marked = text[start : min(end, start + EXCERPT_MAX_CHARS + 1)]
    after = text[end : end + half + 1]
    segments = (
        DiffSegment("equal", before),
        DiffSegment("mark", truncate(marked, EXCERPT_MAX_CHARS)),
        DiffSegment("equal", truncate(after, half)),
    )
    return Side(
        source,
        tuple(segment for segment in segments if segment.text),
        kind,
        image,
    )


def marked_side(
    source: SourceRef,
    text: str,
    mark: str = "",
    *,
    kind: BlockKind | None = None,
    image: ImageRef | None = None,
) -> Side:
    """Return a side that marks the first occurrence of a part of a text.

    Args:
        source: Where the text comes from.
        text: The text, such as the text of a block.
        mark: The part of the text that a finding points at, if any.
        kind: Kind of the block of the text, if any.
        image: Image of the block of the text, if any.

    Returns:
        Without ``mark``, or if the text does not hold it, a side whose one
        segment is the text, cut to `EXCERPT_MAX_CHARS` characters; else
        the side of `marked_span` at the first occurrence of ``mark``.
    """
    start = text.find(mark) if mark else -1
    if start < 0:
        segments = (DiffSegment("equal", truncate(text, EXCERPT_MAX_CHARS)),)
        return Side(source, segments, kind, image)
    return marked_span(
        source,
        text,
        (start, start + len(mark)),
        kind=kind,
        image=image,
    )
