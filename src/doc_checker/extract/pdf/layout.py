# Copyright 2026 Christophe Le Douarec
"""Order the segments of a PDF page for reading, and join them into lines.

The page is cut in two, and each half again, until no cut is left: between
columns first, then between lines. The parts that are left, the leaves, are
read one after the other. Inside a leaf, the segments that share a baseline
make a line. Segments read as a block of their own, such as the cells of a
table, take part in the cuts, so that the text around them is read in the
same order as without them, but in no line.
"""

from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from doc_checker.extract.pdf.segments import Segment

# The thresholds of the cut between columns are multiples of the body size.
COLUMN_GAP: Final = 1.0  # Narrowest gap between two columns.
COLUMN_WIDE: Final = 8.0  # Width that one side of a cut needs somewhere.
COLUMN_NARROW: Final = 4.0  # Width that the other side needs somewhere.
# The tolerance on the baselines is a multiple of the size of a segment.
LINE_TOLERANCE: Final = 0.5  # Largest baseline shift inside a line.

type _Halves = tuple[list[Segment], list[Segment]]


@dataclass(frozen=True, slots=True)
class Line:
    """A line of text, made of the segments that share a baseline.

    The coordinates are those of the segments: points, with the y axis
    pointing up, from the lower left corner of the crop box of the page
    turned upright.

    Attributes:
        page: Position of the page in the document, from 0.
        text: The text of the segments, from left to right, with a space
            between them.
        x0: Left edge of the leftmost segment.
        x1: Right edge of the segment that reaches furthest to the right.
        baseline: Position of the baseline of the leftmost segment.
        size: Font size in points of the segment that has the longest text.
            The leftmost one wins a tie. The segments of digits only that are
            not monospaced do not count if the line has others: a number
            beside text is a line number, a footnote mark or a cell set like
            the text, while a monospaced number is code.
        mono: Whether every segment is monospaced, not counting a leading
            number that is the line number of a code sample.
        bold: Whether every segment is bold.
    """

    page: int
    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    mono: bool
    bold: bool


def _is_column_cut(
    left: Sequence[Segment],
    right: Sequence[Segment],
    body: float,
) -> bool:
    """Return whether the two sides of a cut are columns.

    The cells of a table, or the label and the text of a list item, are far
    apart too, but all of their segments are short. Columns have a segment
    that is wide on one side, and one that is not too narrow on the other.

    Args:
        left: The segments on the left of the cut, at least one.
        right: The segments on the right of the cut, at least one.
        body: The body size of the document, in points.

    Returns:
        Whether the widest segment of one side is at least `COLUMN_WIDE` body
        sizes wide, and the widest segment of the other side is at least
        `COLUMN_NARROW` body sizes wide.
    """
    widths = (max(s.width for s in left), max(s.width for s in right))
    return (
        max(widths) >= COLUMN_WIDE * body
        and min(widths) >= COLUMN_NARROW * body
    )


def _column_cut(segments: Sequence[Segment], body: float) -> _Halves | None:
    """Cut the segments at the widest gap between two columns.

    A gap is a stretch that no segment covers. It is cut at its middle, and
    qualifies if it is at least `COLUMN_GAP` body sizes wide and the segments
    on its two sides are columns. A gap that does not qualify never hides a
    narrower one that does.

    Args:
        segments: The segments to cut.
        body: The body size of the document, in points.

    Returns:
        The segments on the left of the widest gap that qualifies, which is
        the leftmost one on a tie, and those on its right. None if no gap
        qualifies.
    """
    ordered = sorted(segments, key=attrgetter("x0"))
    reach = ordered[0].x1  # The furthest right edge so far.
    # A cut needs a gap wider than 0, so that each side keeps a segment and
    # the cuts of a page come to an end.
    widest = 0.0  # The width of the best gap so far.
    halves: _Halves | None = None
    for segment in ordered[1:]:
        gap = segment.x0 - reach
        if gap > widest and gap >= COLUMN_GAP * body:
            cut = (reach + segment.x0) / 2
            left = [s for s in segments if s.x1 <= cut]
            right = [s for s in segments if s.x0 >= cut]
            if _is_column_cut(left, right, body):
                widest, halves = gap, (left, right)
        reach = max(reach, segment.x1)
    return halves


def _row_cut(segments: Sequence[Segment]) -> _Halves | None:
    """Cut the segments at the largest gap between two lines.

    A gap is a stretch that no line box covers, so that a segment taller than
    the lines next to it keeps them together. It is cut at its middle.

    Args:
        segments: The segments to cut.

    Returns:
        The segments above the largest gap, which is the highest one on a
        tie, and those below it. None if there is no gap.
    """
    ordered = sorted(segments, key=attrgetter("top"), reverse=True)
    low = ordered[0].bottom  # The lowest bottom edge so far.
    # As for the columns, a gap has to be wider than 0.
    largest = 0.0  # The size of the best gap so far.
    cut: float | None = None
    for segment in ordered[1:]:
        gap = low - segment.top
        if gap > largest:
            largest = gap
            cut = (segment.top + low) / 2
        low = min(low, segment.bottom)
    if cut is None:
        return None
    return (
        [s for s in segments if s.bottom >= cut],
        [s for s in segments if s.top <= cut],
    )


def _cut(segments: Sequence[Segment], body: float) -> _Halves | None:
    """Cut segments in two, between columns if possible, else between lines.

    A cut that leaves every segment in one of its halves splits nothing: the
    half would be cut the same way again, and the cutting would never end. The
    segments are then a leaf.

    Args:
        segments: The segments to cut.
        body: The body size of the document, in points.

    Returns:
        The two halves in reading order. None if the segments are a leaf: a
        single segment, no cut at all, or a cut that splits nothing.
    """
    if len(segments) <= 1:
        return None
    halves = _column_cut(segments, body) or _row_cut(segments)
    if halves is None or any(len(half) == len(segments) for half in halves):
        return None
    return halves


def xy_cut(
    segments: Sequence[Segment],
    *,
    body: float,
) -> list[list[Segment]]:
    """Order the segments of a page for reading, with a recursive XY cut.

    The segments are cut in two, and each half again, until no cut is left.
    A cut between columns comes first. It falls in a gap of at least
    `COLUMN_GAP` body sizes, if one side has a segment at least `COLUMN_WIDE`
    body sizes wide and the other side one at least `COLUMN_NARROW` body sizes
    wide: the cells of a table, or the labels of a list, are too short to make
    columns, and are read line by line. Without such a gap, the segments are
    cut between lines. Both cuts choose the widest gap, and the leftmost or
    the highest one if several are as wide.

    A cut that leaves every segment in one of its halves splits nothing, and
    would be found again and again. It is no cut: the segments make a leaf.
    Segments whose line box is upside down, as it is with a negative font
    size, can lead to such a cut.

    Args:
        segments: The segments of one page.
        body: The body size of the document, in points.

    Returns:
        The leaves, which are the sets of segments that no cut splits, in
        reading order: left before right, and top before bottom. The segments
        of a leaf come in the order they were given.
    """
    leaves: list[list[Segment]] = []
    # A stack, not a recursion: the lines of a tall page are cut one at a
    # time, and a thousand of them would exhaust the stack of the interpreter.
    pending = [list(segments)] if segments else []
    while pending:
        part = pending.pop()
        halves = _cut(part, body)
        if halves is None:
            leaves.append(part)
        else:
            # The first half is read first, so it goes on top of the stack.
            pending.extend(reversed(halves))
    return leaves


def _is_loose_number(segment: Segment) -> bool:
    """Return whether a segment is a number that is not monospaced.

    Args:
        segment: The segment.

    Returns:
        Whether its text, without the blanks in it, is made of decimal digits,
        and its font is not monospaced.
    """
    return not segment.mono and "".join(segment.text.split()).isdecimal()


def _line(row: Sequence[Segment]) -> Line:
    """Join the segments of one line.

    Args:
        row: The segments of the line, at least one.

    Returns:
        The line. Its size is that of the segment with the longest text, the
        leftmost one on a tie, leaving out the segments of digits only that
        are not monospaced if there are others: a monospaced number is code.
        It is monospaced if all of its segments are, but for a first segment
        that is only digits and is followed by others.
    """
    parts = sorted(row, key=attrgetter("x0"))
    first = parts[0]
    # A number alone is not the line number of a code sample: it needs the
    # code next to it.
    counted = parts[1:] if len(parts) > 1 and first.text.isdecimal() else parts
    # A number beside text is a line number, a footnote mark or a cell set
    # like the text: it does not give the size of the line. A monospaced
    # number is code, as it is for `mono`.
    sized = [part for part in parts if not _is_loose_number(part)] or parts
    return Line(
        page=first.page,
        text=" ".join(part.text for part in parts),
        x0=first.x0,
        x1=max(part.x1 for part in parts),
        baseline=first.baseline,
        size=max(sized, key=lambda part: len(part.text)).size,
        mono=all(part.mono for part in counted),
        bold=all(part.bold for part in parts),
    )


def leaf_lines(leaf: Sequence[Segment]) -> list[Line]:
    """Join the segments of a leaf into lines.

    The segments are taken from the top down, and from left to right on one
    baseline. A segment joins the current line if its baseline is within
    `LINE_TOLERANCE` sizes of its own of the baseline of the first segment of
    that line. Otherwise it starts a new line.

    Args:
        leaf: The segments of a leaf, as `xy_cut` returns them.

    Returns:
        The lines, from top to bottom. A line puts its segments from left to
        right and takes its baseline and its page from the leftmost one. It
        is monospaced if all of its segments are, but for a first segment that
        is only digits and is followed by others, taken for a line number. Its
        size is that of the segment with the longest text, the leftmost one on
        a tie, leaving out the segments of digits only, blanks apart, that are
        not monospaced if there are others.
    """
    rows: list[list[Segment]] = []
    for segment in sorted(leaf, key=lambda s: (-s.baseline, s.x0)):
        if rows and (
            abs(segment.baseline - rows[-1][0].baseline)
            <= LINE_TOLERANCE * segment.size
        ):
            rows[-1].append(segment)
        else:
            rows.append([segment])
    return [_line(row) for row in rows]


def page_lines(
    segments: Sequence[Segment],
    *,
    body: float,
    hidden: Collection[Segment] = (),
) -> list[Line]:
    """Return the lines of a page, in reading order.

    Args:
        segments: The segments of one page.
        body: The body size of the document, in points.
        hidden: Segments among them that take part in the cuts, but in no
            line, as they are read apart.

    Returns:
        The lines of each leaf that `xy_cut` finds, one leaf after another,
        the hidden segments left out.
    """
    return [
        line
        for leaf in xy_cut(segments, body=body)
        for line in leaf_lines(
            [segment for segment in leaf if segment not in hidden]
            if hidden
            else leaf,
        )
    ]
