# Copyright 2026 Christophe Le Douarec
"""Group the characters of a PDF page into line segments.

A segment is a stretch of text on one line, such as the words of a sentence or
a cell of a table. A new segment starts where the baseline changes, where the
text jumps back, or where a wide gap opens. The segments are what the next
steps work on, so that the characters of a page can be dropped once they are
grouped.
"""

from collections import Counter
from dataclasses import dataclass, replace
from itertools import chain, dropwhile
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from doc_checker.extract.pdf.reader import PdfChar, PdfPageText

# Every threshold is a multiple of the font size. For the limits of a segment,
# the size is the larger one of the two neighbouring characters.
ASCENT: Final = 0.85  # Height of a line above its baseline.
DESCENT: Final = 0.25  # Depth of a line below its baseline.
BASELINE_TOLERANCE: Final = 0.5  # Largest baseline shift inside a segment.
BACKWARD_TOLERANCE: Final = 0.5  # Largest move back inside a segment.
SEGMENT_GAP: Final = 2.0  # Largest gap inside a segment.
SPACE_GAP: Final = 0.3  # Smallest gap that gets a space.
DEFAULT_BODY_SIZE: Final = 10.0  # In points, for a document without text.


@dataclass(frozen=True, slots=True)
class Segment:
    """A stretch of text on one line of a page.

    The coordinates are those of the characters it comes from: points, with
    the y axis pointing up, from the lower left corner of the crop box of
    the page turned upright.

    Attributes:
        page: Position of the page in the document, from 0.
        text: The text, without trailing blanks. A space stands where the
            characters leave a gap.
        x0: Left edge of the leftmost character that is not a blank.
        x1: Right edge of the rightmost character that is not a blank.
        baseline: Position of the baseline of the first character.
        size: Font size in points, which is the commonest one among the
            characters that are not blank, rounded to the half point, not
            counting the number that `mono` leaves out. The first one seen
            wins a tie.
        mono: Whether the font of every character that is not blank is
            monospaced, not counting the run of decimal digits that opens the
            segment and is not monospaced, when other characters follow it:
            this is the number of a line of code, which a listing may set in a
            small proportional font. A segment made only of digits keeps the
            plain rule.
        bold: Whether the font of every character that is not blank is bold.
    """

    page: int
    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    mono: bool
    bold: bool

    @property
    def top(self) -> float:
        """Top edge of the line box, above the baseline."""
        return self.baseline + ASCENT * self.size

    @property
    def bottom(self) -> float:
        """Bottom edge of the line box, below the baseline."""
        return self.baseline - DESCENT * self.size

    @property
    def width(self) -> float:
        """Width of the text."""
        return self.x1 - self.x0


def _is_blank(char: PdfChar) -> bool:
    """Return whether a character is a blank."""
    return char.text.isspace()


def _starts_segment(previous: PdfChar, char: PdfChar) -> bool:
    """Return whether a character starts a segment after the previous one.

    Args:
        previous: The character before it, in the order of PDFium.
        char: The character.

    Returns:
        Whether the baseline changes, the text moves back or a wide gap
        opens, measured with the larger size of the two characters.
    """
    size = max(previous.size, char.size)
    return (
        abs(char.baseline - previous.baseline) > BASELINE_TOLERANCE * size
        or char.x0 < previous.x0 - BACKWARD_TOLERANCE * size
        or char.x0 - previous.x1 > SEGMENT_GAP * size
    )


def _needs_space(previous: PdfChar, char: PdfChar) -> bool:
    """Return whether a space is missing between two joined characters.

    Args:
        previous: The character before it, in the order of PDFium.
        char: The character.

    Returns:
        Whether they leave a gap that a space fills, and neither is a blank.
    """
    size = max(previous.size, char.size)
    return (
        char.x0 - previous.x1 > SPACE_GAP * size
        and not _is_blank(previous)
        and not _is_blank(char)
    )


def _space_before(previous: PdfChar, char: PdfChar) -> PdfChar:
    """Return the space that fills the gap between two characters."""
    return replace(char, text=" ", x0=previous.x1, x1=char.x0)


def _is_loose_digit(char: PdfChar) -> bool:
    """Return whether a character is a decimal digit that is not monospaced."""
    return char.text.isdecimal() and not char.mono


def _after_number(chars: Sequence[PdfChar]) -> list[PdfChar]:
    """Return the characters of a run that follow the number that opens it.

    The number of a line of code may be set in a small proportional font, in
    the segment of the code that follows it. It is the run of decimal digits
    that opens the segment and is not monospaced.

    Args:
        chars: The characters of the run, with its spaces. The first one is
            not a blank.

    Returns:
        The characters that are not blank, but for the number that opens the
        run. That is all of them if the run has no such number, and none if
        the run is made of that number alone.
    """
    # A blank ends the digits, so that numbers set apart are no line number.
    return [
        char
        for char in dropwhile(_is_loose_digit, chars)
        if not _is_blank(char)
    ]


def _segment(page: int, chars: Sequence[PdfChar]) -> Segment:
    """Make the segment of a run of characters.

    Args:
        page: Position of the page, from 0.
        chars: The characters of the run, with its spaces. The first one is
            not a blank.

    Returns:
        The segment, which has no trailing blank. Its size and its `mono`
        leave out the number that opens it, unless the segment has nothing
        else.
    """
    solid = [char for char in chars if not _is_blank(char)]
    # The number of a line of code gives neither the font nor the size of the
    # code that follows it. A number alone is all that its segment has.
    counted = _after_number(chars) or solid
    # Counter keeps the first size seen among the sizes that are as common.
    sizes = Counter(round(char.size * 2) / 2 for char in counted)
    return Segment(
        page=page,
        text="".join(char.text for char in chars).rstrip(),
        x0=min(char.x0 for char in solid),
        x1=max(char.x1 for char in solid),
        baseline=chars[0].baseline,
        size=sizes.most_common(1)[0][0],
        mono=all(char.mono for char in counted),
        bold=all(char.bold for char in solid),
    )


def build_segments(page: PdfPageText) -> list[Segment]:
    """Group the characters of a page into line segments.

    The characters are taken in the order of PDFium. A character joins the
    segment of the one before it, unless the baseline shifts by more than
    `BASELINE_TOLERANCE` sizes, the text moves back by more than
    `BACKWARD_TOLERANCE` sizes, or the gap between them exceeds `SEGMENT_GAP`
    sizes, a size being the larger one of the two characters. A gap that
    exceeds `SPACE_GAP` sizes between two characters that are not blank gets a
    space. A blank never starts a segment, and a segment loses its trailing
    blanks.

    Args:
        page: The page, as read by `read_page`.

    Returns:
        The segments of the page, in the order of their first character.
    """
    segments: list[Segment] = []
    run: list[PdfChar] = []
    for char in page.chars:
        if run and not _starts_segment(run[-1], char):
            if _needs_space(run[-1], char):
                run.append(_space_before(run[-1], char))
            run.append(char)
            continue
        if run:
            segments.append(_segment(page.index, run))
        run = [] if _is_blank(char) else [char]
    if run:
        segments.append(_segment(page.index, run))
    return segments


def body_size(pages: Iterable[Sequence[Segment]]) -> float:
    """Return the font size of the body text of a document.

    This is the size that most of the characters have, blanks left out, among
    the segments that are not monospaced, so that code samples do not count.
    The smaller size wins a tie.

    Args:
        pages: The segments of each page of the document.

    Returns:
        The body size in points, or `DEFAULT_BODY_SIZE` if there is no segment
        that is not monospaced.
    """
    weights: Counter[float] = Counter()
    for segment in chain.from_iterable(pages):
        if not segment.mono:
            solid = sum(not char.isspace() for char in segment.text)
            weights[segment.size] += solid
    if not weights:
        return DEFAULT_BODY_SIZE
    heaviest = max(weights.values())
    return min(size for size, weight in weights.items() if weight == heaviest)
