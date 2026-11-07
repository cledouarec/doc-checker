# Copyright 2026 Christophe Le Douarec
"""Group the lines of a PDF into draft blocks, across page and column breaks.

A draft block is a paragraph, a list item or a code sample, before its lines
are joined into text and before the headings are told from the paragraphs. The
spacing of the lines, their style, the bullets and numbers of lists, and the
indent of the first line of a paragraph tell where a block ends; an indent only
counts where the lines around it show that first lines are indented, rather
than hanging. The spacing is weighed against the pitch of the lines inside the
paragraphs, measured on the document, as some documents set their lines one and
a half or two lines apart. A line that holds only the next number of the
numbered code before it is an empty line of that code, which it stays in, even
if it opens a page or a column. A caption starts a block of its own: its label
and its number are followed by a colon or a dash, or by a full stop or a
capital if it does not wrap a sentence of the line before. A reference to a
figure in the text goes on in lower case, as in "Figure 3 shows", or ends a
sentence, as in "as shown in Figure 3.". A page or a column break ends a block
too, even in the middle of a sentence, so a block that follows such a break is
joined back to the block it continues, unless it starts with a caption.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import chain, pairwise
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.extract.pdf.chrome import page_number
from doc_checker.text.numbers import CAPTION_NUMBER

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from doc_checker.extract.pdf.layout import Line
    from doc_checker.model.document import BlockKind

# The pitch is the distance between the baselines of two lines. Its limits, the
# rise of a baseline that breaks the flow and the indent of the first line of a
# paragraph are multiples of the font size, except PITCH_SPREAD: a paragraph
# allows the wider of PARAGRAPH_PITCH sizes and of PITCH_SPREAD times the pitch
# measured on the paragraphs of its size. The spread is a tenth only, so that
# the small space after a paragraph, about a sixth of the pitch in wide lines,
# still sets it apart.
PARAGRAPH_PITCH: Final = 1.5  # Widest pitch inside a paragraph, in sizes.
PITCH_SPREAD: Final = 1.1  # The same, in measured paragraph pitches.
CODE_PITCH: Final = 2.5  # Widest pitch inside a code sample.
RISE_TOLERANCE: Final = 0.5  # Highest rise of the baseline inside a flow.
SIZE_TOLERANCE: Final = 0.5  # Largest gap of size inside a block, in points.
INDENT_MIN: Final = 0.8  # Narrowest first line or hanging indent.
WRAP_MIN_WORDS: Final = 5  # Fewest words of a line that wraps a paragraph.

# The marker of a list item is a bullet, or a number, a letter or a roman
# numeral that ends with a dot or a parenthesis, such as 3., a) or (iv). Blanks
# must follow it, so that a decimal such as 3.14 or an abbreviation such as
# e.g. is no marker. U+F0B7, U+F0A7, U+F076, U+F0D8 and U+F0FC are the round
# bullet, the square one, the diamond, the arrow and the check mark that Word
# sets in a symbol font, Symbol or Wingdings: PDFium returns them as private
# use characters, which are no encoding damage. Word may set no blank after
# them, so they need a character that is not blank instead.
BULLET: Final = re.compile(
    r"^(?:[\U0000f0b7\U0000f0a7\U0000f076\U0000f0d8\U0000f0fc]\s*(?=\S)"
    r"|[\U00002022\U000025e6\U000025aa\U000025ab\U00002023\U00002043"
    r"\U000025cf\U000025cb\U000025a0\U000025a1\U00002013\U00002014"
    r"\xb7*-]\s+)",
)
LIST_NUMBER: Final = re.compile(
    r"^(?:\(?\d{1,3}[.)]|\(?[a-z][.)]|\(?[ivx]{1,4}[.)])\s+",
)
# The end of a sentence: a full stop, or a mark that stands for one, which
# closing quotes and brackets may follow.
TERMINAL: Final = re.compile(
    r"[.!?:;][\"'\U00002019\U0000201d\xbb)\]]*$",
)
# The number of a line of numbered code has nine digits at most, as in
# `code_text`: a longer run is no line number.
_LEADING_NUMBER: Final = re.compile(r"^\s*(\d{1,9})")


@dataclass(frozen=True, slots=True)
class DraftBlock:
    """Consecutive lines that make a paragraph, a list item or a code sample.

    Attributes:
        lines: The lines of the block, in reading order. There is at least
            one.
        code: Whether every line is monospaced.
        flow_start: Whether the first line opens a page or a column.
    """

    lines: tuple[Line, ...]
    code: bool
    flow_start: bool


def strip_list_marker(
    text: str,
) -> tuple[str, Literal["bullet", "list_number"] | None]:
    """Remove the bullet or the number that starts a list item.

    Args:
        text: The text of a line or of a block.

    Returns:
        The text without its marker, and the key of the counter of that kind of
        marker: ``"bullet"`` for a `BULLET`, ``"list_number"`` for a
        `LIST_NUMBER`. The text as it is, and None, if it starts with neither.
    """
    if match := BULLET.match(text):
        return text[match.end() :], "bullet"
    if match := LIST_NUMBER.match(text):
        return text[match.end() :], "list_number"
    return text, None


def _has_marker(text: str) -> bool:
    """Return whether a text starts with a bullet or a list number."""
    return strip_list_marker(text)[1] is not None


def _is_flow_break(previous: Line, line: Line) -> bool:
    """Return whether a line opens a page or a column.

    Args:
        previous: The line before it, in reading order.
        line: The line.

    Returns:
        Whether the line is on another page, rises above the baseline of the
        previous one by more than `RISE_TOLERANCE` sizes, or has no horizontal
        overlap with it. The size is the larger of the two.
    """
    size = max(previous.size, line.size)
    return (
        line.page != previous.page
        or line.baseline > previous.baseline + RISE_TOLERANCE * size
        or line.x0 > previous.x1
        or line.x1 < previous.x0
    )


def _is_style_change(previous: Line, line: Line) -> bool:
    """Return whether the style of a line differs from that of the previous.

    Args:
        previous: The line before it, in reading order.
        line: The line.

    Returns:
        Whether the sizes differ by more than `SIZE_TOLERANCE` points, or one
        line is monospaced and the other one is not, or one is bold and the
        other one is not.
    """
    return (
        abs(line.size - previous.size) > SIZE_TOLERANCE
        or line.mono != previous.mono
        or line.bold != previous.bold
    )


def _half_point(value: float) -> float:
    """Return a size or a pitch, in points, rounded to half a point."""
    return round(value * 2) / 2


def _is_wrap(previous: Line, line: Line) -> bool:
    """Return whether two consecutive lines are wrapped lines of a paragraph.

    Args:
        previous: The line before it, in reading order.
        line: The line.

    Returns:
        Whether the line neither breaks the flow nor changes the style,
        neither line is monospaced, the previous line has `WRAP_MIN_WORDS`
        words or more and does not end like a sentence, and the line does not
        start with a bullet or a number.
    """
    return not (
        _is_flow_break(previous, line)
        or _is_style_change(previous, line)
        or previous.mono
        or line.mono
        or len(previous.text.split()) < WRAP_MIN_WORDS
        or TERMINAL.search(previous.text) is not None
        or _has_marker(line.text)
    )


def _mode(counts: Counter[float]) -> float:
    """Return the most frequent value of a count, the smallest one on a tie."""
    most = max(counts.values())
    return min(value for value, count in counts.items() if count == most)


def paragraph_pitches(lines: Sequence[Line]) -> dict[float, float]:
    """Measure the pitch of the lines inside the paragraphs, for each size.

    Only the pairs of consecutive lines that wrap a paragraph are measured. A
    pair, made of a line and the line before it, counts if its pitch is
    positive and if all of these hold:

    - The line neither breaks the flow nor changes the style.
    - Neither line is monospaced.
    - The line before has `WRAP_MIN_WORDS` words or more, and does not end
      like a sentence: it does not match `TERMINAL`.
    - The line does not start with a bullet or a number.

    The other pairs, such as the end of a paragraph and the start of the next
    one, the short rows of a table or the items of a list, may be set closer
    or wider apart than the lines of a paragraph.

    Args:
        lines: The lines of the document, in reading order.

    Returns:
        For each size, the most frequent pitch of the pairs of that size, the
        smallest one if several are as frequent. The size of a pair is the
        larger one of its two lines. The sizes and the pitches are rounded to
        half a point. A size without a pair that counts has no pitch.
    """
    counts: defaultdict[float, Counter[float]] = defaultdict(Counter)
    for previous, line in pairwise(lines):
        pitch = previous.baseline - line.baseline
        if pitch > 0 and _is_wrap(previous, line):
            size = _half_point(max(previous.size, line.size))
            counts[size][_half_point(pitch)] += 1
    return {size: _mode(pitches) for size, pitches in counts.items()}


def _is_gap(
    previous: Line,
    line: Line,
    pitches: Mapping[float, float],
) -> bool:
    """Return whether the space between two lines separates two blocks.

    A code sample is set more loosely than a paragraph: its lines get a higher
    limit. The limit of a paragraph grows with the pitch of the paragraphs of
    its size, so that lines set one and a half or two lines apart stay in
    their paragraph.

    Args:
        previous: The line before it, in reading order.
        line: The line.
        pitches: The pitch of the paragraphs of each size, as
            `paragraph_pitches` returns it.

    Returns:
        Whether the pitch exceeds `CODE_PITCH` sizes if both lines are
        monospaced. If not, whether it exceeds both `PARAGRAPH_PITCH` sizes
        and `PITCH_SPREAD` times the pitch that `pitches` gives for the size,
        rounded to half a point, if it gives one. The size is the larger one
        of the two lines.
    """
    size = max(previous.size, line.size)
    pitch = previous.baseline - line.baseline
    if previous.mono and line.mono:
        return pitch > CODE_PITCH * size
    measured = pitches.get(_half_point(size), 0.0)
    return pitch > max(PARAGRAPH_PITCH * size, PITCH_SPREAD * measured)


def _is_list_start(line: Line) -> bool:
    """Return whether a line starts a list item.

    Args:
        line: The line.

    Returns:
        Whether it is not monospaced, as a line of code may look like an item,
        and starts with a bullet or a list number.
    """
    return not line.mono and _has_marker(line.text)


def _caption_head(text: str) -> Literal["mark", "stop", "capital"] | None:
    """Tell what follows the label and the number that start a caption.

    A reference to a figure in the text goes on in lower case, as in
    ``Figure 3 shows`` or ``Figure 4 on the next page``: it is no caption.

    Args:
        text: The text of a line or of a block.

    Returns:
        ``"mark"`` if the label and the number, as `CAPTION_NUMBER` reads
        them, are followed by a colon, a hyphen, an en dash or an em dash,
        ``"stop"`` by a full stop, ``"capital"`` by blanks and a capital
        letter. None if the text starts with no label and number, or if
        blanks and another character follow them.
    """
    match = CAPTION_NUMBER.match(text)
    if match is None:
        return None
    separator = text[match.end(1) : match.end()].strip()
    if separator == ".":
        return "stop"
    if separator:
        return "mark"
    return "capital" if text[match.end()].isupper() else None


def _is_caption_start(previous: Line, line: Line) -> bool:
    """Return whether a line starts a caption.

    A layout may set a caption right under the line before it, such as the
    last row of a table or another caption, or a column break may part it
    from the block it follows in reading order. A reference to a figure that
    ends a sentence, or a capital after the number, may start the line that
    wraps a sentence of the line before, as in ``as shown in`` then
    ``Figure 3. The``: such a line is no caption. A colon or a dash after
    the number is one, and so is any caption under a line that starts a
    caption itself, so that stacked captions split on a page as they do
    across a break.

    Args:
        previous: The line before it, in reading order.
        line: The line.

    Returns:
        Whether the line is not monospaced, as a line of code may look like a
        caption, and its label and number, as `_caption_head` tells, are
        followed by a colon or a dash, or by a full stop or blanks and a
        capital letter if the line does not wrap the previous line, as
        `_is_wrap` tells, or if the previous line starts a caption too.
    """
    if line.mono:
        return False
    head = _caption_head(line.text)
    return head == "mark" or (
        head is not None
        and (
            not _is_wrap(previous, line)
            or _caption_head(previous.text) is not None
        )
    )


def _indents_first_lines(group: Sequence[Line], margin: float) -> bool:
    """Return whether a group of lines indents the first lines of paragraphs.

    The wrapped lines of a paragraph whose first line is indented go back to
    the left, to the margin. Those of a hanging indent, as in a bibliography,
    go to the right. With as many wraps of each kind, where the lines start
    tells: most lines of a text with indented first lines start at the
    margin, as each paragraph has one indented line at most, while most lines
    of a list with hanging indents start at the indent, or as many for
    entries of two lines.

    Args:
        group: Consecutive lines that only the sixth rule of `group_lines`
            may separate.
        margin: The left edge of the leftmost line of the group.

    Returns:
        Whether more pairs of its lines that wrap a paragraph, as `_is_wrap`
        tells, start `INDENT_MIN` sizes or more to the left of the line before
        than to its right, the size being the larger one of the two lines. If
        there are as many of each, whether more of its lines start at the
        margin, less than `INDENT_MIN` times their size to the right of it,
        than at an indent, that far or farther.
    """
    left = right = 0
    for previous, line in pairwise(group):
        if not _is_wrap(previous, line):
            continue
        shift = line.x0 - previous.x0
        indent = INDENT_MIN * max(previous.size, line.size)
        if shift <= -indent:
            left += 1
        elif shift >= indent:
            right += 1
    if left != right:
        return left > right
    indented = sum(
        1 for line in group if line.x0 - margin >= INDENT_MIN * line.size
    )
    return len(group) - indented > indented


def _is_indent(block: Sequence[Line], line: Line, margin: float) -> bool:
    """Return whether a line is the indented first line of a paragraph.

    Args:
        block: The lines of the block in progress, which the line follows.
        line: The line.
        margin: The left edge of the leftmost line of their group.

    Returns:
        Whether the line is not monospaced, the first line of the block does
        not start with a bullet or a number, the last line of the block ends
        like a sentence, as `TERMINAL` tells, and the line starts `INDENT_MIN`
        sizes or more to the right of the last line of the block, or of the
        margin if the block has a single line. The size is the larger one of
        the line and the last line of the block.
    """
    previous = block[-1]
    start = previous.x0 if len(block) > 1 else margin
    return (
        not line.mono
        and not _has_marker(block[0].text)
        and TERMINAL.search(previous.text) is not None
        and line.x0 - start >= INDENT_MIN * max(previous.size, line.size)
    )


def _split_indents(group: Sequence[Line]) -> list[list[Line]]:
    """Split a group of lines before the indented first lines of paragraphs.

    Args:
        group: Consecutive lines that only the sixth rule of `group_lines`
            may separate.

    Returns:
        The lines of each block, in reading order. The group stays whole if
        it does not indent the first lines of paragraphs, as
        `_indents_first_lines` tells. If it does, a block ends before each
        line that `_is_indent` tells.
    """
    margin = min(line.x0 for line in group)
    if not _indents_first_lines(group, margin):
        return [list(group)]
    blocks = [[group[0]]]
    for line in group[1:]:
        if _is_indent(blocks[-1], line, margin):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def _is_numbered_blank(previous: Line, line: Line) -> bool:
    """Return whether a line is the empty line of a numbered code sample.

    Args:
        previous: The line before it, in reading order.
        line: The line.

    Returns:
        Whether the line is not monospaced and its text, without the blanks
        around it, is a number of one to nine digits; the previous line is
        monospaced; and the text of the previous line starts with a number
        that is one less than the number of the line. A page or a column
        break may lie between the two lines: the run of the numbers tells
        that the line is the empty line that opens the page or the column.
    """
    if not previous.mono or line.mono:
        return False
    number = _LEADING_NUMBER.fullmatch(line.text.strip())
    opening = _LEADING_NUMBER.match(previous.text)
    return (
        number is not None
        and opening is not None
        and int(number[1]) == int(opening[1]) + 1
    )


def _numbered_blanks(lines: Sequence[Line]) -> list[Line]:
    """Make monospaced the lines that are the empty lines of numbered code.

    A listing numbers its empty lines too. If the numbers are set in a small
    proportional font, the empty line has nothing but its number, which is not
    monospaced and not of the size of the code, and would cut the sample. The
    empty line may open a page or a column. Left as text, it would stand
    between the two parts of the sample, which `merge_flows` could then not
    join.

    Args:
        lines: The lines of the document, in reading order.

    Returns:
        The lines, in the same order. A line that `_is_numbered_blank` tells
        from the line before it, as this step has left that one, is
        monospaced and has the size of that line. The others are as they were.
    """
    kept: list[Line] = []
    for line in lines:
        previous = kept[-1] if kept else None
        if previous is not None and _is_numbered_blank(previous, line):
            kept.append(replace(line, mono=True, size=previous.size))
        else:
            kept.append(line)
    return kept


def group_lines(lines: Sequence[Line]) -> list[DraftBlock]:
    """Group the lines of a document into draft blocks.

    First, the empty lines of numbered code are told. A listing numbers its
    empty lines too, and if it sets the numbers in a small proportional font,
    such a line has nothing but its number. A line is made monospaced, with
    the size of the line before it, if it is not monospaced, its text is a
    number of one to nine digits, the line before it is monospaced, as this
    step has left it, and the text of that line starts with a number that is
    one less. A page or a column break may lie between them: the line is then
    the empty line that opens the page or the column.

    Then a line opens a new block if it is the first one, or if any of these
    holds, the pitch being the distance between its baseline and that of the
    previous line, and the size the larger one of the two:

    1. It breaks the flow: it is on another page, its baseline rises by more
       than `RISE_TOLERANCE` sizes, or it has no horizontal overlap with the
       previous line. The block that it opens gets `flow_start`.
    2. It changes the style: the sizes differ by more than `SIZE_TOLERANCE`
       points, or one line is monospaced and the other one is not, or one is
       bold and the other one is not.
    3. The pitch exceeds `CODE_PITCH` sizes if both lines are monospaced. If
       not, it exceeds `PARAGRAPH_PITCH` sizes, and also `PITCH_SPREAD` times
       the pitch that `paragraph_pitches` measures on the lines for the size,
       rounded to half a point, if it measures one.
    4. It is not monospaced and starts with a bullet or a number.
    5. It is not monospaced and starts with the label and the number of a
       caption, as `CAPTION_NUMBER` reads them, followed by a colon, a
       hyphen, an en dash or an em dash. Or they are followed by a full stop,
       or by blanks and a capital letter, and the line does not wrap the
       previous line: the previous line is monospaced, has fewer than
       `WRAP_MIN_WORDS` words, ends like a sentence, as `TERMINAL` tells, or
       starts with the label and the number of a caption itself.
       A reference to a figure goes on in lower case, as in ``Figure 3
       shows``, or ends a sentence that the line before wraps, as in ``as
       shown in`` then ``Figure 3. The``: it is no caption.
    6. It is the indented first line of a paragraph, in a group of lines that
       indents them. The groups are the blocks that rules 1 to 5 make. A group
       indents first lines if more of its pairs of lines that wrap a
       paragraph, as `paragraph_pitches` describes them, start `INDENT_MIN`
       sizes or more to the left of the line before than to its right. With
       as many of each, it does if more of its lines start at its margin than
       at an indent: less than `INDENT_MIN` times their size to the right of
       its leftmost line, rather than that far or farther. In such a group, a
       line opens a block if it is not monospaced, the first line of the block
       does not start with a bullet or a number, the previous line matches
       `TERMINAL`, and the line starts `INDENT_MIN` sizes or more to the right
       of the previous line, or of the leftmost line of the group if the block
       has a single line.

    A list item is told by its marker, not by its indentation: the lines under
    a marker go on with their item, indented or not, until the next marker.
    The wrapped lines of a paragraph whose first line is indented go back to
    the left, while those of a hanging indent, as in a bibliography, go to the
    right. Without such wraps, most lines of a text with indented first lines
    start at the margin, and most lines of a list with hanging indents start
    at the indent. A list of hanging indents whose entries mostly fit on one
    line, and never wrap in mid-sentence, is still cut after an entry whose
    first line ends a sentence.

    Args:
        lines: The lines of the document, in reading order: the lines of a
            page, as `page_lines` returns them, one page after the other.

    Returns:
        The blocks, in reading order.
    """
    if not lines:
        return []
    lines = _numbered_blanks(lines)
    pitches = paragraph_pitches(lines)
    groups = [[lines[0]]]
    flows = [False]  # Whether each group opens a page or a column.
    for previous, line in pairwise(lines):
        opens_flow = _is_flow_break(previous, line)
        if (
            opens_flow
            or _is_style_change(previous, line)
            or _is_gap(previous, line, pitches)
            or _is_list_start(line)
            or _is_caption_start(previous, line)
        ):
            groups.append([line])
            flows.append(opens_flow)
        else:
            groups[-1].append(line)
    return [
        DraftBlock(
            lines=tuple(run),
            code=all(line.mono for line in run),
            flow_start=flow and index == 0,
        )
        for group, flow in zip(groups, flows, strict=True)
        for index, run in enumerate(_split_indents(group))
    ]


def _continues(
    previous: DraftBlock,
    block: DraftBlock,
    page_kinds: Mapping[int, BlockKind],
) -> bool:
    """Return whether a block that opens a flow goes on with the one before.

    Args:
        previous: The block before it.
        block: The block, which opens a page or a column.
        page_kinds: The kind of each contents or index page, by position of
            the page.

    Returns:
        Whether the rules of `merge_flows` join the two blocks.
    """
    last, first = previous.lines[-1], block.lines[0]
    if page_kinds.get(last.page) is not page_kinds.get(first.page):
        return False
    text = " ".join(line.text for line in previous.lines)
    if page_number(text) is not None:
        return False
    if previous.code and block.code:
        return True
    return (
        not (previous.code or block.code)
        and not _has_marker(first.text)
        and _caption_head(first.text) is None
        and TERMINAL.search(last.text) is None
        and abs(last.size - first.size) <= SIZE_TOLERANCE
        and not (last.bold or first.bold)
    )


def _join(group: Sequence[DraftBlock]) -> DraftBlock:
    """Join consecutive blocks into one.

    Args:
        group: The blocks, at least one.

    Returns:
        The block that holds all of their lines, with the `code` and the
        `flow_start` of the first one.
    """
    first = group[0]
    if len(group) == 1:
        return first
    lines = tuple(chain.from_iterable(block.lines for block in group))
    return replace(first, lines=lines)


def merge_flows(
    blocks: Sequence[DraftBlock],
    counter: Counter[str],
    *,
    page_kinds: Mapping[int, BlockKind] | None = None,
) -> list[DraftBlock]:
    """Join the blocks that a page or a column break splits.

    A block that opens a flow is added to the block before it if both are
    code. If neither is code, it is added if all of these hold:

    - It does not start with a bullet or a number.
    - It does not start with the label and the number of a caption followed
      by a colon, a dash, a full stop or blanks and a capital letter, as
      `_caption_head` tells. A figure that floats to the top of a page cuts
      the sentence that ends the page before, and the reading order of the XY
      cut may put a caption after a narrow table that it stands above: the
      caption opens the page or the column, and starts a block of its own. A
      reference to a figure that goes on in lower case, as in ``Figure 4 on
      the next page``, goes on with the sentence.
    - The last line of the block before does not end like a sentence: it does
      not match `TERMINAL`.
    - The sizes of that line and of the first line of the block differ by
      `SIZE_TOLERANCE` points at most.
    - Neither of these two lines is bold.

    Whatever the case, a block is never added across a change of page kind:
    the page of the last line of the block before and the page of the first
    line of the block must have the same kind in `page_kinds`, or neither of
    them a kind. Without this, the first paragraph after a contents page would
    join the last entry of the contents, which does not end like a sentence.
    Nor is a block ever added after a page number: a block before it whose
    lines, joined by blanks, make a text that `page_number` reads. Such a
    block is a page number that the chrome has missed, which does not end like
    a sentence either.

    The joined block opens a flow if the first of its blocks does.

    Args:
        blocks: The blocks of the document, as `group_lines` returns them.
        counter: The counts of the normalizations, in which `page_break` goes
            up by one for each block that is added to another.
        page_kinds: The kind of each contents or index page, by position of
            the page, as `toc_pages` returns it. The other pages have no
            kind. None means that no page has a kind.

    Returns:
        The blocks after the joins, in reading order.
    """
    kinds = page_kinds or {}
    groups: list[list[DraftBlock]] = []
    for index, block in enumerate(blocks):
        # The block before stands for the group that it ends: they have the
        # same last line, hence the same last page, and the blocks of a group
        # are all code or none is.
        if (
            index
            and block.flow_start
            and _continues(blocks[index - 1], block, kinds)
        ):
            groups[-1].append(block)
            counter["page_break"] += 1
        else:
            groups.append([block])
    return [_join(group) for group in groups]
