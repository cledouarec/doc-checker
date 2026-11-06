# Copyright 2026 Christophe Le Douarec
"""Tell the headings of a PDF from its other blocks, and give their level.

When a PDF has an outline, its bookmarks are the word of its authors: a block
that a bookmark points to is a heading, as deep as the bookmark. The outline
may stop short, as the bookmarks of LaTeX often stop at the chapters, so the
styles complete it: a block set like a marked heading, or smaller than all of
them, is a heading too. Without an outline, the headings are told by how they
look: a short block that is set
larger than the body text, or in bold, is a heading, and the more its style
stands out, the higher its level. The number that starts a title, such as
``1.2.3``, tells the level better than the style does.
"""

import re
from bisect import bisect_left, bisect_right
from collections import Counter
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.extract.pdf.chrome import HEADING_MIN_SIZE, title_key
from doc_checker.text.lexicon import is_admonition_title

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.extract.pdf.blocks import DraftBlock
    from doc_checker.extract.pdf.reader import Bookmark

HEADING_MAX_LINES: Final = 3  # Most lines of a heading.
HEADING_MAX_WORDS: Final = 15  # Most words of a heading.
MAX_LEVEL: Final = 6  # Deepest level of a heading, as in HTML.

# The number of a title: one number, or several around dots, which a dot may
# follow, and then blanks and the title itself.
_NUMBERING: Final = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
# A text that ends with one of these is a sentence, not a heading.
_SENTENCE_ENDS: Final = (".", ",", ";", ":")

# Matches a number of at least two parts, such as ``2.1``, at the start of a
# title.
_SUB_NUMBER: Final = re.compile(r"^\d+(?:\.\d+)+\.?\s+\S")
# How far two sizes may differ, in points, for one style.
STYLE_TOLERANCE: Final = 0.5

# What found a heading: a bookmark, or the look of the block.
type HeadingSource = Literal["outline", "style"]

# The style of a text: its size in points, and whether it is bold.
type _Style = tuple[float, bool]


def bookmark_window(
    pages: Sequence[int],
    pointer: int,
    page: int | None,
) -> range:
    """Return the positions of the blocks that a bookmark can point to.

    The title may be set on the page after the one that the destination of
    the bookmark names, so a block can be pointed to if its first line is on
    that page or on the next one. The blocks come in the order of the
    document, so the pages of their first lines never decrease, and the blocks
    of the two pages are next to each other: a bisection finds them, where a
    walk down the blocks from the pointer goes to the end of the document for
    a bookmark that matches nothing.

    Args:
        pages: The page of the first line of each block, from 0. It never
            decreases from one block to the next.
        pointer: The position of the first block that can be pointed to.
        page: The page that the bookmark points to, from 0, if it points to
            one.

    Returns:
        The positions of the blocks, from the pointer on, that start on the
        page or on the next one. If the bookmark has no page, all the
        positions from the pointer to the last block. The range is empty if
        there is no such block.
    """
    if page is None:
        return range(pointer, len(pages))
    return range(
        max(pointer, bisect_left(pages, page)),
        bisect_right(pages, page + 1),
    )


def bookmark_levels(
    blocks: Sequence[DraftBlock],
    texts: Sequence[str],
    bookmarks: Sequence[Bookmark],
) -> dict[int, int]:
    """Find the blocks that the bookmarks of a document point to.

    The bookmarks come in the order of the outline, and the blocks in the
    order of the document, so a pointer goes down the blocks as the bookmarks
    go down the outline. It starts at the first block. A bookmark is looked
    for among the blocks of `bookmark_window`: from the pointer on, those that
    start on the page that the bookmark points to or on the next one, if it
    points to a page. Its block is the first of them that:

    - is not code;
    - has the same `title_key` as the bookmark.

    The pointer then goes to the block after it, so that two bookmarks with
    the same title match two blocks. A bookmark whose title has no key, and
    one that finds no block, are ignored and leave the pointer where it is.

    Args:
        blocks: The blocks of the document, in reading order.
        texts: The text of each block, in the same order.
        bookmarks: The outline of the document, in the order of its contents.

    Returns:
        The level of each block that a bookmark matches, by position of the
        block: one more than the depth of the bookmark, which starts at 0, and
        at most `MAX_LEVEL`.
    """
    # A block of code has no key, so that it matches no title.
    keys = [
        None if block.code else title_key(text)
        for block, text in zip(blocks, texts, strict=True)
    ]
    pages = [block.lines[0].page for block in blocks]
    levels: dict[int, int] = {}
    pointer = 0
    for bookmark in bookmarks:
        key = title_key(bookmark.title)
        if not key:
            continue
        found = next(
            (
                index
                for index in bookmark_window(pages, pointer, bookmark.page)
                if keys[index] == key
            ),
            None,
        )
        if found is not None:
            levels[found] = min(bookmark.level + 1, MAX_LEVEL)
            pointer = found + 1
    return levels


def _is_candidate(block: DraftBlock, text: str, body: float) -> bool:
    """Return whether a block looks like a heading.

    Args:
        block: The block.
        text: The text of the block.
        body: The body size of the document, in points.

    Returns:
        Whether all of these hold:

        - It is not code.
        - It has at most `HEADING_MAX_LINES` lines and `HEADING_MAX_WORDS`
          words.
        - Its text does not end with a full stop, a comma, a semicolon or a
          colon.
        - Its text is not the default title of an admonition, such as
          ``Note``, which stays a paragraph.
        - Its first line is at least `HEADING_MIN_SIZE` body sizes large, or
          is bold and at least as large as the body, with no other line in
          the block.
    """
    first = block.lines[0]
    is_large = first.size >= HEADING_MIN_SIZE * body
    is_bold_line = first.bold and first.size >= body and len(block.lines) == 1
    return (
        not block.code
        and len(block.lines) <= HEADING_MAX_LINES
        and len(text.split()) <= HEADING_MAX_WORDS
        and not text.endswith(_SENTENCE_ENDS)
        and not is_admonition_title(text)
        and (is_large or is_bold_line)
    )


def _style_order(style: _Style) -> tuple[float, bool]:
    """Return the sort key of a style: large before small, bold first."""
    size, bold = style
    return -size, not bold


def _level(text: str, rank: int) -> int:
    """Return the level of a heading, from its number or else its rank.

    Args:
        text: The text of the heading.
        rank: The rank of its style among the styles of the headings, from 1.

    Returns:
        The count of the parts of the number that `_NUMBERING` finds at the
        start of the text, such as 3 for ``3.1.4``, or the rank if there is no
        such number. At most `MAX_LEVEL`.
    """
    match = _NUMBERING.match(text)
    return min(match[1].count(".") + 1 if match else rank, MAX_LEVEL)


def style_levels(
    blocks: Sequence[DraftBlock],
    texts: Sequence[str],
    *,
    body: float,
) -> dict[int, int]:
    """Find the headings of a document from the look of its blocks.

    A block is a candidate if `_is_candidate` says so. The style of a
    candidate is the size and the weight of its first line. The distinct
    styles of the candidates are put in order, from the largest size down and
    bold before regular, and the style of rank ``r``, from 1, gives the level
    ``r``. A candidate whose text starts with a number such as ``1.2.3`` has
    a level of its own instead, the count of the parts of that number. No
    level exceeds `MAX_LEVEL`.

    Args:
        blocks: The blocks of the document, in reading order.
        texts: The text of each block, in the same order.
        body: The body size of the document, in points.

    Returns:
        The level of each candidate, by position of the block.
    """
    candidates: dict[int, _Style] = {
        index: (block.lines[0].size, block.lines[0].bold)
        for index, (block, text) in enumerate(zip(blocks, texts, strict=True))
        if _is_candidate(block, text, body)
    }
    # The styles are told apart by a total order, so that the set is not
    # walked in an order of its own.
    styles = sorted(set(candidates.values()), key=_style_order)
    ranks = {style: rank for rank, style in enumerate(styles, start=1)}
    return {
        index: _level(texts[index], ranks[style])
        for index, style in candidates.items()
    }


def _same_style(first: _Style, second: _Style) -> bool:
    """Return whether two styles are one: same weight, close sizes."""
    return (
        first[1] == second[1] and abs(first[0] - second[0]) <= STYLE_TOLERANCE
    )


def heading_levels(
    blocks: Sequence[DraftBlock],
    texts: Sequence[str],
    bookmarks: Sequence[Bookmark],
    *,
    body: float,
) -> dict[int, tuple[int, HeadingSource]]:
    """Find the headings of a document: the outline, completed by the styles.

    The bookmarks make the headings that `bookmark_levels` finds, and they
    say what a heading is. If none of them matches a block, or there are
    none, `style_levels` tells all the headings. Otherwise, every other block
    that `_is_candidate` accepts becomes a heading by its style, where
    ``first`` is its first line:

    - If a marked heading has its style, that is the same weight and a size
      within `STYLE_TOLERANCE`, its level is the one that most marked
      headings of that style have, the smallest on a tie.
    - Otherwise it is a heading if its size is at least `HEADING_MIN_SIZE`
      body sizes and smaller than the smallest marked heading by more than
      `STYLE_TOLERANCE`, or if its text starts with a number of two parts or
      more, such as ``2.1``. The count of the parts of that number is its
      level; without a number, the level is that of the deepest marked
      heading plus the rank of its style, from 1, among the styles of the
      blocks promoted this way, large before small and bold first.

    No level exceeds `MAX_LEVEL`.

    Args:
        blocks: The blocks of the document, in reading order.
        texts: The text of each block, in the same order.
        bookmarks: The outline of the document.
        body: The body size of the document, in points.

    Returns:
        The level and the source of each heading, by position of the block.
    """
    marked = bookmark_levels(blocks, texts, bookmarks) if bookmarks else {}
    if not marked:
        return {
            index: (level, "style")
            for index, level in style_levels(blocks, texts, body=body).items()
        }
    styles: dict[int, _Style] = {
        index: (blocks[index].lines[0].size, blocks[index].lines[0].bold)
        for index in marked
    }
    smallest = min(size for size, _ in styles.values())
    deepest = max(marked.values())
    levels: dict[int, tuple[int, HeadingSource]] = {
        index: (level, "outline") for index, level in marked.items()
    }
    same: dict[int, int] = {}
    others: dict[int, _Style] = {}
    for index, (block, text) in enumerate(zip(blocks, texts, strict=True)):
        if index in marked or not _is_candidate(block, text, body):
            continue
        first = block.lines[0]
        style = (first.size, first.bold)
        twins = [
            marked[i]
            for i, other in styles.items()
            if _same_style(style, other)
        ]
        if twins:
            counts = Counter(twins)
            same[index] = min(counts, key=lambda lv: (-counts[lv], lv))
        elif (
            HEADING_MIN_SIZE * body <= first.size < smallest - STYLE_TOLERANCE
            or _SUB_NUMBER.match(text)
        ):
            others[index] = style
    ordered = sorted(set(others.values()), key=_style_order)
    ranks = {style: rank for rank, style in enumerate(ordered, start=1)}
    levels.update((index, (level, "style")) for index, level in same.items())
    levels.update(
        (index, (_level(texts[index], deepest + ranks[style]), "style"))
        for index, style in others.items()
    )
    return levels
