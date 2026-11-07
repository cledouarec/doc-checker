# Copyright 2026 Christophe Le Douarec
"""Build normalized blocks and their section paths, whatever the format.

The numbers of the headings, such as ``2.1`` or ``Chapter 3``, and those of
the captions, such as ``Figure 1``, are set apart from their texts, as
generators number them differently. The images repeated on many pages are
chrome, and the images that cannot be read are warnings.
"""

import math
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Final

from doc_checker.images.pixels import EXTERNAL
from doc_checker.model.document import NUMBER_ATTR, Block, BlockKind
from doc_checker.text.lexicon import is_chapter_label
from doc_checker.text.normalize import loose_key, normalize_display, strict_key
from doc_checker.text.numbers import CAPTION_NUMBER, HEADING_NUMBER

if TYPE_CHECKING:
    import re
    from collections.abc import Mapping, Sequence

    from doc_checker.model.document import SourceRef

NUMBERED_MIN: Final = 3  # Fewest numbered headings of a numbered document.
NUMBERED_SHARE: Final = 0.5  # Least share of the headings that are numbered.
CHROME_MIN_PAGES: Final = 3  # Fewest pages on which chrome repeats.
_WARNING_NAMES: Final = 10  # Most names of unreadable images in a warning.
_INLINE_IMAGE: Final = "inline image"  # Name of an image without a name.


def normalized_block(  # ruff: ignore[too-many-arguments] - one argument per field of a block
    kind: BlockKind,
    raw_text: str,
    source: SourceRef,
    counter: Counter[str],
    *,
    level: int | None = None,
    attrs: Mapping[str, str] | None = None,
) -> Block:
    """Return the normalized block of a raw text, with its keys.

    Args:
        kind: Kind of the block.
        raw_text: Text as extracted, before any normalization.
        source: Where the block comes from.
        counter: Normalization counter to update.
        level: Heading level, for headings.
        attrs: Extra extractor attributes, copied into the block.

    Returns:
        The block, with its display text and its strict and loose keys.
    """
    text = normalize_display(raw_text, counter)
    key_strict = strict_key(text, counter, code=kind is BlockKind.CODE)
    return Block(
        kind=kind,
        text=text,
        raw_text=raw_text,
        source=source,
        level=level,
        attrs=dict(attrs or {}),
        key_strict=key_strict,
        key_loose=loose_key(key_strict),
    )


def assign_section_paths(blocks: Sequence[Block]) -> None:
    """Set the titles of the enclosing headings of every block.

    Args:
        blocks: Blocks in reading order, updated in place.
    """
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind is BlockKind.HEADING and block.level is not None:
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
        block.section_path = tuple(text for _, text in stack)


def _numbered(block: Block, number: str) -> Block:
    """Return a heading with a number, kept in its attributes."""
    return replace(block, attrs={**block.attrs, NUMBER_ATTR: number})


def _without_labels(
    blocks: Sequence[Block],
    counter: Counter[str],
) -> list[Block]:
    """Fold each chapter label into the heading that follows it.

    The heading is looked for past the image blocks that follow the label,
    such as an ornament, which stay in place, so that the text does not
    depend on whether the images are read. The anchors of a label go,
    without duplicates, before those of the heading.

    Args:
        blocks: Blocks in reading order.
        counter: Normalization counter, counting the labels folded.

    Returns:
        The blocks, without the labels.
    """
    size = len(blocks)
    # first_other[i] is the position of the first block from i on that is
    # not an image, or size when there is none: one pass, from the end.
    first_other = [size] * (size + 1)
    for position in range(size - 1, -1, -1):
        first_other[position] = (
            first_other[position + 1]
            if blocks[position].kind is BlockKind.IMAGE
            else position
        )
    result: list[Block] = []
    index = 0
    while index < size:
        block = blocks[index]
        after = first_other[index + 1]
        following = blocks[after] if after < size else None
        if (
            following is not None
            and following.kind is BlockKind.HEADING
            and block.kind is not BlockKind.IMAGE
            and is_chapter_label(block.text)
        ):
            anchors = dict.fromkeys((*block.anchors, *following.anchors))
            heading = replace(following, anchors=tuple(anchors))
            result.extend(blocks[index + 1 : after])
            result.append(_numbered(heading, block.text))
            counter["heading_number"] += 1
            index = after + 1
        else:
            result.append(block)
            index += 1
    return result


def _without_number(
    block: Block,
    match: re.Match[str],
    counter: Counter[str],
    key: str = "heading_number",
) -> Block:
    """Return a block without the number that starts its text."""
    text = block.text[match.end() :]
    key_strict = strict_key(text, Counter())
    counter[key] += 1
    return replace(
        _numbered(block, match[1]),
        text=text,
        key_strict=key_strict,
        key_loose=loose_key(key_strict),
    )


def strip_heading_numbers(
    blocks: Sequence[Block],
    counter: Counter[str],
) -> list[Block]:
    """Set the numbers of the headings apart from their texts.

    A chapter label, such as ``Chapter 2``, that is followed by a heading
    is removed, and becomes the number of that heading, which takes its
    anchors first. Then, if the document numbers its headings, which is the
    case when enough of them start with a number and these are a large
    share of the headings, the numbers leave the texts of the headings.
    Each number is kept in the attributes of its heading.

    Args:
        blocks: Blocks in reading order.
        counter: Normalization counter, counting the numbers set apart.

    Returns:
        The blocks, without the labels and the numbers in their texts.
    """
    result = _without_labels(blocks, counter)
    matches = {
        id(b): m
        for b in result
        if b.kind is BlockKind.HEADING and (m := HEADING_NUMBER.match(b.text))
    }
    headings = sum(b.kind is BlockKind.HEADING for b in result)
    if len(matches) < max(NUMBERED_MIN, NUMBERED_SHARE * headings):
        return result
    return [
        _without_number(b, matches[id(b)], counter) if id(b) in matches else b
        for b in result
    ]


def strip_caption_numbers(
    blocks: Sequence[Block],
    counter: Counter[str],
) -> list[Block]:
    """Set the labels and numbers of the captions apart from their texts.

    A caption that starts with a label and a number, such as ``Figure 1:``,
    loses them. The number is kept in its attributes, and is never compared.

    Args:
        blocks: Blocks in reading order.
        counter: Normalization counter, counting the captions changed.

    Returns:
        The blocks, the captions without their labels and numbers.
    """
    return [
        _without_number(b, m, counter, "caption_number")
        if b.kind is BlockKind.CAPTION and (m := CAPTION_NUMBER.match(b.text))
        else b
        for b in blocks
    ]


def repeated_images(
    pages: Sequence[Sequence[Block]],
    *,
    share: float,
) -> frozenset[str]:
    """Return the digests of the images repeated on many pages, as chrome.

    Args:
        pages: Blocks of each page, in page order.
        share: Least share of the pages on which an image is chrome.

    Returns:
        The digests of the images of `IMAGE` blocks found on at least
        ``max(CHROME_MIN_PAGES, ceil(share * pages))`` pages, a page
        counting once.
    """
    counts: Counter[str] = Counter()
    for blocks in pages:
        digests = {
            block.image.digest
            for block in blocks
            if block.kind is BlockKind.IMAGE and block.image is not None
        }
        counts.update(digests)
    threshold = max(CHROME_MIN_PAGES, math.ceil(share * len(pages)))
    return frozenset(
        digest for digest, count in counts.items() if count >= threshold
    )


def image_warnings(blocks: Sequence[Block]) -> list[str]:
    """Return the warning about the images whose pixels cannot be read.

    An external image is never read, and is no warning.

    Args:
        blocks: Blocks of a document.

    Returns:
        No warning if every image can be read, else one: the number of
        images that cannot be read, then their names, ``inline image`` for
        an image without a name, each name once, in order, the first
        `_WARNING_NAMES` of them, then an ellipsis if there are more.
    """
    names = [
        block.image.name or _INLINE_IMAGE
        for block in blocks
        if block.image is not None
        and block.image.problem is not None
        and block.image.problem != EXTERNAL
    ]
    if not names:
        return []
    distinct = list(dict.fromkeys(names))
    shown = ", ".join(distinct[:_WARNING_NAMES])
    if len(distinct) > _WARNING_NAMES:
        shown += ", \U00002026"
    return [f"{len(names)} image(s) could not be read: {shown}"]
