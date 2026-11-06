# Copyright 2026 Christophe Le Douarec
"""Build normalized blocks and their section paths, whatever the format.

The numbers of the headings, such as ``2.1`` or ``Chapter 3``, are set apart
from the texts of the headings, as generators number them differently.
"""

import re
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Final

from doc_checker.model.document import Block, BlockKind
from doc_checker.text.normalize import loose_key, normalize_display, strict_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from doc_checker.model.document import SourceRef

NUMBERED_MIN: Final = 3  # Fewest numbered headings of a numbered document.
NUMBERED_SHARE: Final = 0.5  # Least share of the headings that are numbered.
NUMBER_ATTR: Final = "number"  # Attribute that keeps the number of a heading.
HEADING_NUMBER: Final = re.compile(
    r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\.?\s+(?=\S)",
)
CHAPTER_LABEL: Final = re.compile(
    r"^(?:chapter|chapitre|part|partie|appendix|annexe)"
    r"\s+(?:\d+|[ivxlc]+|[a-z])$",
    re.IGNORECASE,
)


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


def _is_label(block: Block) -> bool:
    """Return whether the whole text of a block is a chapter label."""
    return CHAPTER_LABEL.fullmatch(block.text) is not None


def _numbered(block: Block, number: str) -> Block:
    """Return a heading with a number, kept in its attributes."""
    return replace(block, attrs={**block.attrs, NUMBER_ATTR: number})


def _without_labels(
    blocks: Sequence[Block],
    counter: Counter[str],
) -> list[Block]:
    """Fold each chapter label into the heading that follows it.

    Args:
        blocks: Blocks in reading order.
        counter: Normalization counter, counting the labels folded.

    Returns:
        The blocks, without the labels.
    """
    result: list[Block] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if (
            following is not None
            and following.kind is BlockKind.HEADING
            and _is_label(block)
        ):
            result.append(_numbered(following, block.text))
            counter["heading_number"] += 1
            index += 2
        else:
            result.append(block)
            index += 1
    return result


def _without_number(
    block: Block,
    match: re.Match[str],
    counter: Counter[str],
) -> Block:
    """Return a heading without the number that starts its text."""
    text = block.text[match.end() :]
    key_strict = strict_key(text, Counter())
    counter["heading_number"] += 1
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
    is removed, and becomes the number of that heading. Then, if the
    document numbers its headings, which is the case when enough of them
    start with a number and these are a large share of the headings, the
    numbers leave the texts of the headings. Each number is kept in the
    attributes of its heading.

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
