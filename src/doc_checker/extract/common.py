# Copyright 2026 Christophe Le Douarec
"""Build normalized blocks and their section paths, whatever the format.

The default titles of admonitions are kept here too, as the HTML reader and
the PDF reader both need them: the first leaves them out of its blocks, and
the second does not take them for headings.
"""

from typing import TYPE_CHECKING, Final

from doc_checker.model.document import Block, BlockKind
from doc_checker.text.normalize import loose_key, normalize_display, strict_key

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Mapping, Sequence

    from doc_checker.model.document import SourceRef

# The default titles of admonitions, as a text is compared with them: case
# folded, and without a colon at the end.
DEFAULT_ADMONITION_TITLES: Final[frozenset[str]] = frozenset(
    {
        "note",
        "notes",
        "warning",
        "tip",
        "hint",
        "important",
        "caution",
        "danger",
        "attention",
        "error",
        "info",
        "example",
        "see also",
        "todo",
        "abstract",
        "summary",
        "success",
        "question",
        "failure",
        "bug",
        "quote",
        "remarque",
        "avertissement",
        "astuce",
        "erreur",
        "conseil",
        "exemple",
        "voir aussi",
        "à faire",
    },
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
