# Copyright 2026 Christophe Le Douarec
"""Tests for the block building shared by the HTML and PDF loaders."""

from collections import Counter
from types import MappingProxyType

from doc_checker.extract.common import (
    DEFAULT_ADMONITION_TITLES,
    assign_section_paths,
    normalized_block,
)
from doc_checker.model.document import BlockKind, SourceRef

SOURCE = SourceRef("doc.pdf", page=2)


def _paths(
    *items: tuple[BlockKind, str, int | None],
) -> list[tuple[str, ...]]:
    """Return the section paths of blocks made of (kind, text, level) items."""
    counter: Counter[str] = Counter()
    blocks = [
        normalized_block(kind, text, SOURCE, counter, level=level)
        for kind, text, level in items
    ]
    assign_section_paths(blocks)
    return [block.section_path for block in blocks]


def test_normalized_block_computes_keys() -> None:
    """The block gets its display text, its two keys and a page link."""
    counter: Counter[str] = Counter()
    block = normalized_block(
        BlockKind.PARAGRAPH,
        "l\U00002019option  \U0000fb01x",
        SOURCE,
        counter,
    )
    assert (block.text, block.key_strict, block.key_loose) == (
        "l\U00002019option  fix",
        "l'option fix",
        "l option fix",
    )
    assert (counter["ligature"], counter["quote"], block.attrs) == (1, 1, {})
    assert block.source.href == "doc.pdf#page=2"


def test_normalized_block_keeps_code_lines() -> None:
    """A code block keeps its lines, without trailing blanks, in its key."""
    block = normalized_block(
        BlockKind.CODE,
        "x = 1  \n  y = 2\n",
        SourceRef("a.html"),
        Counter(),
    )
    assert block.key_strict == "x = 1\n  y = 2"


def test_normalized_block_joins_lines_of_other_kinds() -> None:
    """A block that is not code has its lines joined in its strict key."""
    block = normalized_block(
        BlockKind.PARAGRAPH,
        "x = 1\n  y = 2",
        SOURCE,
        Counter(),
    )
    assert (block.text, block.key_strict) == ("x = 1\n  y = 2", "x = 1 y = 2")


def test_normalized_block_keeps_raw_text_kind_source_and_level() -> None:
    """The block keeps its raw text, its kind, its source and its level."""
    raw = "l\U00002019option  \U0000fb01x"
    block = normalized_block(
        BlockKind.HEADING,
        raw,
        SOURCE,
        Counter(),
        level=3,
    )
    assert block.raw_text == raw
    assert block.source is SOURCE
    assert (block.kind, block.level) == (BlockKind.HEADING, 3)


def test_normalized_block_copies_attrs() -> None:
    """The attributes of a block are a dict copied from any mapping."""
    given = {"role": "note"}
    block = normalized_block(
        BlockKind.ADMONITION,
        "Note.",
        SOURCE,
        Counter(),
        attrs=MappingProxyType(given),
    )
    given["role"] = "tip"
    assert (type(block.attrs), block.attrs) == (dict, {"role": "note"})


def test_normalized_block_attrs_are_not_shared() -> None:
    """Each block that is given no attributes has a dict of its own."""
    first = normalized_block(BlockKind.PARAGRAPH, "A.", SOURCE, Counter())
    second = normalized_block(BlockKind.PARAGRAPH, "B.", SOURCE, Counter())
    first.attrs["role"] = "note"
    assert second.attrs == {}


def test_assign_section_paths() -> None:
    """Each block gets the titles of the headings that enclose it."""
    counter: Counter[str] = Counter()
    blocks = [
        normalized_block(kind, text, SOURCE, counter, level=level)
        for kind, text, level in (
            (BlockKind.HEADING, "Guide", 1),
            (BlockKind.HEADING, "Install", 2),
            (BlockKind.PARAGRAPH, "Text.", None),
            (BlockKind.HEADING, "Use", 2),
            (BlockKind.PARAGRAPH, "More.", None),
        )
    ]
    assign_section_paths(blocks)
    assert [b.section_path for b in blocks] == [
        ("Guide",),
        ("Guide", "Install"),
        ("Guide", "Install"),
        ("Guide", "Use"),
        ("Guide", "Use"),
    ]


def test_section_path_is_empty_before_the_first_heading() -> None:
    """A block that comes before every heading has an empty section path."""
    assert _paths(
        (BlockKind.PARAGRAPH, "Intro.", None),
        (BlockKind.HEADING, "Guide", 1),
    ) == [(), ("Guide",)]


def test_heading_closes_the_sections_of_its_level_and_deeper() -> None:
    """A heading drops every open section that is as deep, or deeper."""
    assert _paths(
        (BlockKind.HEADING, "Guide", 1),
        (BlockKind.HEADING, "Install", 2),
        (BlockKind.HEADING, "Linux", 3),
        (BlockKind.HEADING, "Use", 2),
        (BlockKind.HEADING, "Reference", 1),
        (BlockKind.PARAGRAPH, "Text.", None),
    ) == [
        ("Guide",),
        ("Guide", "Install"),
        ("Guide", "Install", "Linux"),
        ("Guide", "Use"),
        ("Reference",),
        ("Reference",),
    ]


def test_skipped_heading_level_nests_under_the_open_section() -> None:
    """A heading two levels down nests right under the section above it."""
    assert _paths(
        (BlockKind.HEADING, "Guide", 1),
        (BlockKind.HEADING, "Details", 3),
    ) == [("Guide",), ("Guide", "Details")]


def test_heading_without_level_does_not_open_a_section() -> None:
    """A heading with no level sits in its section but opens none."""
    assert _paths(
        (BlockKind.HEADING, "Guide", 1),
        (BlockKind.HEADING, "Loose", None),
        (BlockKind.PARAGRAPH, "Text.", None),
    ) == [("Guide",), ("Guide",), ("Guide",)]


def test_default_admonition_titles() -> None:
    """The default titles of admonitions are shared by HTML and PDF."""
    assert {"note", "warning", "tip", "see also"} <= DEFAULT_ADMONITION_TITLES
