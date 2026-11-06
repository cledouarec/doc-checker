# Copyright 2026 Christophe Le Douarec
"""Tests for the block building shared by the HTML and PDF loaders."""

from collections import Counter
from types import MappingProxyType

import pytest

from doc_checker.extract.common import (
    NUMBERED_MIN,
    NUMBERED_SHARE,
    assign_section_paths,
    normalized_block,
    strip_heading_numbers,
)
from doc_checker.model.document import Block, BlockKind, SourceRef

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


def _heading(text: str, level: int = 1) -> Block:
    """Make a heading of a PDF page."""
    return normalized_block(
        BlockKind.HEADING,
        text,
        SOURCE,
        Counter(),
        level=level,
    )


def _paragraph(text: str) -> Block:
    """Make a paragraph of a PDF page."""
    return normalized_block(BlockKind.PARAGRAPH, text, SOURCE, Counter())


def test_numbers_of_a_numbered_document_are_set_apart() -> None:
    """The numbers of the headings leave their texts and are kept apart."""
    counter: Counter[str] = Counter()
    blocks = [
        _heading("1 Getting started"),
        _heading("1.1. Installation", 2),
        _paragraph("Text of the section."),
        _heading("A.1 Annex", 2),
        _heading("Unnumbered", 2),
    ]
    result = strip_heading_numbers(blocks, counter)
    assert [(b.text, b.attrs.get("number")) for b in result] == [
        ("Getting started", "1"),
        ("Installation", "1.1"),
        ("Text of the section.", None),
        ("Annex", "A.1"),
        ("Unnumbered", None),
    ]
    assert (result[1].key_strict, result[1].raw_text) == (
        "Installation",
        "1.1. Installation",
    )
    assert counter["heading_number"] == 3


@pytest.mark.parametrize(
    ("extra", "numbers"),
    [(0, ["1", "2", "3", "4"]), (1, [None, None, None, None])],
    ids=["at-the-share", "below-the-share"],
)
def test_numbered_share_decides_a_numbered_document(
    extra: int,
    numbers: list[str | None],
) -> None:
    """Numbers leave the headings at the numbered share, not below it."""
    numbered = [_heading(f"{n} Step {n}") for n in range(1, 5)]
    assert len(numbered) > NUMBERED_MIN
    plain = round(len(numbered) / NUMBERED_SHARE) - len(numbered) + extra
    others = [
        _heading(f"Unnumbered topic {chr(97 + i)}") for i in range(plain)
    ]
    result = strip_heading_numbers([*numbered, *others], Counter())
    assert [b.attrs.get("number") for b in result[:4]] == numbers


def test_numbers_stay_where_headings_are_not_numbered() -> None:
    """A document that does not number its headings keeps their numbers."""
    blocks = [
        _heading("2026 roadmap"),
        _heading("Goals", 2),
        _heading("Risks", 2),
        _heading("3 ways to start", 2),
    ]
    assert strip_heading_numbers(blocks, Counter()) == blocks


def test_a_chapter_label_becomes_the_number_of_its_heading() -> None:
    """A chapter label before a heading is the number of that heading."""
    counter: Counter[str] = Counter()
    blocks = [
        _paragraph("Chapter 2"),
        _heading("Getting started"),
        _paragraph("Chapter 3"),
        _paragraph("Not a heading after it."),
    ]
    result = strip_heading_numbers(blocks, counter)
    assert [(b.text, b.attrs.get("number")) for b in result] == [
        ("Getting started", "Chapter 2"),
        ("Chapter 3", None),
        ("Not a heading after it.", None),
    ]
    assert counter["heading_number"] == 1


def test_a_chapter_label_set_as_a_heading_is_taken_too() -> None:
    """A chapter label that is a heading goes, and attrs are merged."""
    counter: Counter[str] = Counter()
    title = normalized_block(
        BlockKind.HEADING,
        "Getting started",
        SOURCE,
        counter,
        level=2,
        attrs={"heading_source": "style"},
    )
    result = strip_heading_numbers([_heading("Chapter 1", 2), title], counter)
    assert [b.text for b in result] == ["Getting started"]
    assert result[0].attrs == {
        "heading_source": "style",
        "number": "Chapter 1",
    }
