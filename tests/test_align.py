# Copyright 2026 Christophe Le Douarec
"""Tests for the global alignment of blocks."""

from typing import TYPE_CHECKING

import doc_checker.compare.align as align_module
from doc_checker.compare.align import Pair, align
from doc_checker.model.document import Block, BlockKind
from tests.builders import SPLIT_PARTS, SPLIT_WHOLE, make_block

if TYPE_CHECKING:
    import pytest

P = [
    f"Paragraph number {w} talks about topic {w}."
    for w in ("one", "two", "three", "four", "five", "six")
]

type Kind = tuple[str, tuple[int, ...], tuple[int, ...], bool]


def _b(*texts: str) -> list[Block]:
    return [make_block(t) for t in texts]


def _kinds(pairs: list[Pair]) -> list[Kind]:
    return [(p.kind.value, p.ref, p.new, p.moved) for p in pairs]


def _align(ref: list[Block], new: list[Block]) -> list[Pair]:
    return align(ref, new, match=0.6, moved_match=0.8)


def test_identical_documents() -> None:
    """Identical documents align block by block."""
    assert _kinds(_align(_b(*P), _b(*P))) == [
        ("equal", (i,), (i,), False) for i in range(6)
    ]


def test_deleted_and_inserted() -> None:
    """Removed and added blocks are reported as such."""
    new = _b(*P[:2], *P[3:], "A brand new paragraph appears right here.")
    kinds = _kinds(_align(_b(*P), new))
    assert ("deleted", (2,), (), False) in kinds
    assert ("inserted", (), (5,), False) in kinds


def test_modified_block() -> None:
    """A slightly changed block is paired as modified."""
    new = _b(P[0], "Paragraph number two talks about subject two.", *P[2:])
    assert ("modified", (1,), (1,), False) in _kinds(_align(_b(*P), new))


def test_moved_block() -> None:
    """A block that changed place is paired as moved."""
    kinds = _kinds(_align(_b(*P), _b(*P[1:], P[0])))
    assert ("equal", (0,), (5,), True) in kinds
    assert not [k for k in kinds if k[0] in {"deleted", "inserted"}]


def test_split_and_merge() -> None:
    """A block split in two, or two blocks merged, are paired."""
    split = _align(_b(P[0], SPLIT_WHOLE, P[1]), _b(P[0], *SPLIT_PARTS, P[1]))
    assert ("split", (1,), (1, 2), False) in _kinds(split)
    merge = _align(_b(P[0], *SPLIT_PARTS, P[1]), _b(P[0], SPLIT_WHOLE, P[1]))
    assert ("merge", (1, 2), (1,), False) in _kinds(merge)


def test_moved_and_modified_block() -> None:
    """A block that moved and changed is paired by the leftover pass."""
    new = _b(*P[1:], "Paragraph number one talks about subject one.")
    assert ("modified", (0,), (5,), True) in _kinds(_align(_b(*P), new))


def test_repeated_short_blocks() -> None:
    """Repeated short blocks align in order."""
    blocks = _b("Note", P[0], "Note", P[1], "Note")
    same = _b("Note", P[0], "Note", P[1], "Note")
    assert {k[0] for k in _kinds(_align(blocks, same))} == {"equal"}


def test_empty_sides() -> None:
    """An empty side gives only insertions or deletions."""
    assert _kinds(_align([], _b(*P[:2]))) == [
        ("inserted", (), (0,), False),
        ("inserted", (), (1,), False),
    ]
    assert _kinds(_align(_b(*P[:2]), [])) == [
        ("deleted", (0,), (), False),
        ("deleted", (1,), (), False),
    ]


def test_greedy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large gaps are aligned greedily."""
    monkeypatch.setattr(align_module, "MAX_DP_CELLS", 4)
    items = [f"Item {i}" for i in range(30)]
    assert _kinds(_align(_b(*items), _b(*items))) == [
        ("equal", (i,), (i,), False) for i in range(30)
    ]


def test_long_code_block() -> None:
    """A very long code block with one changed line is modified."""
    lines = [f"line_{i} = {i}" for i in range(3000)]
    ref = [make_block("\n".join(lines), BlockKind.CODE)]
    lines[1500] = "line_1500 = -1"
    new = [make_block("\n".join(lines), BlockKind.CODE)]
    assert _kinds(_align(ref, new)) == [("modified", (0,), (0,), False)]


def test_heading_matches_paragraph_with_same_text() -> None:
    """A heading turned into a paragraph is still paired."""
    ref = [
        make_block(
            "Configuration options for the server",
            BlockKind.HEADING,
            level=2,
        ),
    ]
    new = [make_block("Configuration options for the server")]
    assert _kinds(_align(ref, new)) == [("equal", (0,), (0,), False)]
