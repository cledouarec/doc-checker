# Copyright 2026 Christophe Le Douarec
"""Tests for the diff of two texts of lines, line by line first."""

import pytest

from doc_checker.compare.classify import HunkContext, classify_hunk
from doc_checker.compare.textdiff import Hunk, diff_tokens
from doc_checker.text.tokenize import Token, tokenize

MODULE = "doc_checker.compare.textdiff"


def _lined(*lines: str) -> tuple[str, list[Token], list[int]]:
    """Return a text of lines, its tokens and the first token of each line."""
    text = "\n".join(lines)
    tokens = tokenize(text)
    starts: list[int] = []
    count = 0
    for line in lines:
        starts.append(count)
        count += len(tokenize(line))
    return text, tokens, starts


ROWS = tuple(
    f"P-{row:03d} | valve | {row % 7} | 1.{row:02d}" for row in range(40)
)


def test_a_changed_cell_is_one_hunk_by_lines() -> None:
    """Lines diffed first still find the one cell that changed."""
    changed = (*ROWS[:10], ROWS[10].replace("| 3 |", "| 4 |"), *ROWS[11:])
    _, ref, ref_lines = _lined(*ROWS)
    _, new, new_lines = _lined(*changed)
    hunks = diff_tokens(ref, new, lines=(ref_lines, new_lines))
    assert hunks == diff_tokens(ref, new)
    (hunk,) = hunks
    assert (ref[hunk.ref_start].text, new[hunk.new_start].text) == ("3", "4")


def test_lines_removed_and_added_are_one_hunk_each() -> None:
    """A run of lines removed, and one added, are each one hunk."""
    changed = (*ROWS[:5], *ROWS[8:30], "P-999 | pump | 1 | 2.00", *ROWS[30:])
    _, ref, ref_lines = _lined(*ROWS)
    _, new, new_lines = _lined(*changed)
    hunks = diff_tokens(ref, new, lines=(ref_lines, new_lines))
    assert [
        (h.op, h.ref_end - h.ref_start, h.new_end - h.new_start) for h in hunks
    ] == [
        ("delete", 21, 0),
        ("insert", 0, 7),
    ]
    assert not any(hunk.whole for hunk in hunks)


def test_a_long_run_of_replaced_lines_is_one_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replaced lines past the bound on their tokens are one whole hunk."""
    monkeypatch.setattr(f"{MODULE}.RUN_MAX_CELLS", 100)
    changed = (
        *ROWS[:10],
        *(row.replace("| 1.", "| 2.") for row in ROWS[10:20]),
        *ROWS[20:],
    )
    ref_text, ref, ref_lines = _lined(*ROWS)
    new_text, new, new_lines = _lined(*changed)
    (hunk,) = diff_tokens(ref, new, lines=(ref_lines, new_lines))
    # The run starts at the price of row 10, its first token that differs.
    assert hunk == Hunk("replace", 76, 140, 76, 140, whole=True)
    ctx = HunkContext(ref_text, new_text, ref, new, frozenset(), "en", "en")
    assert classify_hunk(hunk, ctx).rule == "text.modified"


@pytest.mark.parametrize(("gap", "whole"), [(1_000_000, False), (1, True)])
def test_a_gap_past_its_bound_is_one_run_of_lines(
    monkeypatch: pytest.MonkeyPatch,
    gap: int,
    whole: bool,  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
) -> None:
    """Lines between two unique lines past the bound form one run."""
    monkeypatch.setattr(f"{MODULE}.GAP_MAX_CELLS", gap)
    monkeypatch.setattr(f"{MODULE}.RUN_MAX_CELLS", 100)
    # Row 11 repeats row 0: neither is a unique line.
    rows = (*ROWS[:11], ROWS[0], *ROWS[12:])
    changed = (
        *rows[:10],
        rows[10].replace("| 3 |", "| 4 |"),
        rows[11],
        rows[12].replace("| 5 |", "| 6 |"),
        *rows[13:],
    )
    _, ref, ref_lines = _lined(*rows)
    _, new, new_lines = _lined(*changed)
    hunks = diff_tokens(ref, new, lines=(ref_lines, new_lines))
    if whole:
        # From the count of row 10 to that of row 12.
        assert hunks == [Hunk("replace", 74, 89, 74, 89, whole=True)]
    else:
        assert [new[h.new_start].text for h in hunks] == ["4", "6"]


def test_repeated_lines_are_diffed_between_unique_ones() -> None:
    """Lines found many times are diffed in the gaps between unique lines."""
    lines = ("begin", *(["}"] * 30), "middle", *(["}"] * 30), "end")
    changed = ("begin", *(["}"] * 29), "middle", *(["}"] * 31), "end")
    _, ref, ref_lines = _lined(*lines)
    _, new, new_lines = _lined(*changed)
    hunks = diff_tokens(ref, new, lines=(ref_lines, new_lines))
    assert [
        (h.op, h.ref_end - h.ref_start, h.new_end - h.new_start) for h in hunks
    ] == [
        ("delete", 1, 0),
        ("insert", 0, 1),
    ]


def test_equal_tokens_of_replaced_lines_are_no_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines cut otherwise around the same tokens give no hunk."""
    monkeypatch.setattr(f"{MODULE}.RUN_MAX_CELLS", 100)
    rows = [row.replace(" | ", " ") for row in ROWS]
    _, ref, ref_lines = _lined(*rows)
    _, new, new_lines = _lined(" ".join(rows[:20]), " ".join(rows[20:]))
    assert diff_tokens(ref, new, lines=(ref_lines, new_lines)) == []
