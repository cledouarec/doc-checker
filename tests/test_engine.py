# Copyright 2026 Christophe Le Douarec
"""Tests for the comparison engine."""

from typing import TYPE_CHECKING

import pytest

from doc_checker.compare.engine import (
    CompareResult,
    CompareStats,
    compare_documents,
)
from doc_checker.config import Config, IgnoreConfig
from doc_checker.errors import InputError
from doc_checker.model.document import Block, BlockKind
from doc_checker.model.findings import DiffSegment, Severity
from doc_checker.pipeline import run_compare
from tests.builders import (
    PARAGRAPHS,
    SAMPLE_EN,
    SPLIT_PARTS,
    SPLIT_WHOLE,
    build_mkdocs_like,
    make_block,
    make_document,
)

if TYPE_CHECKING:
    from pathlib import Path

P = PARAGRAPHS
ADDED = "An extra paragraph was added here."


def _rules(result: CompareResult) -> list[tuple[str, Severity]]:
    return [(f.rule, f.severity) for f in result.findings]


def _b(*texts: str) -> list[Block]:
    return [make_block(t) for t in texts]


def _heading(text: str, level: int) -> Block:
    return make_block(text, BlockKind.HEADING, level=level)


def test_identical_documents_have_no_findings() -> None:
    """Identical documents produce no finding."""
    ref, new = make_document(*_b(*P)), make_document(*_b(*P))
    assert compare_documents(ref, new, Config()).findings == []


def test_missing_block_is_critical() -> None:
    """A missing block is a critical finding with its reference text."""
    ref = make_document(*_b(*P))
    new = make_document(*_b(P[0], P[2], P[3]))
    result = compare_documents(ref, new, Config())
    assert _rules(result) == [("block.missing", Severity.CRITICAL)]
    assert result.findings[0].ref is not None
    assert result.findings[0].ref.text == P[1]
    assert result.findings[0].new is None


def test_missing_section_is_one_finding() -> None:
    """A section missing with all its content is a single finding."""
    ref = make_document(
        _heading("Intro chapter", 1),
        make_block(P[0]),
        _heading("Old section", 2),
        make_block(P[1]),
        make_block(P[2]),
        _heading("Kept section", 2),
        make_block(P[3]),
    )
    new = make_document(
        _heading("Intro chapter", 1),
        make_block(P[0]),
        _heading("Kept section", 2),
        make_block(P[3]),
    )
    result = compare_documents(ref, new, Config())
    assert _rules(result) == [("section.missing", Severity.CRITICAL)]
    assert result.findings[0].detail == "3 blocks missing in this section"


def test_added_and_moved_blocks() -> None:
    """Added blocks are minor and moved blocks are informational."""
    ref = make_document(*_b(*P))
    added = compare_documents(ref, make_document(*_b(*P, ADDED)), Config())
    assert _rules(added) == [("block.added", Severity.MINOR)]
    moved_new = make_document(*_b(*P[1:], P[0]))
    moved = compare_documents(ref, moved_new, Config())
    assert _rules(moved) == [("block.moved", Severity.INFO)]


def test_hunk_finding_has_diff_segments() -> None:
    """A word level finding shows the deleted and inserted words."""
    text = (
        "Install the package with the default configuration before you start."
    )
    ref = make_document(make_block(text))
    new = make_document(
        make_block(text.replace("configuration", "configuraiton")),
    )
    (finding,) = compare_documents(ref, new, Config()).findings
    assert (finding.rule, finding.severity) == ("text.typo", Severity.MAJOR)
    assert finding.ref is not None
    assert finding.new is not None
    assert DiffSegment("delete", "configuration") in finding.ref.segments
    assert DiffSegment("insert", "configuraiton") in finding.new.segments


def test_split_is_info_only() -> None:
    """A block split in two without text change is informational."""
    ref = make_document(*_b(P[0], SPLIT_WHOLE, P[1]))
    new = make_document(*_b(P[0], *SPLIT_PARTS, P[1]))
    assert _rules(compare_documents(ref, new, Config())) == [
        ("block.split_merged", Severity.INFO),
    ]


def test_punctuation_only_blocks_are_ignored() -> None:
    """Blocks without letters or digits are not compared."""
    ref = make_document(*_b(P[0], "\U00002014"))
    new = make_document(*_b(P[0]))
    assert compare_documents(ref, new, Config()).findings == []


def test_chrome_keys_are_removed_on_both_sides() -> None:
    """Chrome found in one document is ignored in both."""
    banner = make_block("Docs for version 2.0.")
    ref = make_document(banner, *_b(P[0]))
    new = make_document(
        *_b(P[0]),
        chrome_keys=frozenset({banner.key_strict}),
    )
    assert compare_documents(ref, new, Config()).findings == []


def test_severity_override_disable_and_min_severity() -> None:
    """Severity overrides, disabled rules and the minimum severity apply."""
    ref, new = make_document(*_b(*P)), make_document(*_b(*P, ADDED))
    override = Config(severity={"block.added": Severity.INFO})
    assert _rules(compare_documents(ref, new, override)) == [
        ("block.added", Severity.INFO),
    ]
    disabled = Config(ignore=IgnoreConfig(rules=("block.added",)))
    assert compare_documents(ref, new, disabled).findings == []
    minimum = Config(min_severity=Severity.MAJOR)
    assert compare_documents(ref, new, minimum).findings == []


def test_ignore_patterns_and_finding_ids() -> None:
    """Ignored patterns and ignored finding identifiers are honored."""
    ref = make_document(
        make_block("Version 1.2.3 is installed by default here."),
    )
    new = make_document(
        make_block("Version 1.2.4 is installed by default here."),
    )
    patterns = Config(ignore=IgnoreConfig(patterns=(r"\d+\.\d+\.\d+",)))
    assert compare_documents(ref, new, patterns).findings == []
    (finding,) = compare_documents(ref, new, Config()).findings
    ignored = Config(ignore=IgnoreConfig(findings=(finding.id,)))
    assert compare_documents(ref, new, ignored).findings == []


def test_stats() -> None:
    """Statistics count blocks, matches and moves."""
    ref = make_document(*_b(*P))
    new = make_document(*_b(*P[1:], P[0]))
    stats = compare_documents(ref, new, Config()).stats
    assert stats == CompareStats(
        ref_blocks=4,
        new_blocks=4,
        matched=4,
        moved=1,
    )


def test_run_compare_rejects_mixed_formats(tmp_path: Path) -> None:
    """Comparing an HTML site with a PDF is an input error."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    with pytest.raises(InputError, match="same format"):
        run_compare(site, pdf, Config())
