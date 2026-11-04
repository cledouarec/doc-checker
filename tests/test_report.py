# Copyright 2026 Christophe Le Douarec
"""Tests for the Markdown report and the terminal summary."""

from typing import TYPE_CHECKING

import click
import pytest

from doc_checker.compare.engine import CompareResult, compare_documents
from doc_checker.config import Config
from doc_checker.errors import ConfigError
from doc_checker.model.findings import DiffSegment
from doc_checker.pipeline import write_reports
from doc_checker.report.markdown import md_cell, relative_link, render_markdown
from doc_checker.report.terminal import render_summary
from tests.builders import PARAGRAPHS, make_block, make_document

if TYPE_CHECKING:
    from pathlib import Path

REF_TEXT = (
    "Install the package with the default configuration before you start."
)


@pytest.fixture
def result() -> CompareResult:
    """Compare documents with one missing block and one typo."""
    ref = make_document(make_block(REF_TEXT), make_block(PARAGRAPHS[0]))
    new = make_document(
        make_block(REF_TEXT.replace("configuration", "configuraiton")),
    )
    return compare_documents(ref, new, Config())


def test_markdown_report_sections(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The report has its sections and the highlighted differences."""
    path = render_markdown(result, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# doc-checker report: compare")
    for expected in (
        "## Inputs",
        "## Summary",
        "## Critical (1)",
        "### Block missing (`block.missing`)",
        "## Major (1)",
        "<ins>configuraiton</ins>",
        "<del>configuration</del>",
    ):
        assert expected in text


def test_report_without_findings(tmp_path: Path) -> None:
    """A comparison without findings says so."""
    doc = make_document(make_block(PARAGRAPHS[0]))
    same = compare_documents(doc, doc, Config())
    path = render_markdown(same, tmp_path / "r.md")
    assert "No findings." in path.read_text(encoding="utf-8")


def test_md_cell_escapes() -> None:
    """Table cells escape pipes, markup and newlines."""
    segments = (
        DiffSegment("equal", "a|b <c>\nd"),
        DiffSegment("delete", "x"),
    )
    assert md_cell(segments) == "a\\|b &lt;c&gt;<br>d<del>x</del>"


def test_relative_link_non_ascii(tmp_path: Path) -> None:
    """Links are relative to the report and percent encoded."""
    target = tmp_path / "ref" / "guide" / "café.html"
    link = relative_link(target, "x", tmp_path / "out")
    assert link == "../ref/guide/caf%C3%A9.html#x"


def test_terminal_summary(tmp_path: Path, result: CompareResult) -> None:
    """The terminal summary counts findings and lists critical ones."""
    text = click.unstyle(render_summary(result, [tmp_path / "report.md"]))
    assert "findings: 1 critical, 1 major, 0 minor, 0 info" in text
    assert "CRITICAL block.missing" in text
    assert "reports: " in text


def test_write_reports_creates_output_dir(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """Reports are written in a newly created output directory."""
    cfg = Config(output=tmp_path / "out" / "nested")
    (path,) = write_reports(result, cfg)
    assert path == tmp_path / "out" / "nested" / "report.md"
    assert path.is_file()


def test_unwritable_report_is_a_config_error(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """A report file that cannot be written is a configuration error."""
    out = tmp_path / "out"
    (out / "report.md").mkdir(parents=True)
    with pytest.raises(ConfigError, match="cannot write reports"):
        write_reports(result, Config(output=out))
