# Copyright 2026 Christophe Le Douarec
"""Tests for the command line interface."""

from typing import TYPE_CHECKING

from click.testing import CliRunner

from doc_checker import __version__
from doc_checker.cli import main
from tests.builders import (
    SAMPLE_EN,
    build_mkdocs_like,
    build_sphinx_like,
    replace_text,
)
from tests.pdf_builders import (
    build_fpdf2_pdf,
    build_raw_pdf,
    build_reportlab_pdf,
    build_scanned_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_main_without_arguments_shows_help() -> None:
    """Running the CLI without a command prints the help and succeeds."""
    result = CliRunner().invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_version_flag_prints_version() -> None:
    """The ``--version`` flag prints the package version."""
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_unknown_command_is_usage_error() -> None:
    """An unknown command is reported as a usage error."""
    result = CliRunner().invoke(main, ["nope"])
    assert result.exit_code == 2


def test_extract_lists_blocks(tmp_path: Path) -> None:
    """The extract command prints the normalized blocks."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["extract", str(site)])
    assert result.exit_code == 0, result.output
    assert "heading" in result.output
    assert "Install the package" in result.output


def test_extract_pdf_lists_blocks(tmp_path: Path) -> None:
    """The extract command prints the blocks of a PDF with their pages."""
    pdf = build_reportlab_pdf(SAMPLE_EN, tmp_path / "doc.pdf")
    result = CliRunner().invoke(main, ["extract", str(pdf)])
    assert result.exit_code == 0, result.output
    assert ": pdf, 9 page(s), generator ReportLab" in result.output
    assert "tagged no" in result.output
    assert "doc.pdf#page=2" in result.output


def test_extract_password_protected_pdf_exits_3(tmp_path: Path) -> None:
    """A PDF protected by a user password exits with code 3."""
    pdf = build_raw_pdf(
        tmp_path / "secret.pdf",
        ["BT /F1 10 Tf 72 700 Td (Hi) Tj ET"],
        encrypt="secret",
    )
    result = CliRunner().invoke(main, ["extract", str(pdf)])
    assert result.exit_code == 3
    assert "password protected" in result.output


def test_extract_unsupported_input_exits_3(tmp_path: Path) -> None:
    """An unsupported input exits with code 3."""
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    result = CliRunner().invoke(main, ["extract", str(tmp_path / "notes.txt")])
    assert result.exit_code == 3
    assert "unsupported input" in result.output


def test_extract_empty_site_exits_3(tmp_path: Path) -> None:
    """A directory without pages exits with code 3."""
    result = CliRunner().invoke(main, ["extract", str(tmp_path)])
    assert result.exit_code == 3
    assert "no HTML pages" in result.output


def test_extract_bad_config_exits_2(tmp_path: Path) -> None:
    """An invalid configuration exits with code 2."""
    conf = tmp_path / "c.toml"
    conf.write_text("[general]\nlang = 'de'\n", encoding="utf-8")
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["extract", str(site), "-c", str(conf)])
    assert result.exit_code == 2


def test_compare_writes_markdown_report(tmp_path: Path) -> None:
    """The compare command writes the report and prints a summary."""
    ref = build_sphinx_like(SAMPLE_EN, tmp_path / "ref")
    new = build_mkdocs_like(SAMPLE_EN, tmp_path / "new")
    out = tmp_path / "out"
    args = ["compare", str(ref), str(new), "-o", str(out)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert (out / "report.md").is_file()
    assert "findings:" in result.output


def test_compare_mixed_formats_exits_3(tmp_path: Path) -> None:
    """Comparing builds of different formats exits with code 3."""
    ref = build_mkdocs_like(SAMPLE_EN, tmp_path / "ref")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    result = CliRunner().invoke(main, ["compare", str(ref), str(pdf)])
    assert result.exit_code == 3
    assert "same format" in result.output


def test_compare_pdf_writes_report_with_page_links(tmp_path: Path) -> None:
    """Comparing two PDFs writes a report that links to their pages."""
    ref = build_reportlab_pdf(SAMPLE_EN, tmp_path / "ref.pdf")
    new = build_fpdf2_pdf(
        replace_text(SAMPLE_EN, "100 requests", "10 requests"),
        tmp_path / "new.pdf",
    )
    out = tmp_path / "out"
    args = ["compare", str(ref), str(new), "-o", str(out)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "CRITICAL text.number_changed  ref.pdf#page=9" in result.output
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "[ref.pdf#page=9](../ref.pdf#page=9)" in report


def test_compare_pdf_with_html_exits_3(tmp_path: Path) -> None:
    """Comparing a PDF with an HTML build exits with code 3."""
    pdf = build_reportlab_pdf(SAMPLE_EN, tmp_path / "ref.pdf")
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["compare", str(pdf), str(site)])
    assert result.exit_code == 3
    assert "same format" in result.output


def test_compare_scanned_pdf_warns(tmp_path: Path) -> None:
    """Comparing a scanned PDF warns about its pages without text."""
    scan = build_scanned_pdf(tmp_path / "scan.pdf", pages=2)
    new = build_fpdf2_pdf(SAMPLE_EN, tmp_path / "new.pdf")
    args = ["compare", str(scan), str(new), "-o", str(tmp_path / "out")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert (
        "warning: ref: 2 page(s) without text, probably scanned: 1, 2"
        in result.output
    )


def test_compare_unknown_format_exits_2(tmp_path: Path) -> None:
    """An unknown report format is a usage error."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "s")
    args = ["compare", str(site), str(site), "--format", "pdf"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert "unknown format" in result.output


def test_compare_missing_path_exits_2(tmp_path: Path) -> None:
    """A missing input path is a usage error."""
    args = ["compare", str(tmp_path / "a"), str(tmp_path / "b")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert "does not exist" in result.output
