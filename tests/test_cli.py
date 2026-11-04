# Copyright 2026 Christophe Le Douarec
"""Tests for the command line interface."""

from typing import TYPE_CHECKING

from click.testing import CliRunner

from doc_checker import __version__
from doc_checker.cli import main
from tests.builders import SAMPLE_EN, build_mkdocs_like, build_sphinx_like

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
