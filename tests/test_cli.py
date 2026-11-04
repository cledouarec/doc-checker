# Copyright 2026 Christophe Le Douarec
"""Tests for the command line interface."""

from typing import TYPE_CHECKING

from click.testing import CliRunner

from doc_checker import __version__
from doc_checker.cli import main
from tests.builders import SAMPLE_EN, build_mkdocs_like

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
