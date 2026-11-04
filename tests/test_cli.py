# Copyright 2026 Christophe Le Douarec
"""Tests for the command line interface."""

from click.testing import CliRunner

from doc_checker import __version__
from doc_checker.cli import main


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
