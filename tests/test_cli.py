# Copyright 2026 Christophe Le Douarec
"""Tests for the command line interface."""

import pytest

from doc_checker.cli import main


def test_main_without_arguments_succeeds() -> None:
    """Running the CLI without arguments returns a success exit code."""
    assert main([]) == 0


def test_version_flag_exits() -> None:
    """The ``--version`` flag prints the version and exits."""
    with pytest.raises(SystemExit):
        main(["--version"])
