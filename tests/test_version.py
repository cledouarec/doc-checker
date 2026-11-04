# Copyright 2026 Christophe Le Douarec
"""Tests for the package metadata."""

import re

from doc_checker import __version__


def test_version_is_semantic() -> None:
    """The package exposes an ``X.Y.Z`` version string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
