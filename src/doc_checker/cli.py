# Copyright 2026 Christophe Le Douarec
"""Command line interface for Doc Checker."""

import argparse

from doc_checker import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the doc-checker command line interface.

    Args:
        argv: Command line arguments, or ``None`` to use ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="doc-checker",
        description="Check documentation format",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.parse_args(argv)
    return 0
