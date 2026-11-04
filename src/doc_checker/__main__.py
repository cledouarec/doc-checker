# Copyright 2026 Christophe Le Douarec
"""Entry point for ``python -m doc_checker``."""

import sys

from doc_checker.cli import main

if __name__ == "__main__":
    sys.exit(main())
