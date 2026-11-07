# Copyright 2026 Christophe Le Douarec
"""Detect the page chrome repeated at the edges of the pages of a site."""

import math
from collections import Counter
from typing import TYPE_CHECKING, Final

from doc_checker.extract.common import CHROME_MIN_PAGES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.model.document import Block

CHROME_MIN_SHARE: Final = 0.5
CHROME_EDGE_BLOCKS: Final = 5


def find_chrome(pages: Sequence[Sequence[Block]]) -> frozenset[str]:
    """Return the strict keys of the blocks repeated at page edges.

    A block is chrome when it appears among the first or last
    ``CHROME_EDGE_BLOCKS`` blocks of at least
    ``max(CHROME_MIN_PAGES, ceil(CHROME_MIN_SHARE * pages))`` pages.

    Args:
        pages: Blocks of each page, in page order.

    Returns:
        The strict keys of the chrome blocks.
    """
    if len(pages) < CHROME_MIN_PAGES:
        return frozenset()
    counts: Counter[str] = Counter()
    for blocks in pages:
        edges = [
            *blocks[:CHROME_EDGE_BLOCKS],
            *blocks[-CHROME_EDGE_BLOCKS:],
        ]
        counts.update({block.key_strict for block in edges})
    threshold = max(
        CHROME_MIN_PAGES,
        math.ceil(CHROME_MIN_SHARE * len(pages)),
    )
    return frozenset(
        key for key, count in counts.items() if count >= threshold
    )
