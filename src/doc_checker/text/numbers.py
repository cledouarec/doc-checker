# Copyright 2026 Christophe Le Douarec
"""Language aware canonical form of numbers, and the numbers of headings."""

import re
from typing import Final

# The number that starts the title of a heading, in its group 1: numbers
# joined by dots, such as ``2.1.3``, or a capital letter followed by them,
# such as ``A.1`` in an appendix. A dot may follow it, then blanks and the
# title, which the match leaves out.
HEADING_NUMBER: Final = re.compile(
    r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\.?\s+(?=\S)",
)

_GROUPING_SPACES: Final = (" ", "\xa0", "\U0000202f")
_GROUP_DIGITS: Final = 3


def canonical_number(text: str, lang: str | None) -> str:
    """Return a language independent form of a number token.

    Grouping separators are removed and the decimal separator becomes a dot.
    Texts with several dots, such as versions, are kept as they are because
    ``3.10`` and ``3.1`` are different versions.

    Args:
        text: Number token, such as ``1 000,50`` or ``1,000.50``.
        lang: Language of the document, which decides what a single comma
            means.

    Returns:
        The canonical form, such as ``1000.50``.
    """
    for space in _GROUPING_SPACES:
        text = text.replace(space, "")
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        grouping = "." if decimal == "," else ","
        return text.replace(grouping, "").replace(decimal, ".")
    if "," in text:
        return _comma_number(text, lang)
    return text


def _comma_number(text: str, lang: str | None) -> str:
    """Return the canonical form of a number that only contains commas.

    Args:
        text: Number token without grouping spaces.
        lang: Language of the document.

    Returns:
        The canonical form.
    """
    if text.count(",") > 1:
        return text.replace(",", "")
    before, after = text.split(",")
    if lang == "fr":
        is_decimal = True
    elif lang == "en":
        is_decimal = len(after) != _GROUP_DIGITS
    else:
        is_decimal = not (
            len(after) == _GROUP_DIGITS and len(before) <= _GROUP_DIGITS
        )
    return f"{before}.{after}" if is_decimal else before + after
