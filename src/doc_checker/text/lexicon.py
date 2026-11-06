# Copyright 2026 Christophe Le Douarec
"""French and English words whose change alters the meaning of a text.

The module also knows the words that a generator adds to a text: the
default titles of admonitions and the labels of chapters.
"""

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

NEGATIONS: Final[frozenset[str]] = frozenset(
    {
        "ne",
        "n'",
        "pas",
        "jamais",
        "aucun",
        "aucune",
        "rien",
        "sans",
        "ni",
        "non",
        "nullement",
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "without",
        "nor",
        "neither",
        "cannot",
    },
)

MODALS: Final[frozenset[str]] = frozenset(
    {
        "doit",
        "doivent",
        "devez",
        "devons",
        "dois",
        "devrait",
        "devraient",
        "devriez",
        "faut",
        "peut",
        "peuvent",
        "pouvez",
        "pouvons",
        "peux",
        "pourrait",
        "pourraient",
        "pourriez",
        "obligatoire",
        "obligatoires",
        "facultatif",
        "facultative",
        "optionnel",
        "optionnelle",
        "requis",
        "requise",
        "recommandé",
        "recommandée",
        "must",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "required",
        "optional",
        "recommended",
        "mandatory",
    },
)

_TYPOGRAPHIC_APOSTROPHE: Final = "\U00002019"
_NEGATIVE_CONTRACTION: Final = "n't"


def _fold(word: str) -> str:
    """Return the casefolded word with a straight apostrophe."""
    return word.casefold().replace(_TYPOGRAPHIC_APOSTROPHE, "'")


def is_negation(word: str) -> bool:
    """Tell whether a word expresses a negation.

    Args:
        word: A single word token.

    Returns:
        Whether the word is a negation marker or a negative contraction.
    """
    folded = _fold(word)
    return folded in NEGATIONS or folded.endswith(_NEGATIVE_CONTRACTION)


def negation_count(words: Iterable[str]) -> int:
    """Count the negation markers among words.

    Args:
        words: Word tokens.

    Returns:
        The number of negation markers.
    """
    return sum(is_negation(word) for word in words)


def is_modal(word: str) -> bool:
    """Tell whether a word expresses an obligation or a possibility.

    Args:
        word: A single word token.

    Returns:
        Whether the word is a modal.
    """
    return _fold(word) in MODALS


# The default titles of admonitions, as a text is compared with them: case
# folded, and without a colon at the end.
DEFAULT_ADMONITION_TITLES: Final[frozenset[str]] = frozenset(
    {
        "note",
        "notes",
        "warning",
        "tip",
        "hint",
        "important",
        "caution",
        "danger",
        "attention",
        "error",
        "info",
        "example",
        "see also",
        "todo",
        "abstract",
        "summary",
        "success",
        "question",
        "failure",
        "bug",
        "quote",
        "remarque",
        "avertissement",
        "astuce",
        "erreur",
        "conseil",
        "exemple",
        "voir aussi",
        "à faire",
    },
)


def is_admonition_title(text: str) -> bool:
    """Tell whether a text is the default title of an admonition.

    The case is ignored, and so is a colon after the title.

    Args:
        text: The text of a block.

    Returns:
        Whether the text is one of the default titles.
    """
    return text.strip().casefold().rstrip(":").strip() in (
        DEFAULT_ADMONITION_TITLES
    )


# A label that a generator sets above the title of a chapter, a part or an
# appendix, such as ``Chapter 3`` or ``Annexe B``.
CHAPTER_LABEL: Final = re.compile(
    r"^(?:chapter|chapitre|part|partie|appendix|annexe)"
    r"\s+(?:\d+|[ivxlc]+|[a-z])$",
    re.IGNORECASE,
)


def is_chapter_label(text: str) -> bool:
    """Tell whether a text is a chapter label, such as ``Chapter 3``.

    Args:
        text: The text of a block.

    Returns:
        Whether the whole text is a label word followed by a number, a roman
        numeral or a letter.
    """
    return CHAPTER_LABEL.fullmatch(text) is not None
