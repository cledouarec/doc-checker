# Copyright 2026 Christophe Le Douarec
"""French and English words whose change alters the meaning of a text."""

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
